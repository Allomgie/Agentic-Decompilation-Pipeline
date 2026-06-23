"""reorder Experte (deterministisch wo moeglich, sonst Permuter-Befund).

reorder zerfaellt (Quellen + Empirie 2026-06) in ZWEI Unterarten:
- BLOCK/STATEMENT-ORDER: ganze Berechnungen in anderer Reihenfolge. C-HEBEL: Statements
  umsortieren (Thesis §3.6 Code Motion / §3.9 Store Positions = datenfluss-/reihenfolge-getrieben).
  -> deterministisch via STATEMENT-REORDER (adjazente Statements tauschen, orakel-gegated).
- SAVE-SCHEDULING: Platzierung von `sw ra`/Save-Instruktionen (±1). Code-Gen-Scheduling (ugen),
  KEIN C-Statement dafuer -> KEIN C-Hebel -> echter Permuter (decomp-permuter reorder-Pass).

Der Experte versucht die deterministische Scheibe (Statement-Reorder + kleine Perturbationen),
orakel-gegated (sicher per Konstruktion). Was er nicht loest, ist per Befund Permuter-Territorium.
Verifikation IMMER ueber modules/ido_compiler (PC-Freeze-Regel).
"""
import re


def _body_lines(c_code):
    i = c_code.find("{"); j = c_code.rfind("}")
    if i < 0 or j < 0 or j < i:
        return None
    return c_code[:i + 1], c_code[i + 1:j].splitlines(), c_code[j:]


def _is_simple_stmt(line):
    s = line.strip()
    if not s or s in ("{", "}"):
        return False
    if s.startswith(("return", "if", "else", "for", "while", "switch", "case", "default",
                     "goto", "do", "{", "}")):
        return False
    return s.endswith(";") and "=" in s or re.match(r"^[A-Za-z_]\w*\s*\(.*\)\s*;$", s) is not None


def statement_reorder_candidates(c_code):
    """Tausche adjazente einfache Statements (orakel-gegated -> nur korrekte Tausche ueberleben).
    Faengt die BLOCK/STATEMENT-ORDER-Unterart."""
    parts = _body_lines(c_code)
    if not parts:
        return []
    pre, body, suf = parts
    # Indizes einfacher Statements
    idxs = [k for k, l in enumerate(body) if _is_simple_stmt(l)]
    out = []
    for a, b in zip(idxs, idxs[1:]):
        if b != a + 1:
            continue  # nur direkt benachbart
        nb = body[:]
        nb[a], nb[b] = nb[b], nb[a]
        out.append((f"swap:{a}<->{b}", pre + "\n".join(nb) + "\n" + suf))
    return out


def temp_perturbation_candidates(c_code):
    """Kleine Live-Range-Perturbationen fuer Save-Scheduling-Residual (meist wirkungslos =
    Permuter-Befund; vollstaendigkeitshalber, orakel-gegated)."""
    parts = _body_lines(c_code)
    if not parts:
        return []
    pre, body, suf = parts
    out = []
    # ungenutzte Variable am Anfang (Allocations-/Scheduling-Druck)
    out.append(("pad_pre", pre + "\n  s32 _pad;\n" + "\n".join(body) + "\n" + suf))
    return out


def _calls(c_code):
    return list(re.finditer(r"([A-Za-z_]\w*)\s*\(([^;{}]*)\)\s*;", c_code))


def _split_args(s):
    out = []; depth = 0; cur = ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def stage2_schedule_candidates(c_code):
    """STUFE 2: diff-skopierter Scheduling-Permuter fuer das Save-Scheduling-Residual.
    Scheduling-beeinflussende Zuege (wie decomp-permuter, aber gezielt), orakel-gegated:
    - ARG-TEMPS: Call-Argumente in Temps materialisieren (aendert Prolog-Scheduling von
      Arg-Setup vs Save).  - Reihenfolge der Arg-Temps variieren.  - SAMELINE/SPLIT.
    Empirie zeigt: meist wirkungslos (Save-Scheduling = ugen, beyond C) -> ehrliches Permuter-
    Verdikt. Vollstaendigkeitshalber + falls einzelne Faelle doch ziehen."""
    out = []
    calls = _calls(c_code)
    for ci, m in enumerate(calls):
        fn, arglist = m.group(1), m.group(2)
        args = [a.strip() for a in _split_args(arglist)]
        if len(args) < 2:
            continue
        # ARG-TEMPS in normaler + umgekehrter Deklarations-Reihenfolge
        for order, tag in ((range(len(args)), "fwd"), (range(len(args) - 1, -1, -1), "rev")):
            decls = "".join(f"  s32 _t{k} = {args[k]};\n" for k in order)
            newcall = f"{fn}({', '.join('_t%d' % k for k in range(len(args)))});"
            ins = c_code[:m.start()] + decls.lstrip() + "  " + newcall + c_code[m.end():]
            out.append((f"argtemp_{tag}:{fn[:16]}", ins))
    return out


def reorder_candidates(c_code, stage2=False):
    gens = statement_reorder_candidates(c_code) + temp_perturbation_candidates(c_code)
    if stage2:
        gens = gens + stage2_schedule_candidates(c_code)
    seen = set(); out = []
    for lbl, code in gens:
        if code != c_code and code not in seen:
            seen.add(code); out.append((lbl, code))
    return out


def reorder_expert(c_code, func_name, target_s_path, count_fn=None, max_iter=6):
    """Iterativer orakel-gegateter Reorder-Reduzierer. Returns {applied, new_code, diffs_before,
    diffs_after, steps, verdict}. verdict='permuter' wenn kein deterministischer Zug greift."""
    before = count_fn(c_code, func_name, target_s_path)
    cur, cur_d = c_code, before
    steps = []
    for it in range(max_iter):
        if cur_d == 0:
            break
        # Stufe 1 (Statement-Reorder) zuerst; Stufe 2 (Scheduling-Permuter) nur wenn Stufe1 stockt
        best_code, best_d, best_lbl = None, cur_d, None
        for stage2 in (False, True):
            for lbl, cand in reorder_candidates(cur, stage2=stage2):
                d = count_fn(cand, func_name, target_s_path)
                if d is not None and (best_d is None or d < best_d):
                    best_code, best_d, best_lbl = cand, d, lbl
            if best_lbl is not None:
                break  # Stufe1-Treffer -> nicht erst Stufe2 brauchen
        if best_lbl is None:
            break
        cur, cur_d = best_code, best_d
        steps.append((best_lbl, best_d))
    applied = cur != c_code
    verdict = "solved" if cur_d == 0 else ("improved" if applied else "permuter")
    return {"applied": applied, "new_code": cur, "diffs_before": before,
            "diffs_after": cur_d, "steps": steps, "verdict": verdict}
