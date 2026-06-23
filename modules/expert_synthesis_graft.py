"""Synthese-Graft-Kern (FOUNDATION fuer die missing-block / instr-block Experten).

Wenn der Draft KOMPILIERT, aber eine Berechnung/Call FEHLT (missing-block), wird das fehlende
Stueck aus einem DONOR (m2c-vom-Target, spaeter AI) chirurgisch eingefuegt -- der gute AI-Draft
(IDO-naher Stil) bleibt erhalten, nur die Luecke wird gefuellt. Orakel-gegated.

Mechanismus (donor-agnostisch -- m2c ODER AI liefert donor_c):
  1. Donor-Namen an den Draft reconcilen: Parameter positionsweise (arg0->1.Param-Name),
     Globals/Callees sind ohnehin gleich (korrekt aus dem Target/Header).
  2. Donor-Body in Statements zerlegen; Kandidaten = Statements, die eine Funktion aufrufen, die
     im Draft NICHT vorkommt (= fehlend), bzw. die der Draft nicht hat.
  3. Jeden Kandidaten an jeder Body-Position einfuegen, ORAKEL (count_fn) waehlt den besten.
  Iterativ (mehrere fehlende Stuecke).

KEIN AI hier -- der Donor liefert den Code. Die zwei Synthese-Experten reichen statt m2c einen
AI-erzeugten Donor-Block; der Graft+Orakel-Mechanismus bleibt identisch.
"""
import re


def _sig_params(c_code, fn):
    """Parameter-Namen aus der Signatur von fn (in Reihenfolge)."""
    m = re.search(r"\b" + re.escape(fn) + r"\s*\(([^;{)]*)\)\s*\{", c_code)
    if not m:
        return []
    inside = m.group(1).strip()
    if inside in ("", "void"):
        return []
    out = []
    for p in inside.split(","):
        names = re.findall(r"[A-Za-z_]\w*", p)
        out.append(names[-1] if names else None)
    return out


def remap_donor_params(donor_c, draft_c, fn):
    """Donor `arg<i>` (m2c) -> i-ter Draft-Parametername. Globals/Callees unveraendert."""
    dp = _sig_params(draft_c, fn)
    out = donor_c
    for i, pname in enumerate(dp):
        if pname:
            out = re.sub(r"\barg" + str(i) + r"\b", pname, out)
    return out


def _body(c_code, fn):
    m = re.search(r"\b" + re.escape(fn) + r"\s*\([^;{)]*\)\s*\{", c_code)
    if not m:
        return None, None, None
    start = m.end() - 1
    depth = 0
    for j in range(start, len(c_code)):
        if c_code[j] == "{":
            depth += 1
        elif c_code[j] == "}":
            depth -= 1
            if depth == 0:
                return c_code[:start + 1], c_code[start + 1:j], c_code[j:]
    return None, None, None


def _top_statements(body):
    """Body in Top-Level-Statements zerlegen (brace-depth 0, an ';' bzw. '}'-Bloecken)."""
    out = []; depth = 0; cur = ""
    for ch in body:
        cur += ch
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out.append(cur.strip()); cur = ""
        elif ch == ";" and depth == 0:
            out.append(cur.strip()); cur = ""
    if cur.strip():
        out.append(cur.strip())
    return [s for s in out if s]


def _called_funcs(text):
    return set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", text))


def missing_statements(draft_c, donor_c, fn):
    """Donor-Statements, die eine im Draft FEHLENDE Funktion aufrufen (Kandidaten zum Graften).
    Donor wird zuvor namens-reconciled."""
    donor = remap_donor_params(donor_c, draft_c, fn)
    pre, dbody, suf = _body(donor, fn)
    if dbody is None:
        return []
    draft_calls = _called_funcs(draft_c)
    out = []
    for st in _top_statements(dbody):
        callees = _called_funcs(st)
        # mind. ein Callee, der im Draft NICHT vorkommt + keine Donor-Locals (var_/sp/temp_)
        if callees - draft_calls and not re.search(r"\b(?:var_|temp_|sp[0-9A-Fa-f]{2})", st):
            out.append(st if st.endswith((";", "}")) else st + ";")
    return out


def graft_candidates(draft_c, donor_stmts, fn):
    """Fuer jedes fehlende Statement: Draft-Varianten mit Einfuegung an jeder Body-Position."""
    pre, body, suf = _body(draft_c, fn)
    if body is None:
        return []
    stmts = _top_statements(body)
    out = []
    for ms in donor_stmts:
        for pos in range(len(stmts) + 1):
            ns = stmts[:pos] + [ms] + stmts[pos:]
            cand = pre + "\n  " + "\n  ".join(ns) + "\n" + suf
            out.append((f"graft@{pos}:{ms[:28]}", cand))
    return out


def synthesis_graft(draft_c, fn, sp, donor_c, count_fn, max_iter=4):
    """Iterativer orakel-gegateter Graft. count_fn(c,fn,sp)->Diff-Anzahl (None=compile-fail).
    Returns dict {applied, new_code, diffs_before, diffs_after, steps}."""
    before = count_fn(draft_c, fn, sp)
    cur, cur_d = draft_c, before
    steps = []
    for _ in range(max_iter):
        if cur_d == 0:
            break
        cands = graft_candidates(cur, missing_statements(cur, donor_c, fn), fn)
        best, best_d, best_lbl = None, cur_d, None
        for lbl, cand in cands:
            d = count_fn(cand, fn, sp)
            if d is not None and (best_d is None or d < best_d):
                best, best_d, best_lbl = cand, d, lbl
        if best is None:
            break
        cur, cur_d = best, best_d
        steps.append((best_lbl, best_d))
    return {"applied": cur != draft_c, "new_code": cur, "diffs_before": before,
            "diffs_after": cur_d, "steps": steps}
