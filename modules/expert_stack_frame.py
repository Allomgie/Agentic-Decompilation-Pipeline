"""stack-frame Experte (deterministisch, orakel-verifiziert).

BEFUND (2026-06): stack-frame-Diffs sind ueberwiegend DOWNSTREAM-SYMPTOME ueberfluessiger
Stack-Locals, KEINE eigenstaendigen Frame-Hebel. Quelle: Coloring §4.3 (aliasierte Live-Unit
= "as if spilled" -> Stack-Slot -> Frame waechst). Der Experte erzeugt deterministische
Kandidaten, die solche Locals entfernen, und laesst das ORAKEL entscheiden (nur Verbesserung
behalten -> sicher per Konstruktion: Fehlgriff erhoeht Diff -> verworfen).

Transform-Klassen (Kandidaten):
- ANTI-ALIAS: `local=X; f(...,&local)` -> `f(...,X)` (address-of zwingt Spill; Coloring §4.3).
- REDUNDANT-COPY / PARAM-ALIAS: `T L = Y; ...L...` -> Nutzungen durch Y ersetzen, L weg.
- UNUSED: nie genutztes Local entfernen.

Verifikation IMMER ueber modules/ido_compiler (PC-Freeze-Regel). KEIN Permuter.
"""
import re

_DECL = re.compile(r"^\s*([A-Za-z_][\w ]*?\**)\s+([A-Za-z_]\w*)\s*;\s*$")


def _body_bounds(c_code):
    i = c_code.find("{")
    j = c_code.rfind("}")
    if i < 0 or j < 0 or j < i:
        return None
    return i, j


def _decls(c_code):
    """-> Liste (zeilenindex, typ, name) der einfachen lokalen Deklarationen."""
    out = []
    lines = c_code.splitlines()
    started = False
    for idx, ln in enumerate(lines):
        if "{" in ln:
            started = True
        if not started:
            continue
        m = _DECL.match(ln)
        if m and m.group(1).strip() not in ("return", "else", "case"):
            out.append((idx, m.group(1).strip(), m.group(2)))
    return out


def _uses(c_code, name):
    return re.findall(r"(?<![\w])" + re.escape(name) + r"(?![\w])", c_code)


def antialias_candidates(c_code):
    """`local = X;` + `&local` als Call-Arg -> `X` einsetzen, local entfernen."""
    out = []
    lines = c_code.splitlines()
    for idx, typ, name in _decls(c_code):
        # &name als Argument (in Klammern, nicht in Deklaration)
        if not re.search(r"&\s*" + re.escape(name) + r"\b", c_code):
            continue
        # einzige Zuweisung name = rhs;
        assigns = re.findall(r"\b" + re.escape(name) + r"\s*=\s*([^;]+);", c_code)
        if len(assigns) != 1:
            continue
        rhs = assigns[0].strip()
        new = c_code
        # &name -> (rhs)
        new = re.sub(r"&\s*" + re.escape(name) + r"\b", f"({rhs})", new)
        # Zuweisungs-Zeile entfernen
        new = re.sub(r"^\s*" + re.escape(name) + r"\s*=\s*[^;]+;\s*$", "", new, flags=re.M)
        # Deklaration entfernen
        new = re.sub(r"^\s*" + re.escape(typ).replace(r"\ ", r"\s+") + r"\s+" + re.escape(name) + r"\s*;\s*$",
                     "", new, flags=re.M)
        # Reste von name uebrig? dann unsicher -> ueberspringen
        if _uses(new, name):
            continue
        out.append((f"antialias:{name}", _clean(new)))
    return out


def redundant_copy_candidates(c_code):
    """`T L = Y; ...L...` -> Nutzungen durch Y ersetzen, L weg (Y = Param oder andere Var)."""
    out = []
    for idx, typ, name in _decls(c_code):
        # Deklaration+Init in einer Zeile: `T name = Y;`
        m = re.search(r"\b" + re.escape(typ).replace(" ", r"\s+") + r"\s+" + re.escape(name)
                      + r"\s*=\s*([A-Za-z_]\w*)\s*;", c_code)
        if not m:
            continue
        src = m.group(1)
        if src == name:
            continue
        # name darf nicht woanders neu zugewiesen werden (nur diese eine Definition)
        if len(re.findall(r"\b" + re.escape(name) + r"\s*=", c_code)) != 1:
            continue
        if re.search(r"&\s*" + re.escape(name) + r"\b", c_code):
            continue
        new = re.sub(r"\b" + re.escape(typ).replace(" ", r"\s+") + r"\s+" + re.escape(name)
                     + r"\s*=\s*" + re.escape(src) + r"\s*;", "", c_code)
        new = re.sub(r"(?<![\w])" + re.escape(name) + r"(?![\w])", src, new)
        out.append((f"redundant:{name}->{src}", _clean(new)))
    return out


def unused_candidates(c_code):
    """Nie (ausser Deklaration) genutztes Local entfernen."""
    out = []
    for idx, typ, name in _decls(c_code):
        if len(_uses(c_code, name)) <= 1:  # nur die Deklaration
            new = re.sub(r"^\s*" + re.escape(typ).replace(" ", r"\s+") + r"\s+" + re.escape(name)
                         + r"\s*;\s*$", "", c_code, flags=re.M)
            out.append((f"unused:{name}", _clean(new)))
    return out


def _clean(code):
    # Leere Doppel-Zeilen eindampfen
    return re.sub(r"\n\s*\n\s*\n", "\n\n", code)


def stack_frame_candidates(c_code):
    seen = set(); cands = []
    for lbl, code in (antialias_candidates(c_code) + redundant_copy_candidates(c_code)
                      + unused_candidates(c_code)):
        if code != c_code and code not in seen:
            seen.add(code); cands.append((lbl, code))
    return cands


def _sp_frame(op):
    m = re.search(r"sp,sp,(-?\d+)", str(op).replace(" ", ""))
    return abs(int(m.group(1))) if m else None


def _frame_grow_bytes(ents):
    """Byte-Delta, um den der TARGET-Frame GROESSER ist als der Draft-Frame (>0 = zu klein). Sonst 0."""
    for e in ents or []:
        if e.get("type") == "Stack Frame Mismatch":
            t, d = _sp_frame(e.get("target")), _sp_frame(e.get("draft"))
            if t is not None and d is not None and t > d:
                return t - d
    return 0


def grow_candidates(c_code, grow):
    """Zu-kleinen Frame wachsen lassen: UNBENUTZTES Padding-Local als ERSTE Deklaration im Koerper (nicht-letzte
    -> wird nicht getrimmt, Slot bleibt reserviert; Research: Decomp_Infosheet + fredchow-Thesis Front-End-Layout).
    Mehrere Groessen (exakt + auf 8 aufgerundet), Orakel waehlt. Nur sinnvoll bei zu-kleinem Frame."""
    i = c_code.find("{")
    if i < 0 or grow <= 0:
        return []
    out = []
    for n in sorted({grow, (grow + 7) // 8 * 8}):
        cand = c_code[:i + 1] + f"\n  s8 _sfpad[{n}];" + c_code[i + 1:]
        out.append((f"grow:pad{n}", cand))
    return out


def stack_frame_expert(c_code, func_name, target_s_path, count_fn=None, diff_fn=None):
    """Generiert deterministische Kandidaten (Local entfernen = Frame schrumpfen; oder bei zu-kleinem Frame
    Padding-Local = wachsen, wenn diff_fn vorhanden), Orakel waehlt den besten.
    Returns dict {applied, new_code, diffs_before, diffs_after, label}."""
    before = count_fn(c_code, func_name, target_s_path)
    best_code, best_d, best_lbl = c_code, before, None
    cands = stack_frame_candidates(c_code)
    if diff_fn is not None:                          # GROW-Transform fuer zu-kleine Frames
        try:
            _t, ents = diff_fn(c_code, func_name, target_s_path)
            grow = _frame_grow_bytes(ents)
            if grow:
                cands = cands + grow_candidates(c_code, grow)
        except Exception:
            pass
    for lbl, cand in cands:
        d = count_fn(cand, func_name, target_s_path)
        if d is not None and (best_d is None or d < best_d):
            best_code, best_d, best_lbl = cand, d, lbl
    applied = best_lbl is not None and (before is None or best_d < before)
    return {"applied": applied, "new_code": best_code if applied else c_code,
            "diffs_before": before, "diffs_after": best_d if applied else before,
            "label": best_lbl}
