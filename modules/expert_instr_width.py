"""instr-wrong-width Experte (deterministisch, orakel-verifiziert, KEIN AI).

Der Draft greift mit FALSCHER Breite zu (gleiche Basis+Offset+Ziel, anderer Load/Store-Opcode):
z.B. `lw` statt `lbu`, `sh` statt `sw`. Die Ziel-Breite + Signedness steckt im Opcode -> exakter
C-Typ via _OPTYPE. Transform = Cast-/Typ-Breite am Zugriff anpassen. Geschwister des memory-offset-
Experten (gleiche Maschinerie, andere Achse: Breite statt Offset).

ECHTER width-diff (eng definiert, gegen Misalignments): beide Loads ODER beide Stores, gleiche
Basis+Offset+Ziel, nur Breiten-Opcode abweichend. (lw vs sw = Load/Store-RICHTUNG, KEIN width.)

Zwei Transform-Klassen (orakel-gegated -> sicher per Konstruktion):
- CAST-SUBST: `*((TYPE *) EXPR)` -> TYPE durch Ziel-Typ ersetzen.
- DEREF-INJECT: `*(EXPR)` (Pointer-Arithmetik, impliziter Typ) -> `*((TYPE *)(EXPR))`.
Verifikation IMMER ueber modules/ido_compiler (PC-Freeze-Regel). KEIN Permuter.
"""
import re

_OPTYPE = {"lb": "s8", "lbu": "u8", "sb": "u8", "lh": "s16", "lhu": "u16", "sh": "u16",
           "lw": "s32", "sw": "s32", "lwc1": "f32", "swc1": "f32"}
_LOADS = {"lw", "lh", "lhu", "lb", "lbu", "lwc1"}
_STORES = {"sw", "sh", "sb", "swc1"}


def _parse(s):
    s = re.sub(r"/\*.*?\*/", "", str(s))
    s = re.sub(r"^\s*[0-9a-f]+:\s*[0-9a-f]+\s*", "", s)
    s = s.replace("\t", " ").strip()
    m = re.match(r"([a-z][a-z0-9.]*)\s+\$?(\w+)\s*,\s*(-?\d+|0x[0-9A-Fa-f]+)\(\$?(\w+)\)", s)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3), 0), m.group(4)


def width_targets(entries):
    """-> Liste (target_op, draft_op, off, base, target_type). ECHTE width-diffs:
    gleiche Basis+Offset, beide Loads ODER beide Stores, nur Breite abweichend. ZIEL-Register
    darf abweichen (andere Breite -> anderer Wert -> ggf. andere Allokation; das ist register-
    order, separat). Auch in Multi-Instruktions-Bloecken (Kreuzprodukt target x draft).
    (lw-vs-sw = Load/Store-RICHTUNG, KEIN width -> ausgeschlossen.)"""
    out = []
    for e in entries:
        if e.get("type") != "Instruction Mismatch":
            continue
        t, d = e.get("target"), e.get("draft")
        if not (isinstance(t, list) and isinstance(d, list)):
            continue
        tp = [p for p in (_parse(x) for x in t) if p]
        dp = [p for p in (_parse(x) for x in d) if p]
        for (mt, rt, ot, bt) in tp:
            for (md, rd, od, bd) in dp:
                if ot != od or bt != bd or mt == md:
                    continue
                both_load = mt in _LOADS and md in _LOADS
                both_store = mt in _STORES and md in _STORES
                if both_load or both_store:
                    out.append((mt, md, ot, bt, _OPTYPE.get(mt, "s32")))
    return out


def cast_subst_candidates(c_code, target_types):
    """`(TYPE *) ...` / `(TYPE **) ...` -> Cast-Typ durch Ziel-Typ ersetzen, Stern-ANZAHL erhalten
    (orakel waehlt). GAP-FIX (2026-06): auch Mehrfach-Stern `(u8 **)` (Pointer-zu-Pointer / Array-of-Ptr,
    z.B. `((u8**)arg0)[i][0]`) -- vorher nur Einfach-Stern -> verfehlte verschachtelte Zugriffe."""
    out = []
    for m in re.finditer(r"\(\s*([A-Za-z_]\w*)(\s*\*+\s*)\)", c_code):
        cur, stars = m.group(1), m.group(2)
        for tt in target_types:
            if tt == cur:
                continue
            cand = c_code[:m.start()] + f"({tt}{stars})" + c_code[m.end():]
            if cand != c_code:
                out.append((f"cast:{cur}{stars.strip()}->{tt}", cand))
    return out


def deref_inject_candidates(c_code, target_types):
    """bare `*(EXPR)` (kein direkter Cast nach `*`) -> `*((TYPE *)(EXPR))`. EXPR balanciert."""
    out = []
    for m in re.finditer(r"\*\s*\(", c_code):
        # pruefe: nach `*(` kommt NICHT direkt ein `(TYPE *)`-Cast (sonst macht cast_subst das)
        start = m.end() - 1  # Position des '('
        expr = _balanced(c_code, start)
        if expr is None:
            continue
        inner = c_code[start:start + expr]
        if re.match(r"\(\s*\(\s*[A-Za-z_]\w*\s*\*", inner):
            continue  # schon ein Cast direkt drin
        for tt in target_types:
            cand = c_code[:m.start()] + f"*(({tt} *)" + c_code[start:start + expr] + ")" + c_code[start + expr:]
            if cand != c_code:
                out.append((f"inject:{tt}", cand))
    return out


def _balanced(s, i):
    """Laenge der balancierten Klammer ab s[i]=='(' inkl. Klammern, oder None."""
    if i >= len(s) or s[i] != "(":
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0:
                return j - i + 1
    return None


def decl_type_candidates(c_code, target_types):
    """STUFE 2: Breite aus DEKLARIERTEM Typ (Pointer/Array/Param) statt Cast. Deklarations-Typ
    durch Ziel-Typ ersetzen -> `f32 *p`->`s32 *p`, `s16 arr[N]`->`s32 arr[N]`, Param `(u8 *p)`.
    Orakel-gegated (sicher trotz breiter Wirkung)."""
    out = []
    # Pointer-Deklaration:  TYPE * name   (Body-Zeile mit = oder ; , oder im Param-Kontext)
    for m in re.finditer(r"(?:(?<=[\(,;{])|^)\s*([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)\b", c_code, flags=re.M):
        cur = m.group(1)
        if cur in ("return", "sizeof"):
            continue
        for tt in target_types:
            if tt == cur:
                continue
            cand = c_code[:m.start(1)] + tt + c_code[m.end(1):]
            if cand != c_code:
                out.append((f"decl_ptr:{cur}->{tt}:{m.group(2)}", cand))
    # Array-Deklaration:  TYPE name[
    for m in re.finditer(r"(?:(?<=[;{])|^)\s*([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\[", c_code, flags=re.M):
        cur = m.group(1)
        for tt in target_types:
            if tt == cur:
                continue
            cand = c_code[:m.start(1)] + tt + c_code[m.end(1):]
            if cand != c_code:
                out.append((f"decl_arr:{cur}->{tt}:{m.group(2)}", cand))
    return out


def instr_width_candidates(c_code, entries):
    tgts = width_targets(entries)
    if not tgts:
        return []
    types = list({tt for *_, tt in tgts})
    seen = set(); out = []
    for lbl, code in (cast_subst_candidates(c_code, types)
                      + deref_inject_candidates(c_code, types)
                      + decl_type_candidates(c_code, types)):
        if code != c_code and code not in seen:
            seen.add(code); out.append((lbl, code))
    return out[:120]


def instr_width_expert(c_code, func_name, target_s_path, eval_fn=None, max_iter=6, ai_call=None):
    """eval_fn(c, fn, sp) -> (metric, entries). metric = LEXIKOGRAFISCH (synth_count, total) ODER None.
    GATE (2026-06): akzeptiere Kandidaten, der die METRIK senkt -- ein Breiten-Fix senkt eine
    Instruction-Mismatch (= synth-Anteil), also faellt die Metrik AUCH wenn total via Hand-off
    (Register/Reorder/...) steigt. Sekundaerer width_targets-Tiebreak bei gleicher Metrik. Loest den
    total-Gate-Bug; loop-sicher, weil die Metrik streng faellt. diffs_after = Metrik (Tupel).

    STUFE 2 (ai_call!=None, 2026-06): wenn die DETERMINISTIK den Rest nicht loest (width_targets > 0),
    EIN Oneshot-KI-Versuch (modules.expert_hybrid_ai, getunte Prompts) -- als zusaetzlicher Kandidat
    durch DASSELBE Orakel-Gate. Nie Verschlechterung. Stufe 3 (permuter/garbage) macht die Pipeline."""
    before, ent = eval_fn(c_code, func_name, target_s_path)
    cur, cur_d, cur_ent = c_code, before, ent
    steps = []
    for _ in range(max_iter):
        cur_w = len(width_targets(cur_ent))
        if cur_w == 0:
            break
        best = None
        for lbl, cand in instr_width_candidates(cur, cur_ent):
            d, m = eval_fn(cand, func_name, target_s_path)
            if d is None or d > cur_d:                      # compile-fail oder Metrik schlechter
                continue
            w = len(width_targets(m))
            if not ((d < cur_d) or (d == cur_d and w < cur_w)):
                continue
            key = (d, w)
            if best is None or key < best[0]:
                best = (key, cand, lbl, m)
        if best is None:
            break
        cur_d, cur, cur_ent = best[0][0], best[1], best[3]
        steps.append((best[2], cur_d))
    # STUFE 2: KI-Oneshot. Laeuft auf dem ORIGINAL (c_code, ent) und KONKURRIERT mit dem det. Ergebnis
    # durch DASSELBE Gate. WICHTIG (sonst f32-Return-Bug): triggern auf "Original hatte width-Ziele UND
    # mm noch > 0" -- NICHT auf "width_targets jetzt > 0". Sonst blockiert ein det. Lateral-Cast (mm flat,
    # width_target->0, z.B. (s32*)->(f32*) bei int-Return) die ueberlegene f32-Return-Loesung der KI.
    if (ai_call is not None and cur_d is not None and cur_d[-1] > 0 and len(width_targets(ent)) > 0):  # cur_d[-1]=mm (md,im,tier,mm)
        from modules import expert_hybrid_ai as _hy
        cur_w = len(width_targets(cur_ent))
        for lbl, cand in _hy.hybrid_ai_candidate(c_code, func_name, ent, "width", ai_call):
            d, m = eval_fn(cand, func_name, target_s_path)
            if d is None or d > cur_d:
                continue
            w = len(width_targets(m))
            if (d < cur_d) or (d == cur_d and w < cur_w):
                cur_d, cur, cur_ent = d, cand, m
                steps.append((lbl, cur_d))
    return {"applied": cur != c_code, "new_code": cur, "diffs_before": before,
            "diffs_after": cur_d, "steps": steps}
