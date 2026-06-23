"""memory-offset Experte (deterministisch, orakel-verifiziert, KEIN AI).

memory-offset = falscher Struct-Feld-/Array-Offset: der Draft greift auf draft_offset zu, das
Target auf target_offset (gleiche Basis, gleiche Breite). URSACHE: KI hat falschen Index/Offset
(oft halluzinierte Struct-Felder -> Offsets stimmen nicht). LOESUNG: Offset-Korrektur direkt aus
dem Diff (kein Struct-Verstaendnis noetig) -- explizit oder via Index. Struct-Layout (offsetof)
liegt in modules/struct_layout.py, wird hier aber meist gar nicht gebraucht, weil der Diff den
Ziel-Offset SCHON liefert.

Zwei deterministische Transform-Klassen (orakel-gegated -> sicher per Konstruktion):
- LITERAL-SUBST: ein Integer-Literal in `[K]`/`+ K` (das draft_offset erzeugt) durch den Wert
  ersetzen, der target_offset erzeugt (T, T//W als Index-Interpretation).
- DEREF-OFFSET-INJEKTION: bare `*(WT*)EXPR` / `EXPR[0]` (Offset 0) -> `*(WT*)((char*)EXPR + T)`.

Verifikation IMMER ueber modules/ido_compiler (PC-Freeze-Regel). KEIN Permuter.
ABI-Alignment (See MIPS Run §9.5.3/§11.5: halfword even, word 4-aligned) ist implizit durch
den echten IDO-Compiler abgedeckt -- wir raten keine Offsets, wir uebernehmen target_offset.
"""
import re

_WIDTH = {"lb": 1, "lbu": 1, "sb": 1, "lh": 2, "lhu": 2, "sh": 2,
          "lw": 4, "sw": 4, "lwc1": 4, "swc1": 4}


def memory_offset_targets(mem_entries):
    """Aus Memory-Access-Diff-Eintraegen -> Liste (width, target_off, draft_off, op).
    Nur echte OFFSET-Diffs (Basis-Register gleich, Offset abweichend)."""
    out = []
    for e in mem_entries:
        op = e.get("op", "")
        w = _WIDTH.get(op)
        if w is None:
            continue
        t = _parse_acc(e.get("target", "")); d = _parse_acc(e.get("draft", ""))
        if not t or not d:
            continue
        (toff, tbase), (doff, dbase) = t, d
        if tbase == dbase and toff != doff:   # echte Offset-Abweichung, gleiche Basis
            out.append((w, toff, doff, op))
    return out


def _parse_acc(s):
    """'t6,8(a0)' -> (8, 'a0')  /  't6,-4(a0)' -> (-4,'a0'). None wenn kein Match."""
    m = re.search(r"(-?\d+)\(([^)]+)\)", str(s))
    if not m:
        return None
    return int(m.group(1)), m.group(2).strip()


def _int_literals(c_code):
    """Integer-Literale in Index-/Offset-Kontext: in [..] oder nach +/-. -> Liste (match-span, wert, text)."""
    out = []
    for m in re.finditer(r"\[\s*(0x[0-9A-Fa-f]+|\d+|0x[0-9A-Fa-f]{8,16}|0xF{8,16})\s*\]", c_code):
        try: out.append((m.span(1), _to_int(m.group(1)), m.group(1), "index"))
        except ValueError: pass
    for m in re.finditer(r"\+\s*(0x[0-9A-Fa-f]+|\d+)\b", c_code):
        try: out.append((m.span(1), _to_int(m.group(1)), m.group(1), "offset"))
        except ValueError: pass
    return out


def _to_int(s):
    v = int(s, 0)
    if v >= (1 << 63):       # 0xFFFFFFFF.. = negativer Index (64-bit)
        v -= (1 << 64)
    return v


def literal_subst_candidates(c_code, targets):
    out = []
    lits = _int_literals(c_code)
    for w, toff, doff, op in targets:
        for (span, val, text, kind) in lits:
            for newval in {toff, toff // w if toff % w == 0 else None}:
                if newval is None or newval == val:
                    continue
                cand = c_code[:span[0]] + str(newval) + c_code[span[1]:]
                if cand != c_code:
                    out.append((f"lit:{text}->{newval}@{op}{toff}", cand))
    return out


def deref_offset_candidates(c_code, targets):
    """bare `*((WT *) EXPR)` (Offset 0 im Draft) -> `*((WT *)((char *)(EXPR) + T))`."""
    out = []
    offs0 = [(w, t) for (w, t, d, op) in targets if d == 0]
    if not offs0:
        return out
    for m in re.finditer(r"\*\s*\(\s*\(\s*([A-Za-z_]\w*\s*\*)\s*\)\s*([A-Za-z_]\w*)\s*\)", c_code):
        wt, expr = m.group(1), m.group(2)
        for w, t in offs0:
            cand = c_code[:m.start()] + f"*(({wt})((char *)({expr}) + {t}))" + c_code[m.end():]
            if cand != c_code:
                out.append((f"deref:{expr}+{t}", cand))
    # EXPR[0] -> EXPR[T//W]
    for m in re.finditer(r"([A-Za-z_]\w*)\s*\[\s*0\s*\]", c_code):
        expr = m.group(1)
        for w, t in offs0:
            if t % w == 0:
                cand = c_code[:m.start()] + f"{expr}[{t // w}]" + c_code[m.end():]
                if cand != c_code:
                    out.append((f"idx0:{expr}[{t // w}]", cand))
    return out


def inline_pointer_copy_candidates(c_code):
    """STUFE 2 (addressing-mode): `T *NAME = <pointer-expr>;` + Nutzung via *NAME / NAME[i] ->
    NAME-Nutzungen durch (<pointer-expr>) ersetzen, Deklaration weg. Der Draft praekomputiert oft
    den Pointer in ein Register (andere BASIS als Target) -> Inlining erzwingt base+offset-
    Adressierung (= Target). Orakel-gegated."""
    out = []
    for m in re.finditer(r"^[ \t]*([A-Za-z_][\w ]*\*)\s*([A-Za-z_]\w*)\s*=\s*([^;]+);[ \t]*$",
                         c_code, flags=re.M):
        typ, name, rhs = m.group(1).strip(), m.group(2), m.group(3).strip()
        # RHS muss eine Pointer-Berechnung sein (Cast / Arithmetik), nicht nur ein Bezeichner
        if not re.search(r"[)+]", rhs) or "(" not in rhs:
            continue
        # NAME darf nur EINMAL definiert werden (keine Re-Assignments)
        if len(re.findall(r"(?<![\w])" + re.escape(name) + r"\s*=(?!=)", c_code)) != 1:
            continue
        new = c_code[:m.start()] + c_code[m.end():]                       # Deklaration entfernen
        new = re.sub(r"(?<![\w])" + re.escape(name) + r"(?![\w])", f"({rhs})", new)
        if new != c_code:
            out.append((f"inline_ptr:{name}", new))
    return out


# Opcode -> exakter C-Typ (Breite + Signedness aus dem Diff)
_OPTYPE = {"lb": "s8", "lbu": "u8", "sb": "u8", "lh": "s16", "lhu": "u16", "sh": "u16",
           "lw": "s32", "sw": "s32", "lwc1": "f32", "swc1": "f32"}


def ptr_field_candidates(c_code, targets):
    """`ptr->field` -> `*(TYPE *)((char *)(ptr) + T)` mit Ziel-Offset T + exaktem Typ aus dem
    Opcode. Faengt halluzinierte Struct-Felder (Offset steckt in der Typdefinition, kein Literal).
    Enumeriert (Zugriff x Ziel-Offset), orakel-gegated."""
    out = []
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)", c_code):
        base = m.group(1)
        for w, toff, doff, op in targets:
            ty = _OPTYPE.get(op, "s32")
            rep = f"(*({ty} *)((char *)({base}) + {toff}))"
            cand = c_code[:m.start()] + rep + c_code[m.end():]
            if cand != c_code:
                out.append((f"field:{base}->..+{toff}({op})", cand))
    return out


def universal_byteoffset_candidates(c_code, targets):
    """Universal-Fallback: bare `*ident` und `ident[CONST]` als explizite Byte-Offset-Form fuer
    jeden Ziel-Offset umschreiben (zwingt base+offset, exakter Typ aus Opcode). Orakel-gegated."""
    out = []
    accesses = []
    for m in re.finditer(r"(?<![\w\)])\*\s*([A-Za-z_]\w*)\b", c_code):   # bare *ident
        accesses.append((m.span(), m.group(1)))
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\[\s*(0x[0-9A-Fa-f]+|\d+)\s*\]", c_code):  # ident[CONST]
        accesses.append((m.span(), m.group(1)))
    for (span, base) in accesses:
        for w, toff, doff, op in targets:
            ty = _OPTYPE.get(op, "s32")
            rep = f"(*({ty} *)((char *)({base}) + {toff}))"
            cand = c_code[:span[0]] + rep + c_code[span[1]:]
            if cand != c_code:
                out.append((f"univ:{base}+{toff}({op})", cand))
    return out


def memory_offset_candidates(c_code, mem_entries):
    targets = memory_offset_targets(mem_entries)
    cands = inline_pointer_copy_candidates(c_code)              # Stufe 2 (addressing-mode)
    if targets:
        cands = (literal_subst_candidates(c_code, targets)
                 + deref_offset_candidates(c_code, targets)
                 + ptr_field_candidates(c_code, targets)
                 + universal_byteoffset_candidates(c_code, targets) + cands)
    seen = set(); out = []
    for lbl, code in cands:
        if code != c_code and code not in seen:
            seen.add(code); out.append((lbl, code))
    return out[:80]   # Deckel gegen Kandidaten-Explosion (Zugriffe x Ziel-Offsets)


def memory_offset_expert(c_code, func_name, target_s_path, eval_fn=None, max_iter=6):
    """eval_fn(c, fn, sp) -> (diff_count, memory_access_entries). Iterativer orakel-gegateter
    Offset-Korrigierer. Returns {applied, new_code, diffs_before, diffs_after, steps}."""
    before, mem = eval_fn(c_code, func_name, target_s_path)
    cur, cur_d, cur_mem = c_code, before, mem
    steps = []
    for _ in range(max_iter):
        if cur_d == 0:
            break
        cur_off = len(memory_offset_targets(cur_mem))
        # VEREINHEITLICHTES GATE: akzeptiere wenn (Gesamt sinkt) ODER (Gesamt flach UND Offset-
        # Diffs sinken). Nie Gesamt-Regress -> sicher per Konstruktion. Faengt Stufe1 (Offset-Fix,
        # oft flach=Ping-Pong) UND Stufe2 (Inline-Pointer/addressing-mode, Gesamt sinkt).
        best = None  # (total, off, code, lbl, mem)
        for lbl, cand in memory_offset_candidates(cur, cur_mem):
            d, m = eval_fn(cand, func_name, target_s_path)
            if d is None or d > cur_d:
                continue
            off = len(memory_offset_targets(m))
            accept = (d < cur_d) or (d == cur_d and off < cur_off)
            if not accept:
                continue
            key = (d, off)
            if best is None or key < (best[0], best[1]):
                best = (d, off, cand, lbl, m)
        if best is None:
            break
        cur_d, cur, cur_mem = best[0], best[2], best[4]
        steps.append((best[3], best[0]))
    return {"applied": cur != c_code, "new_code": cur, "diffs_before": before,
            "diffs_after": cur_d, "steps": steps}
