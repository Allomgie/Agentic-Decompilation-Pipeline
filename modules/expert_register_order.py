"""register-order Experte (Zwei-Stufen, orakel-verifiziert).

Stufe 1 (DETERMINISTISCH): Reverse-Strength-Reduction. Erkennt im TARGET eine
sll/addu/subu-Akkumulator-Kette, die einen Array-Index speist (= vom Compiler
strength-reduzierte Multiplikation), berechnet den Multiplikator K und schreibt den
C-Body als Hochsprachen-Operation `SYM[param * K]` neu -> IDOs eigene Strength-Reduction
erzeugt die Kette mit IHREM Register-Reuse neu (Register passen).

Stufe 2 (interner SpecialPermuter): Hook fuer das Residual (freie Allocator-Register-Wahl
auf einer Live-Range). NOCH NICHT implementiert -> dokumentierter Einstiegspunkt.

Quellen: Coloring §5.2 (Prioritaet=Ersparnis/Groesse), Thesis §3.1.x (Strength Reduction).
Verifikation IMMER ueber modules/ido_compiler (PC-Freeze-Regel) + diff_generator.

WICHTIG v1-SCOPE: Stufe 1 deckt die nachgewiesene Form `return SYM[<arg>*K]` ab
(Einzel-Ausdruck-Funktion). Multiplikator-Berechnung aus der Kette ist GENERELL; der
C-Rewriter ist v1-eng. Erweiterung: Sub-Ausdruck in groesseren Funktionen lokalisieren.
"""
import os, re, json

_REG = r"\$(\w+)"
_LOAD_SIZE = {"lw": 4, "lwc1": 4, "sw": 4, "swc1": 4,
              "lh": 2, "lhu": 2, "sh": 2, "lb": 1, "lbu": 1, "sb": 1}
_AREG = {"a0": 0, "a1": 1, "a2": 2, "a3": 3}


def _parse_s(target_s_path):
    """-> Liste (mnem, [ops]) ohne Kommentare/Labels."""
    out = []
    for ln in open(target_s_path, encoding="utf-8", errors="replace"):
        s = re.sub(r"/\*.*?\*/", "", ln)
        s = s.strip()
        if not s or s.startswith(".") or s.startswith("glabel") or s.startswith("endlabel") \
           or s.endswith(":"):
            continue
        m = re.match(r"([a-z][a-z0-9.]*)\s+(.*)", s)
        if not m:
            continue
        mn = m.group(1)
        ops = [o.strip() for o in m.group(2).split(",")]
        out.append((mn, ops))
    return out


def analyze_target_chain(target_s_path):
    """Erkennt eine strength-reduzierte Multiplikations-Kette, die einen Array-Index speist.
    Returns dict {K, sym, elem_size, base_arg, mask} oder None (sicher: nur bei klarer Form)."""
    insns = _parse_s(target_s_path)
    # 1) base: andi $B,$Aarg,0xNN  ODER  move/or $B,$Aarg,$zero
    base_reg = base_arg = mask = None
    chain_start = None
    for i, (mn, ops) in enumerate(insns):
        rm = [re.match(_REG, o) for o in ops]
        if mn == "andi" and len(ops) == 3 and rm[0] and rm[1]:
            src = rm[1].group(1)
            if src in _AREG:
                base_reg = rm[0].group(1); base_arg = _AREG[src]
                mask = ops[2]; chain_start = i + 1; break
        if mn in ("move", "or") and rm and rm[0]:
            # move $B,$Aarg  /  or $B,$Aarg,$zero
            if len(ops) >= 2 and rm[1] and rm[1].group(1) in _AREG:
                if len(ops) == 2 or (len(ops) == 3 and ops[2] in ("$zero", "$0")):
                    base_reg = rm[0].group(1); base_arg = _AREG[rm[1].group(1)]
                    mask = None; chain_start = i + 1; break
    if base_reg is None:
        return None
    # 2) Akkumulator-Kette ab chain_start: sll/addu/subu auf einem ACC, gespeist von base
    acc_reg = None
    mult = None
    sym = None; elem = None
    i = chain_start
    while i < len(insns):
        mn, ops = insns[i]
        rm = [re.match(_REG, o) for o in ops if re.match(_REG, o)]
        if mn == "sll" and len(ops) == 3 and re.match(_REG, ops[0]) and re.match(_REG, ops[1]):
            dst = re.match(_REG, ops[0]).group(1); src = re.match(_REG, ops[1]).group(1)
            try: sh = int(ops[2], 0)
            except ValueError: return None
            if acc_reg is None:
                if src != base_reg:
                    return None
                acc_reg = dst; mult = (1 << sh)
            else:
                if src != acc_reg or dst != acc_reg:
                    return None
                mult <<= sh
        elif mn in ("addu", "add", "subu", "sub") and len(ops) == 3:
            dst = re.match(_REG, ops[0]); a = re.match(_REG, ops[1]); b = re.match(_REG, ops[2])
            if not (dst and a and b):
                # koennte lui/addu-Array-Phase sein -> Kette zu Ende
                break
            dst, a, b = dst.group(1), a.group(1), b.group(1)
            if acc_reg and dst == acc_reg and a == acc_reg and b == base_reg:
                mult += 1 if mn.startswith("add") else -1
            else:
                break
        else:
            break
        i += 1
    if acc_reg is None or mult is None:
        return None
    # 3) Array-Phase: lui $R,%hi(SYM) ; addu $R,$R,$ACC ; load $D,%lo(SYM)($R)
    rest = insns[i:]
    for j, (mn, ops) in enumerate(rest):
        if mn == "lui" and len(ops) == 2:
            mh = re.search(r"%hi\(([^)]+)\)", ops[1])
            if not mh:
                continue
            cand_sym = mh.group(1)
            # suche das Load mit %lo(cand_sym) und ACC im addu davor
            for mn2, ops2 in rest[j:]:
                if mn2 in _LOAD_SIZE:
                    ml = re.search(r"%lo\(([^)]+)\)", " ".join(ops2))
                    if ml and ml.group(1) == cand_sym:
                        sym = cand_sym; elem = _LOAD_SIZE[mn2]; break
            if sym:
                break
    if sym is None or elem is None:
        return None
    if mult % elem != 0:
        return None
    K = mult // elem
    # mult_full = voller Multiplikator (inkl. Elem-Scale) = Byte-Offset-Faktor
    _elem_type = {4: "s32", 2: "s16", 1: "s8"}.get(elem, "s32")
    return {"K": K, "sym": sym, "elem_size": elem, "base_arg": base_arg, "mask": mask,
            "mode": "load", "mult_full": mult, "elem_type": _elem_type, "field_off": 0}


def analyze_target_addr_chain(target_s_path):
    """Adress-Rueckgabe-Variante: Kette -> `&SYM_bytes + base*M + FIELD` (lui+addiu=&SYM,
    optional addiu acc,FIELD, addu RET,*,&SYM). Returns dict {mode:addr, mult_full, sym,
    field_off, base_arg, mask} oder None. (b)-Erweiterung; deckt die Reihenfolge ab, in der
    die KETTE VOLLSTAENDIG vor dem lui steht, wie func_80017810.)"""
    insns = _parse_s(target_s_path)
    base_reg = base_arg = mask = None; chain_start = None
    for i, (mn, ops) in enumerate(insns):
        rm = [re.match(_REG, o) for o in ops]
        if mn == "andi" and len(ops) == 3 and rm[0] and rm[1] and rm[1].group(1) in _AREG:
            base_reg = rm[0].group(1); base_arg = _AREG[rm[1].group(1)]; mask = ops[2]
            chain_start = i + 1; break
        if mn == "sll" and len(ops) == 3 and rm[0] and rm[1] and rm[1].group(1) in _AREG:
            base_reg = rm[1].group(1); base_arg = _AREG[base_reg]; mask = None
            chain_start = i; break
    if base_reg is None:
        return None
    acc = None; mult = None; i = chain_start
    while i < len(insns):
        mn, ops = insns[i]; r = [re.match(_REG, o) for o in ops]
        if mn == "sll" and len(ops) == 3 and r[0] and r[1]:
            d, s = r[0].group(1), r[1].group(1)
            try: sh = int(ops[2], 0)
            except ValueError: return None
            if acc is None:
                if s != base_reg: return None
                acc = d; mult = 1 << sh
            else:
                if s != acc or d != acc: return None
                mult <<= sh
        elif mn in ("addu", "add", "subu", "sub") and len(ops) == 3 and r[0] and r[1] and r[2]:
            d, a, b = r[0].group(1), r[1].group(1), r[2].group(1)
            if acc and d == acc and a == acc and b == base_reg:
                mult += 1 if mn.startswith("add") else -1
            else:
                break
        else:
            break
        i += 1
    if acc is None:
        return None
    rest = insns[i:]
    sym = None; field_off = 0
    for k, (mn, ops) in enumerate(rest):
        if mn == "lui" and len(ops) == 2:
            mh = re.search(r"%hi\(([^)]+)\)", ops[1])
            if not mh: continue
            cand = mh.group(1)
            # naechste addiu %lo(cand) => &SYM ; suche FIELD-Offset addiu acc,acc,K
            has_lo = any(re.search(r"%lo\(" + re.escape(cand) + r"\)", " ".join(o2))
                         for _m2, o2 in rest[k:k + 3])
            if not has_lo: continue
            for mn2, ops2 in rest:
                if mn2 == "addiu" and len(ops2) == 3:
                    r2 = [re.match(_REG, o) for o in ops2]
                    if r2[0] and r2[1] and r2[1].group(1) == acc:
                        try: field_off = int(ops2[2], 0)
                        except ValueError: pass
            sym = cand; break
    if sym is None:
        return None
    return {"mode": "addr", "mult_full": mult, "sym": sym, "field_off": field_off,
            "base_arg": base_arg, "mask": mask}


def _signature_and_externs(c_code):
    """Extrahiert extern-Zeilen + die Funktions-Signaturzeile (vor dem ersten '{')."""
    externs = [l for l in c_code.splitlines() if l.strip().startswith("extern ")]
    m = re.search(r"([A-Za-z_][\w \*]*\b\w+\s*\([^;{]*\))\s*\{", c_code)
    sig = m.group(1).strip() if m else None
    return externs, sig


def _param_names(sig):
    inside = sig[sig.index("(") + 1:sig.rindex(")")]
    if inside.strip() in ("", "void"):
        return []
    out = []
    for p in inside.split(","):
        nm = re.findall(r"[A-Za-z_]\w*", p)
        out.append(nm[-1] if nm else None)
    return out


def _ret_pointer_type(sig):
    """Rueckgabetyp aus Signatur, z.B. 's16 *func(...)' -> 's16 *'. Default 's32 *'."""
    m = re.match(r"\s*([A-Za-z_][\w ]*\*?)\s*[A-Za-z_]\w*\s*\(", sig)
    t = m.group(1).strip() if m else "s32"
    return t if t.endswith("*") else t + " *"


def stage1_candidates(c_code, target_s_path):
    """Liste (label, code) deterministischer Reverse-Strength-Reduction-Kandidaten.
    Bevorzugt die BYTE-POINTER-Form (aus (a)/Stufe2 als generell erkannt). Deckt load- und
    addr-Modus ab ((b)-Erweiterung). Orakel waehlt den besten."""
    externs, sig = _signature_and_externs(c_code)
    if not sig:
        return []
    params = _param_names(sig)
    def base_of(info):
        ba = info["base_arg"]
        return params[ba] if ba < len(params) and params[ba] else None
    head = "\n".join(externs) + ("\n\n" if externs else "") + sig + "\n{\n"
    out = []
    # LOAD-Modus
    info = analyze_target_chain(target_s_path)
    if info:
        b = base_of(info)
        if b:
            M = info["mult_full"]; et = info["elem_type"]; sym = info["sym"]
            # Byte-Pointer-Form (generell, schliesst das Residual ohne Stufe2)
            out.append(("load_byteptr",
                        head + f"  return *({et} *)((char *){sym} + {b} * {M});\n}}\n"))
            # Array-Form (Fallback)
            out.append(("load_array",
                        head + f"  return {sym}[{b} * {info['K']}];\n}}\n"))
    # ADDR-Modus
    ai = analyze_target_addr_chain(target_s_path)
    if ai:
        b = base_of(ai)
        if b:
            M = ai["mult_full"]; sym = ai["sym"]; fo = ai["field_off"]; rt = _ret_pointer_type(sig)
            off = f" + {fo}" if fo else ""
            out.append(("addr_byteptr",
                        head + f"  return ({rt})((char *){sym} + {b} * {M}{off});\n}}\n"))
    return out


def stage1_reverse_strength_reduction(c_code, target_s_path):
    """Rueckwaertskompatibel: erster Kandidat oder None."""
    cands = stage1_candidates(c_code, target_s_path)
    return cands[0][1] if cands else None


def _index_return_variants(c_code):
    """Erzeugt register-beeinflussende C-Varianten fuer die Form `return SYM[EXPR];`.
    Jede Variante aendert die Live-Range-Struktur der Index-Berechnung anders -> der Allocator
    trifft potenziell die Reuse-Entscheidung des Targets. Diff-skopiert (nur die Index-Naht).
    Returns Liste (label, code) oder []."""
    m = re.search(r"return\s+([A-Za-z_]\w*)\s*\[\s*(.+?)\s*\]\s*;", c_code)
    if not m:
        return []
    sym, expr = m.group(1), m.group(2)
    full = m.group(0)
    externs, sig = _signature_and_externs(c_code)
    head = ("\n".join(externs) + ("\n\n" if externs else "") + sig + "\n{\n")
    def wrap(body):
        return head + body + "\n}\n"
    variants = []
    # M1: Index in s32-Temp
    variants.append(("temp_s32", wrap(f"  s32 idx = {expr};\n  return {sym}[idx];")))
    # M2: Index in u32-Temp
    variants.append(("temp_u32", wrap(f"  u32 idx = {expr};\n  return {sym}[idx];")))
    # M3: Pointer-Arithmetik (Index-Scale anders verdrahtet)
    variants.append(("ptr_arith", wrap(f"  return *({sym} + ({expr}));")))
    # M4: Pointer-Temp
    variants.append(("ptr_temp", wrap(f"  s32 *p = {sym} + ({expr});\n  return *p;")))
    # M5: ungenutzte Variable VOR dem return (Allocations-Druck/Reihenfolge)
    variants.append(("unused_pre", wrap(f"  s32 pad;\n  {full}")))
    # M6: Index-Temp + sofortige Nutzung in Pointer
    variants.append(("idx_ptr", wrap(f"  s32 idx = {expr};\n  return *({sym} + idx);")))
    # M7: Self-Assign (Live-Range potenziell verschmelzen)
    variants.append(("self_assign", wrap(f"  s32 idx = {expr};\n  idx = idx;\n  return {sym}[idx];")))
    # M8: int statt s32
    variants.append(("temp_int", wrap(f"  int idx = {expr};\n  return {sym}[idx];")))
    # M9: Ergebnis in Temp vor return
    variants.append(("res_temp", wrap(f"  s32 idx = {expr};\n  s32 r = {sym}[idx];\n  return r;")))
    # M10: explizite Byte-Offset-Pointer (Index-Scale im C sichtbar)
    variants.append(("byte_off", wrap(f"  return *(s32 *)((char *){sym} + ({expr}) * 4);")))
    return variants


def stage2_register_permute(c_code, func_name, target_s_path, count_fn):
    """Interner diff-skopierter Mini-Permuter fuer das Register-Residual.
    Probiert die Varianten, behaelt die mit den WENIGSTEN Diffs (Orakel). Returns
    dict {applied, new_code, diffs_before, diffs_after, label} (applied nur bei echtem Gewinn)."""
    before = count_fn(c_code, func_name, target_s_path)
    best_code, best_diffs, best_label = c_code, before, None
    for label, cand in _index_return_variants(c_code):
        d = count_fn(cand, func_name, target_s_path)
        if d is not None and (best_diffs is None or d < best_diffs):
            best_code, best_diffs, best_label = cand, d, label
    applied = best_label is not None and (before is None or best_diffs < before)
    return {"applied": applied, "new_code": best_code if applied else c_code,
            "diffs_before": before, "diffs_after": best_diffs if applied else before,
            "label": best_label}


def register_order_expert(c_code, func_name, target_s_path, compile_fn=None,
                          count_fn=None):
    """Zwei-Stufen-Experte. compile_fn/count_fn werden injiziert (Orakel = ido_compiler).
    Returns dict {applied, stage, new_code, diffs_before, diffs_after, note}."""
    before = count_fn(c_code, func_name, target_s_path)
    cur_code, cur_diffs = c_code, before
    stages = []
    # --- Stufe 1: deterministische Reverse-Strength-Reduction (mehrere Kandidaten, Orakel waehlt) ---
    best1, best1_d, best1_lbl = None, cur_diffs, None
    for lbl, cand in stage1_candidates(cur_code, target_s_path):
        d = count_fn(cand, func_name, target_s_path)
        if d is not None and (best1_d is None or d < best1_d):
            best1, best1_d, best1_lbl = cand, d, lbl
    if best1 is not None:
        cur_code, cur_diffs = best1, best1_d
        stages.append((f"1:{best1_lbl}", best1_d))
    # --- Stufe 2: interner diff-skopierter Mini-Permuter (Register-Residual) ---
    if cur_diffs is None or cur_diffs[-1] > 0:    # cur_diffs[-1]=mm (Metrik ist Tupel, nicht int -- Crash-Fix)
        s2 = stage2_register_permute(cur_code, func_name, target_s_path, count_fn)
        if s2["applied"]:
            cur_code, cur_diffs = s2["new_code"], s2["diffs_after"]
            stages.append((f"2:{s2['label']}", cur_diffs))
    applied = cur_code != c_code
    return {"applied": applied, "stage": [s[0] for s in stages] or None,
            "new_code": cur_code, "diffs_before": before, "diffs_after": cur_diffs,
            "note": " -> ".join(f"{n} ({d})" for n, d in stages) or
                    "kein Treffer (weder Stufe 1 noch Stufe 2)"}
