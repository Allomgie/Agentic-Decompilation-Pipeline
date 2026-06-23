"""immediate-value Experte (deterministisch, orakel-verifiziert, KEIN AI).

Der Draft nutzt eine FALSCHE Konstante (gleiche Register, anderes Immediate). Der ZIELWERT steht
im Diff -> deterministische Literal-Substitution. KEIN AI noetig. (Quellen-Befund: Konstanten-
Materialisierung ist Standard -- 16-bit signed=addiu, unsigned=ori, 32-bit=lui+ori, Float=IEEE754-
Bitmuster; der Wert kommt aus dem C, die Regeln sind bekannt. Keine Transkription noetig.)

ECHTER immediate-diff: Register IDENTISCH, numerisches Immediate verschieden (NICHT register-order!).

Stufe 1 (direkt): addiu/ori/sltiu/sll/andi -> Draft-Literal im C durch Ziel-Literal ersetzen
  (dezimal+hex; bei sll auch die 2^n-Multiplikator-Interpretation).
Stufe 2 (lui+ori / Float): High-16-Bit-Konstante mit dem gepaarten ori/lo zum 32-bit-Wert
  kombinieren, ggf. als IEEE754-Float decodieren, Float-/Int-Literal im C ersetzen.
Alle orakel-gegated -> sicher per Konstruktion. Verifikation ueber modules/ido_compiler.
"""
import re, struct

_NUM = re.compile(r"^-?(0x[0-9A-Fa-f]+|\d+)$")
_SINGLE_IMM_OPS = {"addiu", "addi", "ori", "andi", "xori", "sltiu", "slti", "sll", "srl", "sra",
                   "sllv"}


def _toks(s):
    return [t.strip() for t in str(s).split(",")]


def imm_targets(entries):
    """-> Liste (op, draft_val, target_val). Register identisch, EIN numerisches Immediate weicht ab."""
    out = []
    for e in entries:
        if e.get("type") not in ("Register/Immediate", "Address Load"):
            continue
        t, d = e.get("target"), e.get("draft")
        if not (isinstance(t, str) and isinstance(d, str)):
            continue
        tt, dt = _toks(t), _toks(d)
        if len(tt) != len(dt):
            continue
        if any(x in ("sp", "fp") for x in tt + dt):
            continue  # sp/fp-Offset = Frame (p4), KEIN editierbares C-Literal -> nicht immediate
        diffs = []
        ok = True
        for a, b in zip(tt, dt):
            an, bn = _NUM.match(a), _NUM.match(b)
            if an and bn:
                if int(a, 0) != int(b, 0):
                    diffs.append((int(b, 0), int(a, 0)))  # (draft, target)
            elif a != b:
                ok = False; break
        if ok and len(diffs) == 1:
            out.append((e.get("op", ""), diffs[0][0], diffs[0][1]))
    return out


def _lit_forms(v):
    """Plausible C-Literal-Schreibweisen eines Integers (dezimal, hex, vorzeichen)."""
    forms = {str(v), hex(v) if v >= 0 else "-" + hex(-v)}
    if v >= 0:
        forms.add(hex(v).upper().replace("0X", "0x"))
    if 0 <= v <= 0xFFFF:
        forms.add(f"0x{v:X}"); forms.add(f"0x{v:x}")
    return {f for f in forms if f}


def _subst_literal(c_code, draft_v, target_v):
    """Ersetze Literal(e) mit Wert draft_v durch target_v. Liefert PER-VORKOMMEN-Kandidaten UND
    einen REPLACE-ALL-Kandidaten (geteilte Konstanten werden einmal geladen, muessen ueberall
    geaendert werden -- z.B. zweimal `128.0f`)."""
    if target_v == draft_v:
        return []
    out = []
    spans = []
    for m in re.finditer(r"(?<![\w.])(-?0x[0-9A-Fa-f]+|-?\d+)(?![\w.])", c_code):
        try:
            val = int(m.group(1), 0)
        except ValueError:
            continue
        if val == draft_v:
            rep = str(target_v) if not m.group(1).lower().startswith(("0x", "-0x")) else \
                  (hex(target_v) if target_v >= 0 else "-" + hex(-target_v))
            out.append(c_code[:m.start()] + rep + c_code[m.end():])
            spans.append((m.start(), m.end(), rep))
    if len(spans) > 1:                                  # REPLACE-ALL
        nc = c_code; off = 0
        for s, e, rep in spans:
            nc = nc[:s + off] + rep + nc[e + off:]; off += len(rep) - (e - s)
        out.append(nc)
    return out


def stage1_candidates(c_code, targets):
    out = []
    for op, dv, tv in targets:
        for cand in _subst_literal(c_code, dv, tv):
            out.append((f"imm:{dv}->{tv}({op})", cand))
        if op in ("sll", "srl", "sra", "sllv"):           # Shift: auch 2^n-Multiplikator
            if 0 <= dv <= 31 and 0 <= tv <= 31:
                for cand in _subst_literal(c_code, 1 << dv, 1 << tv):
                    out.append((f"shiftmul:{1<<dv}->{1<<tv}", cand))
    return out


def lui_pair_targets(entries):
    """lui-Immediate-Diffs (High-16-Bit). -> Liste (draft_hi, target_hi)."""
    out = []
    for e in entries:
        if e.get("op") != "lui":
            continue
        t, d = e.get("target"), e.get("draft")
        if not (isinstance(t, str) and isinstance(d, str)):
            continue
        tt, dt = _toks(t), _toks(d)
        if len(tt) == 2 and _NUM.match(tt[1]) and _NUM.match(dt[1]):
            dh, th = int(dt[1], 0), int(tt[1], 0)
            if dh != th:
                out.append((dh, th))
    return out


def _as_float(hi, lo):
    bits = ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def stage2_lui_candidates(c_code, lui_tgts):
    """High-16-Bit-Konstante: probiere Float-Decode (hi<<16, lo aus C-Float geschaetzt) sowie
    direkte hi-Substitution. Heuristik: ersetze C-Float-Literale, deren obere 16 Bits == draft_hi,
    durch den Float mit target_hi. Plus Integer-Form (hi<<16)."""
    out = []
    floats = list(re.finditer(r"(?<![\w.])(-?\d+\.\d+f?|-?\d+\.f?|-?\d+\.\d+e[+-]?\d+f?)", c_code))
    for dh, th in lui_tgts:
        # Float: finde C-Floats, deren Bitmuster oben dh hat -> ersetze durch Float mit th (lo gleich).
        # PER-VORKOMMEN + REPLACE-ALL (geteilte Float-Konstante wird einmal geladen).
        hits = []
        for m in floats:
            try:
                fv = float(m.group(1).rstrip("f"))
            except ValueError:
                continue
            bits = struct.unpack(">I", struct.pack(">f", fv))[0]
            if (bits >> 16) == (dh & 0xFFFF):
                newf = _as_float(th, bits & 0xFFFF)
                rep = repr(newf) + "f" if m.group(1).endswith("f") else repr(newf)
                out.append((f"luif:{fv}->{newf}", c_code[:m.start()] + rep + c_code[m.end():]))
                hits.append((m.start(), m.end(), rep, fv, newf))
        if len(hits) > 1:                               # REPLACE-ALL gleicher Float
            nc = c_code; off = 0
            for s, e, rep, _fv, _nf in hits:
                nc = nc[:s + off] + rep + nc[e + off:]; off += len(rep) - (e - s)
            out.append((f"luif-all:{hits[0][3]}->{hits[0][4]}", nc))
        # Integer-Form (hi<<16) als Literal
        for cand in _subst_literal(c_code, dh << 16, th << 16):
            out.append((f"luii:{dh<<16}->{th<<16}", cand))
    return out


def dbl2flt_suffix_candidates(c_code, lui_tgts):
    """double->float-Suffix: ein als DOUBLE geschriebenes C-Literal (`1.0`) materialisiert IDO mit dem
    double-High-Wort (lui 0x3FF0), das Target will aber float (lui 0x3F80). stage2_lui (float-Bitmuster-
    Match) verfehlt das strukturell, weil float32(1.0)-hi != draft-hi. Fix: an Double-Literale ein `f`
    anhaengen (orakel-gegated). Nur wenn lui-Immediate-Diffs vorliegen (= Konstanten-Materialisierung)."""
    if not lui_tgts:
        return []
    out = []
    # Floats OHNE f-Suffix (Double-Literale): 1.0 / .5 / 1. / 1e3 / 1.5e-2
    for m in re.finditer(r"(?<![\w.])(\d*\.\d+|\d+\.\d*|\d+(?:\.\d*)?[eE][+-]?\d+)(?![fF\w.])", c_code):
        lit = m.group(1)
        out.append((f"dbl2flt:{lit}f", c_code[:m.end()] + "f" + c_code[m.end():]))
    return out


def immediate_candidates(c_code, entries):
    cands = stage1_candidates(c_code, imm_targets(entries))
    cands += stage2_lui_candidates(c_code, lui_pair_targets(entries))
    cands += dbl2flt_suffix_candidates(c_code, lui_pair_targets(entries))
    seen = set(); out = []
    for lbl, code in cands:
        if code != c_code and code not in seen:
            seen.add(code); out.append((lbl, code))
    return out[:120]


def _count_imm(entries):
    return len(imm_targets(entries)) + len(lui_pair_targets(entries))


def immediate_expert(c_code, func_name, target_s_path, eval_fn=None, max_iter=8):
    """eval_fn(c, fn, sp) -> (total, register_immediate_and_addrload_entries). Iterativ, orakel-
    gegated. GATE: immediate-Diffs sinken UND Gesamt steigt nicht -> sicher."""
    before, ent = eval_fn(c_code, func_name, target_s_path)
    cur, cur_d, cur_ent = c_code, before, ent
    steps = []
    for _ in range(max_iter):
        if cur_d == 0:
            break
        cur_i = _count_imm(cur_ent)
        best = None
        for lbl, cand in immediate_candidates(cur, cur_ent):
            d, m = eval_fn(cand, func_name, target_s_path)
            if d is None or d > cur_d:
                continue
            i = _count_imm(m)
            if not ((d < cur_d) or (d == cur_d and i < cur_i)):
                continue
            key = (d, i)
            if best is None or key < (best[0], best[1]):
                best = (d, i, cand, lbl, m)
        if best is None:
            break
        cur_d, cur, cur_ent = best[0], best[2], best[4]
        steps.append((best[3], best[0]))
    return {"applied": cur != c_code, "new_code": cur, "diffs_before": before,
            "diffs_after": cur_d, "steps": steps}
