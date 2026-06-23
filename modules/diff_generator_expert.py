"""Experten-Variante des Diff-Generators (Test-Kopie, NICHT produktiv).

Idee (aus der Thinking-Block-Analyse 2026-06-07): die Experten scheitern primaer an einer
LOKALISIERUNGS-Luecke — die KI versteht den Fix, kann den Diff aber nicht auf die C-Stelle
zurueckfuehren. Diese Variante reichert Diff-Eintraege um einen SEMANTISCHEN ANKER an, der aus
dem asm-KONTEXT selbst abgeleitet ist (robust, ohne fragile -O2-Zeilennummern):

  - immediate/address in ein Argument-Register (a0-a3) gesetzt + folgt ein `jal` ->
    "argument N to the call <callee>"  (loest die 'mehrere identische Call-Sites'-Ambiguitaet).

Wrapper um modules.diff_generator (kein Fork): nutzt dessen Parser, haengt nur 'anchor' an.
A/B: gleiche Funktionen, oneshot no-think, anchored vs plain.
"""
import json, re, struct
from modules import diff_generator as _dg

# re-export, damit Aufrufer diff_generator_expert wie das Original nutzen koennen
create_json_diff = _dg.create_json_diff
DIFF_GROUP_THRESHOLD = _dg.DIFF_GROUP_THRESHOLD

_AREG = {"a0": 1, "a1": 2, "a2": 3, "a3": 4}

def _callee(raw: str):
    """Callee-Symbol aus einer jal-Rohzeile (spimdisasm 'jal func_x' / objdump '<func_x>')."""
    if not raw:
        return None
    m = re.search(r"\bjal\b\s+([A-Za-z_.][\w.]*)", raw)
    if m:
        return m.group(1)
    m = re.search(r"<([^>+]+)", raw)
    if m:
        return m.group(1)
    return None

def _first_areg(operands: str):
    """Erstes Argument-Register (a0-a3) im Operanden-String, oder None."""
    if not operands:
        return None
    for tok in operands.split(","):
        t = tok.strip().lstrip("$")
        if t in _AREG:
            return t
    return None

def arg_anchor(target_insns, pos, window=8):
    """Anker fuer einen immediate/address-Eintrag an Ziel-Index pos: in welches Arg-Register
    geht der Wert und welcher Call folgt? -> menschenlesbarer String oder None."""
    if not isinstance(pos, int) or not (0 <= pos < len(target_insns)):
        return None
    reg = _first_areg(target_insns[pos].get("operands", ""))
    if not reg:
        return None
    argn = _AREG[reg]
    for k in range(pos, min(pos + window, len(target_insns))):
        op = target_insns[k].get("opcode", "")
        if op in ("jal", "jalr"):
            c = _callee(target_insns[k].get("raw", ""))
            if op == "jalr":
                return f"this value is argument {argn} (register {reg}) to an indirect call (jalr)"
            if c:
                return f"this value is argument {argn} (register {reg}) of the call to {c}()"
            return f"this value is argument {argn} (register {reg}) of the following call"
    return None

def _last_imm(operands: str):
    """Letzter Operand als int (Immediate), oder None."""
    if not operands:
        return None
    tok = operands.split(",")[-1].strip()
    try:
        return int(tok, 0)
    except ValueError:
        return None

def _as_float(hi: int, lo: int):
    """lui(hi)+ori(lo) -> IEEE754-float + Hex-Wort. (hi,lo = 16-bit Haelften)"""
    word = ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)
    f = struct.unpack(">f", struct.pack(">I", word))[0]
    return f, word

def _fmt_float(f):
    return f"{f:g}f"

def _reg_of(operands):
    """Erstes Register im Operanden-String."""
    if not operands:
        return None
    return operands.split(",")[0].strip().lstrip("$") or None

def merge_float_pairs(entries, target_insns=None):
    """Dekodiert Float-Konstanten (lui+ori ODER lui-only, untere Haelfte=0) zu EINER
    'Float Constant'-Sicht mit dekodiertem Wert (Target vs Draft). Adressiert die ZWEITE
    Barriere (Wert-Dekodierung). Nutzt target_insns, um die ECHTE untere Haelfte zu finden,
    auch wenn die ori-Instruktion selbst matcht (nicht im Diff steht)."""
    by_pos = {e.get("pos"): e for e in entries if isinstance(e.get("pos"), int)}
    consumed = set()
    out = []
    for e in entries:
        p = e.get("pos")
        if p in consumed:
            continue
        if e.get("type") == "Address Load" and e.get("op") == "lui" and isinstance(p, int):
            reg = _reg_of(e.get("target", ""))
            t_hi, d_hi = _last_imm(e.get("target", "")), _last_imm(e.get("draft", ""))
            nxt = by_pos.get(p + 1)
            t_lo = d_lo = 0
            consume_next = False
            if nxt is not None and nxt.get("op") == "ori" and _reg_of(nxt.get("target", "")) == reg:
                # ori weicht ebenfalls ab -> beide Haelften aus dem Diff
                t_lo, d_lo = _last_imm(nxt.get("target", "")), _last_imm(nxt.get("draft", ""))
                consume_next = True
            elif target_insns is not None and p + 1 < len(target_insns):
                # ori matcht (nicht im Diff) -> echte (gleiche) untere Haelfte aus target_insns
                ni = target_insns[p + 1]
                if ni.get("opcode") == "ori" and _reg_of(ni.get("operands", "")) == reg:
                    lo = _last_imm(ni.get("operands", ""))
                    if lo is not None:
                        t_lo = d_lo = lo
            if None not in (t_hi, d_hi):
                tf, tw = _as_float(t_hi, t_lo); df, dw = _as_float(d_hi, d_lo)
                merged = {
                    "type": "Float Constant",
                    "pos": p,
                    "target_float": f"{_fmt_float(tf)} (0x{tw:08X})",
                    "draft_float": f"{_fmt_float(df)} (0x{dw:08X})",
                    "note": "this is a float literal (emitted as lui/lui+ori) — change the C float "
                            "literal from the draft value to the target value (if it is not a float, "
                            "it is the 32-bit constant shown in hex)",
                }
                if e.get("anchor"):
                    merged["anchor"] = e["anchor"]
                out.append(merged)
                consumed.add(p)
                if consume_next:
                    consumed.add(p + 1)
                continue
        out.append(e)
    return out

_MEMRE = re.compile(r"^([^,]+),(-?\d+)\(([^)]+)\)$")
def _mem_parse(op):
    m = _MEMRE.match(str(op).strip())
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None

def _field_desc(struct, off):
    """'name' (exakt), 'within name' (nested), oder None — via verifiziertem offsetof-Layout."""
    try:
        from modules import struct_layout as SL
    except Exception:
        return None
    name = SL.field_at(struct, off)
    if name:
        return name
    cont, base = SL.field_containing(struct, off)
    if cont is not None:
        return f"within {cont} (+0x{off-base:X})"
    return None

def _draft_struct_anchor(draft_layouts, toff, doff):
    """Idee B: Anker aus dem DRAFT-Struct-Layout. Findet die Struct, die real einen Member bei
    Offset doff hat (= aktuell zugegriffener), deckt die Name-vs-Realitaet-Diskrepanz auf und nennt
    das Ziel-Offset. -> str oder None."""
    for sname, L in (draft_layouts or {}).items():
        byoff = L["byoff"]
        if doff not in byoff:
            continue
        import re as _re
        cur = byoff[doff]
        tgt_member = byoff.get(toff)
        byname = L["byname"]
        # kompakte Ist-Layout-Tabelle (Name -> ECHTER Offset), Diskrepanz markiert
        items = []
        for n, o, s in L["fields"]:
            note = ""
            mm = _re.match(r"unk([0-9A-Fa-f]+)$", n)
            if mm:
                try:
                    implied = int(mm.group(1), 16)
                    if implied != o:
                        note = f"(name implies 0x{implied:X})"
                except ValueError:
                    pass
            items.append(f"{n}@0x{o:X}{note}")
        layout_str = ", ".join(items)
        named = f"unk{toff:X}"            # Konvention: unk<HEX> ist fuer offset <HEX> gedacht
        if tgt_member:
            goal = (f"actual offset 0x{toff:X} already holds member '{tgt_member}' — change the access "
                    f"to '{tgt_member}'")
        elif named in byname:
            goal = (f"member '{named}' is meant for offset 0x{toff:X} but is currently MISPLACED at "
                    f"0x{byname[named]:X} — fix the struct padding so '{named}' lands at 0x{toff:X}")
        else:
            goal = (f"actual offset 0x{toff:X} currently has no member — the struct is mis-laid-out; "
                    f"add/pad a member at 0x{toff:X} (name unkNN encodes its INTENDED hex offset)")
        return (f"draft struct {sname} ACTUAL layout: [{layout_str}]. This access currently hits "
                f"actual offset 0x{doff:X} (member '{cur}'); the TARGET needs offset 0x{toff:X}: {goal}. "
                f"(Offsets HEX; member NAMES do not match their real offsets — fix the struct.)")
    return None

def annotate_memory(entries, mem_list, draft_src=None):
    """Anker fuer Memory-Access-Eintraege: Offset in HEX + Struct des Basis-Registers (techenv 'mem')
    + — wenn der Struct in den Headern aufloesbar ist — den ECHTEN FELDNAMEN (verifiziert via
    offsetof, modules.struct_layout). Behebt: (a) dezimal/hex-Falle, (b) Layout-HALLUZINATION des
    Modells (z.B. 0x14 ist drvr, nicht uspt). Konservativ: ohne Struct/Aufloesung nur Hex."""
    base2struct = {}
    for m in (mem_list or []):
        b = str(m.get("base", "")).lstrip("$")
        if b and m.get("struct"):
            base2struct[b] = m["struct"]
    # Idee B: Draft-eigene Struct-Layouts (ad-hoc, kein Header).
    draft_layouts = {}
    if draft_src:
        try:
            from modules import struct_layout as SL
            draft_layouts = SL.draft_layout(draft_src)
        except Exception:
            draft_layouts = {}
    for e in entries:
        if e.get("type") != "Memory Access":
            continue
        pt, pd = _mem_parse(e.get("target", "")), _mem_parse(e.get("draft", ""))
        if not pt or not pd:
            continue
        base = pt[2]
        st = base2struct.get(base.lstrip("$"))
        toff, doff = pt[1], pd[1]
        # (1) GETEILTER Header-Struct mit aufloesbarem Feldnamen -> echter Name (Real-Typ-Pfad)
        if st:
            tf = _field_desc(st, toff)
            if tf:
                df = _field_desc(st, doff)
                dpart = f"'{df}' (offset 0x{doff:X})" if df else f"offset 0x{doff:X}"
                e["anchor"] = (f"struct {st} (base {base}): the TARGET accesses field '{tf}' "
                               f"(offset 0x{toff:X}); your DRAFT accesses {dpart}. Change the C "
                               f"member access to '{tf}'. (Verified struct layout — offsets are HEX.)")
                continue
        # (2) Idee B: Draft-eigene Struct -> ECHTES Layout aufdecken + Ziel-Offset
        da = _draft_struct_anchor(draft_layouts, toff, doff)
        if da:
            e["anchor"] = da
            continue
        # (3) Struct bekannt, Feld nicht aufloesbar -> Hex+Struct
        if st:
            e["anchor"] = (f"struct {st} (base {base}): TARGET field at offset 0x{toff:X}, your DRAFT "
                           f"at 0x{doff:X} — change the C member access to the 0x{toff:X} field "
                           f"(offsets are HEX).")
        else:
            e["anchor"] = (f"base {base}: TARGET reads offset 0x{toff:X}, your DRAFT reads 0x{doff:X} "
                           f"— change the access offset accordingly (offsets are HEX).")
    return entries

# Mnemonic -> C-Bedeutung (statisch, mechanisch). Adressiert wrong-op-Diagnose: Modell muss die
# asm-Semantik nicht mehr selbst herleiten, nur noch die C-Stelle finden + Operator/Cast tauschen.
_OPMEANING = {
    "c.lt.s": "< (less-than, float)", "c.lt.d": "< (less-than, double)",
    "c.le.s": "<= (less-or-equal, float)", "c.le.d": "<= (less-or-equal, double)",
    "c.eq.s": "== (equal, float)", "c.eq.d": "== (equal, double)",
    "add": "+ (signed add)", "addu": "+ (add)", "sub": "- (signed sub)", "subu": "- (subtract)",
    "mult": "* (signed multiply)", "multu": "* (unsigned multiply)",
    "div": "/ or % (signed)", "divu": "/ or % (unsigned)",
    "sll": "<< (left shift)", "sra": ">> (arithmetic right shift, SIGNED operand)",
    "srl": ">> (logical right shift, UNSIGNED operand)",
    "and": "& (bitwise and)", "or": "| (bitwise or)", "xor": "^ (bitwise xor)", "nor": "~(a|b)",
    "slt": "< (signed compare)", "sltu": "< (unsigned compare)",
    "slti": "< (signed compare, immediate)", "sltiu": "< (unsigned compare, immediate)",
    "trunc.w.s": "(s32) cast — float -> int truncation", "cvt.w.s": "(s32) cast — float -> int",
    "cvt.d.s": "(double) cast / promotion — float -> double",
    "cvt.s.d": "(float) cast — double -> float", "cvt.s.w": "(float) cast — int -> float",
    "cvt.d.w": "(double) cast — int -> double",
    "neg.s": "unary - (float negate)", "neg.d": "unary - (double negate)",
    "negu": "unary - (negate)", "mul.s": "* (float multiply)", "div.s": "/ (float divide)",
    "add.s": "+ (float add)", "sub.s": "- (float subtract)",
}
def annotate_wrongop(entries):
    """Anker fuer Instruction-Mismatch (echter Op-Swap): nennt die C-BEDEUTUNG beider Mnemonics,
    damit das Modell nur noch die C-Stelle finden + Operator/Cast tauschen muss."""
    for e in entries:
        if e.get("type") != "Instruction Mismatch":
            continue
        tg, dr = e.get("target"), e.get("draft")
        if not (isinstance(tg, list) and isinstance(dr, list) and len(tg) == 1 and len(dr) == 1):
            continue
        tmn = re.match(r"\s*(?:/\*.*?\*/)?\s*([a-z][a-z0-9.]*)", tg[0])
        dmn = re.match(r"^\s*[0-9a-f]+:\s*[0-9a-f]+\s*([a-z][a-z0-9.]*)", dr[0]) or \
              re.match(r"\s*([a-z][a-z0-9.]*)", dr[0])
        tk = tmn.group(1) if tmn else None
        dk = dmn.group(1) if dmn else None
        td, dd = _OPMEANING.get(tk), _OPMEANING.get(dk)
        if td or dd:
            e["anchor"] = (f"operation difference: the TARGET uses {tk} = {td or '?'}; your DRAFT "
                           f"uses {dk} = {dd or '?'}. Change the C operation to the TARGET meaning "
                           f"(same operands).")
    return entries

# Load/Store-Breite -> C-Typ. Adressiert wrong-width: Breite/Signedness ist deterministisch aus dem
# Opcode; das Modell muss nur noch den Typ des zugegriffenen Felds/Variable aendern.
_WIDTHTYPE = {
    "lb": "s8 (signed 8-bit)", "lbu": "u8 (unsigned 8-bit)",
    "lh": "s16 (signed 16-bit)", "lhu": "u16 (unsigned 16-bit)",
    "lw": "s32/u32 (32-bit word)", "lwc1": "f32 (float)",
    "sb": "8-bit (u8/s8)", "sh": "16-bit (u16/s16)", "sw": "32-bit (s32/u32)", "swc1": "f32 (float)",
    "ld": "s64 (64-bit)", "ldc1": "f64 (double)", "sd": "64-bit", "sdc1": "f64 (double)",
}
def _mn(line):
    s = re.sub(r"/\*.*?\*/", "", str(line))
    s = re.sub(r"^\s*[0-9a-f]+:\s*[0-9a-f]+\s*", "", s).replace("\t", " ").strip()
    m = re.match(r"([a-z][a-z0-9.]*)", s)
    return m.group(1) if m else None

def annotate_width(entries):
    """wrong-width-Anker: nennt die Ziel-BREITE/SIGNEDNESS als C-Typ (aus dem Opcode)."""
    for e in entries:
        if e.get("type") != "Instruction Mismatch":
            continue
        tg, dr = e.get("target"), e.get("draft")
        if not (isinstance(tg, list) and isinstance(dr, list) and len(tg) == 1 and len(dr) == 1):
            continue
        tm, dm = _mn(tg[0]), _mn(dr[0])
        tt, dt = _WIDTHTYPE.get(tm), _WIDTHTYPE.get(dm)
        if tt and dt and tm != dm:
            e["anchor"] = (f"access WIDTH/type is wrong: the TARGET uses {tm} = {tt}; your DRAFT uses "
                           f"{dm} = {dt}. Change the C TYPE of the accessed variable/field (or add a "
                           f"cast) so it is {tt}. (Same address; only the load/store size/sign is wrong.)")
    return entries

def _describe_extra(instr_list):
    """Charakterisiert eine 'Extra in Draft'-Instruktionsliste fuer einen Entfernungs-Hinweis.
    nop wird NICHT als entfernbar gezaehlt (= Scheduling/Delay-Slot, Permuter)."""
    calls, stores, loads, arith, nops = [], [], [], 0, 0
    for ln in instr_list:
        mn = _mn(ln) or ""
        if mn == "nop":
            nops += 1
        elif mn == "jal":
            m = re.search(r"\bjal\b\s+([A-Za-z_.][\w.]*)", str(ln))
            calls.append(m.group(1) if m else "a function")
        elif mn in ("sw", "sh", "sb", "swc1"):
            stores.append(mn)
        elif mn in ("lw", "lh", "lb", "lhu", "lbu", "lwc1"):
            loads.append(mn)
        elif mn:
            arith += 1
    parts = []
    if calls: parts.append("an extra call to " + " / ".join(sorted(set(calls))) + "()")
    if stores: parts.append(f"{len(stores)} extra store(s) (a redundant assignment)")
    if loads: parts.append(f"{len(loads)} extra load(s)")
    if arith: parts.append(f"{arith} extra computation(s)")
    desc = ", ".join(parts) if parts else "extra instructions"
    if nops:
        desc += f" (plus {nops} scheduling nop(s) — IGNORE those, the permuter handles delay slots)"
    return desc

def annotate_extra(entries):
    """extra-code-Anker: charakterisiert die ueberzaehligen Instruktionen + Entfernungs-Richtung."""
    for e in entries:
        if e.get("type") != "Extra in Draft":
            continue
        ex = e.get("extra") or []
        desc = _describe_extra(ex)
        e["anchor"] = (f"your DRAFT produces {desc} that the TARGET does NOT have. Find the C that "
                       f"emits this (a redundant assignment, an unused temporary/call, a superfluous "
                       f"statement or computation) and REMOVE it. Do not remove anything the target "
                       f"still needs.")
    return entries

def annotate_immediate(entries, target_s_path):
    """Haengt 'anchor' an immediate/address-Eintraege an (Argument-Register -> Call) und fasst
    lui+ori-Float-Paare zusammen. entries muss FLACH sein (integer 'pos')."""
    target_insns = _dg._parse_s_file(target_s_path)
    if target_insns:
        for e in entries:
            if e.get("type") in ("Register/Immediate", "Address Load"):
                a = arg_anchor(target_insns, e.get("pos"))
                if a:
                    e["anchor"] = a
    return merge_float_pairs(entries, target_insns)
