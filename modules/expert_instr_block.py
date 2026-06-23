"""instr-block Experte (KI, Reasoning) — PROMPT-BUILDER mit ANKER.

instr-block = der Draft ist strukturell nah (gleiche Laenge, meiste Positionen matchen), schrieb
aber an EINEM Block die FALSCHE Operation -> falsche Opcodes. Das Target-ASM ist Ground Truth.

Strategie (Lukas): volle Funktion als Input (Kontext fuer korrektes C), ABER ein ANKER so spezifisch
wie moeglich pro Funktion -> die KI weiss exakt WO + WAS. Erzwingen kommt aus dem ORAKEL-LOOP
(kompilieren+diffen, nur Match passiert; Best-of-N). Quellen-Prinzip (Fred-Chow §3.7): die KI
schreibt die HIGH-LEVEL-OPERATION, IDOs Optimierer expandiert sie zu den Instruktionen.

Dieses Modul baut Prompts; der eigentliche KI-Call ist orakel-gegated (Pod noetig).
"""
import re

# ---- ASM-Zeilen aus den Diff-Eintraegen parsen (ZWEI Formate) --------------------
# TARGET: '/* hex hex hex */  lhu  $t8, 0x12($a0)'  (Offsets HEX)
# DRAFT:  '14:\\t88820012 \\tlwl\\tv0,18(a0)'        (objdump, Offsets DEZIMAL)
_ASM_T = re.compile(r"\*/\s*([a-z][a-z0-9.]*)\s*(.*)$")


def _parse(asm_line):
    """-> (mnemonic, operands ohne $). Beide Formate."""
    if "*/" in asm_line:
        m = _ASM_T.search(asm_line)
        if m:
            return m.group(1), m.group(2).strip().replace("$", "")
        return None, None
    parts = asm_line.split("\t")
    if len(parts) >= 3:                       # ['14:', 'hex ', 'lwl', 'v0,18(a0)']
        ops = parts[3].strip() if len(parts) > 3 else ""
        return parts[2].strip(), ops.replace("$", "")
    return None, None


# ---- Decoder: Instruktion -> kurze C-Bedeutung (Scaffolding) ----------------------
_LOAD = {"lb": "s8", "lbu": "u8", "lh": "s16", "lhu": "u16", "lw": "s32",
         "lwc1": "f32", "ldc1": "f64"}
_STORE = {"sb": "s8", "sh": "s16", "sw": "s32", "swc1": "f32", "sdc1": "f64"}
_FOP = {"add.s": "f32 +", "sub.s": "f32 -", "mul.s": "f32 *", "div.s": "f32 /",
        "neg.s": "f32 negate", "add.d": "f64 +", "mul.d": "f64 *"}
_OFF = re.compile(r"(-?(?:0x[0-9a-fA-F]+|\d+))\(")


def _f32_from_lui(val):
    """Float-Wert aus dem lui-Operanden. Voll-32-Bit (0x40400000) direkt; 16-Bit-Hi (0x4040) <<16."""
    import struct
    try:
        v = int(val, 16)
        if v <= 0xFFFF:
            v <<= 16
        return round(struct.unpack(">f", struct.pack(">I", v & 0xFFFFFFFF))[0], 6)
    except Exception:
        return None


def _decode(mnem, ops):
    if mnem in _LOAD:
        off = _OFF.search(ops)
        return f"load {_LOAD[mnem]} at offset {off.group(1) if off else '?'}"
    if mnem in _STORE:
        off = _OFF.search(ops)
        return f"store {_STORE[mnem]} at offset {off.group(1) if off else '?'}"
    if mnem in _FOP:
        return _FOP[mnem]
    if mnem == "lui":
        m = re.search(r",\s*(?:\(?\s*)?(0x[0-9a-fA-F]+)", ops)
        if m:
            f = _f32_from_lui(m.group(1))
            return f"constant hi={m.group(1)}" + (f" (= {f}f as float)" if f is not None else "")
    if mnem in ("jal", "jalr"):
        return f"call {ops}"
    if mnem in ("mtc1", "mfc1", "cvt.s.w", "cvt.w.s", "cvt.s.d", "trunc.w.s"):
        return f"float conversion ({mnem})"
    if mnem in ("sll", "sra", "srl"):
        return f"shift {mnem} {ops}"
    if mnem in ("c.eq.s", "c.lt.s", "c.le.s"):
        return f"float compare {mnem}"
    return None


def decode_block(asm_lines):
    """-> lesbare Operations-Zusammenfassung des Target-Blocks (Scaffolding fuer die KI)."""
    out = []
    for a in asm_lines:
        mn, ops = _parse(a)
        if not mn:
            continue
        d = _decode(mn, ops)
        out.append(f"{mn} {ops}" + (f"   // {d}" if d else ""))
    return out


# ---- ANKER: die C-Stelle finden, die den FALSCHEN Block erzeugt -------------------
def _off_forms(raw):
    """Offset-Repraesentationen fuer C-Suche: dez->hex normalisieren (18 -> 0x12), beide behalten."""
    forms = {raw}
    try:
        v = int(raw, 16) if raw.lower().startswith("0x") else int(raw)
        forms.add(hex(v)); forms.add(str(v)); forms.add(f"0x{v:X}"); forms.add(f"0x{v:x}")
    except Exception:
        pass
    return forms


def _off_in_deref(off, line):
    """Offset NUR als echten Anker werten, wenn er in Pointer-Arithmetik/Index steht (`+ off`, `[off]`),
    NICHT als nackte RHS-Konstante (`new_var = 4`). Wort-Grenze verhindert Substring-Treffer (4 in 46).
    KALIBRIER-BEFUND: kleine Offsets (4,8) matchten sonst unzusammenhaengende Literale -> KI loeschte
    falsche Statements (func_8011607C: miss 1->2)."""
    o = re.escape(off)
    return bool(re.search(r"\+\s*" + o + r"\b", line) or re.search(r"\[\s*" + o + r"\s*\]", line))


def _anchor_tokens(asm_lines):
    """Distinktive Tokens: Call-Namen, Offset-Formen (dez+hex), grosse Immediate-Konstanten."""
    calls, offs, imms = [], set(), set()
    for a in asm_lines:
        mn, ops = _parse(a)
        if not ops:
            continue
        if mn in ("jal", "jalr"):
            calls.append(ops.split()[0].split(",")[0])
        for m in _OFF.finditer(ops):
            offs |= _off_forms(m.group(1))
        for m in re.finditer(r"(0x[0-9a-fA-F]{3,}|\b\d{4,})\b", ops):
            imms.add(m.group(1))
    return calls, offs, imms


def find_anchor(c_code, draft_asm_lines, target_asm_lines):
    """-> (anchor_text, level): EIN Anker (Rueckwaerts-Kompat). Nutzt find_anchors."""
    al = find_anchors(c_code, draft_asm_lines, target_asm_lines)
    if al:
        return al[0][0], al[0][1]
    return None, "no unique anchor (positional / model-localized)"


_ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)\s*=(?!=)")              # NUR plain `=` (Ueberschreibung)
_LHS_VAR = re.compile(r"^\s*([A-Za-z_]\w*)\s*[-+*/&|^%]?=(?!=)")  # LHS-Var, auch Compound (`+=`,`^=`...)


def overwrite_conflicts(c_code, anchor_texts):
    """DETERMINISTISCHE Kodierung des Thinking-Schritts (Wurzel A): findet die Zeile, die den Wert des
    Anker-Loads UEBERSCHREIBT. Muster (in allen 3 verifizierten Faellen identisch): Anker `var = ...`,
    die UNMITTELBAR folgende Anweisung ist `var = <expr ohne var>` -> reine Ueberschreibung -> IDO
    eliminiert den Load. Statt die KI das erschliessen zu lassen (gelingt nur mit Thinking), zeigen wir
    die Zeile dem Oneshot EXPLIZIT als 'loeschen'. Gate-geschuetzt: falsche Loeschung -> missing-block waechst."""
    lines = [ln.strip() for ln in c_code.splitlines() if ln.strip()]
    out = []
    for a in anchor_texts:
        m = _LHS_VAR.match(a)            # Anker-LHS (auch `+=`)
        if not m or a not in lines:
            continue
        var = m.group(1)
        i = lines.index(a)
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            mn = _ASSIGN.match(nxt)
            rhs = nxt.split("=", 1)[1] if "=" in nxt else ""
            if mn and mn.group(1) == var and not re.search(r"\b" + re.escape(var) + r"\b", rhs):
                if nxt not in out:
                    out.append(nxt)                  # `var = <ohne var>` direkt nach Anker = tote Ueberschreibung
    return out


def find_anchors(c_code, draft_asm_lines, target_asm_lines):
    """-> Liste (anchor_text, level) ALLER C-Zeilen, die zum divergenten Block gehoeren.
    WICHTIG (Kalibrier-Befund): ein Block kann MEHRERE Statements umfassen (z.B. 4 Float-Stores)
    -- alle muessen gefixt werden, nicht nur eines. Sammelt Treffer ueber alle Block-Tokens
    (Call-Namen, Offsets dez/hex, grosse Konstanten)."""
    lines = [ln.strip() for ln in c_code.splitlines() if ln.strip()]
    d_calls, d_offs, d_imms = _anchor_tokens(draft_asm_lines)
    t_calls, t_offs, t_imms = _anchor_tokens(target_asm_lines)
    out = {}  # line-text -> level (erste Begruendung)
    for call in dict.fromkeys(d_calls + t_calls):
        if not call:
            continue
        for ln in lines:
            if call in ln:
                out.setdefault(ln, f"call '{call}'")
    for off in sorted(d_offs | t_offs, key=len, reverse=True):
        if off in ("0", "0x0"):
            continue
        for ln in lines:
            if _off_in_deref(off, ln):       # nur Deref-Kontext, kein nacktes Literal/Substring
                out.setdefault(ln, f"offset {off}")
    for imm in (d_imms | t_imms):
        for ln in lines:
            if imm in ln:
                out.setdefault(ln, f"constant {imm}")
    return [(ln, lvl) for ln, lvl in out.items()]


# ---- PROMPT ----------------------------------------------------------------------
_SYS = (
    "You fix MIPS->C decompilations for the SGI IDO 5.3 -O2 compiler (Banjo-Tooie N64).\n"
    "The TARGET ASSEMBLY is ground truth. The C function already matches the target EVERYWHERE\n"
    "EXCEPT the anchored spot(s). Rewrite ONLY the anchored C so IDO compiles it to the target\n"
    "instructions; keep every other statement EXACTLY (byte-for-byte).\n"
    "RULES (violating them makes it worse):\n"
    "- Write the HIGH-LEVEL C OPERATION the target computes; IDO's optimizer emits the instructions.\n"
    "  Do NOT transcribe instructions literally, do NOT write expanded shift-add chains.\n"
    "- Match TYPE/WIDTH exactly: lb=s8 lbu=u8 lh=s16 lhu=u16 lw=s32; lwc1/swc1 = f32 (FLOAT).\n"
    "- CRITICAL: a swc1/lwc1 (float load/store) MUST use an f32* cast. NEVER store/load a float through\n"
    "  an s32* cast -- `*(s32*)p = 1.0f` is WRONG (it truncates to int); use `*(f32*)p = 1.0f`.\n"
    "- A float constant from `lui 0xHHHH` is the f32 whose top 16 bits are 0xHHHH (e.g. 0x4040=3.0f).\n"
    "- BYTE OFFSETS: always cast the base to (char*) FIRST: `*(T*)((char*)ptr + 0xN)`. Writing\n"
    "  `*(T*)(ptr + 0xN)` is WRONG for a typed pointer -- it scales 0xN by sizeof(*ptr).\n"
    "- A target block can span SEVERAL C statements (e.g. four float stores). Fix ALL anchored\n"
    "  statements so the ENTIRE target block is reproduced -- not just one of them.\n"
    "- Do NOT add or remove statements outside the block. Do NOT invent types, fields, or headers.\n"
    "Reply with ONLY the full corrected C function. No markdown, no code fences, no explanation."
)

# Chirurgische Lockerung (Wurzel A, Kalibrier-Befund): manchmal SCHREIBT die KI den richtigen Load,
# aber eine BENACHBARTE Anweisung im selben Datenfluss UEBERSCHREIBT/toetet den Wert -> IDOs Optimierer
# eliminiert die Ziel-Instruktion (Dead-Code-Elimination) -> kein Fortschritt, jeder Versuch identisch.
# Beispiel: Anker soll `x = *(s32*)(p+0x30)` laden, aber die naechste Zeile `x = param_0;` killt das.
# Erlaube das ENTFERNEN/VERSCHMELZEN genau dieser kollidierenden Anweisung. Gate-geschuetzt: loescht die
# KI faelschlich etwas Noetiges, waechst missing-block -> Gate verwirft -> kein Schaden.
_RELAX_RULE = (
    "\n- DEAD/CONFLICTING STATEMENT EXCEPTION: If a statement in the SAME data flow as the anchor\n"
    "  OVERWRITES or KILLS the value the target instruction produces (so IDO's optimizer would\n"
    "  eliminate the target instruction), you MAY delete or merge THAT ONE statement. Example: the\n"
    "  anchor should load into `x`, but the next line `x = param_0;` overwrites it -- remove that line\n"
    "  so the load's result is actually used. Touch ONLY the statement in direct conflict; add/remove\n"
    "  nothing else."
)


def build_prompt(c_code, func_name, diff_entries, relax_conflicts=False):
    """(system, user). diff_entries = die 'Instruction Mismatch'-Eintraege (mit target/draft asm-Listen).
    relax_conflicts=True haengt die DEAD/CONFLICTING-STATEMENT-Ausnahme an (Wurzel A, gate-geschuetzt)."""
    sysm = _SYS + (_RELAX_RULE if relax_conflicts else "")
    blocks = []
    anchors = []
    for e in diff_entries:
        if e.get("type") != "Instruction Mismatch":
            continue
        tgt = e.get("target") or []
        drf = e.get("draft") or []
        if not isinstance(tgt, list):
            continue
        dec = decode_block(tgt)
        blocks.append("// Block @ " + str(e.get("pos", "?")) + "\n// TARGET should produce:\n  "
                      + "\n  ".join(dec))
        for anch, lvl in find_anchors(c_code, drf if isinstance(drf, list) else [], tgt):
            ax = f"- {anch}\n    => {lvl}"
            if ax not in anchors:
                anchors.append(ax)
    parts = []
    parts.append("// === TARGET ASSEMBLY (ground truth — the C must compile to THESE instructions) ===\n"
                 + "\n\n".join(blocks))
    if anchors:
        parts.append("// === ANCHOR: ALL these C spots currently produce the WRONG block — fix them ALL\n"
                     "// so the ENTIRE target block above is reproduced (keep the rest of the function exact) ===\n"
                     + "\n".join(anchors))
    else:
        parts.append("// === ANCHOR ===\n// no unique anchor found — locate the divergent spot yourself "
                     "from the target block and the full function.")
    # Root A, concrete: show detected overwrites explicitly as DELETE (relax mode only) -- removes the
    # reasoning step from the oneshot (thinking infers it, oneshot does not).
    if relax_conflicts:
        anchor_lines = [a.split("\n")[0][2:] for a in anchors]   # "- <line>" -> "<line>"
        kill = overwrite_conflicts(c_code, anchor_lines)
        if kill:
            parts.append("// === DELETE THESE LINE(S) — they overwrite the loaded anchor value, so IDO\n"
                         "// eliminates the target load. Remove EXACTLY these line(s), nothing else ===\n"
                         + "\n".join(f"- DELETE: {k}" for k in kill))
    parts.append("// === FULL C FUNCTION (change only the anchor spot, keep the rest exact) ===\n" + c_code)
    return sysm, "\n\n".join(parts)


# ---- EXPERTE (orakel-gegateter KI-Loop, pod-ready) --------------------------------
def instr_block_expert(code, func_name, target_s_path, ai_call, diff_fn,
                       count_fn=None, max_rounds=4, samples=1, relax_conflicts=True):
    """Iterativer KI-Loop, ORAKEL-gegated (nur Verbesserung bleibt; Best-of-N via samples).
    diff_fn(code, fn, sp) -> (total, entries). count_fn default = diff_fn-total.
    ai_call(system, user) -> c_code (oder None). Returns {applied, new_code, diffs_before, diffs_after, steps}.
    Greift nur auf 'Instruction Mismatch'-Bloecke; bei 0 solchen -> kein Job hier (z.B. missing-block).
    relax_conflicts=True: ESKALATION -- erst STRIKT (saubere Faelle unberuehrt), bei Stillstand EINMAL auf
    die DEAD/CONFLICTING-Lockerung (Wurzel A: tote Ueberschreibung loeschen) umschalten. Gate-geschuetzt."""
    if count_fn is None:
        count_fn = lambda c, f, s: (diff_fn(c, f, s)[0])
    before = count_fn(code, func_name, target_s_path)
    cur, cur_d, steps = code, before, []
    relax = False                          # strikt zuerst; bei Stillstand auf Lockerung eskalieren
    for _ in range(max_rounds):
        if cur_d in (0, None):
            break
        t, ents = diff_fn(cur, func_name, target_s_path)
        im = [e for e in ents if e.get("type") == "Instruction Mismatch" and isinstance(e.get("target"), list)]
        if not im:
            break
        sysm, usr = build_prompt(cur, func_name, ents, relax_conflicts=relax)
        best, best_d = None, cur_d
        for _ in range(max(1, samples)):
            try:
                new = ai_call(sysm, usr)
            except Exception:
                new = None
            if new and new.strip() and new.strip() != cur.strip():
                d = count_fn(new, func_name, target_s_path)
                if d is not None and d < best_d:
                    best, best_d = new, d
        if best is None:
            if relax_conflicts and not relax:    # ESKALATION: stuck -> einmal mit Lockerung (Wurzel A)
                relax = True
                continue
            break
        cur, cur_d = best, best_d
        steps.append(("instr-block-ai", "relax" if relax else "strict", best_d))
    return {"applied": cur != code, "new_code": cur, "diffs_before": before,
            "diffs_after": cur_d, "steps": steps}
