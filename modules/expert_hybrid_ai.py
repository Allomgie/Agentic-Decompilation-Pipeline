"""STUFE-2-KI fuer die Hybrid-Experten instr-wrong-width / instr-wrong-op (Produktion).

Architektur (Lukas, 2026-06): Stufe 1 DETERMINISTISCH (im jeweiligen Experten), Stufe 2 = EIN ONESHOT-
KI-Call NUR auf dem Rest, den die Mechanik nicht loesen konnte. Die KI ist hier ein KANDIDATEN-Generator
wie die deterministischen Transforms -> das vorhandene Orakel-Gate des Experten entscheidet (keine neue
Gate-Logik, nie Verschlechterung). Stufe 3 (Routing permuter/garbage) macht die Pipeline downstream.

Prompts sind das Ergebnis der Kalibrierung (oneshot vs thinking, Reasoning-Analyse) -> siehe
memory/hybrid_op_width_calibration.md. ENGLISCH (Lukas-Regel). Basis = ib.build_prompt (Anker+Decode),
davor der Detektor-Constraint (det. Versagen als Constraint: width_targets/op_targets liefern das Ziel
exakt, die KI schreibt nur die richtige Form). Lazy imports gegen Zirkularitaet (width/op importieren dies).
"""

_WIDTH_RULE = (
    "// === DETECTOR FINDING (ground truth; the mechanical fix FAILED) ===\n"
    "// The target requires exactly this width/type at these spots; the draft produces the WRONG one.\n"
    "// A mechanical cast/decl change was tried and does NOT reproduce it -> the value lives in a\n"
    "// struct field / multi-word type / nested or array-index access, OR the return/usage type blocks\n"
    "// the simple cast. Write the EXACT typed form:\n"
    "{targets}\n"
    "// RULES: lb=s8 lbu=u8 lh=s16 lhu=u16 lw=s32 lwc1/swc1=f32 (FLOAT). Byte offset: *(T*)((char*)p+0xN).\n"
    "// MINIMAL EDIT: change ONLY the cast/width token at the anchor (e.g. (u8 *) -> (s32 *)). Keep the\n"
    "//   ENTIRE rest of the address expression byte-for-byte: every ->field, every nested *(T*), every\n"
    "//   + 0xN. Do NOT simplify, drop an inner dereference, or restructure the address -- that creates\n"
    "//   new missing/extra instructions. Copy the address expression CHARACTER-FOR-CHARACTER even if it\n"
    "//   looks redundant or strange (e.g. (0, x), ((0, p))->field + 0xN, comma expressions): ONLY the\n"
    "//   cast TYPE token changes, nothing else -- no cleanup, no rename.\n"
    "// WHICH CAST: locate the anchor by the DRAFT instruction in the finding, NOT by guessing. The draft\n"
    "//   mnemonic maps to exactly one cast width: lbu->(u8 *)  lb->(s8 *)  lhu->(u16 *)  lh->(s16 *)\n"
    "//   lw->(s32 *)/(u32 *)/pointer  lwc1->(f32 *). Find the cast in the draft matching the DRAFT\n"
    "//   mnemonic and retype THAT one to the target width. Example: finding 'target lw, draft lbu' with\n"
    "//   *(u8 *)(*(u32 *)(p + 0xC)) -> the lbu is the OUTER (u8 *) (the (u32 *) is already lw); change the\n"
    "//   (u8 *) to (s32 *), leave the (u32 *) alone. If several casts match the draft mnemonic, use the\n"
    "//   offset 0xN (a deref with no + is offset 0) to pick the right one.\n"
    "// NEVER delete or omit an existing statement (especially the return). Keep every statement; only\n"
    "//   retype/recast at the anchor.\n"
    "// NO CAST TO CHANGE? (array index or declaration-typed access) -- the minimal-edit rule above does\n"
    "//   NOT apply when the wrong width comes from a DECLARATION, not a cast. Two transforms are then\n"
    "//   REQUIRED (do NOT return unchanged):\n"
    "//   (a) ARRAY INDEX `arr[i]` with no cast: rewrite it as a byte-offset deref at the finding's offset,\n"
    "//       PRESERVING the address: `arr[i]` -> `*(T*)((char*)arr + 0xN)` (N = the load displacement in\n"
    "//       the finding; T per the width: s16/u16/s8/u8/s32/f32). Do NOT use `((T*)arr)[i]` -- that\n"
    "//       changes the index stride and is WRONG.\n"
    "//   (b) GLOBAL ARRAY whose ELEMENT type is wrong (e.g. `extern int D[]` loads lw but target is lh):\n"
    "//       change the `extern` DECLARATION element type to match (`extern s16 D[];`). Allowed here.\n"
    "// FLOAT FLOW (lwc1/swc1): the value must FLOW as a float, NOT via an int cast (an int cast forces\n"
    "//   cvt.w.s = wrong). If the anchored float load flows into the RETURN value, CHANGE THE FUNCTION\n"
    "//   RETURN TYPE to f32 (the original int signature is a wrong decompiler guess; lwc1 into the\n"
    "//   return register means the function returns float). Likewise make the carrying variable/field f32.\n"
    "// INTEGER RETURN: if the load is WIDER than the function's return type (e.g. lw=s32 into a\n"
    "//   u8-returning function), KEEP the return type and add NO return cast -- just fix the deref width;\n"
    "//   IDO truncates on return. Only FLOAT loads change the return type.\n"
)
_OP_RULE = (
    "// === DETECTOR FINDING (ground truth; the mechanical fix FAILED) ===\n"
    "{targets}\n"
    "// SHIFT SIGNEDNESS (the MOST COMMON case here): srl vs sra (and sll/srl/sra mismatches) are NOT a\n"
    "//   different C operator. The `>>` / `<<` is ALREADY in the code -- DO NOT change the operator and\n"
    "//   DO NOT return the code unchanged. The fix is the OPERAND TYPE via a cast:\n"
    "//     - target srl (LOGICAL >>)  -> make the shifted operand UNSIGNED: (u32)x >> n\n"
    "//     - target sra (ARITHMETIC >>) -> make the shifted operand SIGNED:   (s32)x >> n\n"
    "//   Reason: a u16/u8 or int operand is promoted to signed int -> sra; casting to (u32) forces srl.\n"
    "//   CRITICAL: even if the operand is ALREADY a narrow unsigned type like *(u16 *) or *(u8 *), C\n"
    "//     integer promotion converts it to SIGNED int before the shift, so >> still emits sra. The\n"
    "//     existing u16/u8 is NOT enough -- you MUST add an explicit (u32) cast: (u32)(*(u16 *)p) >> n.\n"
    "//     Do NOT conclude 'already unsigned, no change needed' -- that is the most common mistake here.\n"
    "//   A signed operand like *(s32 *)p >> n is sra; wrap it (u32)(*(s32 *)p) >> n for srl. Keep any\n"
    "//     surrounding arithmetic (e.g. (... >> 25) - 1) intact -- wrap ONLY the shifted operand.\n"
    "//   For a bitfield extract ((x << a) >> b), cast the SAME operand: ((u32)x << a) >> b for srl.\n"
    "//   Add ONLY the cast; keep every other token (the shift amounts, the field access) byte-for-byte.\n"
    "// REAL OPERATOR SWAP (only when both are arithmetic/logical ops, NOT shifts): and/or/xor = &/|/^\n"
    "//   (e.g. xori vs andi -> change & to ^). slt/sltu = signed/unsigned compare (cast the operands).\n"
    "// MISSING OP (target is srl/sll/andi but draft is `move`): the draft dropped the operation entirely\n"
    "//   (the operand made it a no-op, e.g. >>0 or &0xFFFFFFFF). Restore the real shift amount / mask so\n"
    "//   the instruction is emitted; do NOT leave it as a plain copy.\n"
    "// STRENGTH-REDUCTION ONLY (addu vs subu / add vs sub from a multiply, NO shift and NO logical op in\n"
    "//   the C): this is pure codegen -> do NOT guess, return the code UNCHANGED (the permuter solves it).\n"
    "//   This escape applies ONLY to addu/subu/add/sub pairs -- NEVER to shifts (srl/sra/sll).\n"
)


def detector_targets_text(ents, kind):
    """Human-readable description of width_targets/op_targets for the constraint block (ENGLISH = prompt)."""
    from modules import expert_instr_width as eiw, expert_instr_op as eio, expert_instr_block as ib
    out = []
    if kind == "width":
        for (tgt_mn, drf_mn, off, base, typ) in eiw.width_targets(ents):
            ld = "load" if tgt_mn in ib._LOAD else "store"
            out.append(f"//   - at offset 0x{off:X} (base reg {base}): {ld} {typ}  (target {tgt_mn}, draft {drf_mn})")
    else:
        _SHIFT = {"srl", "sra", "sll"}
        for (a, b) in eio.op_targets(ents):
            if a in _SHIFT or b in _SHIFT:
                if a == "srl":
                    hint = "target srl = LOGICAL >> -> cast the shifted operand to (u32) ((u32)x >> n)"
                elif a == "sra":
                    hint = "target sra = ARITHMETIC >> -> cast the shifted operand to (s32) ((s32)x >> n)"
                elif a == "sll":
                    hint = "target sll = << -> ensure the operand width/value emits the left shift (do not drop it)"
                else:
                    hint = f"target {a}"
                out.append(f"//   - SHIFT at this spot: draft emits '{b}', target needs '{a}'. {hint}. "
                           f"The >>/<< is already in the code; fix the operand CAST, not the operator.")
            else:
                out.append(f"//   - operator: target '{a}' instead of draft '{b}' (real operator swap: &|^ etc.)")
    return "\n".join(out) if out else "//   (no unique detector target)"


def build_hybrid_prompt(code, func_name, ents, kind):
    """(system, user) fuer width/op-Stufe-2. Basis = ib.build_prompt (Anker+Decode), davor Detektor-
    Constraint. kind in {'width','op'}."""
    from modules import expert_instr_block as ib
    sysm, base_usr = ib.build_prompt(code, func_name, ents, relax_conflicts=True)
    rule = _WIDTH_RULE if kind == "width" else _OP_RULE
    constraint = rule.format(targets=detector_targets_text(ents, kind))
    return sysm, constraint + "\n" + base_usr


def _invoke(ai_call, sysm, usr):
    """ai_call-Konvention wie compile_fix: think-kwarg optional. Produktion = ONESHOT (kein thinking)."""
    try:
        return ai_call(sysm, usr, think=False)
    except TypeError:
        return ai_call(sysm, usr)


def make_ai_call(func_name="hybrid-ai", temperature=0.2, max_tokens=6144):
    """Produktions-Adapter: wickelt ai_agent's lokalen Multi-Pod-Chat in die (system,user,think)-Konvention
    der Experten. Lazy import (der no-AI-Pfad braucht ai_agent/openai NICHT). Nutzt das echte Pod-Backend
    (_local_chat_create = least-in-flight + Failover), Thinking-Flag via _thinking_extra_body, robustes
    Code-Extract via _extract_c_code. Verwendung: orchestrate(..., ai=expert_hybrid_ai.make_ai_call())."""
    import modules.ai_agent as aa

    def ai_call(system, user, think=False):
        resp = aa._local_chat_create(
            func_name, model=aa.LOCAL_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature, max_tokens=max_tokens,
            extra_body=aa._thinking_extra_body(aa.MODEL_FAMILY, enable_thinking=think))
        raw = (resp.choices[0].message.content or "")
        return aa._extract_c_code(raw)

    return ai_call


def hybrid_ai_candidate(code, func_name, ents, kind, ai_call):
    """EIN Oneshot-KI-Versuch auf dem Rest. Gibt [(label, new_code)] (0 oder 1) zurueck -- als Kandidat
    fuer das Orakel-Gate des Experten. KEINE eigene Gate-Entscheidung hier (der Experte gated). Bei Fehler
    oder unveraendert/leer -> []."""
    if ai_call is None:
        return []
    try:
        sysm, usr = build_hybrid_prompt(code, func_name, ents, kind)
        new = _invoke(ai_call, sysm, usr)
    except Exception:
        return []
    if not new or not new.strip() or new.strip() == code.strip():
        return []
    return [(f"ai-{kind}", new)]
