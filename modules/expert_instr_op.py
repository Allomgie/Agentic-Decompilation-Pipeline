"""instr-wrong-op Experte (deterministisch, orakel-verifiziert, KEIN AI).

Der Draft nutzt die falsche OPERATION (gleiche Operanden, anderer ALU/Branch-Opcode). Mnemonic
-> C-Operator ist deterministisch -> Operator im C tauschen. KEIN AI noetig.

WICHTIG (Befund 2026-06): die "wrong-op"-Klassifikation UEBERZAEHLT massiv durch (a) Misalignments
(versetzte Positionen), (b) Pseudo-op-Aequivalenzen (or x,y,zero=move; addiu x,zero,0=li=move =
codegen/register-order, KEIN Operator-Fehler), (c) move<->addiu (~70% = fehlender Immediate =
immediate-value-Domaene). GENUINE Operator-Fehler (&|^, Shift-Richtung, Vergleich, Branch-Polaritaet)
sind selten, aber voll deterministisch. Detektor verlangt daher GLEICHE OPERANDEN + andere Operation.

Drei Transform-Klassen (orakel-gegated -> sicher per Konstruktion):
- OPERATOR-SWAP: `&`<->`|`<->`^`, `<<`<->`>>`, `+`<->`-` (and/or/xor, sll/srl, addu/subu).
- BRANCH-INVERT: beq<->bne (`==`<->`!=`), blez<->bgtz, bltz<->bgez (Bedingung umdrehen).
- SIGN-CAST: srl<->sra, slt<->sltu, mult<->multu (gleicher Operator, Operand-Signedness via Cast).
Verifikation IMMER ueber modules/ido_compiler (PC-Freeze-Regel). KEIN Permuter.
"""
import re

# Mnemonic -> binaerer C-Operator (nur eindeutig swap-bare)
_BINOP = {"and": "&", "andi": "&", "or": "|", "ori": "|", "xor": "^", "xori": "^",
          "sll": "<<", "sllv": "<<", "srl": ">>", "srlv": ">>", "sra": ">>", "srav": ">>",
          "addu": "+", "add": "+", "subu": "-", "sub": "-", "mult": "*", "multu": "*"}
# Branch-Polaritaets-Paare (C-Operator)
_BRANCH = {"beq": "==", "bne": "!=", "beqz": "== 0", "bnez": "!= 0",
           "blez": "<= 0", "bgtz": "> 0", "bltz": "< 0", "bgez": ">= 0"}
_INVERT = {"==": "!=", "!=": "==", "<=": ">", ">": "<=", "<": ">=", ">=": "<"}
# Signedness pro Mnemonic: sra=arithmetic=SIGNED, srl=logical=UNSIGNED; slt/mult/div signed, *u unsigned
_SIGN = {"sra": "s", "srl": "u", "slt": "s", "sltu": "u", "mult": "s", "multu": "u",
         "div": "s", "divu": "u"}

_ALU = set(_BINOP) | set(_SIGN) | {"nor", "negu", "move", "not", "neg"}


def _parse(s):
    s = re.sub(r"/\*.*?\*/", "", str(s))
    s = re.sub(r"^\s*[0-9a-f]+:\s*[0-9a-f]+\s*", "", s)
    s = s.replace("\t", " ").strip()
    m = re.match(r"([a-z][a-z0-9.]*)\s+(.*)", s)
    if not m:
        return None
    ops = re.sub(r"\$", "", m.group(2))
    ops = tuple(o.strip() for o in ops.split(","))
    return m.group(1), ops


def op_targets(entries):
    """-> Liste (target_op, draft_op). GENUINE Operator-Diffs: beide ALU, GLEICHE Operanden
    (modulo Operation), andere Mnemonic. Filtert Misalignments + Pseudo-op-Aequivalenzen raus."""
    out = []
    for e in entries:
        if e.get("type") != "Instruction Mismatch":
            continue
        t, d = e.get("target"), e.get("draft")
        if not (isinstance(t, list) and isinstance(d, list) and len(t) == 1 and len(d) == 1):
            continue
        pt, pd = _parse(t[0]), _parse(d[0])
        if not pt or not pd:
            continue
        (mt, ot), (md, od) = pt, pd
        if mt == md or mt not in _ALU or md not in _ALU:
            continue
        # GLEICHE REGISTER-Operanden (Immediate darf abweichen: andi-Maske vs ori-Wert vs
        # Shift-Betrag) = echter Operator-Tausch. Filtert Misalignments raus, erlaubt aber
        # den Immediate-Unterschied (Detektor-Lehre: nicht zu streng).
        rt = tuple(o for o in ot if not re.match(r"-?\d|0x", o))
        rd = tuple(o for o in od if not re.match(r"-?\d|0x", o))
        if rt == rd and rt:
            out.append((mt, md))
    return out


def operator_swap_candidates(c_code, targets):
    out = []
    for mt, md in targets:
        tcop, dcop = _BINOP.get(mt), _BINOP.get(md)
        if not tcop or not dcop or tcop == dcop:
            continue
        # finde dcop als Token, ersetze einzeln durch tcop
        pat = re.escape(dcop)
        for m in re.finditer(r"(?<![<>=!&|^+\-*/])" + pat + r"(?![<>=&|^])", c_code):
            cand = c_code[:m.start()] + tcop + c_code[m.end():]
            if cand != c_code:
                out.append((f"swap:{dcop}->{tcop}", cand))
    return out


def branch_invert_candidates(c_code, entries):
    """beq<->bne usw.: Vergleichs-Operator in if/while/return-Bedingungen invertieren."""
    out = []
    pairs = []
    for e in entries:
        if e.get("type") != "Instruction Mismatch":
            continue
        t, d = e.get("target"), e.get("draft")
        if not (isinstance(t, list) and isinstance(d, list) and len(t) == 1 and len(d) == 1):
            continue
        pt, pd = _parse(t[0]), _parse(d[0])
        if pt and pd and pt[0] in _BRANCH and pd[0] in _BRANCH and pt[0] != pd[0]:
            pairs.append((pt[0], pd[0]))
    if not pairs:
        return out
    # invertiere jeden Vergleichsoperator im Code (orakel waehlt den richtigen)
    for m in re.finditer(r"(==|!=|<=|>=|<|>)", c_code):
        op = m.group(1); inv = _INVERT.get(op)
        if not inv:
            continue
        cand = c_code[:m.start()] + inv + c_code[m.end():]
        if cand != c_code:
            out.append((f"branch:{op}->{inv}", cand))
    return out


def _postfix_operand_start(s, end):
    """Laeuft von Index `end` (letztes Zeichen des Operanden, inkl.) rueckwaerts ueber einen
    POSTFIX-Ausdruck und gibt den Start-Index zurueck. Deckt: bare ident, `x[i]`/`x[i][j]`,
    `p->f`, `s.f`, balancierte `(...)`/`[...]`, und fuehrenden Deref `*(T*)EXPR`. Multiply-Guard:
    fuehrendes `*` wird nur als Deref (nicht Multiplikation) gewertet, wenn davor ein Operator/Klammer
    steht. Bewusst grosszuegig -- Orakel-Gate verwirft falsche Captures."""
    j = end
    while j >= 0:
        c = s[j]
        if c in ")]":
            depth = 0
            while j >= 0:
                if s[j] in ")]": depth += 1
                elif s[j] in "([":
                    depth -= 1
                    if depth == 0: break
                j -= 1
            j -= 1
        elif c.isalnum() or c == "_" or c == ".":
            j -= 1
        elif c == ">" and j >= 1 and s[j - 1] == "-":          # -> Member
            j -= 2
        elif c in " \t":                                       # Whitespace nur wenn Member-Kette weitergeht
            k = j
            while k >= 0 and s[k] in " \t": k -= 1
            if k >= 0 and (s[k].isalnum() or s[k] in "_)]." or (s[k] == ">" and k >= 1 and s[k - 1] == "-")):
                j = k
            else:
                break
        else:
            break
    start = j + 1
    # fuehrende Deref/Cast-Praefixe einsammeln: `*`, `(TYPE *)`
    while True:
        k = start - 1
        while k >= 0 and s[k] in " \t": k -= 1
        if k < 0:
            break
        if s[k] == ")":                                        # evtl. Cast `(s32 *)`
            depth = 0; m = k
            while m >= 0:
                if s[m] == ")": depth += 1
                elif s[m] == "(":
                    depth -= 1
                    if depth == 0: break
                m -= 1
            if m < 0:
                break
            start = m
        elif s[k] == "*":                                      # Deref nur wenn davor Operator/Klammer (kein Multiply)
            p = k - 1
            while p >= 0 and s[p] in " \t": p -= 1
            if p < 0 or s[p] in "([{,=<>+-*/%&|^!?:;~":
                start = k
            else:
                break
        else:
            break
    return start


def sign_cast_candidates(c_code, targets):
    """srl<->sra / slt<->sltu / mult<->multu: Operand-Signedness umstellen via Cast `(u32)`/`(s32)`.
    Heuristik: caste die Operanden des betroffenen Ausdrucks. Hier breit: fuer jede Variable einen
    (u32)/(s32)-Cast-Versuch an Shift-/Vergleichs-Stellen. Orakel-gegated."""
    out = []
    want = set()
    for mt, md in targets:
        if mt in _SIGN and md in _SIGN and _SIGN[mt] != _SIGN[md]:
            want.add(_SIGN[mt])
    if not want:
        return out
    castmap = {"u": "u32", "s": "s32"}
    for sgn in want:
        ct = castmap[sgn]
        # (1) Nackte Variable vor dem Operator: `X >> Y` -> `(ct)(X) >> Y`
        for m in re.finditer(r"([A-Za-z_]\w*)\s*(>>|<|>|<=|>=)", c_code):
            var = m.group(1)
            cand = c_code[:m.start(1)] + f"({ct})({var})" + c_code[m.end(1):]
            if cand != c_code:
                out.append((f"sign:{var}->{ct}", cand))
        # (2) GAP-FIX (2026-06): geklammerter AUSDRUCK vor dem Operator: `(EXPR) >> Y` ->
        #     `(ct)(EXPR) >> Y`. Trifft `(*(u16*)(...)) >> 10` (u16->int-Promotion = sra; (u32) = srl).
        for m in re.finditer(r"(>>|<=|>=|<|>)", c_code):
            j = m.start() - 1
            while j >= 0 and c_code[j] == " ":
                j -= 1
            if j < 0 or c_code[j] != ")":
                continue
            depth = 0; k = j
            while k >= 0:                                  # passende offene Klammer rueckwaerts
                if c_code[k] == ")": depth += 1
                elif c_code[k] == "(":
                    depth -= 1
                    if depth == 0: break
                k -= 1
            if k < 0:
                continue
            grp = c_code[k:j + 1]                           # `(...)` inkl. Klammern
            cand = c_code[:k] + f"({ct})" + grp + c_code[j + 1:]
            if cand != c_code:
                out.append((f"sign_expr:->{ct}", cand))
        # (3) GAP-FIX (2026-06): POSTFIX-Operand vor dem Operator: `x[i] >> Y`, `p->f >> Y`,
        #     `*(T*)p >> Y` (Array-Index / Member / unparenthesisierter Deref). Diese Formen treffen
        #     weder (1) (nackte Var) noch (2) (geklammert). Rueckwaerts-Postfix-Extraktor + (ct)-Wrap.
        for m in re.finditer(r"(>>|<=|>=|<|>)", c_code):
            j = m.start() - 1
            while j >= 0 and c_code[j] in " \t":
                j -= 1
            if j < 0 or c_code[j] not in ")]_." and not (c_code[j].isalnum()):
                continue                                       # nur Postfix-Enden (ident/]/)/member)
            st = _postfix_operand_start(c_code, j)
            operand = c_code[st:j + 1]
            if not operand or re.fullmatch(r"[A-Za-z_]\w*", operand):
                continue                                       # bare-var -> (1)
            # Wenn operand EIN einzelner balancierter `(...)`-Block ist, deckt (2) das ab -> skip.
            # ABER cast-praefixierte Operanden `(u16)(*...)` (erster `(` schliesst NICHT am Ende) MUESSEN
            # hier durch (sonst wickelt (2) nur den inneren Deref ohne den (u16)-Cast -> bleibt sra).
            if operand[0] == "(":
                depth = 0; single = False
                for idx, ch in enumerate(operand):
                    if ch == "(": depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            single = (idx == len(operand) - 1); break
                if single:
                    continue
            cand = c_code[:st] + f"({ct})(" + operand + ")" + c_code[j + 1:]
            if cand != c_code:
                out.append((f"sign_postfix:->{ct}", cand))
    return out


def instr_op_candidates(c_code, entries):
    tgts = op_targets(entries)
    cands = []
    if tgts:
        cands += operator_swap_candidates(c_code, tgts)
        cands += sign_cast_candidates(c_code, tgts)
        # strength_reduction_candidates VERWORFEN (2026-06): empirisch 0/106 -- IDO kanonisiert die
        # Arithmetik, aequivalente C-Formen flippen die addu/subu-Wahl NICHT (oder zerschiessen mm).
        # Diese Codegen-Faelle sind EXTERNES-Permuter-Territorium (batch_permuter), kein interner Stage.
    cands += branch_invert_candidates(c_code, entries)
    seen = set(); out = []
    for lbl, code in cands:
        if code != c_code and code not in seen:
            seen.add(code); out.append((lbl, code))
    return out[:120]


def _count_op(entries):
    return len(op_targets(entries)) + sum(
        1 for e in entries if e.get("type") == "Instruction Mismatch"
        and isinstance(e.get("target"), list) and len(e["target"]) == 1
        and (_parse(e["target"][0]) or ("", ""))[0] in _BRANCH)


def instr_op_expert(c_code, func_name, target_s_path, eval_fn=None, max_iter=6, ai_call=None):
    """eval_fn(c, fn, sp) -> (metric, entries). metric = LEXIKOGRAFISCH (synth_count, total) ODER None.
    GATE (2026-06): akzeptiere Kandidaten, der die METRIK senkt -- ein Operator-Swap (subu->addu)
    senkt eine Instruction-Mismatch (= synth), also faellt die Metrik AUCH wenn total via Hand-off
    (Register/Reorder/...) steigt. Sekundaerer _count_op-Tiebreak. Loest den total-Gate-Bug; loop-sicher
    (Metrik streng fallend). diffs_after = Metrik (Tupel).

    STUFE 2 (ai_call!=None, 2026-06): KI-Oneshot auf dem Rest (modules.expert_hybrid_ai) durch DASSELBE
    Gate. Hilft bei op selten (Kalibrierung: op ~mechanisch, Rest = MISSING-OP/Codegen = andere Stufe),
    aber gate-sicher. Stufe 3 verdict 'permuter' -> Pipeline routet (permuter / missing-block)."""
    before, ent = eval_fn(c_code, func_name, target_s_path)
    cur, cur_d, cur_ent = c_code, before, ent
    steps = []
    for _ in range(max_iter):
        cur_op = _count_op(cur_ent)
        if cur_op == 0:
            break
        best = None
        for lbl, cand in instr_op_candidates(cur, cur_ent):
            d, m = eval_fn(cand, func_name, target_s_path)
            if d is None or d > cur_d:                      # compile-fail oder Metrik schlechter
                continue
            o = _count_op(m)
            if not ((d < cur_d) or (d == cur_d and o < cur_op)):
                continue
            key = (d, o)
            if best is None or key < best[0]:
                best = (key, cand, lbl, m)
        if best is None:
            break
        cur_d, cur, cur_ent = best[0][0], best[1], best[3]
        steps.append((best[2], cur_d))
    # STUFE 2: KI-Oneshot. Laeuft auf dem ORIGINAL (c_code, ent), KONKURRIERT mit dem det. Ergebnis
    # durch DASSELBE Gate. Trigger: Original hatte op-Ziele UND mm noch > 0 (analog width, gegen den
    # Lateral-Move-Block). Hilft bei op selten (Rest = MISSING-OP/Codegen = andere Stufe), aber gate-sicher.
    if (ai_call is not None and cur_d is not None and cur_d[-1] > 0 and _count_op(ent) > 0):  # cur_d[-1]=mm (md,im,tier,mm)
        from modules import expert_hybrid_ai as _hy
        cur_op = _count_op(cur_ent)
        for lbl, cand in _hy.hybrid_ai_candidate(c_code, func_name, ent, "op", ai_call):
            d, m = eval_fn(cand, func_name, target_s_path)
            if d is None or d > cur_d:
                continue
            o = _count_op(m)
            if (d < cur_d) or (d == cur_d and o < cur_op):
                cur_d, cur, cur_ent = d, cand, m
                steps.append((lbl, cur_d))
    # verdict wie reorder_expert: 'permuter' wenn op_targets BLEIBEN und kein Swap/KI griff (= Codegen,
    # addu/subu strength-reduction -> externer Permuter). Funktions-level-Routing macht ppw.permuter_pending.
    applied = cur != c_code
    remaining_op = len(op_targets(cur_ent))
    verdict = "solved" if remaining_op == 0 else ("improved" if applied else "permuter")
    return {"applied": applied, "new_code": cur, "diffs_before": before,
            "diffs_after": cur_d, "steps": steps, "verdict": verdict}
