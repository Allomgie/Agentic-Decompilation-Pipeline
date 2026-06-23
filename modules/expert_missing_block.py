"""missing-block Experte SCHICHT 1: DETERMINISTISCHE, ORAKEL-GEGATE Rekonstruktion.

Symptom "Missing in Draft": das Target hat einen Instruktions-Block, den der Draft nicht erzeugt.
SCHICHT 1 behandelt die mechanisch ableitbaren Faelle (der Rest -> Schicht 2 m2c-Donor / Schicht 3 AI):
  - MISSING-STORE: `s{b,h,w,wc1} $zero, off($base)` -> `*(T*)((char*)P + off) = 0;`  (Wert 0 = $zero).
  - MISSING-RETURN: Block setzt nur $v0 am Ende (`or $v0,$zero,$zero` / `addiu $v0,$zero,N` /
    `or $v0,$reg,$zero`) -> fehlender `return <const>;` (+ void->s32).

KERN-PRINZIP (empirisch, siehe memory/missing_block_plan.md): Basis-Reg->C-Var-Mapping und Platzierung
sind statisch unsicher -> Kandidaten BREIT erzeugen (jede Pointer-Var x jede Statement-Position) und das
ORAKEL (eval_fn, synth-bewusste Metrik) filtern. So loesen GENUINE Faelle automatisch, CODEGEN-Artefakte
(fold/materialize) fallen durch (kein Kandidat senkt md) -> Schicht 2/3 / Permuter.

GATE: ein Kandidat zaehlt NUR, wenn die synth-bewusste Metrik faellt (md = Missing-in-Draft sinkt; neue
Register/Memory/Reorder-Diffs = erlaubter Hand-off). Verifikation IMMER ueber eval_fn (ido_compiler).
"""
import os
import re

_STTYPE = {"sb": "u8", "sh": "u16", "sw": "s32", "swc1": "f32"}


def _mnem_ops(line):
    """'/* .. */  sw  $zero, 0x15C($s0)' -> ('sw', ['$zero','0x15C($s0)']) oder (mnem, [ops...])."""
    s = re.sub(r"/\*.*?\*/", "", line).strip()
    m = re.match(r"\.?([a-z][a-z0-9.]*)\s*(.*)", s)
    if not m:
        return None, []
    ops = [o.strip() for o in m.group(2).split(",")] if m.group(2).strip() else []
    return m.group(1), ops


def _parse_store(mnem, ops):
    """Store -> (ctype, off, src) oder None. Nur src=$zero (Wert 0) in Schicht 1."""
    if mnem not in _STTYPE or len(ops) != 2:
        return None
    src = ops[0]
    m = re.match(r"(-?0x[0-9A-Fa-f]+|-?\d+)\(\$(\w+)\)", ops[1])
    if not m:
        return None
    off = int(m.group(1), 0)
    if src not in ("$zero", "$0"):
        return None
    return _STTYPE[mnem], off, 0


def _pointer_vars(c_code):
    """Heuristik: C-Variablen, die als Basis-Pointer in Frage kommen (Params + lokale Pointer/Vars).
    Breit -- das Orakel filtert. Reihenfolge: Params zuerst (haeufigste Store-Basis)."""
    vars_ = []
    seen = set()
    # Param-Liste
    mh = re.search(r"\b\w+\s+\w+\s*\(([^)]*)\)\s*\{", c_code)
    if mh:
        for p in mh.group(1).split(","):
            mm = re.search(r"(\w+)\s*$", p.strip())
            if mm and mm.group(1) not in ("void", "") and mm.group(1) not in seen:
                seen.add(mm.group(1)); vars_.append(mm.group(1))
    # lokale Deklarationen  TYPE [*] name;
    for m in re.finditer(r"(?:^|[;{])\s*[A-Za-z_]\w*\s*\**\s*([A-Za-z_]\w*)\s*[;=]", c_code, flags=re.M):
        if m.group(1) not in seen:
            seen.add(m.group(1)); vars_.append(m.group(1))
    return vars_


def _stmt_positions(c_code):
    """Top-Level-Statement-Grenzen im Funktionskoerper (Brace-Tiefe 1) -> Insert-Indizes. Plus direkt vor
    der letzten schliessenden Klammer."""
    body0 = c_code.find("{")
    if body0 < 0:
        return []
    pos = []
    depth = 0
    i = body0
    while i < len(c_code):
        c = c_code[i]
        if c == "{":
            depth += 1
            if depth == 1:
                pos.append(i + 1)               # direkt nach Funktions-Oeffnungsklammer
        elif c == "}":
            depth -= 1
            if depth == 0:
                pos.append(i)                   # vor schliessender Funktionsklammer
                break
        elif c == ";" and depth == 1:
            pos.append(i + 1)                   # nach jedem Top-Level-Statement
        i += 1
    # dedupe, sortiert
    return sorted(set(pos))


def missing_block_candidates(c_code, entries):
    """-> [(label, new_code)] deterministische Rekonstruktions-Kandidaten (breit; Orakel filtert)."""
    out = []
    mb = [e for e in entries if e.get("type") == "Missing in Draft"]
    if not mb:
        return out
    pvars = _pointer_vars(c_code)
    positions = _stmt_positions(c_code)

    for e in mb:
        instrs = [_mnem_ops(x) for x in e.get("missing", [])]
        instrs = [(m, o) for (m, o) in instrs if m]
        # --- MISSING-STORE: jede Store-Instr im Block -> Insert an jeder Position fuer jede Pointer-Var
        for (mn, ops) in instrs:
            st = _parse_store(mn, ops)
            if not st:
                continue
            ctype, off, val = st
            for P in pvars:
                stmt = f"*({ctype} *)((char *)({P}) + 0x{off:X}) = {val};"
                for pos in positions:
                    cand = c_code[:pos] + "\n  " + stmt + "\n" + c_code[pos:]
                    if cand != c_code:
                        out.append((f"mb-store:{ctype}@0x{off:X}={P}", cand))
        # --- MISSING-RETURN: Block setzt $v0 (typisch am Ende) -> fehlender return
        rconsts = []
        for (mn, ops) in instrs:
            if not ops or ops[0] not in ("$v0", "$2"):
                continue
            if mn == "or" and len(ops) == 3 and ops[1] in ("$zero", "$0") and ops[2] in ("$zero", "$0"):
                rconsts.append("0")
            elif mn in ("addiu", "li", "ori", "addu") and len(ops) >= 2:
                mc = re.match(r"(-?0x[0-9A-Fa-f]+|-?\d+)$", ops[-1])
                if mc:
                    rconsts.append(str(int(mc.group(1), 0)))
        for rc in rconsts:
            out += _return_candidates(c_code, rc)

    # dedupe + Cap
    seen = set(); uniq = []
    for lbl, code in out:
        if code != c_code and code not in seen:
            seen.add(code); uniq.append((lbl, code))
    return uniq[:200]


def _return_candidates(c_code, rc):
    """fehlender `return <rc>;` + ggf. void->s32. Insert vor schliessender Funktionsklammer."""
    out = []
    end = c_code.rfind("}")
    if end < 0:
        return out
    body = c_code[:end] + f"\n  return {rc};\n" + c_code[end:]
    # void-Rueckgabetyp -> s32 (sonst kompiliert `return N;` nicht). NUR die DEFINITION treffen
    # (void NAME(...) {  -- mit Body-Brace), NICHT die `;`-terminierten extern-Decls.
    body_nonvoid = re.sub(r"\bvoid(\s+\w+\s*\([^)]*\)\s*\{)", r"s32\1", body, count=1)
    out.append((f"mb-return:{rc}", body))
    if body_nonvoid != body:
        out.append((f"mb-return:{rc}:s32", body_nonvoid))
    return out


_M2C_CACHE = {}     # func_name -> nutzbarer (kompilierbarer) m2c-Ganzfunktions-C ODER None. Cache, damit
                    # m2c + ggf. Compile-Repair pro Funktion nur EINMAL laeuft (nicht pro Orchestrator-Runde).


def _compile_repair(code, func_name, target_s_path, eval_fn, ai_call=None, context_c=None):
    """Wenn `code` NICHT kompiliert -> durch den COMPILE-ERROR-EXPERTEN (wir haben dafuer einen eigenen;
    use_m2c=False = kein Rekurs). Liefert kompilierbaren code ODER None. Genutzt fuer m2c-Output UND
    KI-Graft-Output (beide koennen non-kompilierbar sein -- m2c-Quirks bzw. KI-Halluzination)."""
    if eval_fn(code, func_name, target_s_path)[0] is not None:
        return code
    try:
        from modules import expert_compile_fix as ecf
        rr = ecf.compile_error_expert(code, func_name, ai_call=ai_call, target_s_path=target_s_path,
                                      context_c=context_c, use_m2c=False)
        if rr.get("ok") and eval_fn(rr["code"], func_name, target_s_path)[0] is not None:
            return rr["code"]
    except Exception:
        pass
    return None


def _m2c_whole(func_name, target_s_path, eval_fn, ai_call=None, context_c=None):
    """m2c-Ganzfunktion holen; non-kompilierbar -> _compile_repair. Liefert kompilierbaren m2c-C/None. Gecacht."""
    if func_name in _M2C_CACHE:
        return _M2C_CACHE[func_name]
    res = None
    try:
        from modules import m2c_donor
        mc = m2c_donor.run_m2c_ido(target_s_path, func_name, context_c=context_c)
    except Exception:
        mc = None
    if mc:
        res = _compile_repair(mc, func_name, target_s_path, eval_fn, ai_call=ai_call, context_c=context_c)
    # SAEUBERN (Lukas): das m2c-Transplantat traegt einen Makro-Header + Kommentare + M2C_FIELD/sp+N/arg-
    # Namen -- das bricht Similar (keine Kommentare/#define erlaubt) und passt nicht zur Konvention. clean_m2c
    # inlined die Makros (-> explizite Offset-Form), streift Header+Kommentare, arg->param. Orakel-gegated:
    # nur uebernehmen, wenn die saubere Version weiter kompiliert (semantisch identisch -> gleiche .o). Nicht
    # saeuberbar (exotische Makros) -> Transplantat VERWERFEN (kein dirty m2c speichern).
    if res:
        from modules import m2c_donor
        cleaned = m2c_donor.clean_m2c(res)
        if cleaned is None:
            res = None
        elif eval_fn(cleaned, func_name, target_s_path)[0] is not None:
            res = cleaned
        else:
            res = None
    _M2C_CACHE[func_name] = res
    return res


_M2C_SPEC_CACHE = {}    # func_name -> rohe m2c-Ganzfunktion als SPEC (Pseudocode-Plan) ODER None.


def _m2c_spec(func_name, target_s_path, context_c=None):
    """SPEC-Getter (Lukas): rohe m2c-Ganzfunktion als faithful Pseudocode-PLAN. ANDERS als _m2c_whole:
    KEIN _compile_repair -- die Spec MUSS NICHT kompilieren. Das ist der Unlock: m2cs einzige echte
    Schwaeche (nicht immer kompilierbarer Code) ist als SPEC irrelevant; wir nutzen nur Kontroll-/Datenfluss.
    Makro-Definitions-Header (_macros(), reines Boilerplate) wird abgestreift -- die KI braucht nur den
    Funktionskoerper; die Makro-Notation wird im Prompt knapp erklaert. Gecacht, None bei m2c-Totalausfall."""
    if func_name in _M2C_SPEC_CACHE:
        return _M2C_SPEC_CACHE[func_name]
    spec = None
    try:
        from modules import m2c_donor
        raw = m2c_donor.run_m2c_ido(target_s_path, func_name, context_c=context_c)
        mh = m2c_donor._macros()
        if raw and raw.startswith(mh):          # run_m2c_ido stellt _macros() voran -> exakt abstreifen
            raw = raw[len(mh):]
        spec = raw.lstrip("\n") if raw else None
    except Exception:
        spec = None
    _M2C_SPEC_CACHE[func_name] = spec
    return spec


_MB_SYS = (
    "You are completing a MIPS-decompiled C function for the SGI IDO 5.3 -O2 compiler. The draft is "
    "MISSING a block of instructions that the target has. The missing block is REAL: you MUST reconstruct "
    "it and add it -- NEVER return the function unchanged. Insert the missing logic, expressed in the "
    "DRAFT's own variables and style, so the function compiles to include the target instructions. "
    "Reply with ONLY the full corrected C function. No markdown, no code fences, no explanation."
)


# Strukturierter Reasoning-Plan fuer den MISSING-Modus -- abgeleitet aus analysis/Research/
# (Decomp_Infosheet: Codegen-Idiome/Casts/Bitfields/Int-Promotion; See MIPS Run: ABI a0-a3/f12-f14,
# Return v0/f0; Chow/fredchow: Regalloc -> die KI soll NICHT exakte Register jagen, nur die Semantik).
_MB_PLAN = (
    "// === HOW TO RECONSTRUCT THE MISSING BLOCK (work through these steps) ===\n"
    "// STEP 1 -- Classify each missing instruction: load (lb/lbu/lh/lhu/lw/lwc1) | store (sb/sh/sw/swc1)\n"
    "//   | call (jal/jalr) | branch (b/beq/bne/blez/bgtz/...) | arithmetic/move.\n"
    "// STEP 2 -- Loads/stores: base register + offset 0xN -> a field access of the RIGHT type. Width->type:\n"
    "//   lb=s8 lbu=u8 lh=s16 lhu=u16 lw=s32 lwc1/swc1=f32 (FLOAT). Form: *(T*)((char*)P + 0xN), where P is\n"
    "//   the DRAFT variable currently held in that base register (trace it from the surrounding code).\n"
    "// STEP 3 -- Calls (jal func_X): the arguments are set up in a0,a1,a2,a3 (f12,f14 for the FIRST float\n"
    "//   args) in the instructions just BEFORE the jal -- read them to get the call args. A float arg that\n"
    "//   comes AFTER an int arg is passed in a GPR (a0-a3), not an f-register. Result returns in v0 (f0 if\n"
    "//   float). Write `func_X(args)` (or `lhs = func_X(args)`).\n"
    "// STEP 4 -- Branches: reconstruct CONTROL FLOW. The condition comes from the compare just before\n"
    "//   (slt/sltu = signed/unsigned <) or from beq/bne (==/!=). The target label tells the shape: forward\n"
    "//   jump over code = if(cond){...}; backward = loop; jump to the epilogue = early `return`. Idiom: a\n"
    "//   single-bit test can appear as load-32 + shift-the-bit-to-the-sign + branch-on-sign.\n"
    "// STEP 5 -- Recognize lowered idioms; write the SOURCE form, not the lowered one:\n"
    "//   u32->u8 = (x<<24)>>24; sign-shift srl=(u32)x>>n vs sra=(s32)x>>n; `x > CONST` often emitted as\n"
    "//   `x >= CONST+1`; a plain global/field load may appear address-materialized -- just write the\n"
    "//   natural access. Int promotion: u8/u16 operands promote to int.\n"
    "// STEP 6 -- Placement: insert the reconstructed C at the position matching the surrounding\n"
    "//   instructions that ALREADY match the target (keep relative order). STRONG HINT: an EMPTY block in\n"
    "//   the draft -- `if (cond) { }`, `else { }`, an empty loop body -- is almost always WHERE the missing\n"
    "//   instructions belong; fill it.\n"
    "// STEP 7 -- Do NOT chase exact registers (t*/s*/f* allocation is fixed downstream). Get the SEMANTICS\n"
    "//   right: correct field+type, correct call+args, correct control flow.\n"
)


_VA_RE = re.compile(r"/\*\s*[0-9A-Fa-f]+\s+([0-9A-Fa-f]{8})\s")


def _anchor_context(target_s_path, entries, window=5):
    """ANKER: lokales Target-Kontextfenster UM die Luecke (matchende Nachbarn + Labels), die fehlenden
    Zeilen mit `>>>` markiert. Gibt der KI die Einfuegestelle + lokalen Register/Kontrollfluss OHNE die
    ganze Funktion (Prompt-Budget). None bei Fehler. (Diff + Anker, wie in unseren anderen Experten.)"""
    try:
        raw = open(target_s_path, encoding="utf-8", errors="replace").read().splitlines()
    except Exception:
        return None
    insns = []                                  # (vaddr|None, text)  -- Labels als Kontext-Marker behalten
    for ln in raw:
        s = re.sub(r"/\*.*?\*/", "", ln).strip()
        if s.startswith("glabel") or s.startswith("endlabel"):
            continue
        if not s:
            continue
        m = _VA_RE.search(ln)
        if m is None and not (s.startswith(".") and s.endswith(":")):
            continue                            # nur Instruktionen (mit vaddr) + Labels
        insns.append((int(m.group(1), 16) if m else None, s))
    miss_va = set()
    for e in entries:
        if e.get("type") != "Missing in Draft":
            continue
        for x in e.get("missing", []):
            m = _VA_RE.search(x)
            if m:
                miss_va.add(int(m.group(1), 16))
    idxs = [i for i, (va, _) in enumerate(insns) if va in miss_va]
    if not idxs:
        return None
    lo, hi = max(0, idxs[0] - window), min(len(insns), idxs[-1] + 1 + window)
    # PLAIN lokales Fenster (>>> = fehlend). Empirisch: MEHR Kontext (Voll-ASM ODER Branch-Ziel-Anhang)
    # senkte die oneshot-Quote (Ablenkung) -> bewusst NUR das lokale Fenster. Control-Flow-`b`-Faelle mit
    # weit entferntem Ziel bleiben hart (Hebel: adaptive Regeln / anderer Experte), aber 10/20 (50%) Basis.
    return "\n".join(("    >>> " if va in miss_va else "        ") + s for va, s in insns[lo:hi])


_ADDR_FOLD = {"addiu", "lui", "ori", "la", "li"}


def _is_branch(m):
    return (m.startswith("b") and m != "break") or m == "j"


def _label_vaddrs(target_s_path):
    """{.Llabel -> vaddr der naechsten Instruktion danach}. Fuer No-op-Branch-Erkennung."""
    try:
        raw = open(target_s_path, encoding="utf-8", errors="replace").read().splitlines()
    except Exception:
        return {}
    out = {}
    pending = []
    for ln in raw:
        s = re.sub(r"/\*.*?\*/", "", ln).strip()
        if s.startswith(".") and s.endswith(":"):
            pending.append(s[:-1])
            continue
        m = _VA_RE.search(ln)
        if m:
            va = int(m.group(1), 16)
            for lab in pending:
                out[lab] = va
            pending = []
    return out


def is_codegen_artifact_gap(entries, target_s_path):
    """True, wenn ALLE 'Missing in Draft'-Bloecke reine CODEGEN-Artefakte sind (kein fehlendes Logik):
      - lone Adress-Fold (nur addiu/lui/ori/li mit %lo/%hi) = lui/lw-Faltung vs Materialisierung;
      - No-op-Branch (`b/j` + nop, dessen Ziel die UNMITTELBAR folgende Instruktion ist = va+8).
    Solche Luecken erzeugt KEIN C -> nicht an die KI, Permuter-Territorium. (Wie op/width-Codegen-Escape.)"""
    mb = [e for e in entries if e.get("type") == "Missing in Draft"]
    if not mb:
        return False
    labels = None
    for e in mb:
        rows = []
        for x in e.get("missing", []):
            mm = _VA_RE.search(x); s = re.sub(r"/\*.*?\*/", "", x).strip()
            mn = re.match(r"\.?([a-z][a-z0-9.]*)", s)
            rows.append((int(mm.group(1), 16) if mm else None, mn.group(1) if mn else "?", s))
        if not rows:
            return False
        mns = [m for _, m, _ in rows]
        # (a) reiner Adress-Fold?
        if all(m in _ADDR_FOLD for m in mns) and any("%lo" in t or "%hi" in t for _, _, t in rows):
            continue
        # (b) reine Branch/nop UND jeder Branch ist No-op (Ziel = va+8 = naechste Instr nach Delay)?
        if all(_is_branch(m) or m == "nop" for m in mns):
            if labels is None:
                labels = _label_vaddrs(target_s_path)
            noop = True
            for va, m, t in rows:
                if _is_branch(m):
                    lab = re.search(r"\.L\w+", t)
                    tgt = labels.get(lab.group(0)) if lab else None
                    if va is None or tgt is None or tgt != va + 8:
                        noop = False; break
            if noop:
                continue
        return False        # dieser Block ist KEIN Codegen-Artefakt -> genuin
    return True             # ALLE Bloecke sind Codegen-Artefakte


def build_mb_prompt(draft, func_name, entries, m2c_hint=None, tech_env=None, anchor=None):
    """(system, user) fuer den KI-Block-Graft (Schicht 3, MISSING-Modus). DIFF (fehlende Instruktionen) +
    ANKER (lokales Target-Kontextfenster um die Luecke, >>> markiert -> Einfuegestelle/Kontrollfluss) +
    STRUKTUR-PLAN (_MB_PLAN, aus Research) + optional m2c-Hinweis + optional TechEnv. Namensraum macht KI."""
    mb = [e for e in entries if e.get("type") == "Missing in Draft"]
    blocks = []
    for e in mb:
        blocks.append("\n".join("    " + re.sub(r"/\*.*?\*/", "", x).strip() for x in e.get("missing", [])))
    miss = "\n\n".join(b for b in blocks if b.strip())
    u = ["// === MISSING TARGET INSTRUCTIONS (in the target, ABSENT in your draft) ===", miss]
    if anchor:
        u += ["", "// === ANCHOR: target context AROUND the gap ( >>> = the missing lines; the others are",
              "//   instructions your draft ALREADY has -> shows WHERE to insert + the local control/data flow) ===",
              anchor]
    u += ["", _MB_PLAN]
    if tech_env:
        u += ["", "// === TECH ENV (struct/type/symbol context for this function) ===", tech_env]
    if m2c_hint:
        u += ["",
              "// === REFERENCE: a mechanical decompiler (m2c) produced this WHOLE function -- faithful but",
              "//   awkward, variable names differ. Use it ONLY to understand the missing logic; express the",
              "//   result in the DRAFT's variables/style, NOT m2c's sp+N/temp_*: ===",
              m2c_hint]
    u += ["",
          "// === RULES ===",
          "// - Your ONE job: make the MISSING instructions appear. New diffs of OTHER kinds elsewhere are",
          "//   FINE (other experts fix them) -- only the missing block must be resolved.",
          "// - Use the DRAFT's existing variables; keep the draft's other statements; place new logic at",
          "//   the matching position.",
          "// - If a draft statement is clearly HALLUCINATED (produces instructions NOT in the target), you",
          "//   MAY remove it -- only when it clearly does not belong (not common).",
          "",
          "// === DRAFT C (complete it) ===",
          draft]
    return _MB_SYS, "\n".join(u)


_SPEC_SYS = (
    "You are completing a MIPS-decompiled C function for the SGI IDO 5.3 -O2 compiler. You are given a "
    "faithful REFERENCE decompilation of the WHOLE target function from the m2c decompiler: its logic and "
    "control flow are CORRECT (but it uses raw names sp+N/temp_*/macros and may not compile). You also get a "
    "DRAFT that has the right variable names, types and signature but is INCOMPLETE and sometimes GARBLED "
    "(nonsense expressions, broken loops, wrong/duplicate returns). Produce a function whose LOGIC MATCHES "
    "THE REFERENCE, written in the DRAFT's variables/types/signature. Keep draft statements that agree with "
    "the reference; fix or drop any that contradict it. Reply with ONLY the full corrected C function. No "
    "markdown, no explanation."
)

# v3 (Lukas, 2026-06): Referenz auch STRUKTUR- + TYP-autoritativ. Reasoning-Diagnose der md-stuck-Fails
# zeigte: die KI OPTIMIERT die exakte Codegen-Struktur weg (gzthread: peeled-iteration+Guard -> sauberes
# do-while, "doesn't matter much") und behaelt FALSCHE Draft-Typen (chhillbilly: s32->sra statt (u16)->srl;
# int+return0 statt void). v3 BEWAHRT Loop-Form/Guards/Signedness/Return-Typ der Referenz; Draft liefert nur
# Variablennamen + Struct/Symbol-Wissen.
_SPEC_SYS_V3 = (
    "You are reconstructing a MIPS-decompiled C function for the SGI IDO 5.3 -O2 compiler so it compiles to "
    "the EXACT target instructions. You are given a faithful REFERENCE decompilation (m2c) of the whole "
    "target function: it defines the EXACT control structure, types, signedness and return type that produce "
    "the target code (it uses raw names sp+N/temp_*/macros). You are also given a DRAFT with good VARIABLE "
    "NAMES and struct/symbol knowledge (real struct types, field and global names) but a wrong/garbled "
    "structure. Reproduce the REFERENCE's EXACT structure and types, written with the DRAFT's variable names "
    "and struct/symbol knowledge. Do NOT simplify, optimize or 'clean up' the reference -- keep peeled "
    "iterations, explicit guards and the exact loop shape; equivalent-looking C compiles to DIFFERENT "
    "instructions. Reply with ONLY the full corrected C function. No markdown, no explanation."
)


def build_mb_spec_prompt(draft, func_name, entries, spec, anchor=None, tech_env=None, variant=None):
    """SPEC-PROMPT (Lukas' Idee): die m2c-Ganzfunktion als faithful PSEUDOCODE-PLAN ist das ZENTRUM, nicht
    eine Fussnote. variant: 'v2' = Referenz LOGIK-autoritativ (Draft-Typen/Stil); 'v3' = Referenz STRUKTUR+
    TYP-autoritativ (exakte Codegen-Form bewahren, Draft nur Namen/Struct-Wissen). variant=None -> env
    MB_SPEC_VARIANT (default 'v2', A/B-bereit)."""
    if variant is None:
        import os as _os
        variant = _os.environ.get("MB_SPEC_VARIANT", "v2")
    mb = [e for e in entries if e.get("type") == "Missing in Draft"]
    miss = "\n".join("    " + re.sub(r"/\*.*?\*/", "", x).strip() for e in mb for x in e.get("missing", []))
    u = ["// === MISSING TARGET INSTRUCTIONS (present in target, ABSENT in your draft) ===", miss]
    if anchor:
        u += ["", "// === ANCHOR: target context around the gap ( >>> = the missing lines; shows WHERE ) ===",
              anchor]
    u += ["",
          "// === REFERENCE DECOMPILATION (m2c, WHOLE target function -- correct logic, raw style, may not compile) ===",
          "// m2c notation: M2C_FIELD(e, T*, off) = *(T*)((char*)e + off) (field at a byte offset); M2C_UNK =",
          "//   unknown/int type; M2C_BITWISE(T, e) = reinterpret e as T; sp+N / temp_* = raw stack slots/temps.",
          spec]
    if tech_env:
        u += ["", "// === TECH ENV (struct/type/symbol context for this function) ===", tech_env]
    if variant == "v3":
        u += ["",
              "// === RULES (match the EXACT codegen, not just the logic) ===",
              "// - The REFERENCE defines the exact CONTROL STRUCTURE: same loop shape (do-while / while / a",
              "//   PEELED first iteration), same guards and branches, same order. Do NOT merge, simplify or",
              "//   optimize it -- a cleaner equivalent loop compiles to DIFFERENT instructions and will NOT match.",
              "// - Use the REFERENCE's TYPES: signedness and width (u16 vs s32 changes srl vs sra) and the",
              "//   RETURN TYPE (void vs int -- a spurious `return 0;` adds instructions). Match them exactly.",
              "// - From the DRAFT take ONLY: variable NAMES and struct/symbol knowledge (real struct types like",
              "//   Actor*, real field and global names). If a draft numeric type or structure CONTRADICTS the",
              "//   reference, the REFERENCE wins.",
              "// - Other new diffs elsewhere are FINE (handoff); resolving the missing block is what matters.",
              "",
              "// === DRAFT C (use its variable NAMES + struct/symbol knowledge; follow the REFERENCE for structure/types) ===",
              draft]
        return _SPEC_SYS_V3, "\n".join(u)
    u += ["",
          "// === RULES ===",
          "// - The REFERENCE is the correct logic and control flow. Produce a function whose logic MATCHES the",
          "//   reference, written in the DRAFT's variables, types and signature (NOT m2c's sp+N/temp_*/macros).",
          "// - Keep draft statements that AGREE with the reference; FIX or DROP any draft statement that",
          "//   CONTRADICTS it -- drafts can be garbled (nonsense expressions, broken loops, wrong/dup returns).",
          "// - Add the missing logic where the anchor marks the gap. Other new diffs elsewhere are FINE",
          "//   (handoff to other experts); resolving the missing block is what matters.",
          "",
          "// === DRAFT C (variable names/types/signature; complete & fix it to match the reference) ===",
          draft]
    return _SPEC_SYS, "\n".join(u)


_GARBAGE_SYS = (
    "You are fixing a MIPS-decompiled C function for the SGI IDO 5.3 -O2 compiler. At one spot the draft "
    "C is WRONG -- it compiles to the wrong instructions (garbage). You are given the target instructions "
    "and the draft's wrong instructions. Rewrite ONLY that spot so it compiles to the target instructions, "
    "keeping the rest of the function. Reply with ONLY the full corrected C function. No markdown, no "
    "explanation."
)


def build_garbage_prompt(draft, func_name, entries, m2c_hint=None):
    """(system,user) fuer den GARBAGE-Reroute (instr-block/width-KI hat hier Muell erzeugt). Zeigt KONKRET
    target-vs-draft am Garbage-Spot (Instruction Mismatch): 'hier ist Muell, das ist der Diff, fix es'."""
    im = [e for e in entries if e.get("type") == "Instruction Mismatch"]
    spots = []
    for e in im:
        tgt = "\n".join("      " + re.sub(r"/\*.*?\*/", "", x).strip() for x in (e.get("target") or []))
        drf = "\n".join("      " + re.sub(r"/\*.*?\*/", "", x).strip() for x in (e.get("draft") or []))
        spots.append("// TARGET should produce:\n" + tgt + "\n// but your DRAFT produced (GARBAGE here):\n" + drf)
    body = "\n\n".join(spots) if spots else "//   (no instruction-mismatch spot)"
    u = ["// === GARBAGE SPOT(S): the draft C here compiles to the WRONG instructions ===",
         body]
    if m2c_hint:
        u += ["", "// === REFERENCE (m2c whole function, faithful but awkward; different var names) ===", m2c_hint]
    u += ["",
          "// === RULES ===",
          "// - Fix ONLY the garbage spot so it compiles to the TARGET instructions above. New diffs of",
          "//   OTHER kinds elsewhere are FINE (other experts fix them).",
          "// - Use the DRAFT's existing variables; keep the rest of the function.",
          "",
          "// === DRAFT C (fix it) ===",
          draft]
    return _GARBAGE_SYS, "\n".join(u)


_RETREG = {"$v0", "$2", "$f0"}
_BRANCH_MN = {"bltz", "blez", "bgtz", "bgez", "beq", "bne", "beqz", "bnez", "b", "j",
              "bnel", "beql", "blezl", "bgtzl", "bltzl", "bgezl", "bc1t", "bc1f"}
_LOAD_MN = {"lw", "lbu", "lb", "lh", "lhu", "lwc1", "ldc1", "lui", "addiu", "addu"}


def gap_class(entries, target_s_path):
    """Klassifiziert den Missing-Gap fuer ADAPTIVE, fokussierte Prompts (Lukas). Reihenfolge:
      'codegen'      -> reines Codegen-Artefakt (Permuter, KI skip).
      'control-flow' -> genuiner Branch (bltz/beq/b...) im Gap = if/else/loop/return-Struktur fehlt.
      'return'       -> Gap schreibt $v0/$f0 (Rueckgabewert fehlt).
      'decl'         -> Gap ist Global-Load (%lo/%hi) ohne Branch/Return -> Pointer-Array/Typ-Sache.
      'other'        -> generischer Plan.
    ERWEITERBAR (weitere Klassen -> Testmenge aendern). Der Prompt wird in Schicht 3 entsprechend gewaehlt."""
    if is_codegen_artifact_gap(entries, target_s_path):
        return "codegen"
    mns, dests, has_global = [], [], False
    for e in entries:
        if e.get("type") != "Missing in Draft":
            continue
        for x in e.get("missing", []):
            s = re.sub(r"/\*.*?\*/", "", x).strip()
            mm = re.match(r"\.?([a-z][a-z0-9.]*)\s*(\$\w+)?", s)
            if mm:
                mns.append(mm.group(1)); dests.append(mm.group(2))
            if "%lo" in s or "%hi" in s:
                has_global = True
    if any(m in _BRANCH_MN for m in mns):
        return "control-flow"
    if any(d in _RETREG for d in dests):
        return "return"
    if has_global and mns and all(m in _LOAD_MN for m in mns):
        return "decl"
    return "other"


_CTRL_SYS = (
    "You are completing a MIPS-decompiled C function for the SGI IDO 5.3 -O2 compiler. The function is "
    "MISSING CONTROL FLOW: a conditional branch (and its body / the return paths). Reconstruct the "
    "if/else (or loop / early return) structure with the CORRECT condition polarity. Reply with ONLY the "
    "full corrected C function. No markdown, no code fences, no explanation."
)


def build_ctrlflow_prompt(draft, func_name, entries, anchor=None, m2c_hint=None):
    """FOKUSSIERT auf fehlende KONTROLLFLUSS-Struktur (Klasse 'control-flow')."""
    mb = [e for e in entries if e.get("type") == "Missing in Draft"]
    miss = "\n".join("    " + re.sub(r"/\*.*?\*/", "", x).strip() for e in mb for x in e.get("missing", []))
    u = ["// === MISSING: control flow (a conditional + its body / returns). Missing instructions: ===", miss]
    if anchor:
        u += ["", "// === ANCHOR ( >>> = missing; surrounding target context) ===", anchor]
    u += ["",
          "// === DECODE CONTROL FLOW (mind the polarity!) ===",
          "// A branch JUMPS PAST the code up to its label when the condition holds -> the code BELOW the",
          "//   branch runs in the OPPOSITE case. So `bltz r,.L` (jump if r<0) => `if (r >= 0) { <code",
          "//   below> }`. blez=>`if(r>0)`, bgtz=>`if(r<=0)`, bgez=>`if(r<0)`, beq a,b,.L=>`if(a!=b)`,",
          "//   bne a,b,.L=>`if(a==b)`.",
          "// jal func_X inside the block -> func_X(args) (args from a0-a3 before it). A branch to the",
          "//   epilogue with $v0=N in its delay slot -> `return N`.",
          "// Build the if (and else / early return) with the right polarity; keep the existing calls.",
          "",
          "// === DRAFT C (add the missing control flow) ===", draft]
    if m2c_hint:
        u += ["", "// === m2c reference (faithful but awkward; understand only) ===", m2c_hint]
    return _CTRL_SYS, "\n".join(u)


_DECL_SYS = (
    "You are completing a MIPS-decompiled C function for the SGI IDO 5.3 -O2 compiler. A value is accessed "
    "through a GLOBAL/ARRAY whose declared type is too narrow: the target does an EXTRA load, meaning the "
    "global holds a POINTER (or is a pointer-array) that is then dereferenced. Fix the access/declaration. "
    "Reply with ONLY the full corrected C function. No markdown, no code fences, no explanation."
)


def build_decl_prompt(draft, func_name, entries, anchor=None, m2c_hint=None):
    """FOKUSSIERT auf Global-/Array-Pointer-Indirektion (Klasse 'decl')."""
    mb = [e for e in entries if e.get("type") == "Missing in Draft"]
    miss = "\n".join("    " + re.sub(r"/\*.*?\*/", "", x).strip() for e in mb for x in e.get("missing", []))
    u = ["// === MISSING: an extra load through a global/array (it holds a POINTER). Missing instr.: ===", miss]
    if anchor:
        u += ["", "// === ANCHOR ( >>> = missing; surrounding target context) ===", anchor]
    u += ["",
          "// === DECODE ===",
          "// Pattern `lui %hi(D); addu idx; lw %lo(D)(..)` then ANOTHER load = D[idx] is a POINTER that",
          "//   gets dereferenced. Write the double access: e.g. `*( *(T**)((char*)D + idx) )`, or declare",
          "//   `extern T *D[];` and index it. The FINAL load's width gives T (lbu=u8, lw=s32, lwc1=f32).",
          "//   Keep the index expression; only fix the global's type / add the extra dereference.",
          "",
          "// === DRAFT C (fix the access) ===", draft]
    if m2c_hint:
        u += ["", "// === m2c reference (faithful but awkward; understand only) ===", m2c_hint]
    return _DECL_SYS, "\n".join(u)


_RETURN_SYS = (
    "You are completing a MIPS-decompiled C function for the SGI IDO 5.3 -O2 compiler. The function is "
    "MISSING its RETURN VALUE: the draft has no return (or a wrong one), but the target sets the return "
    "register and returns. Add the correct `return <expr>;` and fix the function's RETURN TYPE to match. "
    "Reply with ONLY the full corrected C function. No markdown, no code fences, no explanation."
)


def build_return_prompt(draft, func_name, entries, anchor=None, m2c_hint=None):
    """FOKUSSIERTER Prompt fuer fehlenden RUECKGABEWERT (Klasse 'return'). Anders/kompakter als der
    generische Plan -- nur die Return-Decodierung."""
    mb = [e for e in entries if e.get("type") == "Missing in Draft"]
    miss = "\n".join("    " + re.sub(r"/\*.*?\*/", "", x).strip() for e in mb for x in e.get("missing", []))
    u = ["// === MISSING: the function's RETURN VALUE (the target sets $v0/$f0 then returns; your draft",
         "//   does not). These are the missing instructions: ===", miss]
    if anchor:
        u += ["", "// === ANCHOR ( >>> = missing; surrounding target context) ===", anchor]
    u += ["",
          "// === DECODE THE RETURN ===",
          "// or $v0,$zero,$zero -> return 0 ;  addiu $v0,$zero,N -> return N",
          "// lwc1 $f0, off($r) -> the function returns a FLOAT: return *(f32*)((char*)<ptr in $r> + off);",
          "//   set the RETURN TYPE to f32.",
          "// jal func_X with the result then in $v0/$f0 -> return func_X(args)  (return the call's result).",
          "// addiu $v0,$v0,%lo(D) -> return (s32)&D  (the ADDRESS of D, not its value).",
          "// mfc1 $v0,$f0 -> an int bit-copied from a float result.",
          "// Set the function's RETURN TYPE (int / f32) to match; keep every other statement.",
          "",
          "// === DRAFT C (add the missing return) ===", draft]
    if m2c_hint:
        u += ["", "// === m2c reference (faithful but awkward; use only to understand) ===", m2c_hint]
    return _RETURN_SYS, "\n".join(u)


def _invoke_ai(ai_call, sysm, usr):
    try:
        return ai_call(sysm, usr, think=False)
    except TypeError:
        return ai_call(sysm, usr)


def missing_block_expert(c_code, func_name, target_s_path, eval_fn=None, ai_call=None,
                         use_m2c=True, context_c=None, mode="missing", tech_env=None, ai_retries=3,
                         use_spec=None):
    """eval_fn(c,fn,sp) -> (metric, entries); metric LEXIKOGRAFISCH (synth_count zuerst) ODER None.
    GATE: akzeptiere Kandidaten, der die METRIK senkt (md sinkt; Register/Memory/Reorder duerfen via
    Hand-off wachsen -- der Sinn dieses Experten). diffs_after = Metrik.

    SCHICHT 1 (det.): missing_block_candidates (Store/Return, orakel-gegated).
    SCHICHT 2a (m2c-Donor, modular): wenn Synthese-Rest bleibt -> m2c-GANZFUNKTION als EIN orakel-gegateter
    Kandidat (draft-zerstoerend, aber NUR akzeptiert wenn synth strikt sinkt -> sicher; mm-Schutz im
    Orchestrator). m2c ist oft schwaecher (-> meist verworfen), gewinnt aber bei garbage-Drafts. Gecacht,
    per use_m2c steuerbar (langsam). SCHICHT 2b (block-graft, draft-erhaltend) + SCHICHT 3 (AI) spaeter."""
    before, ent = eval_fn(c_code, func_name, target_s_path)
    cur, cur_d, cur_ent = c_code, before, ent
    steps = []
    if cur_d is None:
        return {"applied": False, "new_code": cur, "diffs_before": before, "diffs_after": cur_d, "steps": steps}
    # eval_fn-Metrik = (md, im, tier, mm). EIGENES Ziel: md (missing-mode, Index 0) bzw. im (garbage-mode,
    # Index 1). Gate ist lexikografisch md-primaer -> beide Modi senken ihr eigenes Ziel, Rest = Handoff.
    own = 0 if mode == "missing" else 1
    # SCHICHT 1: deterministische Rekonstruktion (NUR missing-Modus; Store/Return sind fuer fehlende
    # Bloecke, nicht fuer Instruction-Mismatch-Garbage). Ein Satz Kandidaten, Orakel waehlt besten.
    if mode == "missing":
        best = None
        for lbl, cand in missing_block_candidates(cur, cur_ent):
            d, m = eval_fn(cand, func_name, target_s_path)
            if d is None or not (d < cur_d):      # nur strikte Metrik-Senkung (synth_count zuerst)
                continue
            if best is None or d < best[0]:
                best = (d, cand, lbl, m)
        if best is not None:
            cur_d, cur, cur_ent = best[0], best[1], best[3]
            steps.append((best[2], cur_d))
    # use_spec FRUEH bestimmen: beeinflusst die m2c-Reihenfolge (Lukas' Leiter). spec-Modus = draft-
    # erhaltende KI ZUERST (mit m2c nur als SPEC/Wissen), m2c-Whole-Transplantat erst als Fallback NACH der
    # KI. Sonst (Prosa) bisheriges Verhalten: m2c-Whole-Transplant VOR der KI.
    if use_spec is None:
        import os as _os
        # DEFAULT AN (2026-06): A/B bewies Spec 10/20 vs Prosa 3-5/20 (draft-erhaltend), 2 Laeufe stabil.
        # Mit MB_USE_SPEC=0 abschaltbar. m2c-Spec-Ausfall faellt automatisch auf per-Klasse-Prosa zurueck.
        use_spec = _os.environ.get("MB_USE_SPEC", "1") == "1"
    # SCHICHT 2a: m2c-Ganzfunktion (nur wenn noch Synthese-Rest UND .s vorhanden), orakel-gegated.
    mc = None
    mc_hint = None
    if use_m2c and target_s_path and cur_d is not None and cur_d[0] > 0:
        mc = _m2c_whole(func_name, target_s_path, eval_fn, ai_call=ai_call, context_c=context_c)
        if mc:
            mc_hint = mc
            d, m = eval_fn(mc, func_name, target_s_path)
            # NICHT-spec: m2c-Whole sofort transplantieren wenn synth-strikt besser. spec-Modus: NICHT jetzt
            # -- sonst wird der Draft durch m2c ersetzt und Draft==Spec -> KI no-opt (empirisch belegt: 958A4/
            # fxripple/idflasha). Im spec-Modus bleibt m2c-Whole als Fallback nach der KI (s.u.).
            if not use_spec and d is not None and d < cur_d:        # nur wenn m2c synth-strikt besser ist
                cur_d, cur, cur_ent = d, mc, m
                steps.append(("mb-m2c-whole", cur_d))
    # SCHICHT 3: KI-BLOCK-GRAFT (draft-erhaltend, Namensraum macht die KI), wenn noch Synthese-Rest bleibt.
    # REKURSIV/RETRY (Lukas): die KI-Graft-Quote hat HOHE VARIANZ (verschiedene Funcs gelingen je Versuch)
    # -> bis zu ai_retries Versuche (temperature>0 -> andere Samples), JEDER orakel-gegated; nach jedem
    # AKZEPTIERTEN Graft, der den Block noch nicht voll fuellt (synth-Rest), wird mit dem NEUEN Stand und
    # frischem Prompt weiterprobiert (= Selbstaufruf bis Luecke gefuellt oder Versuche erschoepft). m2c-
    # Ganzfunktion als Verstaendnis-Hinweis. Gate = eval_fn (synth-Metrik strikt sinkt) -> sicher.
    # (1) GAP-KLASSE bestimmen (adaptiv): 'codegen' -> KI ueberspringen (Permuter), 'return' -> fokussierter
    # Return-Prompt, sonst generischer Plan. (Erweiterbar: control-flow/decl spaeter.)
    # INITIALE Codegen-Pruefung: entscheidet, ob der KI-Loop ueberhaupt laeuft (reines Artefakt -> Permuter)
    # und ob eine Spec geholt wird. gcls/anchor selbst werden PRO ITERATION neu bestimmt (s.u.).
    init_gcls = gap_class(cur_ent, target_s_path) if (mode == "missing" and target_s_path and cur_d and cur_d[0] > 0) else "other"
    codegen_gap = (init_gcls == "codegen")
    # SPEC-Modus (Lukas): m2c-Ganzfunktion als faithful Pseudocode-PLAN in den Prompt (statt per-Klasse-
    # Prosa). Env-steuerbar fuer A/B (MB_USE_SPEC=1). Nur missing-Modus, nicht-codegen; faellt auf per-Klasse
    # zurueck, wenn m2c keine Spec liefert. Die Spec ist whole-function -> deckt JEDE Rest-Luecke ab (auch nach
    # Schrumpfen), darum NUR einmal geholt (m2c ist teuer). (use_spec ist oben schon bestimmt.)
    spec = (_m2c_spec(func_name, target_s_path, context_c=context_c)
            if (use_spec and mode == "missing" and target_s_path and not codegen_gap) else None)
    gcls, anchor = init_gcls, None
    ai_cands = []                  # kompilierende KI-Grafts SAMMELN -> spaeter Frankenstein-Hunk-Mix (best of both)
    for _att in range(max(1, ai_retries) if (ai_call is not None and not codegen_gap) else 0):
        if cur_d is None or cur_d[own] == 0:
            break                                   # eigenes Ziel (md bzw. im) gefuellt -> fertig
        # PRO-ITERATION (Lukas' Frage): die Luecke schrumpft/wechselt nach jedem akzeptierten Graft die
        # Klasse -> gcls + anchor aus dem AKTUELLEN cur_ent NEU bestimmen, sonst deckt ein stale per-Klasse-
        # Prompt / stale Anker die geschrumpfte Rest-Luecke nicht ab. Wird der Rest reines Codegen-Artefakt
        # -> abbrechen (Permuter, kein KI-Call). (Spec deckt als whole-function ohnehin jede Rest-Luecke.)
        if mode == "missing" and target_s_path:
            gcls = gap_class(cur_ent, target_s_path)
            if gcls == "codegen":
                break
            anchor = _anchor_context(target_s_path, cur_ent)
        if mode == "garbage":
            sysm, usr = build_garbage_prompt(cur, func_name, cur_ent, m2c_hint=mc_hint)
        elif use_spec and spec:
            # v2 (Referenz logik-autoritativ, Draft-Stil) und v3 (Referenz struktur+typ-autoritativ) sind
            # KOMPLEMENTAER (A/B: je 9/20, Union 13/20). Ueber die Retries ALTERNIEREN -> beide Framings
            # pro Fall (greift, wenn der erste Versuch md nicht senkt). v2 zuerst (haeufig gute Drafts).
            sv = "v2" if (_att % 2 == 0) else "v3"
            sysm, usr = build_mb_spec_prompt(cur, func_name, cur_ent, spec, anchor=anchor, tech_env=tech_env,
                                             variant=sv)
        elif gcls == "return":
            sysm, usr = build_return_prompt(cur, func_name, cur_ent, anchor=anchor, m2c_hint=mc_hint)
        elif gcls == "control-flow":
            sysm, usr = build_ctrlflow_prompt(cur, func_name, cur_ent, anchor=anchor, m2c_hint=mc_hint)
        elif gcls == "decl":
            sysm, usr = build_decl_prompt(cur, func_name, cur_ent, anchor=anchor, m2c_hint=mc_hint)
        else:
            sysm, usr = build_mb_prompt(cur, func_name, cur_ent, m2c_hint=mc_hint, tech_env=tech_env,
                                        anchor=anchor)
        try:
            new = _invoke_ai(ai_call, sysm, usr)
        except Exception:
            new = None
        if not new or not new.strip() or new.strip() == cur.strip():
            continue
        # KI-Output kompiliert nicht? -> Compile-Experte drauf (KI halluziniert manchmal).
        rep = _compile_repair(new, func_name, target_s_path, eval_fn, ai_call=ai_call, context_c=context_c)
        if rep:
            new = rep
        d, m = eval_fn(new, func_name, target_s_path)
        if d is not None and new not in ai_cands:   # JEDEN kompilierenden Graft sammeln (auch nicht-akzeptierte)
            ai_cands.append(new)
        if d is not None and d < cur_d:             # akzeptieren; bei Rest weiter (rekursiv) auf neuem Stand
            cur_d, cur, cur_ent = d, new, m
            steps.append((f"mb-ai-graft#{_att + 1}", cur_d))
    # ASM-HUNK (Lukas, 2026-06): zusaetzlich zu v2/v3 EIN asm-zentrierter Kandidat -- der fehlende Block IST
    # ein Stueck Ziel-ASM. Ganzes Ziel-ASM + markierter Missing-Block + Draft zeigen und fuellen lassen (brachte
    # beim Garbage-Experten die meisten Perfects). Greedy-akzeptiert wenn besser, UND in ai_cands fuer den Mix.
    if (os.environ.get("MB_ASM_HUNK", "1") == "1" and ai_call is not None and target_s_path
            and not codegen_gap and cur_d is not None and cur_d[own] > 0):
        try:
            from modules import expert_garbage as _egb
            asm = "\n".join(_egb._target_insns(target_s_path))
            a2 = _anchor_context(target_s_path, cur_ent)
            asm_sys = ("You are decompiling MIPS to C for the SGI IDO 5.3 -O2 compiler. Below is the FULL TARGET "
                       "assembly and a DRAFT C that is MISSING some of the logic. Add the missing logic so the C "
                       "compiles to the target assembly; keep the draft's existing variables, types and "
                       "structure. Reply with ONLY the full C function. No markdown, no explanation.")
            au = ["// === TARGET ASM ===", asm]
            if a2:
                au += ["", "// === MISSING INSTRUCTIONS (>>> = missing in draft) ===", a2]
            if mc_hint:
                au += ["", "// === M2C REFERENCE ===", mc_hint]
            au += ["", "// === DRAFT C (add the missing logic) ===", cur]
            aout = _invoke_ai(ai_call, asm_sys, "\n".join(au))
            if aout and aout.strip():
                arep = _compile_repair(aout, func_name, target_s_path, eval_fn, ai_call=ai_call,
                                       context_c=context_c)
                if arep:
                    aout = arep
                d, m = eval_fn(aout, func_name, target_s_path)
                if d is not None and aout not in ai_cands:
                    ai_cands.append(aout)
                if d is not None and d < cur_d:
                    cur_d, cur, cur_ent = d, aout, m
                    steps.append(("mb-asm-hunk", cur_d))
        except Exception:
            pass
    # FRANKENSTEIN-HUNK-MIX (Lukas, 2026-06): die gesammelten KI-Grafts mischen -- die nicht-ueberlappenden
    # Kombinationen ihrer Hunks (alle vom selben Draft -> kein ASM-Block!=C-Statement-Risiko) enthalten
    # best-of-N (jeder Graft ganz) UND Cross-Kombis (v2 fuellt Teil A, v3 Teil B). KEINE Extra-KI-Calls
    # (reuse), orakel-gegated (eval_fn, md-primaer) -> nur uebernommen, wenn strikt < cur_d. Per MB_FRANKENSTEIN.
    if (os.environ.get("MB_FRANKENSTEIN", "1") == "1" and cur_d is not None and cur_d[own] > 0
            and len(ai_cands) >= 2):
        try:
            from modules import expert_garbage as _egb
            _ev = lambda c: eval_fn(c, func_name, target_s_path)[0]
            _cr = lambda c: _compile_repair(c, func_name, target_s_path, eval_fn, ai_call=ai_call,
                                            context_c=context_c)
            mxd, mxc = _egb._mix(c_code, ai_cands, func_name, target_s_path, _ev, _cr, resolve=None)
            if mxc and mxd is not None and mxd < cur_d:
                cur_d, cur = mxd, mxc
                _, cur_ent = eval_fn(cur, func_name, target_s_path)
                steps.append(("mb-frankenstein", cur_d))
        except Exception:
            pass
    # SCHICHT 2a-FALLBACK (NUR spec-Modus, Lukas' Leiter): hat die draft-erhaltende Spec-KI die Luecke NICHT
    # geschlossen, JETZT doch das m2c-Whole-Transplantat versuchen (letzter Ausweg, draft-zerstoerend, orakel-
    # gegated). Im Prosa-Modus wurde m2c-Whole bereits VOR der KI transplantiert -> hier nichts zu tun.
    if use_spec and mc is not None and cur_d is not None and cur_d[0] > 0:
        d, m = eval_fn(mc, func_name, target_s_path)
        if d is not None and d < cur_d:
            cur_d, cur, cur_ent = d, mc, m
            steps.append(("mb-m2c-whole-fallback", cur_d))
    # Verdikt: 'codegen' = Rest ist Codegen-Artefakt (Permuter, kein KI-Job); 'solved' md==0; sonst 'open'.
    # Auch der durch Grafts ENTSTANDENE Codegen-Rest (Loop brach mit gcls=='codegen' ab) -> 'codegen'.
    remaining_md = cur_d[0] if cur_d is not None else None
    if codegen_gap:
        verdict = "codegen"
    elif remaining_md == 0:
        verdict = "solved"
    elif (remaining_md and mode == "missing" and target_s_path
          and is_codegen_artifact_gap(cur_ent, target_s_path)):
        verdict = "codegen"
    else:
        verdict = "open"
    return {"applied": cur != c_code, "new_code": cur, "diffs_before": before,
            "diffs_after": cur_d, "steps": steps, "verdict": verdict}
