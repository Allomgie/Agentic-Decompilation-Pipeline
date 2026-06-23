"""Compiler-Error-Experte (Phase 0b, NACH micro_permuter-Repair).

Zweck: KI-Drafts der SCHWERSTEN Funktionen kompilierbar machen (Feuerprobe-Befund: 80% der
haertesten kompilieren nicht -> Engpass ist Phase A, nicht Matching).

ZWEI STUFEN:
  STUFE 1  DETERMINISTISCH (kein AI): bekannte Muster mechanisch fixen.
           - unkNN-Feldzugriff (`X->unkNN`, `(*X).unkNN`) -> expliziter Offset-Cast (kein Struct).
           - undefinierte Symbole extern deklarieren (Form aus Nutzung: skalar vs array).
           - doppelte case-Labels entfernen.
  STUFE 2  KI-REPARATUR (nur fuer den Rest, mit Pod): die DOMINANTE Rest-Klasse sind HALLUZINIERTE
           STRUCT-TYPEN (`HallucType *local = ...`). Die KI repariert den C-Code -- mit der
           KRITISCHEN Regel, dass die korrekten Header/Typen SCHON eingebunden sind (sonst erfindet
           die KI Header, um zu "fixen"). build_repair_prompt() kodiert diese Regel.

Verifikation IMMER ueber modules/ido_compiler (compile_code success). KEIN erfundener Header.
"""
import re
import modules.ido_compiler as _ic
import modules.halluc_detect as _hd

# ----------------------------------------------------------------- STUFE 1 (det.)
_LHS = r"([A-Za-z_]\w*(?:\[[^\]]*\])?|\([^()]*\))"
_UNK_ARROW = re.compile(_LHS + r"\s*->\s*unk([0-9A-Fa-f]+)")
_UNK_DEREF_DOT = re.compile(r"\(\s*\*\s*" + _LHS + r"\s*\)\s*\.\s*unk([0-9A-Fa-f]+)")


def unk_rewrite(code):
    prev = None
    while prev != code:
        prev = code
        code = _UNK_DEREF_DOT.sub(lambda m: f"(*(s32 *)((char *)({m.group(1)}) + 0x{m.group(2)}))", code)
        code = _UNK_ARROW.sub(lambda m: f"(*(s32 *)((char *)({m.group(1)}) + 0x{m.group(2)}))", code)
    return code


def declare_undef(code, log):
    add = ""
    for s in set(re.findall(r"'([A-Za-z_]\w*)' undefined", log)):
        if re.search(r"\bextern\b[^;]*\b" + re.escape(s) + r"\b", code):
            continue
        rs = re.escape(s)
        scalar = re.search(rf"\b{rs}\s*=(?!=)", code) and not re.search(rf"\b{rs}\s*\[", code) \
            and not re.search(rf"&\s*{rs}\b", code)
        add += (f"extern s32 {s};\n" if scalar else f"extern u8 {s}[];\n")
    return add + code if add else code


def dedup_cases(code):
    seen = set(); out = []
    for ln in code.splitlines():
        m = re.match(r"\s*case\s+([0-9xXA-Fa-f]+)\s*:\s*$", ln)
        if m:
            k = m.group(1).lower()
            if k in seen:
                continue
            seen.add(k)
        out.append(ln)
    return "\n".join(out)


def deterministic_compile_fix(code, func_name, rounds=8):
    """-> (code, ok). Iteriert: compile -> Muster anwenden -> retry."""
    for _ in range(rounds):
        comp = _ic.compile_code(code, func_name)
        if comp.get("success"):
            _ic.cleanup_temp(comp.get("temp_dir", "")); return code, True
        log = comp.get("error_log", ""); _ic.cleanup_temp(comp.get("temp_dir", ""))
        nc = declare_undef(dedup_cases(unk_rewrite(code)), log)
        if nc == code:
            break
        code = nc
    comp = _ic.compile_code(code, func_name); ok = comp.get("success")
    _ic.cleanup_temp(comp.get("temp_dir", "")); return code, ok


# ----------------------------------------------------------------- STUFE 3 (AI, halluzinations-gegated)
_REPAIR_SYSTEM = (
    "You are a C compile-error fixer for the SGI IDO 5.3 compiler (Banjo-Tooie N64 decompilation).\n"
    "The build system already includes the available headers/types. You may NOT add headers or define\n"
    "types yourself.\n"
    "CRITICAL RULES (violating these makes it WORSE):\n"
    "- Do NOT add any #include directive.\n"
    "- Do NOT define or typedef any struct/union/enum/type yourself.\n"
    "- Some type names in the code are NOT AVAILABLE in THIS compile environment (regardless of whether\n"
    "  they exist elsewhere in the project). You must NOT rely on them. This is NOT a missing header --\n"
    "  do not try to make the type appear; replace it.\n"
    "- A wrong TYPE NAME does NOT mean the memory ACCESS is wrong. The field offsets / dereferences are\n"
    "  most likely correct. KEEP the access pattern and the logic; change ONLY the unavailable type.\n"
    "- To replace an unavailable type: use a base type (s32/u8/f32/...) for the pointer/local, and where\n"
    "  a field is read/written use an EXPLICIT byte offset `*(T*)((char*)ptr + 0xNN)`. (`unkNN` = offset\n"
    "  0xNN.) Prefer explicit offsets over guessing another named type.\n"
    "- Some FIELD NAMES (e.g. ptr->someField) are also not available. Same fix: replace ptr->someField\n"
    "  with an explicit byte offset. The TARGET ASSEMBLY (if given) shows the real load/store offsets\n"
    "  (e.g. `lw $v0, 0x5C($a0)` => offset 0x5C) -- use it to pick the correct offset.\n"
    "- Keep ALL of the function's logic and structure; change ONLY what is needed to compile.\n"
    "Reply with ONLY the corrected C code. No markdown, no code fences, no explanation."
)


def build_repair_prompt(code, error_log, tech_env="", hallucinated=None,
                        hallucinated_fields=None, asm=""):
    """(system, user) fuer die KI-Reparatur.
    hallucinated = unsourced TYPnamen, hallucinated_fields = unsourced FELDnamen (beide vom Detektor),
    asm = Target-ASM (echte Offsets zur Aufloesung erfundener Namen). tech_env = verfuegbare Typen."""
    parts = []
    if hallucinated:
        parts.append("// === UNAVAILABLE TYPE NAMES (not in techenv, asm, or headers -- replace each with a\n"
                     "// base type + explicit byte offset; keep the access) ===\n" + ", ".join(hallucinated))
    if hallucinated_fields:
        parts.append("// === UNAVAILABLE FIELD NAMES (ptr->field not available -- replace with explicit byte\n"
                     "// offset from the ASM; keep the access) ===\n" + ", ".join(hallucinated_fields))
    if tech_env:
        parts.append("// === AVAILABLE TYPES & SIGNATURES (these ARE provided by the build; use ONLY these) ===\n"
                     + tech_env.strip())
    if asm:
        parts.append("// === TARGET ASSEMBLY (ground truth for offsets/structure) ===\n" + asm.strip())
    parts.append("// === COMPILER ERRORS (fix these) ===\n" + error_log.strip())
    parts.append("// === CURRENT C CODE ===\n" + code)
    return _REPAIR_SYSTEM, "\n\n".join(parts)


def _error_count(log):
    return len(re.findall(r"cfe:\s*Error:", log))


def _call_ai(ai_call, sysm, usr, think):
    """ai_call mit optionalem think-Flag aufrufen (robust, falls die Signatur es nicht kennt)."""
    try:
        return ai_call(sysm, usr, think=think)
    except TypeError:
        return ai_call(sysm, usr)


def _ai_repair_loop(code, func_name, tech_env, ai_call, ai_rounds, use_thinking=True):
    """Iterative KI-Reparatur: fix -> neu kompilieren -> naechster Fehler -> ... bis MATCH oder
    bis die KI VOLL VERSAGT (nichts Neues, Runden erschoepft, oder 2 Runden ohne Fortschritt).
    Behandelt 'fix one, reveal another'. Eskaliert in der zweiten Haelfte auf Thinking-Modus
    (Hebel 2, fuer harte/Mehrfach-Faelle) -- NUR wenn use_thinking (Produktion=False=nur oneshots).
    Reicht Typ- + FELD-Halluzinationen + ASM mit (Hebel 1). -> (code, ok)."""
    asm = _hd.target_asm_text(func_name)
    no_progress = 0
    for rnd in range(ai_rounds):
        comp = _ic.compile_code(code, func_name)
        if comp.get("success"):
            _ic.cleanup_temp(comp.get("temp_dir", "")); return code, True
        log = comp.get("error_log", ""); _ic.cleanup_temp(comp.get("temp_dir", ""))
        before = _error_count(log)
        h_types = _hd.unsourced_type_names(log, func_name)
        h_fields = _hd.unsourced_field_names(log, func_name)
        sysm, usr = build_repair_prompt(code, log, tech_env, hallucinated=h_types,
                                        hallucinated_fields=h_fields, asm=asm)
        think = use_thinking and rnd >= (ai_rounds // 2)   # Thinking-Eskalation nur wenn erlaubt
        try:
            new = _call_ai(ai_call, sysm, usr, think)
        except Exception:
            new = None
        if not new or new.strip() == code.strip():
            break  # KI liefert nichts Neues -> voll versagt
        new, ok = deterministic_compile_fix(new, func_name)
        if ok:
            return new, True
        comp2 = _ic.compile_code(new, func_name)
        after = _error_count(comp2.get("error_log", "")); _ic.cleanup_temp(comp2.get("temp_dir", ""))
        no_progress = no_progress + 1 if after >= before else 0
        code = new
        if no_progress >= 2:
            break
    return code, False


def compile_error_expert(code, func_name, ai_call=None, tech_env="", ai_rounds=6,
                         target_s_path=None, context_c=None, use_m2c=True, use_thinking=True):
    """Compiler-Error-Experte -- self-contained, draft-ERHALTEND priorisiert (Lukas: 100% ohne
    Code-Verlust). Stufen, je nur wenn die vorige versagt:
      1 DETERMINISTISCH  deterministic_compile_fix (mechanische Muster).
      2 DETEKTION  halluc_detect: unsourced TYP- + FELD-Namen (sicher erfunden) -> informiert den Prompt.
      3 KI-REPARATUR (ITERATIV, DRAFT-ERHALTEND)  fuer ALLES, was nicht kompiliert (nicht nur
                      Halluzinationen -- Hebel 3): chirurgische Edits, Draft bleibt. Fokus-Prompt mit
                      Typ/Feld-Liste + ASM-Offsets (Hebel 1), Thinking-Eskalation in Runde >=N/2 (Hebel 2).
      4 m2c-WHOLE-RESCUE (LETZTE Rettung, DRAFT-ZERSTOEREND): nur wenn die KI voll versagt; mit
                      context_c (Hebel 4). Wirft den Draft weg -> bewusst ganz am Ende.
    Returns {code, ok, stage, hallucinated_types, hallucinated_fields}.
    stage in 'deterministic'|'ai'|'m2c'|'unfixed'."""
    # tech_env auto-laden (Hebel 4 draft-erhaltend): echte verfuegbare Structs/Felder fuer die KI
    if not tech_env:
        tech_env = _hd.techenv_types_text(func_name)

    # --- Stufe 1: deterministisch -------------------------------------------------
    code, ok = deterministic_compile_fix(code, func_name)
    if ok:
        return {"code": code, "ok": True, "stage": "deterministic",
                "hallucinated_types": [], "hallucinated_fields": []}

    # --- Stufe 2: Detektion (informiert nur, gated NICHT mehr -- Hebel 3) ----------
    comp = _ic.compile_code(code, func_name)
    log = comp.get("error_log", ""); _ic.cleanup_temp(comp.get("temp_dir", ""))
    h_types = _hd.unsourced_type_names(log, func_name)
    h_fields = _hd.unsourced_field_names(log, func_name)

    # --- Stufe 3: iterative KI-Reparatur fuer ALLE Compile-Fails (Draft-Erhalt) ----
    if ai_call is not None:
        rcode, ok = _ai_repair_loop(code, func_name, tech_env, ai_call, ai_rounds, use_thinking)
        if ok:
            return {"code": rcode, "ok": True, "stage": "ai",
                    "hallucinated_types": h_types, "hallucinated_fields": h_fields}
        code = rcode  # bester KI-Stand als Basis (Draft-naeher als m2c)

    # --- Stufe 4: m2c-Whole-Rescue -- LETZTE Rettung (draft-zerstoerend) -----------
    if use_m2c and target_s_path:
        try:
            import modules.m2c_donor as _m2c
            mc = _m2c.run_m2c_ido(target_s_path, func_name, context_c=context_c)
            mc, mok = deterministic_compile_fix(mc, func_name)
            if mok:
                return {"code": mc, "ok": True, "stage": "m2c",
                        "hallucinated_types": h_types, "hallucinated_fields": h_fields}
        except Exception:
            pass
    return {"code": code, "ok": False, "stage": "unfixed",
            "hallucinated_types": h_types, "hallucinated_fields": h_fields}
