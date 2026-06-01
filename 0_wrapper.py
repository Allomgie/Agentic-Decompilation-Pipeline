#!/usr/bin/env python3
# wrapper.py — Agentic IDO Repair & Match Pipeline v2
# Fixes: -non_shared, smart stagnation, temp cap 0.7, early abort,
#        redeclaration auto-fix, rich live view, dynamic rounds

import os
import sys
import re
import json
import logging
import argparse
import threading
from pathlib import Path

from modules import ido_compiler
from modules import ai_agent
from modules import ido_comparison
from modules import diff_generator
from modules import micro_permuter
from modules import session as sess

# Logging
_file_handler = logging.FileHandler("pipeline.log", encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG, handlers=[_console_handler, _file_handler])
for noisy in ["httpx", "httpcore", "openai", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# ANSI Terminal-Farben (fuer Status-Ausgabe)
_C_GREEN  = "\033[1;32m"
_C_RED    = "\033[31m"
_C_YELLOW = "\033[33m"
_C_CYAN   = "\033[1;36m"
_C_DIM    = "\033[2m"
_C_SPLIT  = "\033[0;94m"   # High-Intensity Blue — Splitter-Pfad
_C_WHITE  = "\033[0;37m"   # normales Weiss — Splitter-Tag
_C_RESET  = "\033[0m"

# Konstanten
MAX_SYNTAX_RETRIES = 8
TEMP_START = 0.5
TEMP_MAX = 0.7

# Phase 0 (Cluster-Transplant-Vorflug): ab dieser ASM-Aehnlichkeit (DB-struct_score)
# zu einem GELOESTEN Similar wird VOR Phase A ein Transplantat versucht und per
# wasserdichtem evaluate_match verifiziert. Der DB-Score ist nur Vorfilter — die
# 100%-Entscheidung trifft immer das Live-evaluate_match. Per Env uebersteuerbar.
PHASE0_SIM_THRESHOLD = float(os.environ.get("PHASE0_SIM_THRESHOLD", "90"))


def _get_mode_constants():
    """Liest CONV_MODE dynamisch (nach CLI-Arg-Parsing)."""
    mode = os.environ.get("CONV_MODE", "oneshot")
    # thinking/dynamic/dynamic_batch nutzen alle dieselben Stagnations-Schwellen
    is_thinking_like = mode in ("thinking", "dynamic", "dynamic_batch")
    return {
        "mode": mode,
        "MAX_DIFF_ROUNDS": 20 if is_thinking_like else 50,
        "STAGNATION_BUMP": 3 if is_thinking_like else 5,
        "STAGNATION_ABORT": 8 if is_thinking_like else 15,
    }


def process_function(func_name, initial_c_code, target_s_path, tech_env,
                     ai_backend="local", rel_path="", live_cb=None,
                     similar_context="", cheatsheet_context="", similar_ref_code="",
                     similar_score=0.0, similar_source=""):
    log.info(f"=== Pipeline Start: {func_name} ===")
    _sub = str(Path(rel_path).parent) if rel_path else ""

    # Modus-Konstanten dynamisch lesen (nach CLI-Arg-Parsing)
    _mc = _get_mode_constants()
    _CONV_MODE = _mc["mode"]
    MAX_DIFF_ROUNDS = _mc["MAX_DIFF_ROUNDS"]
    STAGNATION_BUMP = _mc["STAGNATION_BUMP"]
    STAGNATION_ABORT = _mc["STAGNATION_ABORT"]

    # Session laden
    sn = sess.load_session(func_name)

    # Fix 1: 100%-Funktionen sofort ueberspringen
    if sn.get("best_score", 0) >= 100.0 and sn.get("best_code"):
        log.info(f"[{func_name}] Bereits 100% — ueberspringe.")
        out = _save(func_name, sn["best_code"], "output/perfect_matches", _sub)
        return {"success": True, "match_rate": 100.0, "struct_score": 100.0, "output_path": out, "ai_calls": 0}

    # Struct-100%-Funktionen sofort ueberspringen (Relocations, KI kann nicht mehr helfen)
    if sn.get("best_struct", 0) >= 100.0 and sn.get("best_code"):
        log.info(f"[{func_name}] Bereits Struct 100% — ueberspringe.")
        out = _save(func_name, sn["best_code"], "output/struct_matches", _sub)
        return {"success": True, "match_rate": sn.get("best_score", 0), "struct_score": 100.0, "output_path": out, "ai_calls": 0}

    # Baseline setzen: Der ORIGINALE Input, wird nie veraendert
    if not sn.get("baseline_code"):
        sn["baseline_code"] = initial_c_code

    # Session-Recovery: Vom besten Stand weitermachen, aber Baseline bleibt
    if sn.get("best_code") and sn.get("best_score", 0) > 0:
        cur_code = sn["best_code"]
        best_code = sn["best_code"]
        best_score = sn["best_score"]
        best_struct = sn.get("best_struct", 0.0)
        log.info(f"[{func_name}] Setze fort bei Score={best_score}% aus vorheriger Session")
    else:
        cur_code = initial_c_code
        best_code = initial_c_code
        best_score = 0.0
        best_struct = 0.0

    temp_dir = None
    ai_calls = 0

    def _lc(phase, **kw):
        if live_cb:
            live_cb(func_name=func_name, phase=phase, **kw)

    def _confidence(temp):
        """TechEnv-Confidence sinkt mit steigender Temperatur (min 80%)."""
        return max(0.80, 1.0 - (temp - TEMP_START) * 0.5)

    # === PHASE 0: Cluster-Transplant-Vorflug =============================
    # Existiert ein Similar aus einer GELOESTEN Quelle (perfect/struct) mit sehr
    # hoher ASM-Aehnlichkeit, ist diese Funktion oft ein Cluster-Duplikat. Wir
    # bauen das Transplantat (Referenz-Loesung mit den Symbolen DIESER Funktion)
    # und verifizieren es mit dem wasserdichten evaluate_match — BEVOR teure
    # Syntax-AI / Permuter laufen. Trifft es 100% / struct-100% -> sofort
    # speichern (0 AI). Sonst, wenn besser als der aktuelle Stand -> als
    # kompilierenden Seed uebernehmen und Phase A ueberspringen.
    # Der DB-similar_score ist nur Vorfilter; die Entscheidung trifft das Live-
    # evaluate_match (robust gegen veraltete DB-Scores).
    if (similar_ref_code
            and similar_source in ("perfect_matches", "struct_matches")
            and similar_score >= PHASE0_SIM_THRESHOLD):
        log.info(f"[{func_name}] Phase 0: Cluster-Transplant-Vorflug "
                 f"(sim={similar_score:.1f}%, src={similar_source})")
        _lc("phase0", c_code=cur_code)
        try:
            rr = micro_permuter.run_rescue(cur_code, func_name, target_s_path,
                                           similar_ref_code)
        except Exception as e:
            rr = {"compiled": False}
            log.warning(f"[{func_name}] Phase 0 Rescue-Fehler: {e}")
        if rr.get("compiled"):
            t_comp = ido_compiler.compile_code(rr["best_c_code"], func_name)
            if t_comp.get("success"):
                t_rank = ido_comparison.evaluate_match(
                    func_name, rr["best_c_code"], t_comp.get("temp_o_path", ""),
                    target_s_path)
                t_score = t_rank["match_rate"]
                t_struct = t_rank["struct_score"]
                _clean(t_comp.get("temp_dir"))
                log.info(f"[{func_name}] Phase 0 Transplant (Stufe {rr.get('stage')}): "
                         f"match={t_score}% struct={t_struct}%")
                if t_score >= 100.0:
                    sn["best_code"] = rr["best_c_code"]; sn["best_score"] = 100.0
                    sn["best_struct"] = 100.0; sess.save_session(func_name, sn)
                    out = _save(func_name, rr["best_c_code"], "output/perfect_matches", _sub)
                    log.info(f"[{func_name}] Phase 0: PERFEKT via Cluster-Transplant — 0 AI-Calls.")
                    ai_agent.reset_conversation(func_name)
                    return {"success": True, "match_rate": 100.0, "struct_score": 100.0,
                            "output_path": out, "ai_calls": 0}
                if t_struct >= 100.0:
                    sn["best_code"] = rr["best_c_code"]; sn["best_score"] = t_score
                    sn["best_struct"] = 100.0; sess.save_session(func_name, sn)
                    out = _save(func_name, rr["best_c_code"], "output/struct_matches", _sub)
                    log.info(f"[{func_name}] Phase 0: STRUCT-100% via Cluster-Transplant — 0 AI-Calls.")
                    ai_agent.reset_conversation(func_name)
                    return {"success": True, "match_rate": t_score, "struct_score": 100.0,
                            "output_path": out, "ai_calls": 0}
                # Nicht perfekt, aber besser als der aktuelle Stand -> als
                # kompilierenden Seed uebernehmen (Phase A wird damit gespart).
                if t_struct > best_struct or t_score > best_score:
                    cur_code = rr["best_c_code"]; best_code = rr["best_c_code"]
                    best_score = max(best_score, t_score)
                    best_struct = max(best_struct, t_struct)
                    log.info(f"[{func_name}] Phase 0: Transplant als Seed uebernommen "
                             f"(match={t_score}% struct={t_struct}%).")
            else:
                _clean(t_comp.get("temp_dir"))

    # === PHASE A: Syntax ===
    log.info(f"[{func_name}] Phase A: Syntax-Reparatur")
    comp = None
    # Deterministischer Auto-Fix: TechEnv einmal parsen, State ueber Iterationen halten
    _parsed_tech_env = ido_compiler.parse_tech_env(tech_env)
    _autofix_state = {}
    syntax_ai_fails = 0  # Zaehle KI-Versuche die nicht kompilierten
    _dyn_compiler_on = os.environ.get("DYNAMIC_COMPILER", "") == "1"
    for att in range(MAX_SYNTAX_RETRIES):
        comp = ido_compiler.compile_code(cur_code, func_name)
        if comp["success"]:
            temp_dir = comp.get("temp_dir", "")
            log.info(f"[{func_name}] Kompilierung OK (Versuch {att+1})")
            # --dynamic-compiler: Syntax-Loop vorbei -> Thinking-Track deaktivieren
            if _dyn_compiler_on and func_name:
                ai_agent.de_escalate_syntax_thinking(func_name)
            break
        log.warning(f"[{func_name}] Kompilierung fehlgeschlagen ({att+1}/{MAX_SYNTAX_RETRIES})")
        log.info(f"[{func_name}] Compiler-Output:\n{comp['error_log']}")
        _lc("syntax", attempt=att+1, error=comp["error_log"], c_code=cur_code)

        # Deterministischer Auto-Fix (kein KI-Call): redeclaration, Incompatible
        # return type und undefined — nutzt autoritative Typen aus env.ext.
        # Loop-sicher: _autofix_state verhindert dass derselbe Fix mehrfach greift.
        af_code, af_changed, af_applied = ido_compiler.autofix_errors(
            cur_code, comp["error_log"], func_name,
            te=_parsed_tech_env, state=_autofix_state)
        if af_changed:
            cur_code = af_code
            log.info(f"[{func_name}] Auto-Fix angewandt: {', '.join(af_applied)}")
            continue

        syntax_ai_fails += 1

        # --- ESKALATIONS-ENTSCHEIDUNG ab Syntax-Fail #5 ---
        # Reihenfolge: --dynamic-compiler hat Vorrang vor Batch. Begruendung:
        # Bei einem Compile-Loop haengen N parallele Oneshots typischerweise
        # alle am gleichen Fehler. 1 guter Thinking-Call schlaegt N schlechte
        # Oneshots. Wer beide haben will, kriegt: erst Thinking, falls weiter
        # gescheitert -> automatisch wieder Batch beim naechsten Fail-Threshold.
        _global_batch_syn = int(os.environ.get("BATCH_SIZE", "1"))
        # Syntax-Fix wird NICHT gestaffelt (env.ext ist hier autoritativ) — der
        # Batch bleibt klassisches Best-of-N. Kein distinct-Bonus (kein Kontext-
        # Staging), aber dieselbe BASE/reps-Basis wie der Diff-Pfad.
        _dyn_batch_syn = (int(os.environ.get("DYNAMIC_BATCH_BASE", "8"))
                          * int(os.environ.get("DYNAMIC_BATCH_REPS", "1")))

        # 1) --dynamic-compiler Track: ab Fail #5 Syntax-Thinking aktivieren
        _use_syntax_thinking = False
        if _dyn_compiler_on and syntax_ai_fails >= 5 and func_name:
            ai_agent.escalate_syntax_thinking(func_name)
            _use_syntax_thinking = True

        # 2) Batch-Entscheidung: nur wenn kein Syntax-Thinking eskaliert
        _use_batch_syn = False
        _active_batch_syn = 1
        if not _use_syntax_thinking:
            if _global_batch_syn > 1 and syntax_ai_fails >= 5:
                _use_batch_syn = True
                _active_batch_syn = _global_batch_syn
            elif _CONV_MODE == "dynamic_batch" and syntax_ai_fails >= 5:
                _use_batch_syn = True
                _active_batch_syn = _dyn_batch_syn

        if _use_batch_syn:
            log.info(f"[{func_name}] Batch Syntax-Fix: {_active_batch_syn}x parallel (Versuch {att+1})")
            candidates = ai_agent.generate_fix_batch(
                c_code=cur_code, context_data=comp["error_log"],
                task_type="syntax_fix", temperature=0.2,
                tech_env=tech_env, backend=ai_backend,
                func_name=func_name, batch_size=_active_batch_syn,
                cheatsheet_context=cheatsheet_context,
            )
            ai_calls += _active_batch_syn
            found_fix = False
            for cand in candidates:
                if not cand or cand == cur_code:
                    continue
                tc = ido_compiler.compile_code(cand, func_name)
                if tc.get("success"):
                    cur_code = cand
                    comp = tc
                    temp_dir = tc.get("temp_dir", "")
                    found_fix = True
                    log.info(f"[{func_name}] Batch Syntax-Fix erfolgreich!")
                    break
                else:
                    ido_compiler.cleanup_temp(tc.get("temp_dir", ""))
            if found_fix:
                # Falls vorher mal Thinking-Track aktiv war: deeskalieren
                if _dyn_compiler_on and func_name:
                    ai_agent.de_escalate_syntax_thinking(func_name)
                break
            continue

        # Single-Call: oneshot oder thinking — generate_fix entscheidet anhand
        # is_syntax_thinking_escalated(func_name).
        #
        # Temperatur:
        #   Oneshot: 0.1 — deterministisch fuer haeufige Syntax-Patterns (Redecl, extern).
        #   Thinking: 0.6 — Qwens "precise coding tasks" Empfehlung. Niedrige Temp
        #     bei Thinking-Calls verstaerkt Reasoning-Loops weil das Modell im
        #     local minimum festhaengt (gleiche Tokens, gleicher Gedankengang).
        #     0.6 ist hoch genug fuer echte Spread bei Loops, niedrig genug fuer
        #     fokussierte Antworten bei Syntax-Tasks.
        _syn_temp = 0.6 if _use_syntax_thinking else 0.1
        cur_code = ai_agent.generate_fix(c_code=cur_code, context_data=comp["error_log"],
                                         task_type="syntax_fix", temperature=_syn_temp,
                                         tech_env=tech_env, backend=ai_backend,
                                         func_name=func_name,
                                         cheatsheet_context=cheatsheet_context)
        ai_calls += 1
    else:
        log.error(f"[{func_name}] ABBRUCH Phase A nach {MAX_SYNTAX_RETRIES} Versuchen.")
        log.error(f"[{func_name}] Letzter Fehler:\n{comp['error_log']}")
        # Phase-A Rescue: Das Similar IST eine kompilierbare Funktion — sie hat
        # nur die falschen globalen Namen / evtl. Typabweichungen. Wir bauen
        # daraus eine kompilierbare Funktion (3 Stufen, beste gewinnt) und
        # uebergeben sie an Phase B/C, statt aufzugeben.
        _rescued = False
        if similar_ref_code:
            log.info(f"[{func_name}] Phase-A Rescue via Similar...")
            rr = micro_permuter.run_rescue(cur_code, func_name, target_s_path,
                                           similar_ref_code)
            for _ln in rr.get("report", "").splitlines():
                log.info(f"[{func_name}]   {_ln}")
            if rr.get("compiled"):
                cur_code = rr["best_c_code"]
                best_code = cur_code
                _clean(temp_dir)
                comp = ido_compiler.compile_code(cur_code, func_name)
                temp_dir = comp.get("temp_dir", "")
                _rescued = comp.get("success", False)
                if _rescued:
                    log.info(f"[{func_name}] Rescue Stufe {rr['stage']} erfolgreich "
                             f"(Struct={rr['match_rate']:.1f}%) — weiter mit Phase B/C.")
        if not _rescued:
            ai_agent.reset_conversation(func_name)
            if _dyn_compiler_on and func_name:
                ai_agent.de_escalate_syntax_thinking(func_name)
            # compiled=False: einziger Pfad ohne kompilierbaren Kandidaten -> "failed"
            return {"success": False, "match_rate": 0.0, "struct_score": 0.0,
                    "output_path": "", "ai_calls": ai_calls, "compiled": False}
        if _dyn_compiler_on and func_name:
            ai_agent.de_escalate_syntax_thinking(func_name)

    # === PHASE B/C: Diff ===
    log.info(f"[{func_name}] Phase B/C: Diff-Minimierung")
    stag = 0
    stag_total = 0
    temp = TEMP_START
    permuter_report = sn.get("permuter_report", "")  # Aus Session wiederherstellen

    try:
        # Initialer Permuter (nur wenn wir keinen gespeicherten Report haben)
        if not permuter_report:
            log.info(f"[{func_name}] Initialer Permuter...")
            _lc("permuter", c_code=cur_code)
            pr = micro_permuter.run_permuter(cur_code, func_name, target_s_path,
                                             similar_ref=similar_ref_code or None,
                                             strategy="beam")
            permuter_report = pr.get("report", "")
            sn["permuter_report"] = permuter_report  # Persistieren
            if pr["match_rate"] >= 100.0:
                sn["best_code"] = pr["best_c_code"]; sn["best_score"] = 100.0; sn["best_struct"] = 100.0
                sess.save_session(func_name, sn)
                out = _save(func_name, pr["best_c_code"], "output/perfect_matches", _sub)
                _clean(temp_dir); temp_dir = None
                ai_agent.reset_conversation(func_name)
                return {"success": True, "match_rate": 100.0, "struct_score": 100.0, "output_path": out, "ai_calls": ai_calls}
            if pr["match_rate"] > 0:
                log.info(f"[{func_name}] Permuter: {pr['match_rate']:.1f}%")
                cur_code = pr["best_c_code"]
                _clean(temp_dir)
                comp = ido_compiler.compile_code(cur_code, func_name)
                temp_dir = comp.get("temp_dir", "")
        else:
            log.info(f"[{func_name}] Permuter-Report aus Session geladen (ueberspringe Permuter).")

        for rnd in range(MAX_DIFF_ROUNDS):
            temp_o = comp.get("temp_o_path", "")
            if comp.get("success") and temp_o and os.path.exists(temp_o):
                rank = ido_comparison.evaluate_match(func_name, cur_code, temp_o, target_s_path)
            else:
                rank = {"match_rate": 0.0, "struct_score": 0.0, "mismatch_count": 999, "is_permuter_ready": False}
            score = rank["match_rate"]
            struct = rank["struct_score"]
            log.info(f"[{func_name}] Runde {rnd+1}: Score={score}% Mismatches={rank['mismatch_count']} Struct={struct:.1f}% Temp={temp:.1f}")
            _lc("diff", round=rnd+1, score=score, struct_score=struct,
                mismatches=rank["mismatch_count"], c_code=cur_code, temperature=temp)

            # Progress-Tracking: Nutze STRUCT_SCORE (nicht match_rate)
            # weil struct_score robuster ist (ignoriert Relocations)
            if struct > best_struct or score > best_score:
                if score > best_score:
                    best_score = score
                best_struct = struct
                best_code = cur_code
                stag = 0
                stag_total = 0
            else:
                stag += 1
                stag_total += 1

            if score >= 100.0:
                sn["best_code"] = cur_code; sn["best_score"] = 100.0; sn["best_struct"] = 100.0
                sess.save_session(func_name, sn)
                out = _save(func_name, cur_code, "output/perfect_matches", _sub)
                _clean(temp_dir); temp_dir = None
                ai_agent.reset_conversation(func_name)
                return {"success": True, "match_rate": 100.0, "struct_score": 100.0, "output_path": out, "ai_calls": ai_calls}

            # Struct 100% aber match_rate < 100% = Relocations, KI kann nicht mehr helfen
            if struct >= 100.0:
                log.info(f"[{func_name}] Struct 100% erreicht (Score={score:.1f}%) — Relocations, breche ab.")
                sn["best_code"] = cur_code; sn["best_score"] = score; sn["best_struct"] = 100.0
                sess.save_session(func_name, sn)
                out = _save(func_name, cur_code, "output/struct_matches", _sub)
                _clean(temp_dir); temp_dir = None
                ai_agent.reset_conversation(func_name)
                return {"success": True, "match_rate": score, "struct_score": 100.0, "output_path": out, "ai_calls": ai_calls}

            # Early abort bei Parse-Fehler
            # Focus-Logik: Erst Struktur fixen (Stack Frame, Missing/Extra),
            # dann Details (Register, Memory Offsets)
            same_length = (rank["mismatch_count"] <= len(
                ido_comparison.extract_hex_from_original_s(target_s_path)))
            # Wenn Instruktionsanzahl stimmt UND struct_score > 60% -> Detail-Phase
            if rank.get("struct_score", 0) >= 60 and summary_matches_length(comp, target_s_path):
                diff_focus = "detail"
            else:
                diff_focus = "structure"

            diff_json = diff_generator.create_json_diff(comp["temp_o_path"], target_s_path, focus=diff_focus)
            if '"type": "Error"' in diff_json or "Failed to parse" in diff_json:
                log.error(f"[{func_name}] ASM-Parser Fehler — ueberspringe.")
                break

            # Smart Stagnation: Temperatur erhoehen
            if stag >= STAGNATION_BUMP:
                old = temp
                temp = min(temp + 0.1, TEMP_MAX)
                stag = 0
                if temp > old:
                    log.info(f"[{func_name}] Stagnation: Temp {old:.1f} -> {temp:.1f}")

            # Abbruch NUR wenn: Temperatur am Maximum UND immer noch stagniert
            if stag_total >= STAGNATION_ABORT:
                if temp >= TEMP_MAX:
                    # Dynamic: Statt Abbruch -> Thinking einschalten wenn Struct < 80%
                    if _CONV_MODE == "dynamic" and best_struct < 80.0 and func_name:
                        if func_name not in ai_agent._escalated_funcs:
                            ai_agent.escalate_to_thinking(func_name)
                            log.info(f"[{func_name}] Dynamic: Eskaliere zu Thinking "
                                     f"(Struct={best_struct:.1f}% < 80%, {stag_total} Runden stagniert).")
                            stag_total = 0
                            stag = 0
                            temp = TEMP_START  # Reset Temp fuer frischen Thinking-Start
                            continue
                    # Dynamic-Batch: Statt Abbruch -> Batch-Sampling einschalten wenn Struct < 80%
                    if _CONV_MODE == "dynamic_batch" and best_struct < 80.0 and func_name:
                        if not ai_agent.is_batch_escalated(func_name):
                            ai_agent.escalate_to_batch(func_name)
                            log.info(f"[{func_name}] DynamicBatch: Eskaliere zu Batch "
                                     f"(Struct={best_struct:.1f}% < 80%, {stag_total} Runden stagniert).")
                            stag_total = 0
                            stag = 0
                            temp = TEMP_START  # Reset Temp damit Batch frischen Spread hat
                            continue
                    log.warning(f"[{func_name}] Abbruch: {stag_total} Runden ohne Fortschritt bei Max-Temp {TEMP_MAX}.")
                    break
                else:
                    temp = min(temp + 0.1, TEMP_MAX)
                    stag_total = 0
                    stag = 0
                    log.info(f"[{func_name}] Stagnation aber Temp nicht am Max — erhoehe auf {temp:.1f}, reset Counter.")

            # Permuter Fast-Lane
            if rank.get("is_permuter_ready") and sn.get("_permuter_done_for_struct", -1) != round(struct, 1):
                log.info(f"[{func_name}] Fast-Lane Permuter aktiv (Mismatches={rank['mismatch_count']}, Struct={struct:.1f}%)...")
                pr = micro_permuter.run_permuter(cur_code, func_name, target_s_path,
                                                 similar_ref=similar_ref_code or None,
                                                 strategy="beam")
                permuter_report = pr.get("report", permuter_report)
                if pr["match_rate"] >= 100.0:
                    sn["best_code"] = pr["best_c_code"]; sn["best_score"] = 100.0; sn["best_struct"] = 100.0
                    ai_agent.reset_conversation(func_name)
                    sess.save_session(func_name, sn)
                    out = _save(func_name, pr["best_c_code"], "output/perfect_matches", _sub)
                    _clean(temp_dir); temp_dir = None
                    return {"success": True, "match_rate": 100.0, "struct_score": 100.0, "output_path": out, "ai_calls": ai_calls}
                # Vergleiche Permuter-Score mit struct_score (gleiche Metrik!)
                # Permuter nutzt SequenceMatcher = struct_score, NICHT match_rate
                if pr["match_rate"] > struct:
                    cur_code = pr["best_c_code"]
                    _clean(temp_dir)
                    comp = ido_compiler.compile_code(cur_code, func_name)
                    temp_dir = comp.get("temp_dir", "")
                    continue
                else:
                    sn["_permuter_done_for_struct"] = round(struct, 1)
                    log.info(f"[{func_name}] Permuter erschoepft bei Struct={struct:.1f}%.")

            # KI-Call mit History-Kontext und Confidence
            hist_ctx = sess.build_history_context(sn, max_entries=10)
            # Permuter-Report anfuegen (zeigt was deterministisch schon getestet wurde)
            if permuter_report:
                hist_ctx = hist_ctx + "\n\n" + permuter_report if hist_ctx else permuter_report
            conf = _confidence(temp)
            _lc("ai_call", c_code=cur_code, diff=diff_json, temperature=temp)
            _effective_mode_str = ai_agent._effective_mode(func_name) if func_name else _CONV_MODE

            code_before_ai = cur_code

            # Batch-Entscheidung:
            # 1) Explizit via -b/--batch: globaler BATCH_SIZE > 1 (gilt fuer alle Oneshot-Calls)
            # 2) dynamic_batch: Funktion wurde zu Batch eskaliert (nach Stagnation).
            #    Adaptives Budget: BASE Slots + 1 Bonus-Slot pro zusaetzlich
            #    verfuegbarer distinct-Stufe. Mehr Kontext-Zutaten => mehr lohnt
            #    sich Exploration (1 Stufe=BASE, 2=+1, 3=+2, alle 4=+3).
            #    reps multipliziert das Ganze (Experiment-Knopf, Standard 1).
            _global_batch_size = int(os.environ.get("BATCH_SIZE", "1"))
            _dyn_batch_reps = int(os.environ.get("DYNAMIC_BATCH_REPS", "1"))
            _dyn_batch_base = int(os.environ.get("DYNAMIC_BATCH_BASE", "8"))
            _n_distinct = ai_agent._distinct_stage_count(
                bool(cheatsheet_context), bool(similar_context))
            _dyn_batch_size = (_dyn_batch_base + (_n_distinct - 1)) * _dyn_batch_reps

            _use_batch = False
            _active_batch_size = 1
            if _effective_mode_str == "oneshot":
                if _global_batch_size > 1:
                    _use_batch = True
                    _active_batch_size = _global_batch_size
                elif _CONV_MODE == "dynamic_batch" and func_name and ai_agent.is_batch_escalated(func_name):
                    _use_batch = True
                    _active_batch_size = _dyn_batch_size

            if _use_batch:
                # Best-of-N Batch Sampling
                _budget_note = ("global" if _global_batch_size > 1 else
                                f"dynamic_batch escalation, base={_dyn_batch_base}"
                                f"+{_n_distinct - 1} distinct x{_dyn_batch_reps}reps")
                log.info(f"[{func_name}] Batch-Call: {_active_batch_size}x parallel "
                         f"(reason={_budget_note})")
                candidates = ai_agent.generate_fix_batch(
                    c_code=cur_code, context_data=diff_json,
                    task_type="diff_minimize", temperature=temp,
                    tech_env=tech_env, backend=ai_backend,
                    history_context=hist_ctx, confidence=conf,
                    func_name=func_name, batch_size=_active_batch_size,
                    similar_context=similar_context,
                    cheatsheet_context=cheatsheet_context,
                )
                ai_calls += _active_batch_size

                # Bewerte alle Kandidaten: Erst deduplizieren, dann kompilieren
                seen_codes = set()
                unique_candidates = []
                for cand in candidates:
                    if not cand or cand == cur_code:
                        continue
                    cand_stripped = cand.strip()
                    if cand_stripped in seen_codes:
                        continue
                    if sess.is_duplicate(sn, cand):
                        continue
                    seen_codes.add(cand_stripped)
                    unique_candidates.append(cand)

                log.debug(f"[{func_name}] Batch: {len(unique_candidates)} einzigartige von {len(candidates)}")

                best_candidate = None
                best_cand_score = -1
                best_cand_struct = -1

                for cand in unique_candidates:
                    tc = ido_compiler.compile_code(cand, func_name)
                    if not tc.get("success"):
                        ido_compiler.cleanup_temp(tc.get("temp_dir", ""))
                        continue
                    temp_o = tc.get("temp_o_path", "")
                    if not temp_o or not os.path.exists(temp_o):
                        ido_compiler.cleanup_temp(tc.get("temp_dir", ""))
                        continue
                    cand_rank = ido_comparison.evaluate_match(func_name, cand, temp_o, target_s_path)
                    cand_s = cand_rank.get("struct_score", 0)
                    cand_m = cand_rank.get("match_rate", 0)
                    ido_compiler.cleanup_temp(tc.get("temp_dir", ""))
                    if cand_s > best_cand_struct or (cand_s == best_cand_struct and cand_m > best_cand_score):
                        best_candidate = cand
                        best_cand_score = cand_m
                        best_cand_struct = cand_s

                if best_candidate:
                    proposed = best_candidate
                    log.info(f"[{func_name}] Batch: Bester Kandidat Struct={best_cand_struct:.1f}% Score={best_cand_score:.1f}%")
                else:
                    log.info(f"[{func_name}] Batch: Kein valider Kandidat aus {_active_batch_size} Calls.")
                    stag += 1
                    stag_total += 1
                    continue

            else:
                # Einzelner Call (Oneshot oder Thinking)
                proposed = ai_agent.generate_fix(
                    c_code=cur_code, context_data=diff_json,
                    task_type="diff_minimize", temperature=temp,
                    tech_env=tech_env, backend=ai_backend,
                    history_context=hist_ctx, confidence=conf,
                    func_name=func_name,
                    similar_context=similar_context,
                    cheatsheet_context=cheatsheet_context,
                )
                ai_calls += 1

            # Duplicate-Detection
            if sess.is_duplicate(sn, proposed):
                log.info(f"[{func_name}] KI produzierte Duplikat — erhoehe Temp temporaer.")
                temp = min(temp + 0.05, TEMP_MAX)
                stag += 1
                stag_total += 1
                continue

            _clean(temp_dir)
            tc = ido_compiler.compile_code(proposed, func_name)
            if tc["success"]:
                cur_code = proposed
                comp = tc
                temp_dir = tc.get("temp_dir", "")
                sess.record_attempt(sn, proposed, score, rank["struct_score"], temp, "diff",
                                    prev_score=best_score, prev_code=code_before_ai)
            else:
                log.warning(f"[{func_name}] KI brach Syntax. Fallback.")
                sess.record_attempt(sn, proposed, 0.0, 0.0, temp, "syntax_broken",
                                    prev_score=best_score, prev_code=code_before_ai)
                cur_code = best_code
                comp = ido_compiler.compile_code(best_code, func_name)
                temp_dir = comp.get("temp_dir", "")

    except Exception as e:
        log.error(f"[{func_name}] Fehler Phase B/C: {e}")
    finally:
        _clean(temp_dir)
        # Conversation aufraeumen (Speicher freigeben, kein State-Leak)
        ai_agent.reset_conversation(func_name)
        # Session speichern (ueberlebt Neustarts)
        if best_score > sn.get("best_score", 0.0):
            sn["best_code"] = best_code
            sn["best_score"] = best_score
            sn["best_struct"] = best_struct
        sess.save_session(func_name, sn)

    log.info(f"[{func_name}] Fertig. Score={best_score}% AI={ai_calls}")
    out = _save(func_name, best_code, "output/partial_matches", _sub)
    # compiled=True: Phase A war erfolgreich, best_code ist ein kompilierbarer Seed.
    return {"success": False, "match_rate": best_score, "struct_score": best_struct,
            "output_path": out, "ai_calls": ai_calls, "compiled": True}


# Cheatsheet-Modul-State (wird von run_batch gesetzt wenn --cheatsheet aktiv)
_cheatsheet_state = {"results": None, "update_fn": None}

def _save(fn, code, tdir, sub=""):
    d = Path(tdir)
    if sub and sub != ".": d = d / sub
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{fn}.c"
    p.write_text(code, encoding="utf-8")

    # Inkrementelles Cheatsheet-Update bei neuem Perfect Match
    if "perfect_matches" in tdir and _cheatsheet_state["results"] is not None:
        try:
            _cheatsheet_state["update_fn"](
                _cheatsheet_state["results"], fn, code, sub)
        except Exception as e:
            log.debug(f"Cheatsheet update fuer {fn} fehlgeschlagen: {e}")

    return str(p)

def _clean(td):
    if td: ido_compiler.cleanup_temp(td)

def _find_asm(cp, ip, ap):
    rel = cp.relative_to(ip)
    for s in [ap / rel.with_suffix(".s"), ap / rel.with_suffix(".s").name]:
        if s.exists(): return s
    return None

def summary_matches_length(comp, target_s_path):
    """Prueft ob Draft und Target die gleiche Instruktionsanzahl haben."""
    draft_hex = ido_comparison.extract_hex_from_objdump(comp.get("temp_o_path", ""))
    target_hex = ido_comparison.extract_hex_from_original_s(target_s_path)
    return len(draft_hex) == len(target_hex) and len(draft_hex) > 0


# =========================================================
# BATCH
# =========================================================
def run_batch(input_dir, target_asm_dir, tech_env_path, ai_backend="local",
              num_workers=1, live_mode=False, use_cheatsheet=False,
              use_resume=False, use_similar=False):
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ip = Path(input_dir); ap = Path(target_asm_dir)
    if not ip.is_dir() or not ap.is_dir():
        print(f"ERROR: Verzeichnis nicht gefunden: {input_dir} oder {target_asm_dir}")
        return

    # Fix 2: Header-DB VOR dem ThreadPool laden (verhindert Race Condition)
    # Header-Dateien normalisieren (einmalig, persistent)
    ido_compiler.normalize_all_headers()
    # Header-DB laden
    ido_compiler._load_header_db()

    # Fix 6: Connection-Check zum AI Backend
    if ai_backend == "local":
        try:
            import httpx
            r = httpx.get(f"{ai_agent.LOCAL_API_BASE.rstrip('/v1')}/v1/models", timeout=10)
            if r.status_code != 200:
                print(f"ERROR: AI Backend antwortet nicht korrekt (Status {r.status_code})")
                print(f"URL: {ai_agent.LOCAL_API_BASE}")
                return
            print(f"AI Backend OK: {ai_agent.LOCAL_API_BASE}")
        except Exception as e:
            print(f"ERROR: Kann AI Backend nicht erreichen: {e}")
            print(f"URL: {ai_agent.LOCAL_API_BASE}")
            print("Starte die Pipeline NICHT ohne funktionierenden AI-Endpunkt.")
            return
    elif ai_backend == "dashscope":
        if not ai_agent.DASHSCOPE_API_KEY:
            print("ERROR: DASHSCOPE_API_KEY nicht gesetzt.")
            print("Export: DASHSCOPE_API_KEY=sk-xxx")
            return
        print(f"DashScope Backend: Model={ai_agent.DASHSCOPE_MODEL}")
        print(f"  Mode: {os.environ.get('CONV_MODE', 'oneshot')}")

    # Cheatsheet- und Similar-Kontext laden (aus perfect_matches generiert).
    # --cheatsheet: laedt die Pattern-Cheatsheets UND (zur Bequemlichkeit) die
    #               Similar-DB. --similar: laedt NUR die Similar-DB (Referenzen)
    #               ohne den restlichen Cheatsheet-Ballast.
    _cs_results = None
    _similar_db = {}
    if use_cheatsheet or use_similar:
        from modules.cheatsheet import (
            load_cache as _cs_load, full_rebuild as _cs_rebuild,
            sync_cache as _cs_sync,
            generate_prompt_context as _cs_prompt,
            incremental_update as _cs_update,
            load_similar_db as _cs_load_sim,
        )
        if use_cheatsheet:
            _cs_results = _cs_load()
            if _cs_results:
                print(f"Cheatsheet geladen: {_cs_results['total_files']} Patterns aus perfect_matches")
                _cs_results, _synced = _cs_sync(_cs_results)
                if _synced:
                    print(f"  +{_synced} neue Patterns nachgetragen (jetzt {_cs_results['total_files']})")
            else:
                print("Kein Cheatsheet-Cache — baue aus perfect_matches auf ...")
                _cs_results = _cs_rebuild()
                print(f"Cheatsheet erstellt: {_cs_results['total_files']} Patterns")
            # State fuer _save() setzen — ermoeglicht inkrementelle Updates
            _cheatsheet_state["results"] = _cs_results
            _cheatsheet_state["update_fn"] = _cs_update
        # Similar-ASM-DB laden (bei --similar ODER --cheatsheet)
        _similar_db = _cs_load_sim()
        if _similar_db:
            print(f"Similar-ASM-DB geladen: {len(_similar_db)} Referenzen")

    # Fix 5: Bereits fertige Funktionen aus perfect_matches + struct_matches scannen
    done_funcs = set()
    for done_dir in [Path("output/perfect_matches"), Path("output/struct_matches")]:
        if done_dir.exists():
            for f in done_dir.rglob("*.c"):
                done_funcs.add(f.stem)
    if done_funcs:
        print(f"Ueberspringe {len(done_funcs)} bereits fertige Funktionen (perfect + struct matches)")

    ted = Path(tech_env_path) if tech_env_path else None
    gte = ""
    if ted and not ted.is_dir():
        if ted.is_file(): gte = ted.read_text(encoding="utf-8")
        ted = None

    cfiles_all = sorted(ip.rglob("*.c"))
    if not cfiles_all:
        print("Keine .c Dateien gefunden."); return

    # Bereits fertige Funktionen VOR dem ThreadPool rausfiltern
    cfiles = [cf for cf in cfiles_all if cf.stem not in done_funcs]
    skipped_pre = len(cfiles_all) - len(cfiles)
    if skipped_pre:
        print(f"  -> {skipped_pre} von {len(cfiles_all)} Dateien vorgefiltert, {len(cfiles)} verbleiben.")

    # Resume: bereits versuchte Funktionen ueberspringen
    _resume_path = Path("data/pipeline_resume.txt")
    resume_skipped = 0
    if use_resume and _resume_path.exists():
        _resume_attempted = set(ln for ln in _resume_path.read_text(encoding="utf-8").splitlines() if ln)
        _before_resume = len(cfiles)
        cfiles = [cf for cf in cfiles if cf.stem not in _resume_attempted]
        resume_skipped = _before_resume - len(cfiles)
        if resume_skipped:
            print(f"  Resume: {resume_skipped} bereits versuchte uebersprungen, {len(cfiles)} verbleiben.")
        if not cfiles:
            print("Alle Funktionen bereits versucht. data/pipeline_resume.txt loeschen fuer Neustart.")
            return

    # Rich Live
    lp = None
    if live_mode:
        try:
            from modules.live_view import LivePanel
            lp = LivePanel(); lp.start()
        except ImportError:
            print("WARNUNG: pip install rich"); live_mode = False

    lock = threading.Lock()
    st = {"perfect": 0, "struct": 0, "partial": 0, "failed": 0, "skipped": 0}
    mrs = []; sss = []; tac = [0]

    def lcb(**kw):
        if lp: lp.update(**kw)

    def do(cf):
        fn = cf.stem; rp = cf.relative_to(ip)

        ts = _find_asm(cf, ip, ap)
        if not ts: return {"rel_path": str(rp), "status": "skipped", "match_rate": 0.0, "struct_score": 0.0, "ai_calls": 0}
        try: cc = cf.read_text(encoding="utf-8")
        except: return {"rel_path": str(rp), "status": "failed", "match_rate": 0.0, "struct_score": 0.0, "ai_calls": 0}
        if not cc.strip(): return {"rel_path": str(rp), "status": "skipped", "match_rate": 0.0, "struct_score": 0.0, "ai_calls": 0}

        te = gte
        if ted:
            for ext in (".json", ".txt", ".h", ".c"):
                tp = ted / rp.with_suffix(ext)
                if tp.exists():
                    try: te = tp.read_text(encoding="utf-8")
                    except: pass
                    break
            # Fallback: Funktionsname direkt im TechEnv-Verzeichnis suchen
            if not te:
                for ext in (".json", ".txt"):
                    tp = ted / f"{fn}{ext}"
                    if tp.exists():
                        try: te = tp.read_text(encoding="utf-8")
                        except: pass
                        break
        if not te:
            log.warning(f"[{fn}] Keine TechEnv gefunden! KI fliegt blind.")

        # Cheatsheet-Kontext an TechEnv anhaengen, Similar separat durchreichen
        cs_tier = ""
        _sim_ctx = ""
        _cs_ctx = ""
        _sim_ref_code = ""
        _sim_score = 0.0
        _sim_source = ""
        if _cs_results or _similar_db:
            _sn_pre = sess.load_session(fn)
            _best_struct = _sn_pre.get("best_struct", 0.0)
            ov_prefix = ""
            parts = rp.parts
            try:
                ov_idx = parts.index("overlays")
                if ov_idx + 1 < len(parts):
                    ov_prefix = parts[ov_idx + 1]
            except (ValueError, IndexError):
                pass
            sim_entry = _similar_db.get(fn)
            cs_result = _cs_prompt(_cs_results, fn, ov_prefix, sim_entry,
                                   current_struct=_best_struct)
            _sim_ctx = cs_result.get("similar", "")
            _cs_ctx = cs_result.get("cheatsheet", "")
            # _cs_ctx wird NICHT mehr an die TechEnv geklebt — separat durchreichen,
            # damit der Kontext-Fahrplan (generate_fix_batch) ihn pro Stufe togglen kann.
            # Tier-Indikator
            _sim_score = sim_entry.get("struct_score", 0) if sim_entry else 0
            _sim_source = sim_entry.get("best_ref_source", "") if sim_entry else ""
            _has_sim = bool(_sim_score >= 70.0 and _best_struct < _sim_score)
            # Rohen Referenz-C-Code an den Permuter geben, sobald eine brauchbare
            # Referenz existiert — unabhaengig von der Prompt-Unterdrueckung, da der
            # Permuter durch sein Score-Gate gegen irrefuehrende Referenzen geschuetzt ist.
            if sim_entry and _sim_score >= 70.0:
                _sim_ref_code = sim_entry.get("c_code", "") or ""
            _has_cs = False
            _ep_m = re.search(r'_entrypoint_(\d+)', fn)
            if _ep_m and _cs_results:
                _has_cs = int(_ep_m.group(1)) in _cs_results.get("entrypoint_slots", {})
            if not _has_cs and ov_prefix and _cs_results:
                _has_cs = ov_prefix in _cs_results.get("overlay_api_usage", {})
            if _has_sim and _has_cs:
                cs_tier = "SC"
            elif _has_sim:
                cs_tier = "S"
            elif _has_cs:
                cs_tier = "C"
            else:
                cs_tier = "·"

        r = process_function(fn, cc, str(ts), te, ai_backend, str(rp), lcb if live_mode else None,
                             similar_context=_sim_ctx, cheatsheet_context=_cs_ctx,
                             similar_ref_code=_sim_ref_code,
                             similar_score=_sim_score, similar_source=_sim_source)
        # Kategorisierung nach dem einzig ehrlichen Kriterium: hat die Pipeline
        # ueberhaupt einen KOMPILIERBAREN Kandidaten erzeugt? Wenn ja, ist es ein
        # verwertbarer Seed fuer den externen Permuter (egal wie hoch match/struct)
        # -> "partial". "failed" bleibt nur, was nach allen Versuchen nicht einmal
        # kompiliert (Phase A gescheitert, kein Rescue). Das beseitigt die
        # Inkonsistenz "struct 43% = fail, aber struct 32%/match 5% = partial".
        # match-perfect (byte-exakt 100%) vs. struct-perfect (struct 100%, aber
        # match < 100% wegen Relocations) werden getrennt — gruen vs. cyan.
        if r["success"] and r["match_rate"] >= 100.0:
            s = "perfect"
        elif r["success"]:
            s = "struct"
        elif r.get("compiled", (r["match_rate"] > 0 or r["struct_score"] > 0)):
            s = "partial"
        else:
            s = "failed"
        return {"rel_path": str(rp), "status": s, "match_rate": r["match_rate"],
                "struct_score": r["struct_score"], "ai_calls": r.get("ai_calls", 0),
                "cs_tier": cs_tier}

    # Resume-File: immer schreiben (ohne --resume frisch, mit --resume append)
    _resume_path.parent.mkdir(parents=True, exist_ok=True)
    _resume_f = open(_resume_path, "a" if use_resume else "w", encoding="utf-8")

    pbar = tqdm(total=len(cfiles), desc="Pipeline", unit="func", dynamic_ncols=True, disable=live_mode)

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futs = {ex.submit(do, cf): cf for cf in cfiles}
        for f in as_completed(futs):
            try: res = f.result()
            except Exception as e:
                log.error(f"Worker: {e}")
                res = {"rel_path": "?", "status": "failed", "match_rate": 0.0, "struct_score": 0.0, "ai_calls": 0}

            with lock:
                st[res["status"]] = st.get(res["status"], 0) + 1
                tac[0] += res.get("ai_calls", 0)
                if res["status"] != "skipped":
                    mrs.append(res["match_rate"]); sss.append(res["struct_score"])
                am = sum(mrs)/len(mrs) if mrs else 0
                astr = sum(sss)/len(sss) if sss else 0
                # Resume-File: versuchte Funktion merken (immer, nicht nur mit --resume)
                if res["status"] != "skipped":
                    _resume_f.write(Path(res["rel_path"]).stem + "\n")
                    _resume_f.flush()

            # Ergebnis grep-bar ins Log (ohne ANSI); Splitter mit Mutter markiert
            _lp = Path(res["rel_path"]).parts
            _lt = f" [Splitter→{_lp[1]}]" if (len(_lp) >= 2 and _lp[0] == "splits") else ""
            log.info(f"[{Path(res['rel_path']).stem}] RESULT={res['status'].upper()}{_lt} "
                     f"match={res['match_rate']:.1f}% struct={res['struct_score']:.1f}%")

            pbar.update(1)
            pbar.set_postfix_str(f"OK={st['perfect']} St={st['struct']} P={st['partial']} F={st['failed']} S={st['skipped']} avg={am:.1f}%")

            if not live_mode:
                r = res["rel_path"]
                _ct = res.get("cs_tier", "")
                _cs_tag = f" {_C_DIM}[{_ct}]{_C_RESET}" if _ct else ""
                # Splitter-Erkennung rein aus dem Pfad (kein Manifest noetig):
                # splits/<Mutter>/.../<func>.c -> parts[0]=="splits", parts[1]=Mutter
                _parts = Path(r).parts
                _is_split = len(_parts) >= 2 and _parts[0] == "splits"
                _split_tag = f" {_C_WHITE}[Splitter→{_parts[1]}]{_C_RESET}" if _is_split else ""
                _r = f"{_C_SPLIT}{r}{_C_RESET}" if _is_split else r
                if res["status"] == "perfect":
                    tqdm.write(f"  {_C_GREEN}✓ PERFECT{_C_RESET} {_r}{_cs_tag}{_split_tag}")
                elif res["status"] == "struct":
                    tqdm.write(f"  {_C_CYAN}≈ STRUCT{_C_RESET}  {_r} ({res['match_rate']:.1f}%/{res['struct_score']:.1f}%){_cs_tag}{_split_tag}")
                elif res["status"] == "partial":
                    tqdm.write(f"  {_C_YELLOW}~ PARTIAL{_C_RESET} {_r} ({res['match_rate']:.1f}%/{res['struct_score']:.1f}%){_cs_tag}{_split_tag}")
                elif res["status"] == "failed":
                    _rf = _r if _is_split else f"{_C_DIM}{r}{_C_RESET}"
                    tqdm.write(f"  {_C_RED}✗ FAILED{_C_RESET}  {_rf}{_cs_tag}{_split_tag}")

    pbar.close()
    if lp: lp.stop()

    # Resume-File schliessen
    _resume_f.close()

    n_total = len(cfiles_all); n = len(cfiles); p = st["perfect"]+st["struct"]+st["partial"]+st["failed"]
    total_skip = skipped_pre + resume_skipped + st.get("skipped", 0)
    am = sum(mrs)/len(mrs) if mrs else 0; astr = sum(sss)/len(sss) if sss else 0
    print(f"""
{'='*64}
  PIPELINE ERGEBNIS
{'='*64}
  Dateien:       {n_total:>6}   Verarbeitet: {p:>6}   Skip: {total_skip:>6}
  (davon {skipped_pre} perfect/struct, {resume_skipped} resume, {st.get('skipped',0)} kein ASM/leer)
  Workers:       {num_workers:>6}   AI Calls:    {tac[0]:>6}
{'='*64}
  Perfect:       {st['perfect']:>6}  ({st['perfect']/max(p,1)*100:.1f}%)
  Struct:        {st['struct']:>6}  ({st['struct']/max(p,1)*100:.1f}%)
  Partial:       {st['partial']:>6}  ({st['partial']/max(p,1)*100:.1f}%)
  Failed:        {st['failed']:>6}  ({st['failed']/max(p,1)*100:.1f}%)
{'='*64}
  Avg Match:     {am:>8.2f}%
  Avg Struct:    {astr:>8.2f}%
{'='*64}""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="IDO Repair Pipeline v2")
    ap.add_argument("--input", default="data/c_input")
    ap.add_argument("--target-asm", default="data/target_asm")
    ap.add_argument("--tech-env", default="data/tech_env")
    ap.add_argument("--backend", default="local", choices=["local", "claude", "dashscope"])
    ap.add_argument("-j", "--workers", type=int, default=1)
    ap.add_argument("--live", action="store_true", help="Rich Live-View")

    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument("--oneshot", action="store_const", const="oneshot", dest="conv_mode",
                            help="Klassischer One-Shot Modus (jeder Call isoliert)")
    mode_group.add_argument("--thinking", action="store_const", const="thinking", dest="conv_mode",
                            help="Qwen3.6 Multi-Turn mit Thinking Preservation")
    mode_group.add_argument("--dynamic", action="store_const", const="dynamic", dest="conv_mode",
                            help="Dynamisch: Oneshot zuerst, Thinking bei Stagnation")
    mode_group.add_argument("--dynamic-batch", action="store_const", const="dynamic_batch", dest="conv_mode",
                            help="Dynamisch: Oneshot zuerst, dann Kontext-Fahrplan-Batch bei "
                                 "Stagnation (1 Slot je Kontext-Pfad: full/no_cs/hard_env/diff_only). "
                                 "Wiederholungen pro Pfad via --dynamic-batch-reps (Default 1).")
    ap.set_defaults(conv_mode=None)

    ap.add_argument("--dynamic-compiler", action="store_true",
                    help="Orthogonal: Eskaliert Syntax-Fix-Calls zu Thinking ab 5 "
                         "Compile-Fehlern in Folge. Bleibt aktiv bis Compile gruen. "
                         "Kombinierbar mit allen anderen Modes — speziell sinnvoll mit "
                         "--dynamic-batch (Batch fuer Diff-Stagnation, Thinking fuer Syntax-Loop).")

    ap.add_argument("--streaming", action="store_true",
                    help="HTTP-Streaming fuer Thinking-Calls (verhindert Cloudflare Timeout)")
    ap.add_argument("--debug", action="store_true",
                    help="Schreibt Thinking-Streams live in data/debug/{func}.txt")
    ap.add_argument("-b", "--batch", type=int, default=1,
                    help="Best-of-N Sampling: N parallele Oneshot-Calls pro Runde "
                         "(Default 1 = aus). Mit --dynamic-batch wird das zur Batch-Groesse "
                         "auch nach Eskalation; ohne --dynamic-batch wirkt es ab Runde 1.")
    ap.add_argument("--dynamic-batch-reps", type=int, default=1,
                    help="Nur fuer --dynamic-batch: Multiplikator auf das adaptive "
                         "Budget (Default 1, Experiment-Knopf). Gesamt-Calls = "
                         "(base + distinct-Bonus) x reps. Wird ignoriert wenn -b/--batch gesetzt.")
    ap.add_argument("--dynamic-batch-base", type=int, default=8,
                    help="Nur fuer --dynamic-batch: Basis-Slots nach Eskalation (Default 8). "
                         "Pro zusaetzlich verfuegbarer Kontext-Stufe gibt es +1 Slot "
                         "(1 Stufe=base, 2=+1, 3=+2, alle 4=+3). Mehr Kontext => mehr Exploration.")
    ap.add_argument("--cheatsheet", action="store_true",
                    help="Haengt Kontext aus perfect_matches an den KI-Prompt an. "
                         "Vorher einmal 'python3 generate_cheatsheet.py' ausfuehren. "
                         "Laedt zusaetzlich die Similar-ASM-Referenz-DB.")
    ap.add_argument("--similar", action="store_true",
                    help="Laedt NUR die Similar-ASM-Referenz-DB (ohne den restlichen "
                         "Cheatsheet). --cheatsheet laedt beides.")
    ap.add_argument("--resume", action="store_true",
                    help="Setzt den Batch dort fort wo zuletzt aufgehoert wurde. "
                         "Merkt sich versuchte Funktionen in data/pipeline_resume.txt. "
                         "Datei wird automatisch geloescht wenn der Batch komplett durchlaeuft.")

    a = ap.parse_args()

    # Conv-Mode: CLI > ENV > Default
    if a.conv_mode:
        os.environ["CONV_MODE"] = a.conv_mode
    elif not os.environ.get("CONV_MODE"):
        os.environ["CONV_MODE"] = "oneshot"

    if a.streaming or a.debug:
        os.environ["USE_STREAMING"] = "1"

    if a.debug:
        os.environ["STREAM_DEBUG"] = "1"

    if a.batch > 1:
        os.environ["BATCH_SIZE"] = str(a.batch)

    if a.dynamic_batch_reps != 1:
        os.environ["DYNAMIC_BATCH_REPS"] = str(a.dynamic_batch_reps)
    if a.dynamic_batch_base != 8:
        os.environ["DYNAMIC_BATCH_BASE"] = str(a.dynamic_batch_base)

    if a.dynamic_compiler:
        os.environ["DYNAMIC_COMPILER"] = "1"

    # Reload ai_agent damit es die neuen ENV-Vars sieht
    import importlib
    importlib.reload(ai_agent)

    if a.live and a.workers > 1:
        print("HINWEIS: --live erzwingt -j 1 (Live-View ist single-threaded)")
        a.workers = 1

    mode_str = os.environ.get("CONV_MODE", "oneshot")
    stream_str = " +streaming" if (a.streaming or a.debug) else ""
    batch_str = f" +batch={a.batch}" if a.batch > 1 else ""
    if mode_str == "dynamic_batch" and a.batch <= 1:
        _base = a.dynamic_batch_base
        _max = _base + (len(ai_agent.CONTEXT_STAGES) - 1)
        _reps = a.dynamic_batch_reps
        _rep_str = f"x{_reps}reps" if _reps != 1 else ""
        batch_str = f" +dyn_batch={_base}..{_max}{_rep_str} (adaptiv n. Kontext)"
    dc_str = " +dyn_compiler" if a.dynamic_compiler else ""
    cs_str = " +cheatsheet" if a.cheatsheet else ""
    sim_str = " +similar" if (a.similar and not a.cheatsheet) else ""
    rs_str = " +resume" if a.resume else ""
    print(f"Modus: {mode_str}{stream_str}{batch_str}{dc_str}{cs_str}{sim_str}{rs_str}")

    run_batch(a.input, a.target_asm, a.tech_env, a.backend, a.workers, a.live,
              use_cheatsheet=a.cheatsheet, use_resume=a.resume,
              use_similar=a.similar)