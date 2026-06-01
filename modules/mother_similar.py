# modules/mother_similar.py
#
# "SIMILAR" + MICRO-PERMUTER bridge for the mother wrapper.
#
# The user's most important request: reuse the main pipeline's `similar`
# mechanism -- functions cluster by structural similarity, and a similar
# already-matched function is gold both as
#     (a) PROMPT INJECTION   (similar_context -> the AI imitates its shape), and
#     (b) PERMUTER SEED       (similar_ref_code -> deterministic beam search).
#
# A "mother" IS a function, so similarity here works at the MOTHER granularity --
# exactly the per-function granularity the main pipeline already uses.  We reuse
# the existing, proven sourcing (modules.cheatsheet) READ-ONLY; nothing in the
# main pipeline is touched.
#
# The micro_permuter compiles a function STANDALONE (ido_compiler.compile_code)
# and scores its whole-function skeleton against the whole target .s.  For a
# mother that is precisely a whole-mother struct push.  Because the permuter's
# internal compile path can differ from our compile-in-mother backend (PIC mode,
# header injection), we treat the permuter's score as ADVISORY and re-validate
# every candidate through mother_match.evaluate_mother -- our backend stays
# authoritative.

import logging

log = logging.getLogger("mother.similar")

# Mirror the main pipeline's gate: a similar reference is only fed to prompt /
# permuter when its structural score clears this bar (see cheatsheet.py).
SIMILAR_SCORE_THRESHOLD = 70.0


# ============================================================ DB / context load
def load_context_db(use_similar=False, use_cheatsheet=False):
    """Load the similar-ASM DB (+ optional cheatsheet patterns), exactly the way
    0_wrapper.run_batch does.  Returns (similar_db, cs_results).  Either may be
    empty/None when the flag is off or the data is missing."""
    similar_db = {}
    cs_results = None
    if not (use_similar or use_cheatsheet):
        return similar_db, cs_results
    try:
        from modules.cheatsheet import load_cache, load_similar_db
    except Exception as e:
        log.warning(f"cheatsheet module unavailable: {e}")
        return similar_db, cs_results
    similar_db = load_similar_db() or {}
    if similar_db:
        log.info(f"Similar-ASM-DB geladen: {len(similar_db)} Referenzen")
    if use_cheatsheet:
        cs_results = load_cache()
        if cs_results:
            log.info(f"Cheatsheet geladen: {cs_results.get('total_files', '?')} Patterns")
    return similar_db, cs_results


def context_for(name, similar_db, cs_results, current_struct=0.0, overlay_prefix=""):
    """Source (similar_context, cheatsheet_context, similar_ref_code) for one
    mother, reusing cheatsheet.generate_prompt_context (the same call the main
    pipeline makes per function).  similar_ref_code is the raw reference C handed
    to the permuter whenever a usable (>=threshold) reference exists."""
    if not similar_db and not cs_results:
        return "", "", ""
    try:
        from modules.cheatsheet import generate_prompt_context
    except Exception:
        return "", "", ""
    sim_entry = similar_db.get(name) if similar_db else None
    ctx = generate_prompt_context(cs_results, name, overlay_prefix, sim_entry,
                                  current_struct=current_struct)
    similar_context = ctx.get("similar", "")
    cheatsheet_context = ctx.get("cheatsheet", "")
    similar_ref_code = ""
    if sim_entry and sim_entry.get("struct_score", 0) >= SIMILAR_SCORE_THRESHOLD:
        similar_ref_code = sim_entry.get("c_code", "") or ""
    return similar_context, cheatsheet_context, similar_ref_code


# ============================================================ permuter (mother level)
def permuter_pass(c_src, name, target_s_path, target_insns, pic,
                  similar_ref_code="", strategy="beam"):
    """Run the micro-permuter on the WHOLE mother, seeded by the similar
    reference, then VALIDATE its best candidate through our compile-in-mother
    backend.  Returns dict:
        c_src       : adopted C (improved candidate, or the input unchanged)
        improved    : bool -- a strictly better validated candidate was adopted
        whole_exact : bool -- the adopted candidate is byte-exact end-to-end
        struct      : whole-mother struct of the adopted C (our metric)
        report      : permuter_report text (deterministic tested-changes log)
    On any failure the input is returned unchanged."""
    from . import mother_match as MC
    try:
        from modules import micro_permuter
    except Exception as e:
        log.warning(f"[{name}] micro_permuter unavailable: {e}")
        return {"c_src": c_src, "improved": False, "whole_exact": False,
                "struct": _whole_struct(c_src, name, target_insns, pic), "report": ""}

    base_struct = _whole_struct(c_src, name, target_insns, pic)
    try:
        pr = micro_permuter.run_permuter(
            c_src, name, target_s_path,
            similar_ref=similar_ref_code or None, strategy=strategy)
    except Exception as e:
        log.warning(f"[{name}] permuter raised: {e}")
        return {"c_src": c_src, "improved": False, "whole_exact": False,
                "struct": base_struct, "report": ""}

    report = pr.get("report", "") or ""
    cand = pr.get("best_c_code", c_src)
    if not cand or cand.strip() == c_src.strip():
        return {"c_src": c_src, "improved": False, "whole_exact": False,
                "struct": base_struct, "report": report}

    # Authoritative re-validation in OUR backend (permuter score is advisory).
    res = MC.evaluate_mother(cand, name, target_insns, pic=pic)
    if not res["ok"]:
        log.info(f"[{name}] permuter candidate rejected (does not compile in mother backend).")
        return {"c_src": c_src, "improved": False, "whole_exact": False,
                "struct": base_struct, "report": report}
    cand_struct = _struct_from_res(res)
    if res.get("whole_exact") or cand_struct > base_struct:
        log.info(f"[{name}] permuter improved whole-mother struct "
                 f"{base_struct:.2f} -> {cand_struct:.2f}"
                 f"{' (similar)' if similar_ref_code else ''}.")
        return {"c_src": cand, "improved": True,
                "whole_exact": bool(res.get("whole_exact")),
                "struct": cand_struct, "report": report}
    return {"c_src": c_src, "improved": False, "whole_exact": False,
            "struct": base_struct, "report": report}


# ------------------------------------------------------------ whole-mother struct
def _struct_from_res(res):
    """Mean per-arm struct of an evaluate_mother result -> a single whole-mother
    progress scalar (0..100)."""
    arms = res.get("arms", [])
    if not arms:
        return 100.0 if res.get("whole_exact") else 0.0
    return 100.0 * sum(a["struct"] for a in arms) / len(arms)


def _whole_struct(c_src, name, target_insns, pic):
    from . import mother_match as MC
    res = MC.evaluate_mother(c_src, name, target_insns, pic=pic)
    if not res["ok"]:
        return 0.0
    return _struct_from_res(res)
