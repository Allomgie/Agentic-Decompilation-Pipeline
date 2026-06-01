#!/usr/bin/env python3
# 1_wrapper_mother.py
#
# SECOND pipeline wrapper -- MOTHER FUNCTION decomposition.
#
# This wrapper is PARALLEL to 0_wrapper.py and shares NONE of its match logic.
# The existing pipeline (process_function / compile_code / evaluate_match) is
# standalone & non-PIC -- the path proven UNTAUGLICH for arms.  Here we use the
# proven compile-in-mother PIC region-diff backend (modules/mother_*).  The
# existing pipeline is never imported for its match core and never modified.
#
# Per mother .s (dropped into  data/target_asm/mother_functions/):
#   1. SEED   : m2c -> structured C (skeleton + inline arm drafts)   [mother_seed]
#   2. GATE   : only watertight (nested-if, 1 block/arm) mothers; loop
#               mothers are flagged + skipped.                       [mother_seed]
#   3. PHASE A: get the WHOLE mother to compile (PIC,-g3)  -- autofix + AI
#               syntax repair (reused read-only from the existing pipeline).
#   4. PHASE B: FRONT-TO-BACK per-arm matching.  Compile whole mother, extract
#               each arm region via the line map, align to the target stream,
#               byte-compare.  AI diff-minimize the focused arm until byte-exact,
#               then LOCK and move downstream (upstream stays frozen).
#   5. WRITE  : split/<mother>/ -- final mother.c, per-arm C, and a manifest of
#               byte-exact (C-arm <-> target hex slice) pairs.
#
# Reassembly is guaranteed by construction: when every arm region is byte-exact,
# the whole compiled mother equals the target.
#
# MUST run inside WSL (the IDO cc is a Linux binary):
#   wsl.exe -d ubuntu -- bash -c 'cd .../IDO_agent_pipeline && python3 1_wrapper_mother.py [opts]'

import os
import re
import sys
import json
import argparse
import logging
from pathlib import Path

from modules import mother_seed as seed_mod
from modules import mother_match as match_mod
from modules import mother_similar as sim_mod   # similar sourcing + validated permuter
from modules import ido_compiler            # READ-ONLY: autofix + tech_env parse
from modules import session as sess          # READ-ONLY: dedup + history context

# ai_agent pulls in the `openai` SDK at import time; only needed when the AI
# loop actually runs.  Import it lazily so --seed-only works without that dep.
def _ai():
    from modules import ai_agent
    return ai_agent

log = logging.getLogger("mother")

# --- defaults ------------------------------------------------------------------
DEF_TARGET = "data/target_asm/mother_functions"
# Output root MIRRORS the target hierarchy: a mother at
#   <DEF_TARGET>/<overlay>/<addr>/func_X.s
# produces a FOLDER at the same relative position
#   <DEF_SPLIT>/<overlay>/<addr>/func_X/
# holding that mother's per-arm .s + metadata.  Kept OUTSIDE mother_functions so
# the input tree stays clean (only the user's mother .s live there).
DEF_SPLIT  = "data/target_asm/mother_split"
DEF_TECHENV = "data/tech_env"

MAX_SYNTAX_RETRIES = 8
MAX_ARM_ITERS      = 12      # AI diff-minimize attempts per arm

# --- dynamic_batch loop knobs (ported from 0_wrapper.py, same semantics) -------
TEMP_START = 0.5
TEMP_MAX   = 0.7
# Per-arm matching unit -> use the thinking-like stagnation thresholds.
STAGNATION_BUMP  = 3        # bump temperature after N stagnant rounds
STAGNATION_ABORT = 8        # escalate/abort after N stagnant rounds at max temp
# Whole-mother permuter fast-lane only fires once the focused arm is this close.
PERMUTER_STRUCT_GATE = 50.0


def _conv_mode():
    """The mother wrapper defaults to dynamic_batch (the requested behaviour);
    CONV_MODE in the env still overrides it."""
    return os.environ.get("CONV_MODE", "dynamic_batch")


# ============================================================ tech env (optional)
def load_tech_env(tech_env_dir, name):
    """Best-effort: read a per-mother tech_env file if present, else ''."""
    if not tech_env_dir:
        return ""
    d = Path(tech_env_dir)
    if not d.is_dir():
        return d.read_text(encoding="utf-8") if d.is_file() else ""
    for ext in (".json", ".ext", ".txt", ""):
        p = d / f"{name}{ext}"
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return ""


# ============================================================ deterministic autofix
def _sanitize_m2c_types(c_src):
    """m2c emits a bare '?' as the type for any variable/parameter/return it
    cannot infer ('? sp2C;', 's32 f(void *a, ? b)').  cfe rejects that as a
    syntax error.  Replace '?' in clear TYPE positions with a neutral 's32' so
    the seed at least compiles -- byte-exactness is recovered later by the arm
    diff, and any wrong guess is caught when evaluate_mother re-validates.
    Ternaries (cond ? x : y) are left untouched: those never have the '?'
    immediately followed by 'identifier;' / 'identifier,' / 'identifier)'."""
    s = c_src
    # local declaration:  ^   ? name ;     (optionally with array/pointer)
    s = re.sub(r'(?m)^(\s*)\?(\s+\*?\w+\s*(?:\[[^\]]*\])?\s*;)', r'\1s32\2', s)
    # parameter / signature return:  ? name ,   |   ? name )
    s = re.sub(r'\?(\s+\*?\w+\s*[,)])', r's32\1', s)
    # leading return type at a function signature:  ^? name(
    s = re.sub(r'(?m)^(\s*)\?(\s+\w+\s*\()', r'\1s32\2', s)
    return s


def autofix_only(c_src, name, tech_env, pic, max_iters=MAX_SYNTAX_RETRIES):
    """Drive the mother toward a compile using ONLY the deterministic autofix
    (no AI).  Used by the split-asm/seed-only path so we can still recover arm
    alignment for mothers whose raw m2c seed has trivial, mechanically-fixable
    compile errors.  Returns (c_src, ok)."""
    c_src = _sanitize_m2c_types(c_src)
    te = ido_compiler.parse_tech_env(tech_env)
    state = {}
    for _ in range(max_iters):
        res = match_mod.evaluate_mother(c_src, name, [], pic=pic)
        if res["ok"]:
            return c_src, True
        fixed, changed, applied = ido_compiler.autofix_errors(
            c_src, res["error_log"], name, te=te, state=state)
        if not changed:
            return c_src, False
        log.info(f"[{name}]   autofix: {', '.join(applied)}")
        c_src = fixed
    res = match_mod.evaluate_mother(c_src, name, [], pic=pic)
    return c_src, res["ok"]


# ============================================================ Phase A: make it compile
def phase_a_compile(c_src, name, tech_env, backend, pic):
    """Drive the WHOLE mother to a successful compile (matching the target's PIC
    mode) using deterministic autofix first, then AI syntax_fix.
    Returns (c_src, ok)."""
    te = ido_compiler.parse_tech_env(tech_env)
    autofix_state = {}
    for attempt in range(MAX_SYNTAX_RETRIES):
        res = match_mod.evaluate_mother(c_src, name, [], pic=pic)  # only need compile
        if res["ok"]:
            log.info(f"[{name}] Phase A OK (compiles) nach {attempt} Reparaturen.")
            return c_src, True
        err = res["error_log"]
        log.info(f"[{name}] Phase A Versuch {attempt+1}: Compile-Fehler.")
        # 1) deterministic autofix (no AI)
        fixed, changed, applied = ido_compiler.autofix_errors(
            c_src, err, name, te=te, state=autofix_state)
        if changed:
            log.info(f"[{name}]   autofix: {', '.join(applied)}")
            c_src = fixed
            continue
        # 2) AI syntax fix
        c_src = _ai().generate_fix(
            c_code=c_src, context_data=err, task_type="syntax_fix",
            temperature=0.1, tech_env=tech_env, backend=backend, func_name=name)
    # final check
    res = match_mod.evaluate_mother(c_src, name, [], pic=pic)
    return c_src, res["ok"]


# ============================================================ candidate evaluation
def _eval_candidate(cand, name, target_insns, pic, k):
    """Score a full-mother candidate for arm k in OUR backend (authoritative).
    Returns (ok, up_ok, arm_struct_0_100, arm_exact, whole_exact) or
    (False, ...) when it does not compile / arm k is absent."""
    res = match_mod.evaluate_mother(cand, name, target_insns, pic=pic)
    if not res["ok"]:
        return (False, False, -1.0, False, False)
    arms = res["arms"]
    if k >= len(arms):
        return (False, False, -1.0, False, bool(res.get("whole_exact")))
    arm = arms[k]
    up_ok = all(arms[j]["exact"] for j in range(k))
    return (True, up_ok, arm["struct"] * 100.0, bool(arm["exact"]),
            bool(res.get("whole_exact")))


def _better(cand_tuple, best_tuple):
    """Rank candidates: upstream-exact first, then arm struct, then arm exact."""
    _, up, st, ex, _ = cand_tuple
    _, bup, bst, bex, _ = best_tuple
    return (up, st, ex) > (bup, bst, bex)


# ============================================================ Phase B: front-to-back arms
def phase_b_arms(c_src, name, ts_path, target_insns, tech_env, backend, max_iters, pic,
                 similar_context="", cheatsheet_context="", similar_ref_code=""):
    """Match arms front-to-back with the main pipeline's dynamic_batch machinery:
    per-arm progress tracking, temperature/stagnation, escalate-to-batch when the
    focused arm stalls below 80% struct, the whole-mother micro-permuter
    fast-lane (similar-seeded), and best-of-N batch generation with our backend as
    the authoritative scorer.  Returns (c_src, arm_status, whole_exact)."""
    res = match_mod.evaluate_mother(c_src, name, target_insns, pic=pic)
    if not res["ok"]:
        return c_src, [], False
    n_arms = len(res["arms"])
    locked = [False] * n_arms
    mode = _conv_mode()

    # ---- initial whole-mother permuter pass (seeded by the similar reference) --
    pp = sim_mod.permuter_pass(c_src, name, ts_path, target_insns, pic,
                               similar_ref_code=similar_ref_code, strategy="beam")
    if pp["whole_exact"]:
        c_src = pp["c_src"]
        log.info(f"[{name}] permuter solved the whole mother before Phase B.")
        res = match_mod.evaluate_mother(c_src, name, target_insns, pic=pic)
        return c_src, _status(res, [True] * n_arms), True
    if pp["improved"]:
        c_src = pp["c_src"]
    permuter_report = pp["report"]

    for k in range(n_arms):
        # in-memory session for dedup + history (NOT persisted: baseline is the
        # whole mother at this arm's start, not a standalone function).
        sn = {"baseline_code": c_src, "best_code": c_src, "best_score": 0.0,
              "best_struct": 0.0, "history": [], "seen_hashes": []}
        ekey = f"{name}#a{k}"
        temp = TEMP_START
        stag = stag_total = 0
        best_struct = -1.0
        permuted_at = -1.0

        for rnd in range(max_iters):
            res = match_mod.evaluate_mother(c_src, name, target_insns, pic=pic)
            if not res["ok"]:
                log.warning(f"[{name}] arm {k}: compile broke mid-loop; stop.")
                return c_src, _status(res, locked), False
            arms = res["arms"]
            if k >= len(arms):
                break
            arm = arms[k]
            up_ok = all(arms[j]["exact"] for j in range(k))
            arm_struct = arm["struct"] * 100.0

            if arm["exact"] and up_ok:
                locked[k] = True
                log.info(f"[{name}] arm {k} LOCKED byte-exact "
                         f"(rnd={rnd}, {arm['n_our']} insns).")
                break
            if res.get("whole_exact"):
                log.info(f"[{name}] whole mother byte-exact at arm {k}.")
                return c_src, _status(res, [True] * n_arms), True

            # progress on the focused arm's struct
            if arm_struct > best_struct:
                best_struct = arm_struct
                stag = stag_total = 0
            else:
                stag += 1
                stag_total += 1

            log.info(f"[{name}] arm {k} rnd={rnd}: struct={arm_struct:.1f}% "
                     f"({arm['equal']}/{max(arm['n_our'], arm['n_tgt'])}) "
                     f"temp={temp:.2f} up_ok={up_ok}")

            # smart stagnation: bump temperature
            if stag >= STAGNATION_BUMP:
                old = temp
                temp = min(temp + 0.1, TEMP_MAX)
                stag = 0
                if temp > old:
                    log.info(f"[{name}] arm {k}: stagnation, temp {old:.1f}->{temp:.1f}")

            # abort / escalate when stuck at max temp
            if stag_total >= STAGNATION_ABORT:
                if temp >= TEMP_MAX:
                    if (mode == "dynamic_batch" and best_struct < 80.0
                            and not _ai().is_batch_escalated(ekey)):
                        _ai().escalate_to_batch(ekey)
                        log.info(f"[{name}] arm {k}: escalate to Batch "
                                 f"(struct={best_struct:.1f}% < 80%).")
                        stag = stag_total = 0
                        temp = TEMP_START
                    else:
                        log.info(f"[{name}] arm {k}: give up after {stag_total} stagnant "
                                 f"rounds at max temp (struct={best_struct:.1f}%).")
                        break
                else:
                    temp = min(temp + 0.1, TEMP_MAX)
                    stag = stag_total = 0

            # whole-mother permuter fast-lane (similar-seeded), once per struct level
            if arm_struct >= PERMUTER_STRUCT_GATE and round(arm_struct, 1) != permuted_at:
                pp = sim_mod.permuter_pass(c_src, name, ts_path, target_insns, pic,
                                           similar_ref_code=similar_ref_code, strategy="beam")
                permuter_report = pp["report"] or permuter_report
                permuted_at = round(arm_struct, 1)
                if pp["whole_exact"]:
                    c_src = pp["c_src"]
                    return c_src, _status(res, [True] * n_arms), True
                if pp["improved"]:
                    c_src = pp["c_src"]
                    continue

            # ---- AI diff-minimize on the focused arm ----------------------------
            diff = match_mod.arm_diff_json(arm, focus="all")
            hist = sess.build_history_context(sn, max_entries=10)
            if permuter_report:
                hist = (hist + "\n\n" + permuter_report) if hist else permuter_report

            cand = _arm_ai_step(
                c_src, name, k, diff, temp, tech_env, backend, ekey, mode,
                similar_context, cheatsheet_context, target_insns, pic, sn, hist)

            if cand is None or sess.is_duplicate(sn, cand):
                stag += 1
                stag_total += 1
                temp = min(temp + 0.05, TEMP_MAX)
                continue

            ok, up2, st2, ex2, whole2 = _eval_candidate(cand, name, target_insns, pic, k)
            if ok and (up2 or not up_ok):
                # adopt only if it does not regress an already-exact upstream
                sess.record_attempt(sn, cand, st2, st2, temp, "diff",
                                    prev_score=best_struct, prev_code=c_src)
                c_src = cand
                if whole2:
                    res = match_mod.evaluate_mother(c_src, name, target_insns, pic=pic)
                    return c_src, _status(res, [True] * n_arms), True
            else:
                sess.record_attempt(sn, cand, 0.0, 0.0, temp, "syntax_broken",
                                    prev_score=best_struct, prev_code=c_src)
        else:
            log.info(f"[{name}] arm {k} not byte-exact after {max_iters} rounds "
                     f"(best struct={best_struct:.1f}%); best-effort.")

    res = match_mod.evaluate_mother(c_src, name, target_insns, pic=pic)
    return c_src, _status(res, locked), res.get("whole_exact", False)


def _arm_ai_step(c_src, name, k, diff, temp, tech_env, backend, ekey, mode,
                 similar_context, cheatsheet_context, target_insns, pic, sn, hist):
    """One AI step for arm k.  In dynamic_batch+escalated state, runs best-of-N
    batch generation (adaptive budget, staged context) and returns the best
    candidate by our authoritative arm score; otherwise a single generate_fix.
    Returns a candidate C string (or None)."""
    ai = _ai()
    conf = max(0.1, 1.0 - (temp - TEMP_START))

    use_batch = (mode == "dynamic_batch" and ai.is_batch_escalated(ekey))
    if not use_batch:
        return ai.generate_fix(
            c_code=c_src, context_data=diff, task_type="diff_minimize",
            temperature=temp, tech_env=tech_env, backend=backend, func_name=name,
            history_context=hist, confidence=conf,
            similar_context=similar_context, cheatsheet_context=cheatsheet_context)

    # adaptive batch budget (same formula as 0_wrapper.py)
    base = int(os.environ.get("DYNAMIC_BATCH_BASE", "8"))
    reps = int(os.environ.get("DYNAMIC_BATCH_REPS", "1"))
    n_distinct = ai._distinct_stage_count(bool(cheatsheet_context), bool(similar_context))
    batch_size = (base + (n_distinct - 1)) * reps
    log.info(f"[{name}] arm {k}: batch-call {batch_size}x parallel "
             f"(base={base}+{n_distinct - 1} distinct x{reps}).")
    candidates = ai.generate_fix_batch(
        c_code=c_src, context_data=diff, task_type="diff_minimize",
        temperature=temp, tech_env=tech_env, backend=backend,
        history_context=hist, confidence=conf, func_name=name,
        batch_size=batch_size, similar_context=similar_context,
        cheatsheet_context=cheatsheet_context)

    seen = set()
    best_cand = None
    best_tuple = (False, False, -1.0, False, False)
    for c in candidates:
        if not c or c.strip() == c_src.strip():
            continue
        key = c.strip()
        if key in seen or sess.is_duplicate(sn, c):
            continue
        seen.add(key)
        t = _eval_candidate(c, name, target_insns, pic, k)
        if t[0] and _better(t, best_tuple):
            best_tuple = t
            best_cand = c
    if best_cand is not None:
        log.info(f"[{name}] arm {k}: best batch cand up_ok={best_tuple[1]} "
                 f"struct={best_tuple[2]:.1f}% exact={best_tuple[3]}.")
    return best_cand


def _status(res, locked):
    out = []
    for a in res.get("arms", []):
        out.append({"id": a["id"], "head": a["head"], "kind": a["kind"],
                    "exact": a["exact"], "struct": round(a["struct"], 3),
                    "n_our": a["n_our"], "n_tgt": a["n_tgt"],
                    "tgt_range": a["tgt_range"],
                    "tgt_hex": a["tgt_hex"], "stmts": a.get("stmts", []),
                    "locked": locked[a["id"]] if a["id"] < len(locked) else False})
    return out


# ============================================================ output
def write_outputs(mother_dir, name, c_src, arm_status, whole_exact, seed):
    d = Path(mother_dir)               # already the mirrored per-mother folder
    (d / "arms").mkdir(parents=True, exist_ok=True)
    (d / "mother.c").write_text(c_src, encoding="utf-8")
    (d / "seed.c").write_text(seed["c_src"], encoding="utf-8")
    manifest = {
        "name": name,
        "whole_exact": whole_exact,
        "n_arms": len(arm_status),
        "n_exact": sum(1 for a in arm_status if a["exact"]),
        "arms": [],
    }
    for a in arm_status:
        arm_c = "\n".join(a["stmts"])
        (d / "arms" / f"arm_{a['id']:02d}.c").write_text(arm_c, encoding="utf-8")
        manifest["arms"].append({
            "id": a["id"], "head": a["head"], "kind": a["kind"],
            "exact": a["exact"], "struct": a["struct"], "locked": a["locked"],
            "n_insns": a["n_tgt"], "target_range": a["tgt_range"],
            "c_stmts": a["stmts"], "target_hex": a["tgt_hex"],
        })
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return d, manifest


# ============================================================ ASM split export
# For the SIMILAR-DB workflow: the similar database compares ASM-to-ASM (masked
# hex), so each arm only needs its byte-exact TARGET instruction slice rendered
# as a standalone spimdisasm-style .s -- NO C is required.  build_similar_asm_db
# then ingests these as targets and clusters each against already-translated
# references.  Deterministic: uses the seed's arm->target alignment, no AI.
def render_arm_asm(func_label, insns_slice):
    """Render a target instruction slice as a minimal spimdisasm .s that
    extract_hex_from_original_s can read (glabel + /* vaddr hex */ mnem ops)."""
    out = [f"glabel {func_label}"]
    for ins in insns_slice:
        body = f"{ins['mnem']} {ins['ops']}".rstrip()
        out.append(f"/* {ins['vaddr']} {ins['hex']} */  {body}")
    out.append(f"endlabel {func_label}")
    return "\n".join(out) + "\n"


def export_arms_asm(mother_dir, target_insns, arms):
    """Write one standalone .s per arm (byte-exact target hex slice) directly
    into the mirrored per-mother folder `mother_dir`.  Returns list of
    (func_label, n_insns).

    The seed's per-arm aligned ranges can OVERLAP when the m2c draft does not
    byte-match (loose alignment anchors).  For a clean similarity DB we clip the
    arms to a DISJOINT cover: sorted by target order, each arm ends where the
    next begins.  Empty/degenerate ranges are skipped."""
    d = Path(mother_dir)
    d.mkdir(parents=True, exist_ok=True)
    ranges = sorted(
        [a.get("tgt_range", (0, 0)) for a in arms if a.get("tgt_range", (0, 0))[1] >
         a.get("tgt_range", (0, 0))[0]],
        key=lambda r: r[0])
    written = []
    for i, (t0, t1) in enumerate(ranges):
        if i + 1 < len(ranges):
            t1 = min(t1, ranges[i + 1][0])   # disjoint: stop at next arm's start
        if t1 <= t0:
            continue
        sl = target_insns[t0:t1]
        if not sl:
            continue
        label = f"func_{sl[0]['vaddr'].upper()}"
        (d / f"{label}.s").write_text(render_arm_asm(label, sl), encoding="utf-8")
        written.append((label, len(sl)))
    return written


def write_skip(mother_dir, name, reason, seed=None):
    d = Path(mother_dir)              # already the mirrored per-mother folder
    d.mkdir(parents=True, exist_ok=True)
    info = {"name": name, "skipped": True, "reason": reason}
    if seed is not None:
        info["n_arms"] = len(seed["arms"])
        info["loop_arms"] = seed["loop_arms"]
        info["arms"] = [{"id": a["id"], "kind": a["kind"], "head": a["head"]}
                        for a in seed["arms"]]
    (d / "SKIPPED.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


# ============================================================ per-mother driver
def process_mother(ts_path, out_root, target_root, tech_env_dir, backend, max_iters,
                   seed_only, similar_db=None, cs_results=None, export_asm=False):
    name_hint = Path(ts_path).stem
    # Mirror the input hierarchy: the per-mother output FOLDER sits at the same
    # relative position the mother .s occupies under target_root, with the file
    # stem becoming a directory (the slot where the mother function lived).
    try:
        rel = Path(ts_path).resolve().relative_to(Path(target_root).resolve()).with_suffix("")
    except Exception:
        rel = Path(name_hint)
    mother_dir = Path(out_root) / rel
    log.info(f"=== MOTHER {name_hint} :: {ts_path} -> {mother_dir} ===")
    try:
        seed = seed_mod.seed_mother(ts_path)
    except Exception as e:
        log.error(f"[{name_hint}] seed failed: {e}")
        write_skip(mother_dir, name_hint, f"seed/m2c failed: {e}")
        return {"name": name_hint, "status": "seed_failed"}
    name = seed["name"]
    target_insns = seed["target"]["insns"]
    log.info(f"[{name}] arms={len(seed['arms'])} loop_arms={seed['loop_arms']} "
             f"target_insns={len(target_insns)}")

    if not seed_mod.is_watertight(seed):
        log.info(f"[{name}] NICHT wasserdicht (loop/switch arms) -> skip.")
        write_skip(mother_dir, name, "not watertight (loop/switch mother)", seed)
        return {"name": name, "status": "skipped_loop"}

    pic = match_mod.detect_pic(target_insns)
    log.info(f"[{name}] target build mode: {'PIC' if pic else 'non-PIC (-non_shared)'}")

    if seed_only:
        # deterministic: baseline evaluate (no AI), write seed + arm target slices.
        # First a NO-AI autofix pass to recover trivially-broken m2c seeds (the
        # deterministic half of Phase A) -- this is pure yield recovery for the
        # split-all mode and never invokes the model.
        c_seed = seed["c_src"]
        tech_env = load_tech_env(tech_env_dir, name)
        c_seed, fixed_ok = autofix_only(c_seed, name, tech_env, pic)
        if fixed_ok and c_seed != seed["c_src"]:
            log.info(f"[{name}] seed-only: deterministic autofix recovered compile.")
        res = match_mod.evaluate_mother(c_seed, name, target_insns, pic=pic)
        if not res["ok"]:
            write_skip(mother_dir, name, f"seed does not compile: {res['error_log'][:300]}", seed)
            return {"name": name, "status": "seed_no_compile"}
        status = _status(res, [a["exact"] for a in res["arms"]])
        d, man = write_outputs(mother_dir, name, c_seed, status,
                               res["whole_exact"], seed)
        log.info(f"[{name}] seed-only: {man['n_exact']}/{man['n_arms']} arms already exact -> {d}")
        if export_asm:
            written = export_arms_asm(mother_dir, target_insns, res["arms"])
            log.info(f"[{name}] ASM-Split: {len(written)} Arm-.s -> {mother_dir}")
            man["asm_split"] = [{"label": l, "n_insns": n} for (l, n) in written]
        return {"name": name, "status": "seed_only", "manifest": man}

    tech_env = load_tech_env(tech_env_dir, name)

    # SIMILAR sourcing (prompt injection + permuter seed), reused from the main
    # pipeline's cheatsheet module keyed by the mother name.
    similar_context, cheatsheet_context, similar_ref_code = sim_mod.context_for(
        name, similar_db or {}, cs_results, current_struct=0.0)
    if similar_ref_code:
        log.info(f"[{name}] similar reference available -> permuter seed + prompt injection.")
    elif similar_context or cheatsheet_context:
        log.info(f"[{name}] context: similar={bool(similar_context)} cheatsheet={bool(cheatsheet_context)}")

    c_src, ok = phase_a_compile(seed["c_src"], name, tech_env, backend, pic)
    if not ok:
        write_skip(mother_dir, name, "Phase A: mother never compiled", seed)
        return {"name": name, "status": "compile_failed"}

    c_src, status, whole_exact = phase_b_arms(
        c_src, name, ts_path, target_insns, tech_env, backend, max_iters, pic,
        similar_context=similar_context, cheatsheet_context=cheatsheet_context,
        similar_ref_code=similar_ref_code)
    d, man = write_outputs(mother_dir, name, c_src, status, whole_exact, seed)
    log.info(f"[{name}] DONE: {man['n_exact']}/{man['n_arms']} arms byte-exact, "
             f"whole_exact={whole_exact} -> {d}")
    return {"name": name, "status": "done", "manifest": man}


# ============================================================ batch
def find_mothers(target_dir, split_dir):
    """All .s under target_dir, excluding anything inside the split output dir."""
    td = Path(target_dir)
    sd = Path(split_dir).resolve()
    out = []
    for p in sorted(td.rglob("*.s")):
        if sd in p.resolve().parents:
            continue
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description="Mother-function decomposition wrapper")
    ap.add_argument("--target-asm", default=DEF_TARGET,
                    help="dir with mother .s files (default: %(default)s)")
    ap.add_argument("--split", default=DEF_SPLIT,
                    help="output ROOT; mirrors the target hierarchy, one folder "
                         "per mother at its relative position (default: %(default)s)")
    ap.add_argument("--tech-env", default=DEF_TECHENV)
    ap.add_argument("--backend", default="local",
                    help="AI backend (local/claude/none). 'none' => seed-only.")
    ap.add_argument("--seed-only", action="store_true",
                    help="deterministic: seed + compile + baseline arm slices, NO AI")
    ap.add_argument("--max-arm-iters", type=int, default=MAX_ARM_ITERS)
    ap.add_argument("--only", default="", help="process only this mother stem")
    ap.add_argument("--similar", action="store_true",
                    help="use the similar-ASM DB (prompt injection + permuter seed)")
    ap.add_argument("--cheatsheet", action="store_true",
                    help="also load cheatsheet patterns (implies similar DB)")
    ap.add_argument("--split-asm", action="store_true",
                    help="DETERMINISTIC split-all: export one .s per arm (byte-exact "
                         "target hex slice) for the similar-ASM DB. Implies seed-only.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.split, exist_ok=True)
    # --split-asm is the deterministic ASM-decomposition mode -> force seed-only.
    seed_only = args.seed_only or args.split_asm or args.backend == "none"
    export_asm = bool(args.split_asm)

    similar_db, cs_results = ({}, None)
    if not seed_only and (args.similar or args.cheatsheet):
        similar_db, cs_results = sim_mod.load_context_db(
            use_similar=args.similar, use_cheatsheet=args.cheatsheet)

    mothers = find_mothers(args.target_asm, args.split)
    if args.only:
        mothers = [m for m in mothers if m.stem == args.only]
    if not mothers:
        print(f"Keine Mutter-.s in {args.target_asm} (lege Dateien dort ab).")
        return

    print(f"{len(mothers)} Mutter-Funktion(en) gefunden. seed_only={seed_only} "
          f"similar={bool(similar_db)} cheatsheet={bool(cs_results)} "
          f"split_asm={export_asm}")
    results = []
    for ts in mothers:
        results.append(process_mother(str(ts), args.split, args.target_asm,
                                      args.tech_env, args.backend, args.max_arm_iters,
                                      seed_only, similar_db=similar_db,
                                      cs_results=cs_results, export_asm=export_asm))

    print("\n=== ZUSAMMENFASSUNG ===")
    total_arm_s = 0
    for r in results:
        m = r.get("manifest")
        extra = f" {m['n_exact']}/{m['n_arms']} exakt" if m else ""
        if m and m.get("asm_split"):
            n = len(m["asm_split"])
            total_arm_s += n
            extra += f" | {n} Arm-.s"
        print(f"  {r['name']:<24} {r['status']}{extra}")
    if export_asm:
        print(f"\n  {total_arm_s} Arm-.s insgesamt -> {args.split}/ "
              f"(gespiegelte Hierarchie, bereit fuer build_similar_asm_db)")


if __name__ == "__main__":
    main()
