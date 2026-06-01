# modules/mother_seed.py
#
# ARM-SEED step of the mother wrapper.  Given a mother's target .s we have no
# source C, so we recover a structured C draft with m2c (mips_to_c):
#
#     python3 modules/m2c/m2c.py -t mips-ido-c  <mother.s>
#
# The single decompiled C body serves BOTH roles the design needs:
#   * SKELETON  -- signature, frame, control flow (conditions/loops/braces)
#   * ARM DRAFTS-- the payload statements of each leaf arm, already inline
#
# We parse it with the same line-aware leaf-run splitter the validated
# test_group harness uses (mother_common.arms_from_c), so the arm boundaries are
# identical to the proven experiment.

import os
import re
import subprocess

from . import mother_common as MC

_PIPELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
M2C_DIR = os.path.join(os.path.dirname(__file__), "m2c")
M2C_PY  = os.path.join(M2C_DIR, "m2c.py")


def _sanitize_preamble(c_src, fname):
    """m2c emits unknown types as '?' (e.g. '? heap_free(s32);').  That is not
    valid C.  Replace '?' with a concrete default type, but ONLY in the
    declaration preamble (everything before the function definition) so we never
    touch a ternary '?' inside the body.  Default type 's32' compiles cleanly
    under -w; the AI loop refines real signatures later from the tech_env."""
    m = re.search(r'[A-Za-z_][\w ]*\b' + re.escape(fname) + r'\s*\([^;{]*\)\s*\{', c_src)
    cut = m.start() if m else len(c_src)
    preamble, body = c_src[:cut], c_src[cut:]
    preamble = re.sub(r'(^|\n)\s*\?(\s)', lambda mm: mm.group(1) + "s32" + mm.group(2),
                      preamble)
    preamble = preamble.replace("(?)", "(s32)").replace("?,", "s32,").replace(", ?", ", s32")
    return preamble + body


def run_m2c(target_s_path, fname=None, context_c=None, timeout=60):
    """Decompile a mother .s to C via m2c. Returns the C source string with
    referenced globals + callee prototypes declared (--globals used) and
    unknown '?' types sanitized.  Raises RuntimeError on failure."""
    cmd = ["python3", M2C_PY, "-t", "mips-ido-c", "--globals", "used"]
    if context_c and os.path.exists(context_c):
        cmd += ["--context", context_c]
    cmd += [os.path.abspath(target_s_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=M2C_DIR, timeout=timeout)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError("m2c failed:\n" + (proc.stderr or proc.stdout)[:2000])
    out = proc.stdout
    if fname:
        out = _sanitize_preamble(out, fname)
    return out


def seed_mother(target_s_path, context_c=None):
    """Full seed: parse target .s, run m2c, parse the resulting C into arms.

    Returns dict:
        name      : function symbol (func_XXXXXXXX)
        target    : parsed target stream (mother_common.parse_target_s)
        c_src     : m2c C source (the skeleton + inline arm drafts)
        fn        : find_function() result over c_src
        arms      : list of arms (mother_common.arms_from_c)
        loop_arms : count of loop/switch arms (watertight gate)
    """
    tgt = MC.parse_target_s(target_s_path)
    name = tgt["name"]
    if not name:
        raise RuntimeError(f"no glabel/function symbol in {target_s_path}")

    c_src = run_m2c(target_s_path, fname=name, context_c=context_c)

    # m2c may name the function exactly as the symbol; confirm it is present.
    if not re.search(r'\b' + re.escape(name) + r'\s*\(', c_src):
        # fall back: take the first defined function name m2c emitted
        m = re.search(r'\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*\{', c_src)
        if not m:
            raise RuntimeError("m2c produced no function definition")
        name = m.group(1)

    fn, arms = MC.arms_from_c(c_src, name)
    loop_arms = sum(1 for a in arms if a["kind"] == "loop")
    return {
        "name": name,
        "target": tgt,
        "c_src": c_src,
        "fn": fn,
        "arms": arms,
        "loop_arms": loop_arms,
    }


def is_watertight(seed):
    """Watertight gate (user decision: 'Nur wasserdichte Faelle').
    A mother is watertight when every arm is a straight-line leaf run (no
    loop/switch arms) -- the nested-if class proven to give 1 contiguous block
    per arm.  Loop mothers (fragmented arms / multi-exit) are flagged + skipped
    until the loop tooling is hardened."""
    return seed["loop_arms"] == 0 and len(seed["arms"]) > 0
