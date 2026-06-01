# modules/mother_match.py
#
# COMPILE-IN-MOTHER  match backend (the user's decided model).
#
# The register-allocation barrier only bites when an arm is compiled STANDALONE.
# It does NOT bite when the arm's C lives inside the whole mother and the mother
# is compiled as one unit (proven: 82/82 arms byte-identical in test_group).
#
# So per-arm matching here is:
#     take the current mother C (skeleton + all arm drafts)
#       -> substitute the candidate C for ONE arm
#       -> compile the WHOLE mother  (PIC, -g3, keep #line)
#       -> objdump -dl  => instruction stream with a source line per instr
#       -> the line map labels each instruction with the arm that owns its line
#       -> ALIGN our stream to the target .s stream (front-to-back, on hex words)
#       -> the arm's target region = aligned slice
#       -> byte-compare our arm region vs the target slice.
#
# Reassembly is guaranteed: once every arm region is byte-exact, the whole
# compiled mother equals the target by construction.
#
# Compilation reuses the existing pipeline's header DB + include dirs READ-ONLY
# (we never modify ido_compiler).  Two deliberate differences vs the existing
# compile_code:
#   * PIC: no -non_shared            (the reference / locator path is PIC)
#   * keep #line: -g3 and NO -P      (so objdump -dl maps instr -> source line)
# Header decls are injected via -include (a side file) so they do NOT shift the
# mother's own line numbers; arms are re-parsed from the exact compiled text.

import os
import re
import shutil
import signal
import tempfile
import subprocess
from difflib import SequenceMatcher

import json

from . import mother_common as MC
from . import ido_compiler as IC      # READ-ONLY: header DB + include dirs
from . import diff_generator as DG    # READ-ONLY: instruction normalization


# ----------------------------------------------------------------- compile
def _decls_for(func_name):
    """Header/extern block for the mother (reuses the pipeline header DB)."""
    IC._load_header_db()
    headers = IC._header_db.get(func_name, []) if IC._header_db else []
    return "\n".join(headers)


def _run(cmd, cwd=None, env=None, timeout=20):
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                cwd=cwd, env=env, preexec_fn=os.setsid)
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return -1, "", "Timeout"
    except Exception as e:
        return -1, "", str(e)


def detect_pic(target_insns):
    """Decide whether the TARGET was built PIC.  PIC uses the GOT (%got/%call16)
    and gp-relative loads; non-PIC uses absolute lui %hi / %lo.  Banjo-Tooie ROM
    .s files are non-PIC (absolute addressing); the test_group reference was PIC.
    Returns True for PIC, False for non-PIC.  Empty/ambiguous -> False (non-PIC,
    matching the existing pipeline's -non_shared)."""
    got = abso = 0
    for ins in target_insns:
        o = ins.get("ops", "")
        if "%got" in o or "%call16" in o or "%gp_rel" in o:
            got += 1
        if "%hi" in o or "%lo" in o:
            abso += 1
    return got > abso


def compile_mother(c_src, func_name, extra_decls="", pic=False):
    """Compile a whole mother keeping #line, matching the TARGET's PIC mode.
    pic=False adds -non_shared (absolute addressing, like the ROM and the
    existing pipeline); pic=True omits it (GOT/gp-relative, the test_group
    reference recipe).  Returns dict {success, temp_dir, o_path, error_log}."""
    temp_dir = tempfile.mkdtemp(prefix=f"mom_{func_name}_")
    c_file = os.path.join(temp_dir, f"{func_name}.c")
    h_file = os.path.join(temp_dir, "_decls.h")
    i_file = os.path.join(temp_dir, f"{func_name}.i")
    o_file = os.path.join(temp_dir, f"{func_name}.o")

    decls = _decls_for(func_name)
    if extra_decls:
        decls = decls + "\n" + extra_decls
    decls = IC._normalize_includes(decls)
    with open(h_file, "w", encoding="utf-8") as f:
        f.write(decls + "\n")
    with open(c_file, "w", encoding="utf-8") as f:
        f.write(c_src)

    # GCC preprocess: keep #line markers (NO -P) so objdump -dl maps lines.
    gcc = ["gcc", "-E", "-xc", "-D_LANGUAGE_C", "-D_MIPS_SZLONG=32",
           "-D__attribute__(x)=", "-D__extension__=", "-include", h_file]
    for inc in IC.INCLUDE_DIRS:
        gcc += ["-I", inc]
    gcc += [c_file, "-o", i_file]
    rc, _o, err = _run(gcc, timeout=10)
    if rc != 0:
        log = IC._clean_error_log(err, temp_dir, "GCC Preprocessor Error")
        cleanup(temp_dir)
        return {"success": False, "temp_dir": "", "o_path": "", "error_log": log}

    env = os.environ.copy()
    env["COMPILER_PATH"] = MC.IDO_DIR
    env["LD_LIBRARY_PATH"] = f"{MC.IDO_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
    # -g3 retains line info under -O2.  -non_shared = non-PIC (absolute), unless
    # the target is PIC.  This MUST match how the target was built or globals
    # diverge (gp-relative GOT load vs absolute lui %hi/%lo).
    ido = [MC.IDO_CC, "-c", "-O2", "-mips2", "-G", "0", "-w", "-g3"]
    if not pic:
        ido.append("-non_shared")
    ido += [i_file, "-o", o_file]
    rc, out, err = _run(ido, cwd=temp_dir, env=env, timeout=20)
    if rc == 0 and os.path.exists(o_file):
        return {"success": True, "temp_dir": temp_dir, "o_path": o_file, "error_log": ""}
    raw = err or out
    log = IC._clean_error_log(raw, temp_dir, "IDO Compiler Error")
    cleanup(temp_dir)
    return {"success": False, "temp_dir": "", "o_path": "", "error_log": log}


def cleanup(temp_dir):
    if temp_dir and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


# ----------------------------------------------------------------- objdump -dl
def objdump_lined(o_path, c_basename, func_name):
    """Instructions of `func_name` only, each with its source line in
    <c_basename> (the mother .c) or None.  Ported from run_locator_cut."""
    rc, out, err = _run([MC.OBJDUMP, "-dl", "-z", o_path], timeout=20)
    insns = []
    cur_line = None
    cur_in = False
    in_target = False
    base_c = c_basename
    for ln in out.splitlines():
        ms = re.match(r'^[0-9a-f]+\s+<([^>]+)>:\s*$', ln.strip())
        if ms:
            in_target = (ms.group(1) == func_name)
            continue
        m = re.match(r'^(\S+\.[ch]):(\d+)\s*$', ln.strip())
        if m:
            cur_in = (os.path.basename(m.group(1)) == base_c)
            cur_line = int(m.group(2)) if cur_in else None
            continue
        mi = re.match(r'^\s*([0-9a-f]+):\s+([0-9a-f]{8})\s+(\S+)\s*(.*)$', ln)
        if mi and in_target:
            ops = re.split(r'[;#]', mi.group(4))[0].strip()
            insns.append({
                "idx": len(insns),
                "vaddr": mi.group(1),
                "hex": mi.group(2).lower(),
                "mnem": mi.group(3),
                "ops": ops,
                "line": cur_line if cur_in else None,
            })
    return insns


# ----------------------------------------------------------------- partition our stream into arms
def label_arms(insns, c_src, func_name):
    """Re-parse arms from the EXACT compiled text and label each instruction
    with the arm id owning its source line (or 'skel').  Returns (arms, labels)
    where arms[i] gains 'idxs' (instruction indices in our stream)."""
    _fn, arms = MC.arms_from_c(c_src, func_name)
    line2arm = {}
    for a in arms:
        for ln in a["lines"]:
            line2arm.setdefault(ln, a["id"])
    labels = [line2arm.get(ins["line"], "skel") if ins["line"] is not None else "skel"
              for ins in insns]
    for a in arms:
        a["idxs"] = [i for i, lab in enumerate(labels) if lab == a["id"]]
    return arms, labels


# ----------------------------------------------------------------- align our stream <-> target
def align_to_target(our_insns, target_insns):
    """SequenceMatcher on raw hex words (renderer-independent, byte-exact for
    locked regions).  Returns the list of opcodes and a mapper.

    map_range(i0, i1): given a half-open OUR index range, return the
    corresponding half-open TARGET index range, snapped to the surrounding
    matched anchors (the skeleton instructions bounding the arm)."""
    a = [x["hex"] for x in our_insns]
    b = [x["hex"] for x in target_insns]
    sm = SequenceMatcher(a=a, b=b, autojunk=False)
    blocks = sm.get_matching_blocks()  # [(i, j, n), ... , (len_a, len_b, 0)]

    def map_lower(i):
        """Target index where the arm region STARTS: end of the last match block
        lying before i (or exact map if i is inside a match)."""
        res = 0
        for (ai, bj, n) in blocks:
            if n == 0:
                continue
            if ai <= i < ai + n:
                return bj + (i - ai)
            if ai + n <= i:
                res = bj + n
        return res

    def map_upper(i):
        """Target index where the arm region ENDS (exclusive): start of the
        first match block at/after i (or exact map if i is inside a match)."""
        for (ai, bj, n) in blocks:
            if n == 0:
                continue
            if ai <= i < ai + n:
                return bj + (i - ai)
            if ai >= i:
                return bj
        return len(b)

    def map_range(i0, i1):
        t0 = map_lower(i0)
        t1 = map_upper(i1)
        if t1 < t0:
            t1 = t0
        return t0, t1

    return sm.get_opcodes(), map_range


# ----------------------------------------------------------------- per-arm comparison
def compare_arm(our_insns, arm, target_insns, map_range):
    """Compare one arm's OUR instructions against its aligned TARGET slice.
    Returns dict: {exact, n_our, n_tgt, equal, struct, our_hex, tgt_hex,
                   tgt_range}."""
    idxs = arm.get("idxs", [])
    if not idxs:
        return {"exact": False, "n_our": 0, "n_tgt": 0, "equal": 0,
                "struct": 0.0, "our_hex": [], "tgt_hex": [], "tgt_range": (0, 0),
                "our_disasm": [], "tgt_disasm": []}
    i0, i1 = idxs[0], idxs[-1] + 1
    t0, t1 = map_range(i0, i1)
    our_hex = [our_insns[i]["hex"] for i in idxs]
    tgt_hex = [target_insns[t]["hex"] for t in range(t0, t1)]
    our_dis = [(our_insns[i]["hex"], our_insns[i]["mnem"], our_insns[i]["ops"]) for i in idxs]
    tgt_dis = [(target_insns[t]["hex"], target_insns[t]["mnem"], target_insns[t]["ops"])
               for t in range(t0, t1)]

    exact = (our_hex == tgt_hex)
    # struct score: fraction of our instrs that byte-match in aligned position
    sm = SequenceMatcher(a=our_hex, b=tgt_hex, autojunk=False)
    equal = sum(n for (_t, _i, _j, n) in
                [(o[0], o[1], o[3], (o[2] - o[1]))
                 for o in sm.get_opcodes() if o[0] == "equal"])
    denom = max(len(our_hex), len(tgt_hex), 1)
    struct = equal / denom
    return {
        "exact": exact, "n_our": len(our_hex), "n_tgt": len(tgt_hex),
        "equal": equal, "struct": struct,
        "our_hex": our_hex, "tgt_hex": tgt_hex, "tgt_range": (t0, t1),
        "our_disasm": [{"hex": h, "mnem": m, "ops": o} for (h, m, o) in our_dis],
        "tgt_disasm": [{"hex": h, "mnem": m, "ops": o} for (h, m, o) in tgt_dis],
    }


def evaluate_mother(c_src, func_name, target_insns, extra_decls="", pic=None):
    """Compile the whole mother and score EVERY arm against the target.
    pic: True/False forces the mode; None auto-detects from target_insns
    (default).  Returns dict:
        {ok, error_log, arms:[...], whole_exact, our_insns}
    ok=False with error_log when compilation fails."""
    if pic is None:
        pic = detect_pic(target_insns)
    comp = compile_mother(c_src, func_name, extra_decls=extra_decls, pic=pic)
    if not comp["success"]:
        return {"ok": False, "error_log": comp["error_log"]}
    try:
        our = objdump_lined(comp["o_path"], f"{func_name}.c", func_name)
        arms, _labels = label_arms(our, c_src, func_name)
        _ops, map_range = align_to_target(our, target_insns)
        whole_exact = ([x["hex"] for x in our] == [x["hex"] for x in target_insns])
        results = []
        for a in arms:
            r = compare_arm(our, a, target_insns, map_range)
            r.update({"id": a["id"], "head": a["head"], "kind": a["kind"],
                      "stmts": a["stmts"], "lrange": a["lrange"]})
            results.append(r)
    finally:
        cleanup(comp["temp_dir"])
    return {"ok": True, "error_log": "", "arms": results,
            "whole_exact": whole_exact, "our_insns": our}


# ----------------------------------------------------------------- per-arm diff JSON (for the AI)
def arm_diff_json(arm_result, focus="all", max_entries=8):
    """Build the SAME JSON-diff shape that diff_generator.create_json_diff
    produces, but for ONE arm's in-memory region (our vs target hex+disasm).
    Reuses diff_generator's normalization so the AI sees a familiar format."""
    def norm(dis):
        out = []
        for d in dis:
            ins = DG._normalize_instruction(d["mnem"], d["ops"],
                                            f'{d["mnem"]} {d["ops"]}', hex_code=d["hex"])
            if ins:
                out.append(ins)
        return out

    our = norm(arm_result["our_disasm"]) if arm_result.get("our_disasm") is not None else None
    tgt = norm(arm_result["tgt_disasm"]) if arm_result.get("tgt_disasm") is not None else None
    if not our and not tgt:
        # fall back to hex-only entries
        d = [{"type": "Summary", "focus": focus,
              "target_count": arm_result["n_tgt"], "draft_count": arm_result["n_our"],
              "diff": arm_result["n_our"] - arm_result["n_tgt"]}]
        return json.dumps(d, indent=2)

    summary = {"type": "Summary", "focus": focus,
               "target_count": len(tgt), "draft_count": len(our),
               "diff": len(our) - len(tgt)}
    t_ops = [i["opcode"] for i in tgt]
    d_ops = [i["opcode"] for i in our]
    sm = SequenceMatcher(None, t_ops, d_ops)
    entries = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                ti, di = tgt[i1 + k], our[j1 + k]
                if ti["hex"] and di["hex"] and DG._is_reloc_only_diff(ti["hex"], di["hex"]):
                    continue
                if ti["operands"] != di["operands"]:
                    entries.append(DG._analyze_operand_mismatch(ti, di, i1 + k))
        elif tag == "replace":
            entries.append({"type": "Instruction Mismatch",
                            "pos": f"target[{i1}:{i2}] vs draft[{j1}:{j2}]",
                            "target": [i["raw"] for i in tgt[i1:i2]],
                            "draft": [i["raw"] for i in our[j1:j2]]})
        elif tag == "delete":
            entries.append({"type": "Missing in Draft", "pos": f"target[{i1}:{i2}]",
                            "missing": [i["raw"] for i in tgt[i1:i2]]})
        elif tag == "insert":
            entries.append({"type": "Extra in Draft", "pos": f"draft[{j1}:{j2}]",
                            "extra": [i["raw"] for i in our[j1:j2]]})
    log = [summary] + entries[:max_entries]
    if len(entries) > max_entries:
        log.append({"type": "Note", "remaining": len(entries) - max_entries})
    return json.dumps(log, indent=2)


# ----------------------------------------------------------------- source surgery
def replace_arm_stmts(c_src, c_stmts, candidate):
    """Replace an arm's exact STATEMENT span (first stmt start .. last stmt end,
    located by text) with `candidate`.  Enclosing braces untouched.  Newlines
    are preserved so downstream arms keep their line numbers.  Ported from
    run_match_model.replace_arm_stmts."""
    first, last = c_stmts[0], c_stmts[-1]
    i0 = c_src.index(first)
    i1 = c_src.index(last, i0) + len(last)
    removed_nl = c_src.count("\n", i0, i1)
    return c_src[:i0] + candidate + "\n" * removed_nl + c_src[i1:]
