# modules/mother_common.py
#
# Shared primitives for the MOTHER-FUNCTION decomposition wrapper (second
# pipeline).  Three concerns live here:
#
#   1. PARSE a spimdisasm target .s  ->  the byte-exact ground-truth instruction
#      stream of a mother function (vaddr, hex word, mnemonic, operands).
#
#   2. C PARSE (line-aware) that splits a mother C body into leaf "arms" exactly
#      the way the proven test_group harness does (loops/switch kept whole,
#      if/else descended).  Ported verbatim from
#      PreProcess/splitter/test_group/run_locator_cut.py so behaviour is
#      identical to the validated experiment.
#
#   3. ALIGNMENT: a position/allocation-independent canonical key + a
#      SequenceMatcher-based aligner, used to line our freshly compiled mother up
#      against the target stream (front-to-back matching).
#
# NOTHING in here touches the existing pipeline modules.

import os
import re

# --- toolchain (pipeline-local) ------------------------------------------------
_PIPELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IDO_DIR = os.path.join(_PIPELINE_ROOT, "IDO_compiler")
IDO_CC  = os.path.join(IDO_DIR, "cc")
OBJDUMP = "mips-linux-gnu-objdump"

# ==============================================================================
# 1) TARGET .s PARSE  (spimdisasm format)
# ==============================================================================
# Lines look like:
#   glabel func_800B5054
#     /* 1E8E944 800B5054 27BDFFD8 */  addiu      $sp, $sp, -0x28
#   .L800B5090:
# We keep, per instruction: file-offset, vaddr, hex word, mnemonic, operands and
# any label that precedes it.  The hex word IS the byte-exact ground truth.
# spimdisasm comment is either three fields (ROM offset, VRAM vaddr, word) or
# two (vaddr, word).  The instruction WORD is always the LAST 8-hex token before
# '*/'.  Capture trailing (vaddr, word); ROM offset (if present) is ignored.
_TGT_INSN_RE = re.compile(
    r'/\*\s*(?:[0-9A-Fa-f]+\s+)?([0-9A-Fa-f]{6,8})\s+([0-9A-Fa-f]{8})\s*\*/\s*(\S+)\s*(.*)$')
_TGT_LABEL_RE = re.compile(r'^\s*([.\w]+):\s*$')
_GLABEL_RE    = re.compile(r'^\s*glabel\s+(\w+)')


def parse_target_s(path):
    """Parse a spimdisasm .s file. Returns dict:
        {name, insns:[{idx,vaddr,hex,mnem,ops,label}]}
    `hex` is the lowercase 8-char instruction word (byte-exact)."""
    name = None
    insns = []
    pending_label = None
    with open(path, "r", errors="replace") as f:
        for raw in f:
            mg = _GLABEL_RE.match(raw)
            if mg:
                if name is None:
                    name = mg.group(1)
                continue
            mi = _TGT_INSN_RE.search(raw)
            if mi:
                ops = re.split(r'[;#]', mi.group(4))[0].strip()
                insns.append({
                    "idx":   len(insns),
                    "vaddr": mi.group(1).lower(),
                    "hex":   mi.group(2).lower(),
                    "mnem":  mi.group(3),
                    "ops":   ops,
                    "label": pending_label,
                })
                pending_label = None
                continue
            ml = _TGT_LABEL_RE.match(raw)
            if ml:
                pending_label = ml.group(1)
    return {"name": name, "insns": insns}


# ==============================================================================
# 2) C PARSE (line-aware) -- ported from run_locator_cut.py
# ==============================================================================
TYPE_RE = re.compile(
    r'^(?:unsigned\s+|signed\s+)*'
    r'(?:void|char|short|int|long|float|double|u8|s8|u16|s16|u32|s32|u64|s64|f32|f64)\b')


def lineno(src, pos):
    return src.count("\n", 0, pos) + 1


def find_function(src, fname):
    m = re.search(r'(^|\n)([^\n;{}]*\b' + re.escape(fname) + r'\s*\([^;{]*\)\s*)\{', src)
    if not m:
        raise RuntimeError(f"function {fname} not found")
    sig_start = m.start(2)
    brace_open = src.index('{', m.end(2) - 1)
    depth = 0
    i = brace_open
    while i < len(src):
        c = src[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                break
        i += 1
    return {
        "header": src[:sig_start],
        "signature": src[sig_start:brace_open].strip(),
        "body": src[brace_open + 1:i],
        "body_pos": brace_open + 1,
        "body_end": i,           # abs offset of the closing '}'
        "src": src,
    }


def split_top_statements_lined(text, base_off, src):
    """Yield (stmt_text, start_line, end_line) for brace-depth-0 statements."""
    out = []
    depth = 0
    paren = 0
    buf = []
    buf_start = 0
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if not buf:
            buf_start = i
        buf.append(c)
        if c == '(':
            paren += 1
        elif c == ')':
            paren -= 1
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and paren == 0:
                rest = text[i + 1:]
                if not re.match(r'\s*else\b', rest):
                    s = "".join(buf).strip()
                    if s:
                        sl = lineno(src, base_off + buf_start)
                        el = lineno(src, base_off + i)
                        out.append((s, sl, el))
                    buf = []
        elif c == ';' and depth == 0 and paren == 0:
            s = "".join(buf).strip()
            if s:
                sl = lineno(src, base_off + buf_start)
                el = lineno(src, base_off + i)
                out.append((s, sl, el))
            buf = []
        i += 1
    if "".join(buf).strip():
        s = "".join(buf).strip()
        sl = lineno(src, base_off + buf_start)
        el = lineno(src, base_off + (n - 1))
        out.append((s, sl, el))
    return out


def is_decl(stmt):
    s = stmt.rstrip(';').strip()
    if '{' in stmt or '}' in stmt:
        return False
    if not TYPE_RE.match(s):
        return False
    return '(' not in s


def inner_blocks_off(stmt, stmt_off):
    bodies = []
    i = 0
    n = len(stmt)
    while i < n:
        if stmt[i] == '{':
            depth = 0
            j = i
            while j < n:
                if stmt[j] == '{':
                    depth += 1
                elif stmt[j] == '}':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            bodies.append((stmt[i + 1:j], stmt_off + i + 1))
            i = j + 1
        else:
            i += 1
    return bodies


def collect_leaf_runs(text, base_off, src):
    """Return list of runs; each run = list of (stmt, sl, el).
    Loops/switch kept whole (one run); if/else descended; decls skipped."""
    runs = []
    cur = []
    for (st, sl, el) in split_top_statements_lined(text, base_off, src):
        if '{' in st:
            if cur:
                runs.append(cur)
                cur = []
            kw = re.match(r'\s*(\w+)', st)
            kw = kw.group(1) if kw else ''
            if kw in ('for', 'while', 'do', 'switch'):
                runs.append([(st, sl, el)])
            else:
                stmt_off = src.index(st, base_off) if st in src[base_off:] else base_off
                for (b, boff) in inner_blocks_off(st, stmt_off):
                    runs.extend(collect_leaf_runs(b, boff, src))
        else:
            if is_decl(st):
                continue
            cur.append((st, sl, el))
    if cur:
        runs.append(cur)
    return runs


def arms_from_c(src, fname):
    """High-level: parse src, return ordered arms.
    Each arm: {id, lines:set, lrange:(lo,hi), stmts:[str], head, kind}."""
    fn = find_function(src, fname)
    runs = collect_leaf_runs(fn["body"], fn["body_pos"], src)
    arms = []
    for ai, run in enumerate(runs):
        lines = set()
        for (_st, sl, el) in run:
            lines.update(range(sl, el + 1))
        head = re.sub(r'\s+', ' ', run[0][0])[:60]
        kw = re.match(r'\s*(\w+)', run[0][0])
        kw = kw.group(1) if kw else ''
        kind = "loop" if kw in ('for', 'while', 'do', 'switch') else "straight"
        arms.append({
            "id": ai,
            "lines": lines,
            "lrange": (min(lines), max(lines)),
            "stmts": [s for (s, _a, _b) in run],
            "head": head,
            "kind": kind,
        })
    return fn, arms


# ==============================================================================
# 3) ALIGNMENT  (position/allocation-independent canonical key)
# ==============================================================================
# Scratch / ABI-volatile registers the global allocator is free to renumber, and
# PC-relative branch targets that move with code position.  Two instructions
# with the same canon differ at most by the temp-allocator's choice or position.
TEMP_RE = re.compile(r'\b(t[0-9]|v[01]|a[0-3]|at|f[12]?[0-9])\b')
TGT_RE  = re.compile(r'\b[0-9a-f]+ <[^>]+>')
_HEXOFF_RE = re.compile(r'\b0x[0-9a-fA-F]+\b')


def canon(mnem, ops):
    """Position- and allocation-independent instruction key: blank scratch
    registers and branch targets, keep opcode / stack offset / s-register /
    immediate."""
    o = TGT_RE.sub("@", ops)
    o = TEMP_RE.sub("$", o)
    return mnem + " " + o


def canon_target(mnem, ops):
    """Canon for a target (.s) instruction. Target operands render branch
    targets as labels (.L800B5090) and use the same register names, so we only
    blank labels + scratch regs."""
    o = re.sub(r'\.L[0-9A-Fa-f]+', '@', ops)
    o = re.sub(r'\b(func_[0-9A-Fa-f]+|[A-Za-z_]\w+)\b',
               lambda m: m.group(0), o)   # keep symbol names
    o = TEMP_RE.sub("$", o)
    return mnem + " " + o
