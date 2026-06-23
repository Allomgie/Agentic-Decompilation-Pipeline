"""Garbage-Experte (universeller Stall-Fallback). Greift NUR, wenn kein anderer Experte mehr zuendet, die
Funktion aber noch nicht am Ziel ist (mm>0). Diff-unabhaengig, anker-angereichert.

Philosophie (Lukas): kein "besserer Metrik-Wert" als Erfolgsmaß, sondern SACKGASSE VERLASSEN in einen Zustand,
den die deterministischen Experten loesen koennen -- auch wenn der Garbage-Output selbst kurz schlechter ist
(Talfahrt). Drei Wege (min/rw/asm) -> Frankenstein-Hunk-Mix -> Escape-Gate (deterministische Experten ueber
den resolve-Callback). Rueckgabe: ein NEUER Zustand (besser als Eingang via Downstream) oder None.

Produktion = ONESHOTS (ai-Callback, kein Thinking). Reasoning ist nur Analyse-Werkzeug (siehe analysis/).

Spaeter ggf. um weitere Mismatch-Klassen erweitern (Fail-Taxonomie im Memory garbage-expert-plan)."""
import os, re, struct, difflib, itertools
from collections import Counter

_VA = re.compile(r"/\*\s*[0-9A-Fa-f]+\s+([0-9A-Fa-f]{8})\s")
_FREE_MASK = re.compile(r"\b(t\d|s[0-8]|f\d+)\b")       # frei-allozierbar (Permuter); a/v/at/ra/sp/zero = semantisch
_SEM = re.compile(r"\b(a[0-3]|v[01])\b")
_LDST = ("lw", "sw", "lh", "sh", "lb", "sb", "lhu", "lbu", "lwc1", "swc1", "ldc1", "sdc1")
_IMM = ("addiu", "addi", "ori", "andi", "xori", "slti", "sltiu", "li")
_DETAIL = ("Memory Access", "Register/Immediate", "Address Load")
_INT_LIT = re.compile(r"-?0x[0-9A-Fa-f]+[uUlL]*|-?\d+[uUlL]*")


# ------------------------------------------------------------------ Target-ASM / Anker
def _target_insns(sp):
    out = []
    for ln in open(sp, encoding="utf-8", errors="replace"):
        st = re.sub(r"/\*.*?\*/", "", ln).strip()
        if _VA.search(ln) and not (st.startswith(".") and st.endswith(":")):
            out.append(st)
    return out


def _landmark(insns, pos, rng=3):
    best = None
    for d in range(0, rng + 1):
        for j in (pos - d, pos + d):
            if not (0 <= j < len(insns)):
                continue
            t = insns[j]
            m = re.search(r"\bjal\s+([A-Za-z_]\w*)", t)
            if m: return f"call to {m.group(1)}"
            m = re.search(r"%[hl][io]\(([^)]+)\)", t)
            if m: return f"global {m.group(1)}"
            m = re.search(r"\b(?:li|addiu|ori|lui)\b[^,]*,\s*(?:[^,]+,\s*)?(0x[0-9A-Fa-f]{3,}|\d{4,})", t)
            if m and best is None: best = f"constant {m.group(1)}"
    return best or "(no nearby landmark)"


# ------------------------------------------------------------------ Spot-Filter + Klassifikation
def _spot_ai_fixable(e):
    tgt = _FREE_MASK.sub("<r>", str(e.get("target", "")))
    drf = _FREE_MASK.sub("<r>", str(e.get("draft", "")))
    return tgt != drf


def _keep_spot(e):
    if e.get("type") not in _DETAIL or not _spot_ai_fixable(e):
        return False
    blob = str(e.get("target", "")) + str(e.get("draft", ""))
    if "jtbl" in blob or ".rodata" in blob:        # Jump-Table/Switch = strukturell, nicht operand-lokal
        return False
    return True


def _pa(x):
    m = re.match(r"\s*([a-z0-9]+)\s*,\s*(-?\d+)\(([^)]+)\)", str(x))
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def _difftoks(tgt, drf):
    pat = r"[A-Za-z_]\w*|-?0x[0-9A-Fa-f]+|-?\d+"
    tt, dd = re.findall(pat, str(tgt)), re.findall(pat, str(drf))
    return [(tt[i] if i < len(tt) else None, dd[i] if i < len(dd) else None)
            for i in range(max(len(tt), len(dd))) if (tt[i] if i < len(tt) else None) != (dd[i] if i < len(dd) else None)]


def _fmtf(f):
    s = f"{f:.7g}"
    if not any(c in s for c in ".eEnf"):
        s += ".0"
    return s + "f"


def _const32_float(insns, ci):
    if insns is None or ci is None:
        return None
    for j in (ci, ci + 1, ci - 1):
        if 0 <= j < len(insns):
            m = re.search(r"0x([0-9A-Fa-f]{8})", insns[j])
            if m:
                v = int(m.group(1), 16)
                try:
                    return v, struct.unpack(">f", struct.pack(">I", v))[0]
                except Exception:
                    return None
    return None


def _win_floats(insns, ci, rng=6):
    if insns is None or ci is None:
        return []
    seen = []
    for j in range(max(0, ci - rng), min(len(insns), ci + rng + 1)):
        for m in re.finditer(r"0x([0-9A-Fa-f]{8})", insns[j]):
            try:
                f = struct.unpack(">f", struct.pack(">I", int(m.group(1), 16)))[0]
            except Exception:
                continue
            if -1e9 < f < 1e9 and f not in seen:
                seen.append(f)
    return seen


def _spot_hint(e, insns=None, ci=None):
    op = str(e.get("op", "")); tgt = str(e.get("target", "")); drf = str(e.get("draft", ""))
    if op == "lui" and re.fullmatch(r"[a-z0-9]+,\s*-?\d+", tgt.strip()):
        cf = _const32_float(insns, ci)
        if cf:
            v, f = cf
            nearby = _win_floats(insns, ci)
            setnote = ""
            if len(nearby) > 1:
                setnote = (f" The TARGET float constants near here are {{{', '.join(_fmtf(x) for x in nearby)}}}; "
                           f"this spot is {_fmtf(f)} -- change only the draft literal that is not in that set.")
            return (f"HINT (float-const): the TARGET 32-bit constant is 0x{v:08X} (= {_fmtf(f)}). Match it but "
                    f"KEEP the draft's literal STYLE: if the draft uses a raw hex bit-pattern (0x........), write "
                    f"the hex 0x{v:08X} -- writing a decimal float like {_fmtf(f)} would move it to rodata and "
                    f"change the instructions." + setnote)
        return ("HINT (float-const): read the full 0x........ constant from the TARGET window and write it at "
                "this spot in the SAME literal style the draft uses (hex bit-pattern stays a hex bit-pattern).")
    if op in _LDST:
        tp, dp = _pa(tgt), _pa(drf)
        if tp and dp and tp[2] == dp[2] and tp[1] != dp[1]:
            return (f"HINT (offset): same base, only the OFFSET is wrong (TARGET byte offset {tp[1]}). "
                    f"Do NOT add or remove any dereference/load -- keep the exact pointer structure. "
                    f"If a typed base scales the offset, wrap only the base: ((s8*)base + {tp[1]}). "
                    f"If the base is an INDEXED pointer like arr[i] / &arr[i] (the asm computes it via "
                    f"sll+addu before this load), the byte offset is ADDED ON TOP of the index -- write "
                    f"*(T*)((s8*)(&arr[i]) + {tp[1]}); the index scaling must NOT absorb {tp[1]}.")
        if tp and dp and tp[2] != dp[2]:
            return ("HINT (base): the BASE register differs -> wrong pointer/variable is dereferenced. "
                    "Change which variable/field is read; keep the same number of dereferences.")
    dpairs = _difftoks(tgt, drf)
    sem = [x for a, b in dpairs for x in (a, b) if x and _SEM.fullmatch(x)]
    if sem:
        win = insns and ci is not None
        jal_next = win and any("jal" in insns[j] for j in range(ci, min(len(insns), ci + 3)))
        jr_near = win and any(re.search(r"\bjr\b", insns[j]) for j in range(max(0, ci - 1), min(len(insns), ci + 2)))
        if any(re.fullmatch(r"a[0-3]", x) for x in sem) and jal_next:
            return ("HINT (sem-arg): an ARGUMENT register (a0-a3) before a call differs -> reorder the "
                    "call's arguments / change which value is passed into that slot. NOT just a rename.")
        if any(re.fullmatch(r"v[01]", x) for x in sem) and jr_near:
            return ("HINT (sem-return): the RETURN register (v0/v1) differs -> make the returned "
                    "expression the one that lands here; do not rename a temp.")
        return None        # Regalloc -> Permuter, nicht KI
    if any(re.fullmatch(r"-?(0x[0-9a-fA-F]+|\d+)", a or "") for a, _b in dpairs):
        return ("HINT (int-immediate): match the TARGET immediate exactly. Note 'x > C' compiles as "
                "'x >= C+1' for constant C -> an off-by-one often means flipping >/>= or the constant.")
    return "HINT (general): adjust the C so this spot compiles to the TARGET operands; keep the rest."


def _pos_int(pos):
    if isinstance(pos, int):
        return pos
    m = re.search(r"target\[(\d+):", str(pos))
    return int(m.group(1)) if m else None


def _kept_spots(insns, entries):
    out = []
    for e in entries:
        if not _keep_spot(e):
            continue
        ci = _pos_int(e.get("pos"))
        hint = _spot_hint(e, insns, ci)
        if hint is None:
            continue
        out.append((e, ci, hint))
    freg = {m.group(1) for e, _c, h in out if h.startswith("HINT (float-const)")
            for m in [re.match(r"([a-z0-9]+),", str(e.get("target")))] if m}
    res = []
    for e, ci, h in out:
        if str(e.get("op")) == "ori":
            m = re.match(r"([a-z0-9]+),([a-z0-9]+),", str(e.get("target")))
            if m and m.group(1) == m.group(2) and m.group(1) in freg:
                continue
        res.append((e, ci, h))
    return res


def residual_class(sp, ents):
    """Bottleneck-Klassifikation des Residuums (warum kein Experte/welche Klasse blockiert) -- spiegelt die
    _why_stuck-Taxonomie, importierbar fuers Live-Log. Prioritaet: strukturell zuerst, dann garbage-fixbar,
    dann Permuter/IM/sonst."""
    if not ents:
        return "no-diff"
    types = Counter(e.get("type") for e in ents)
    if "Stack Frame Mismatch" in types:
        return "stack-frame"
    md = sum(len(e.get("target", [])) if isinstance(e.get("target"), list) else 1
             for e in ents if e.get("type") == "Missing in Draft")
    if md:
        return f"missing-block(md={md})"
    blob = "".join(str(e.get("target", "")) + str(e.get("draft", "")) for e in ents)
    if "jtbl" in blob or ".rodata" in blob:
        return "jump-table"
    try:
        ks = _kept_spots(_target_insns(sp), ents)
    except Exception:
        ks = []
    if ks:
        cls = Counter(m.group(1) for _e, _c, h in ks for m in [re.match(r"HINT \(([a-z-]+)\)", h)] if m)
        return "ai-fixable:" + (cls.most_common(1)[0][0] if cls else "?")
    if "Instruction Mismatch" in types:
        return "instr-mismatch"
    detail = [e for e in ents if e.get("type") in _DETAIL]
    if detail and all(not _spot_ai_fixable(e) for e in detail):
        return "register-alloc/permuter"
    if "Reordered" in types:
        return "reordered"
    if "Extra in Draft" in types:
        return "extra-in-draft"
    return "other:" + ",".join(sorted(t for t in types if t))


# ------------------------------------------------------------------ deterministische Exact-Fixes (stil-erhaltend)
def _close(a, b):
    return abs(a - b) <= max(1e-4, abs(b) * 1e-4)


def _float_reconcile(draft, insns):
    tgt = []
    for ln in insns:
        for m in re.finditer(r"0x([0-9A-Fa-f]{8})", ln):
            try:
                f = struct.unpack(">f", struct.pack(">I", int(m.group(1), 16)))[0]
            except Exception:
                continue
            if 1e-3 < abs(f) < 1e7 and not any(_close(f, x) for x in tgt):
                tgt.append(f)
    dt = []
    for m in re.finditer(r"-?\d+\.\d+f?|-?0x[0-9A-Fa-f]{8}", draft):
        s = m.group(0)
        try:
            val = (struct.unpack(">f", struct.pack(">I", int(s, 16) & 0xFFFFFFFF))[0]
                   if "x" in s else float(s.rstrip("f")))
        except Exception:
            continue
        if 1e-3 < abs(val) < 1e7:
            dt.append((s, val))
    draft_only = [(s, v) for s, v in dt if not any(_close(v, t) for t in tgt)]
    tgt_only = [t for t in tgt if not any(_close(t, v) for _, v in dt)]
    if len(draft_only) == 1 and len(tgt_only) == 1:
        tok = draft_only[0][0]; tval = tgt_only[0]
        new = (f"0x{struct.unpack('>I', struct.pack('>f', tval))[0]:08X}" if "x" in tok.lower() else _fmtf(tval))
        return [(tok, new)]
    return []


def _imm_of(operand):
    m = re.search(r"(-?\d+)\s*$", str(operand).strip())
    return int(m.group(1)) if m else None


def _lit_val(tok):
    try:
        return int(tok.rstrip("uUlL"), 0) & 0xFFFFFFFF
    except Exception:
        return None


def _int_exact_fixes(draft, entries):
    fixes = []
    for e in entries:
        if e.get("type") != "Register/Immediate" or str(e.get("op", "")) not in _IMM:
            continue
        t, d = _imm_of(e.get("target")), _imm_of(e.get("draft"))
        if t is None or d is None or t == d:
            continue
        toks = {tok for tok in _INT_LIT.findall(draft) if _lit_val(tok) == (d & 0xFFFFFFFF)}
        if len(toks) != 1:
            continue
        tok = next(iter(toks))
        new = (f"0x{t & 0xFFFFFFFF:X}" if "x" in tok.lower() and t >= 0 else str(t))
        if (tok, new) not in fixes:
            fixes.append((tok, new))
    return fixes


# ------------------------------------------------------------------ m2c-Referenz vom Target
def _m2c_ref(sp, fn):
    try:
        from modules import m2c_donor as _md
        txt = _md.run_m2c_ido(sp, fn)
    except Exception:
        return None
    m = re.search(r"(?m)^[A-Za-z_].*\b" + re.escape(fn) + r"\s*\(", txt)
    return txt[m.start():].strip() if m else None


# ------------------------------------------------------------------ Prompt-Bau (3 Wege: min / rw / asm)
def _build_prompt(draft, fn, sp, entries, insns, variant, ref_m2c=None):
    spots = []
    for e, ci, hint in _kept_spots(insns, entries):
        lm = _landmark(insns, ci) if ci is not None else "(unknown)"
        lo, hi = (max(0, ci - 3), min(len(insns), ci + 4)) if ci is not None else (0, 0)
        win = "\n".join(("    >>> " if i == ci else "        ") + insns[i] for i in range(lo, hi))
        spots.append(f"// spot @ {lm}:\n//   TARGET wants:  {e.get('target')}\n"
                     f"//   DRAFT produced: {e.get('draft')}   <<< wrong\n//   {hint}\n{win}")
    if variant == "asm":
        asm = "\n".join(insns)
        sysm = ("You are decompiling MIPS to C for the SGI IDO 5.3 -O2 compiler. Below is the TARGET assembly, "
                "an M2C REFERENCE (faithful auto-decompilation of it), and a DRAFT C that is close but compiles "
                "wrong. Rewrite the draft so it compiles to the TARGET assembly; keep the draft's type names. "
                "Not every asm detail maps directly to C -- match structure, constants, offsets and argument "
                "order. Reply with ONLY the full C function. No markdown, no explanation.")
        u = ["// === TARGET ASM ===", asm]
        if ref_m2c:
            u += ["", "// === M2C REFERENCE ===", ref_m2c]
        u += ["", "// === DRAFT C (fix) ===", draft]
        return sysm, "\n".join(u)
    keep_naming = (variant != "rw")
    refblock = ([" ", "// === M2C REFERENCE (faithful decompilation of the TARGET -- shows the CORRECT "
                 "structure, argument order and constants; align the draft to it at the spots) ===", ref_m2c]
                if ref_m2c else [])
    refnote = (" An M2C REFERENCE of the target is provided: a faithful (differently-named) decompilation "
               "showing the correct structure, argument order and constants -- use it to see what each spot "
               "SHOULD be, but keep the draft's own types and variable names." if ref_m2c else "")
    sysm = ("You are fixing a MIPS-decompiled C function for the SGI IDO 5.3 -O2 compiler. At the spots below "
            "the draft compiles to the WRONG instructions. Each spot is anchored to a nearby landmark (a call, "
            "a global, or a constant) that appears verbatim in the C -- use it to LOCATE the spot." + refnote +
            " Fix the listed spots; reproduce every other line of the draft unchanged. Reply with ONLY the full "
            "corrected C function. No markdown, no explanation.")
    fxlines = ([f"// EXACT FIX (apply verbatim, keep the SAME literal style, change nothing else): "
                f"replace {tok} with {new}" for tok, new in
                (_float_reconcile(draft, insns) + _int_exact_fixes(draft, entries))] if keep_naming else [])
    u = (["// === WRONG SPOTS (anchored; >>> = the wrong target line) ===", "\n\n".join(spots)]
         + (["", "// === EXACT FIXES (deterministic) ==="] + fxlines if fxlines else [])
         + refblock
         + ["", "// === RULES ===",
            "// - Use the landmark + M2C REFERENCE to find each spot in the draft, then fix it to the TARGET.",
            "// - Each spot carries a HINT with the IDO-specific recipe for that kind of mismatch -- apply it."]
         + (["// - Keep the DRAFT's variable names/types; change ONLY the spot tokens. Do not adopt m2c's naming "
             "or rewrite calls that are already correct in the draft. New diffs elsewhere are FINE (handoff)."]
            if keep_naming else
            ["// - The draft may be structurally wrong at the spots: follow the M2C REFERENCE's structure, "
             "argument order and constants there (keep the draft's TYPE names). New diffs elsewhere are FINE."])
         + ["", "// === DRAFT C (fix the wrong spots) ===", draft])
    return sysm, "\n".join(u)


def _strip_fences(s):
    if not s:
        return s
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", s, re.DOTALL)
    if m:
        s = m.group(1)
    return s.strip()


# ------------------------------------------------------------------ Frankenstein-Hunk-Mixer
def _norm_line(l):
    return re.sub(r"\s+", " ", l.strip())


def _hunks(draft, cand):
    a = draft.splitlines(); b = cand.splitlines()
    sm = difflib.SequenceMatcher(None, [_norm_line(x) for x in a], [_norm_line(x) for x in b])
    return [(i1, i2, b[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


def _apply_hunks(draft, hunks):
    a = draft.splitlines()
    for i1, i2, new in sorted(hunks, key=lambda h: -h[0]):
        a[i1:i2] = new
    return "\n".join(a)


def _nonoverlap_combos(hunks, cap=80):
    res = []
    for r in range(1, len(hunks) + 1):
        for idx in itertools.combinations(range(len(hunks)), r):
            rs = sorted((hunks[k] for k in idx), key=lambda h: h[0])
            if all(rs[k][1] <= rs[k + 1][0] for k in range(len(rs) - 1)):
                res.append([hunks[k] for k in idx])
                if len(res) >= cap:
                    return res
    return res


def _mix(draft, outputs, fn, sp, ev, compile_repair, resolve=None, topk=3):
    """Hunks der N Outputs sammeln -> nicht-ueberlappende Kombis brute-forcen -> compile -> ev; top-K
    zusaetzlich Escape-Gate (resolve = deterministische Experten). Baseline d0 -> nie schlechter als Eingang."""
    seen = {}; hunks = []
    for oi, out in enumerate(outputs):
        if not out:
            continue
        for (i1, i2, new) in _hunks(draft, out):
            key = (i1, i2, tuple(new))
            if key not in seen:
                seen[key] = oi
                hunks.append((i1, i2, new))
    if not hunks:
        return None, None
    scored = []
    for combo in _nonoverlap_combos(hunks):
        code = _apply_hunks(draft, combo)
        use = (compile_repair(code) if compile_repair else code) or code
        d = ev(use)
        if d:
            scored.append((d, use))
    scored.sort(key=lambda x: x[0])
    d0 = ev(draft)
    best, best_code = d0, None
    for d, use in scored[:topk]:        # Escape-Gate (resolve) ist teuer -> nur die besten Kombos
        dd, final = d, use
        if resolve:
            try:
                rc = resolve(use)
                dn = ev(rc)
                if dn and dn < dd:
                    dd, final = dn, rc
            except Exception:
                pass
        if best is None or dd < best:
            best, best_code = dd, final
    return best, best_code


# ------------------------------------------------------------------ ENTRY
def garbage_expert(code, fn, sp, ai, ev, diff_fn, resolve=None, compile_repair=None):
    """Stall-Fallback. ai=oneshot-Callable(system,user)->str. ev(code)->(md,im,tier,mm). diff_fn(code,fn,sp)
    ->(_,entries). resolve(code)->deterministisch-aufgeloester Code (Escape-Gate; None -> Sofort-Metrik).
    compile_repair(code)->kompilierbarer Code|None. Gibt einen NEUEN Zustand zurueck (via Downstream besser als
    der Eingang) oder None (nichts gefunden -> echter Stall)."""
    if ai is None:
        return None
    insns = _target_insns(sp)
    try:
        _t, ents = diff_fn(code, fn, sp)
    except Exception:
        return None
    if not ents or not _kept_spots(insns, ents):
        return None
    ref = _m2c_ref(sp, fn)

    def _one(v):
        try:
            sysm, usr = _build_prompt(code, fn, sp, ents, insns, v, ref_m2c=ref)
            return _strip_fences(ai(sysm, usr))
        except Exception:
            return None
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as ex:      # 3 Wege PARALLEL (Netz-I/O) -> nicht 3x seriell warten
        outs = list(ex.map(_one, ("min", "rw", "asm")))
    if not any(outs):
        return None
    d0 = ev(code)
    best, best_code = _mix(code, outs, fn, sp, ev, compile_repair, resolve=resolve)
    if best_code and d0 and best < d0:        # nur, wenn es die Sackgasse (via Downstream) tatsaechlich verlaesst
        return best_code
    return None
