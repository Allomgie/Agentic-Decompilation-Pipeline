"""Stack-Flow-Buchhaltung (deterministisch) -- liefert der KI bei Stack-Frame-Faellen eine vorab-gerechnete
Gleichung, damit sie gezielt aendern kann statt blind zu "optimieren".

MODELL (IDO, fredchow-Thesis; siehe Memory stack_frame_model): Frame = Align8(ArgBuild + SavedRegs +
NonSavedSpace), NonSavedSpace = ForcedSpills (address-taken/volatile/Array/Struct-Locals) + SpilledLocals
(Reg-Alloc-Spills). SavedRegs + SpilledLocals sind Compiler-Entscheidungen -> NICHT aus C berechenbar, ABER:
- SavedRegs exakt aus dem KOMPILAT/Target-.s (Prolog: sw $ra/$sN = 4B, sdc1 $fN = 8B).
- NonSavedSpace = Frame - SavedBytes (- ArgBuild), exakt.
- ForcedSpills best-effort aus C (Regex): &local, volatile, Array/Struct.

Aus Draft (via ido_compiler+objdump) UND Target (.s) gerechnet -> die DELTAS lokalisieren, WAS fehlt/ueber-
zaehlig ist (zusaetzliches Saved-Reg = Wert lebt ueber Call / f32-Param; mehr/weniger NonSaved = Local-Anzahl).
Verifikation IMMER ueber modules/ido_compiler (PC-Freeze-Regel).
"""
import re

_SAVED = re.compile(r"^(ra|s[0-7]|fp|s8|f2[0-9]|f3[01])$")
_FREG = re.compile(r"^f\d+$")


def _saved_bytes(regs):
    return sum(8 if _FREG.match(r) else 4 for r in regs)


def _from_target(sp):
    """(frame, saved_regs:set) aus dem Target-.s."""
    frame, saved = None, set()
    for ln in open(sp, errors="replace"):
        m = re.search(r"addiu\s+\$sp,\s*\$sp,\s*-0x([0-9a-fA-F]+)", ln)
        if m and frame is None:
            frame = int(m.group(1), 16)
        if "($sp)" in ln:
            mm = re.search(r"\b(sw|sdc1|swc1)\s+\$(\w+),", ln)
            if mm and _SAVED.match(mm.group(2)):
                saved.add(mm.group(2))
    return frame, saved


def _from_draft(code, fn):
    """(frame, saved_regs:set) aus dem kompilierten Draft (objdump). None bei compile-fail."""
    from modules import ido_compiler, ido_comparison
    comp = ido_compiler.compile_code(code, fn)
    if not comp.get("success"):
        ido_compiler.cleanup_temp(comp.get("temp_dir", "")); return None, set()
    try:
        insns = ido_comparison._objdump_insns(comp["temp_o_path"])
    finally:
        ido_compiler.cleanup_temp(comp.get("temp_dir", ""))
    frame, saved = None, set()
    for i in insns:
        ops, mn = i.get("ops", ""), i.get("mnem", "")
        m = re.search(r"sp,sp,-(\d+)", ops)
        if mn == "addiu" and m and frame is None:
            frame = int(m.group(1))
        if mn in ("sw", "sdc1", "swc1") and "(sp)" in ops:
            r = ops.split(",")[0]
            if _SAVED.match(r):
                saved.add(r)
    return frame, saved


def _forced_spills(code):
    """Best-effort aus C: Locals, die zwingend auf den Stack muessen (address-taken / volatile / Array / Struct).
    -> Liste (name, grund). Regex-basiert (kein pycparser-Zwang); dient nur als HINWEIS fuer die KI."""
    out = []
    # nur der Funktionskoerper (ab erstem '{'); grobe Decl-Erkennung
    body = code[code.find("{"):] if "{" in code else code
    addr_taken = set(re.findall(r"&\s*([A-Za-z_]\w*)", body))
    for m in re.finditer(r"(?m)^\s*([A-Za-z_][\w\s\*]+?)\b(\w+)\s*(\[[^\]]*\])?\s*[;=]", body):
        typ, name, arr = m.group(1).strip(), m.group(2), m.group(3)
        if name in addr_taken:
            out.append((name, "address-taken (&)"))
        elif "volatile" in typ:
            out.append((name, "volatile"))
        elif arr:
            out.append((name, f"Array {arr}"))
        elif re.search(r"\bstruct\b", typ):
            out.append((name, "struct"))
    return out


def stack_flow(code, fn, sp):
    """Vollstaendige Zerlegung Draft vs Target. -> dict (None-frames bei compile-fail)."""
    df, ds = _from_draft(code, fn)
    tf, ts = _from_target(sp)
    res = {"draft_frame": df, "target_frame": tf, "draft_saved": sorted(ds), "target_saved": sorted(ts),
           "extra_saved": sorted(ds - ts), "missing_saved": sorted(ts - ds), "forced_spills": _forced_spills(code)}
    if df is not None and tf is not None:
        res["draft_nonsaved"] = df - _saved_bytes(ds)
        res["target_nonsaved"] = tf - _saved_bytes(ts)
        res["frame_delta"] = df - tf
        res["nonsaved_delta"] = res["draft_nonsaved"] - res["target_nonsaved"]
    return res


def _classify(r):
    """-> (kategorie, ai_actionable:bool). Reg-Set-Diff: f-Regs = f32-Param/Wert-ueber-Call (AI-aenderbar);
    nur-NonSaved-Diff: Local-Anzahl (AI-aenderbar). Reine int-s-Reg-Diff ohne C-Anker: Reg-Alloc (AI schwer)."""
    if r.get("draft_frame") is None or r.get("target_frame") is None:
        return "compile-fail", False
    if r["frame_delta"] == 0 and not r["extra_saved"] and not r["missing_saved"]:
        return "frame-match", False
    freg_diff = [x for x in r["extra_saved"] + r["missing_saved"] if _FREG.match(x)]
    if r["missing_saved"] or r["extra_saved"]:
        if freg_diff:
            return "reg-diff:float (f32 param/value lives across call)", True
        return "reg-diff:int (value lives across call / regalloc)", True
    return "nonsaved:locals (local-variable count)", True


# Kategorie-bedingte Trick-Liste fuer den KI-Prompt (ENGLISCH; gegroundet in analysis/Research/Decomp_Infosheet.md:
# "Stack placement", "Permanent (callee-save) registers", "Function declarations matter", "Variables matter",
# "Rematerialization of constants"). NUR die zur Diagnose passenden Tricks -> fokussierter Prompt.
_TRICK_GROW = [
    "A stack local is MISSING. Declare the needed local (an array/struct, or take a local's address with &local) so it gets a stack slot.",
    "To reserve slack without changing behavior, add an UNUSED local at the TOP of the declaration order (a last-declared unused local is trimmed and would NOT count).",
    "`void f(void)` uses 4 more stack bytes than `void f()`; calling a value-returning function consumes stack even if the result is ignored.",
]
_TRICK_SHRINK = [
    "First choice: if a local is DEAD (assigned but never read) or an unnecessary &local / volatile / array that "
    "forces a slot, remove its declaration AND every reference to it.",
    "Only if no dead local exists: a local that merely caches a simple expression (e.g. `p = arg0 + 0x18;` or a "
    "field load) reused across a call MAY be inlined -- replace it by that exact expression at every use and delete "
    "the local. Inline ONE such local, do NOT inline call results and do NOT restructure anything else.",
    "Last resort: move register-promotable locals to the END of the declaration order. Never merge or nest calls.",
]
_TRICK_ADD_SAVED = [
    "The target saves MORE registers: a value must live ACROSS a function call and be read/written several times so the compiler puts it in a callee-saved register. Try splitting `a = b + c;` into `a = b; a += c;`, or reuse ONE variable across the call instead of recomputing.",
]
_TRICK_ADD_SAVED_F = [
    "The extra saved register(s) are FLOAT (f20+). A float value/parameter must live across a call: declare the relevant parameter/value as `f32` (not double, and give callees an f32 prototype) so it is kept in a float callee-saved register.",
]
_TRICK_REMOVE_SAVED = [
    "You save MORE registers than the target: a value is needlessly kept across a call. Consume it BEFORE the call, or duplicate the expression on each side of the call instead of storing it in a variable, so it stays in a temporary register.",
]
_TRICK_RELOCATE = [
    "Same frame size and same saved registers, only stack offsets differ: reorder local declarations, or change which values are kept in locals vs recomputed; statement order / `,` vs `;` / whitespace can shift slot assignment.",
]


def _tricks(r, cat):
    out = []
    if r.get("missing_saved"):
        out += _TRICK_ADD_SAVED_F if any(_FREG.match(x) for x in r["missing_saved"]) else _TRICK_ADD_SAVED
    if r.get("extra_saved"):
        out += _TRICK_REMOVE_SAVED
    nd = r.get("nonsaved_delta", 0)
    if nd > 0 and not r.get("extra_saved"):
        out += _TRICK_SHRINK
    elif nd < 0 and not r.get("missing_saved"):
        out += _TRICK_GROW
    if not out and r.get("frame_delta") == 0:
        out += _TRICK_RELOCATE
    return out


def stack_flow_hint(code, fn, sp):
    """Deterministic plain-text hint for the AI PROMPT (ENGLISH) + (category, ai_actionable).
    Gives the model the computed stack-flow equation + the concrete delta diagnosis + category-matched tricks."""
    r = stack_flow(code, fn, sp)
    cat, actionable = _classify(r)
    if r.get("draft_frame") is None or r.get("target_frame") is None:
        return "[stack-flow: compile-fail or no frame readable]", cat, actionable
    lines = [
        f"STACK-FLOW ACCOUNTING for {fn} -- these numbers are MEASURED from the actual compiled object. Treat them "
        f"as ground truth: do NOT recompute byte sizes or guess pointer widths. (Reference: on N64/MIPS32 int and "
        f"pointer = 4 bytes; the compiler reserves one ~8-byte-aligned stack slot per spilled value.) Focus only on "
        f"WHICH value to add/remove, not on the arithmetic.",
        f"  Frame = Align8(SavedRegs + NonSaved[ArgBuild + ForcedSpills + SpilledLocals]).",
        f"  YOUR draft:  Frame={r['draft_frame']}B = SavedRegs{r['draft_saved']} ({_saved_bytes(set(r['draft_saved']))}B) + NonSaved={r['draft_nonsaved']}B",
        f"  TARGET:      Frame={r['target_frame']}B = SavedRegs{r['target_saved']} ({_saved_bytes(set(r['target_saved']))}B) + NonSaved={r['target_nonsaved']}B",
        f"  Frame delta={r['frame_delta']:+d}B  (NonSaved delta={r['nonsaved_delta']:+d}B)",
    ]
    if r["missing_saved"]:
        lines.append(f"  -> TARGET additionally saves {r['missing_saved']}: a value must live across a call"
                     + (" (likely an f32 parameter/value)" if any(_FREG.match(x) for x in r['missing_saved']) else " (a long-lived local/parameter)") + ".")
    if r["extra_saved"]:
        lines.append(f"  -> YOU save excess {r['extra_saved']}: a value lives across a call unnecessarily.")
    if r["nonsaved_delta"] != 0 and not (r["missing_saved"] or r["extra_saved"]):
        lines.append(f"  -> Same registers, but {abs(r['nonsaved_delta'])}B "
                     + ("TOO MUCH local space (excess local variable[s])." if r['nonsaved_delta'] > 0 else "TOO LITTLE local space (a local/array is missing)."))
    if r["forced_spills"]:
        lines.append(f"  Forced stack locals (from your C): " + ", ".join(f"{n}({why})" for n, why in r["forced_spills"]))
    tr = _tricks(r, cat)
    if tr:
        lines.append("  RELEVANT TRICKS (apply the minimal one that fits):")
        lines += [f"    - {t}" for t in tr]
    lines.append(f"  Category: {cat} | AI-actionable: {'YES' if actionable else 'likely NO (pure register allocation)'}")
    return "\n".join(lines), cat, actionable
