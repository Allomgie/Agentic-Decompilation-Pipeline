"""GENERISCHER agentischer Engine (Compile-Feedback-Loop + ReAct-Tools), FOKUS = Parameter.
Domaenen-Schliesser-Architektur: derselbe Engine, pro Aufruf auf EINE Fix-Domaene fokussiert + fuer die
anderen "blind" (Feedback/Goal nur auf der Fokus-Domaene; Rest = Handoff an die det. Experten via Ping-Pong).
  focus="structure"  -> Host expert_garbage; Ziel Struktur-Diffs (Missing/Extra/Reordered/BLOCK)->0; Tool: m2c-Ref.
  focus="typ"        -> Domaenen-Schliesser; Ziel TYP-Diffs->0.
Tools (read-only): TOOL search/read/grep (modules.header_tools) -> echte Feldtypen; nicht gefunden=halluziniert.
EIN Durchgang, multi-turn, Reasoning MITLESEN, KEIN best-of-N. Prompts sind ERSTE ENTWUERFE -> am Reasoning tunen.

PRODUKTIV (modules/agent_engine.py): self-contained, eval_fn INJIZIERT (kein ppw-Import -> kein Zirkular).
Pod aus LOCAL_API_BASE (gleicher Pod wie die Pipeline). Aufruf via run()/solve_recursive(focus, fn, sp, code, eval_fn).
"""
import json, re, urllib.request, urllib.error, os, subprocess, time
from modules import expert_instr_width as e_iw, header_tools as ht, expert_garbage as eg
from modules import m2c_donor as _m2c, ido_compiler as _ic
from modules import expert_immediate as e_imm, expert_instr_op as e_op


_REG = re.compile(r"^\$?(zero|at|v[01]|a[0-3]|t\d|s[0-8]|k[01]|gp|sp|fp|ra|f\d+|hi|lo)$")


def _has_sp_fp(e):
    """True, wenn ein Operand sp/fp ist -> der abweichende Wert ist ein STACK-OFFSET (Frame), kein C-Literal.
    (Lokale Kopie aus ppw._has_sp_fp, damit agent_engine self-contained ist.)"""
    for side in ("target", "draft"):
        s = str(e.get(side) or "")
        if any(x.strip() in ("sp", "fp", "$sp", "$fp") for x in re.split(r"[,\s]+", s)):
            return True
    return False


def _compile_err(code, fn):
    """VOLLER IDO-Compiler-Fehler (error_log) fuer das Feedback -- die KI soll den kompletten Fehler sehen,
    nicht raten. Header weggekuerzt, Cap grosszuegig (4000, IDO-Fehler sind kurz; bei Mehrfach-Fehlern lieber
    die letzten = die echten cfe-Errors). Orakel-sicher (compile->cleanup)."""
    try:
        comp = _ic.compile_code(code, fn)
        log = (comp.get("error_log") or "").strip()
        _ic.cleanup_temp(comp.get("temp_dir", ""))
        log = re.sub(r"^---+\s*IDO Compiler Error\s*---+\s*", "", log)   # Header weg, Platz fuer echten Fehler
        return log[-4000:] if log else "(kein error_log -- evtl. Compile-Timeout/leere .o)"
    except Exception as e:
        return f"(compile-err Ausnahme: {e})"

BASE = (os.environ.get("POD_BASE") or os.environ.get("LOCAL_API_BASE") or "").rstrip("/")
MODEL = os.environ.get("POD_MODEL") or os.environ.get("LOCAL_API_MODEL") or "Qwen/Qwen3.6-35B-A3B"
KEY = os.environ.get("POD_KEY") or os.environ.get("LOCAL_API_KEY") or "EMPTY"
_NUM = re.compile(r"^-?(0x[0-9A-Fa-f]+|\d+)$")


# ----------------------------------------------------------------- Fokus-Definitionen
def _is_block(e):
    t, d = e.get("target"), e.get("draft")
    return isinstance(t, list) and isinstance(d, list) and (len(t) > 1 or len(d) > 1)


def _struct_in_focus(e):
    return e.get("type") in ("Missing in Draft", "Extra in Draft", "Reordered") or \
        (e.get("type") == "Instruction Mismatch" and _is_block(e))


def _missing_mnemonic(ins):
    s = re.sub(r"/\*.*?\*/", "", str(ins)).strip()
    s = re.sub(r"^[0-9a-fA-F]+:\s*[0-9a-fA-F]+\s*", "", s)   # evtl. Adresse/Hex strippen
    m = re.match(r"([a-z][a-z0-9.]*)", s)
    return m.group(1) if m else ""


def _pure_codegen_structure(ents):
    """True, wenn die GESAMTE Struktur-Rest-md reines Codegen ist (nur redundante b/j + nop = Epilog-Branch/
    Scheduling) -> C-seitig NICHT fixbar -> Agent ueberspringen, direkt Handoff (Permuter). KONSERVATIV: jedes
    reale fehlende Statement (jal/sw/lw/addiu/...) ODER Extra/Reordered/BLOCK -> NICHT skippen (Agent laeuft)."""
    fe = [e for e in ents if _struct_in_focus(e)]
    if not fe:
        return False
    for e in fe:
        if e.get("type") != "Missing in Draft":
            return False                       # Extra/Reordered/BLOCK = echte Struktur
        miss = e.get("missing") or []
        if not miss or len(miss) > 2:
            return False
        if any(_missing_mnemonic(x) not in ("b", "j", "nop") for x in miss):
            return False
    return True


def _struct_count(ents):
    md = sum(len(e.get("missing", []) or e.get("target", []) or [1]) for e in ents if e.get("type") == "Missing in Draft")
    other = sum(1 for e in ents if e.get("type") in ("Extra in Draft", "Reordered")
                or (e.get("type") == "Instruction Mismatch" and _is_block(e)))
    return md + other


def _missing_window(ents, n=12):
    """Naechste n FEHLENDE Target-Instruktionen in Reihenfolge (fuer den Leiter-Fix). missing-Liste ist bereits
    target-geordnet (konsekutive Adressen). Adress/Hex-Praefix gestrippt (Token-spar, Mnemonic+Operanden reicht)."""
    out = []
    for e in ents:
        if e.get("type") == "Missing in Draft":
            for ins in (e.get("missing") or []):
                s = re.sub(r"/\*.*?\*/", "", str(ins)).strip()
                if s:
                    out.append(s)
    return out[:n]


def _is_scale(dv, tv):
    if dv == 0 or tv == 0: return False
    a, b = abs(dv), abs(tv); lo, hi = min(a, b), max(a, b)
    return hi % lo == 0 and (hi // lo) in (2, 4, 8, 16)


def _typ_in_focus(e):
    t = e.get("type")
    if t == "Instruction Mismatch":
        return bool(e_iw.width_targets([e]))
    if t == "Memory Access":
        s = str(e.get("target")) + str(e.get("draft")); return not ("RELOC" in s or "@" in s)
    if t in ("Register/Immediate", "Address Load"):
        if _has_sp_fp(e): return False
        tg, dr = e.get("target"), e.get("draft")
        if not (isinstance(tg, str) and isinstance(dr, str)): return False
        tt = [x.strip() for x in tg.split(",")]; dt = [x.strip() for x in dr.split(",")]
        if len(tt) != len(dt): return False
        nd = [(int(a, 0), int(b, 0)) for a, b in zip(tt, dt) if _NUM.match(a) and _NUM.match(b) and int(a, 0) != int(b, 0)]
        nonnum = any(a != b and not (_NUM.match(a) and _NUM.match(b)) for a, b in zip(tt, dt))
        if len(nd) == 1 and not nonnum:
            tv, dv = nd[0]
            return e.get("op", "") in ("sll", "srl", "sra", "sllv", "srlv", "srav") or _is_scale(dv, tv)
    return False


def _typ_count(ents):
    return sum(1 for e in ents if _typ_in_focus(e))


# --- LOGIC = TYP u KONST u OP (eine zusammenhaengende "lokale In-Place-Edit"-Domaene, gleiches Toolset).
# Begruendung: reines Single-Domain ist genauso kontraproduktiv wie "fix this" (Tunnelblick -> Kollateral-
# schaden in Nachbardomaenen, den der Agent gar nicht sieht; vgl. func_800C1568: TYP geloest, dabei p2/p3/p4
# aufgerissen). Wir buendeln, zeigen das VOLLE Diff als Kontext, schliessen aber nur TYP/KONST/OP.
def _konst_in_focus(e):
    """KONST: genau EIN numerisches Literal weicht ab -- aber KEIN Stride/Shift (das ist TYP)."""
    return bool(e_imm.imm_targets([e])) and not _typ_in_focus(e)


def _op_in_focus(e):
    """OP: echter Operator-/Signedness-Tausch (beide ALU, gleiche Register-Operanden)."""
    return bool(e_op.op_targets([e]))


def _logic_in_focus(e):
    return _typ_in_focus(e) or _konst_in_focus(e) or _op_in_focus(e)


def _logic_count(ents):
    return sum(1 for e in ents if _logic_in_focus(e))


def _logic_tips(ents):
    """Diff-ABHAENGIGE Tipps -- nur die Sub-Domaenen, die wirklich im Diff stehen. Inhalt aus unseren
    kalibrierten Experten-Prompts (expert_instr_width / expert_immediate / expert_instr_op)."""
    tips = []
    if any(_typ_in_focus(e) for e in ents):
        tips.append(
            "TYPE/WIDTH/STRIDE: a wrong pointer/element type changes load/store width (lb/lbu=1, lh/lhu=2, "
            "lw=4, ld=8 bytes) and array stride (index<<n: <<0=1B char, <<1=2B short, <<2=4B int/float/pointer, "
            "<<3=8B double/long long). Pick the element type whose size matches the target's width/stride. Use "
            "TOOL search to confirm real struct field types. A value the target treats as float must be f32, "
            "not f64/int.")
    if any(_konst_in_focus(e) for e in ents):
        tips.append(
            "CONSTANT: a numeric literal differs from the target -- set it to the target value. If one constant "
            "is loaded once but reused (shared int/float literal), change EVERY occurrence. Mind hex-vs-decimal "
            "spelling and signedness.")
    if any(_op_in_focus(e) for e in ents):
        tips.append(
            "OPERATOR/SIGNEDNESS: sra=signed >>, srl=unsigned >> (fix via operand cast (s32)/(u32)); slt/sltu and "
            "mult/multu differ only in operand signedness -> cast, do not rewrite the expression. For a genuine "
            "operator swap (+/-, &/|, etc.) change the C operator itself.")
    return tips


def _other_diff_lines(ents, in_focus, cap=8):
    """NICHT-Focus-Diffs als KONTEXT (kurz): der Agent soll sie sehen, um sie nicht zu verschlechtern."""
    out = []
    for e in ents:
        if in_focus(e):
            continue
        t = e.get("type")
        if t == "Missing in Draft":
            out.append(f"  [structure: MISSING {len(e.get('missing') or [])} instr]")
        elif t in ("Extra in Draft", "Reordered"):
            out.append(f"  [structure: {t}]")
        elif t == "Memory Access":
            out.append(f"  [dataref/reloc] target=({str(e.get('target'))[:48]})")
        elif _has_sp_fp(e):
            out.append(f"  [stack/frame] {e.get('op','')} target=({str(e.get('target'))[:40]})")
        else:
            out.append(f"  [{t}] {e.get('op','')}")
        if len(out) >= cap:
            break
    return out


STRUCT_SYS = (
    "The draft C function is STRUCTURALLY wrong vs the target: it is missing statements, has extra ones, has "
    "mis-ordered code, or wrong call arguments. You are given an M2C REFERENCE: a faithful (differently-named) "
    "decompilation of the TARGET that shows the CORRECT structure, statement set and argument order. Make the "
    "draft's STRUCTURE match the target: add the missing statements, remove extra ones, fix ordering and call "
    "arguments. Keep real type names (use tools to check field types). The M2C reference is a GUIDE (plain compilable C) "
    "but may be imperfect or over-inlined; the compile feedback (remaining diff vs the real target) is GROUND TRUTH -- "
    "when they conflict, follow the feedback, not the reference. If a load+store looks 'missing' even though you wrote "
    "the assignment, split it into an explicit local variable (T *p = ...; *p = ...;) to match the separate lw/sw. "
    "Never repeat an attempt that did not change the result. "
    "IDO is an OLD C89 compiler: declare ALL local variables at the TOP of the function, before any statement "
    "(NO declarations after a statement); a void function must not return a value. "
    "Do NOT worry about exact register allocation or minor constant values -- other passes handle those. "
    "Tools (read-only), reply with ONLY tool lines to call them:\n"
    "  TOOL m2c                   -> faithful decompilation (reference structure) of the whole target function\n"
    "  TOOL search <TypeName> | TOOL read <header.h> | TOOL grep <regex>\n"
    "For large functions the M2C reference is NOT inlined -- call TOOL m2c to get it. "
    "If a type name from the draft is NOT found, it is likely hallucinated -> use a real type. When ready, reply "
    "with ONLY the full corrected C in one ```c block. Keep reasoning short."
)
LOGIC_SYS = (
    "The draft C function is ALREADY very close to correct -- usually only a FEW assembly mismatches away. Treat "
    "the draft as the BASE and make the SMALLEST possible edit. Your job: change ONLY the specific TYPE, CONSTANT or "
    "OPERATOR tokens needed to fix the listed mismatches, and change NOTHING else.\n"
    "CRITICAL: do NOT rewrite the function from the assembly. Do NOT change the function signature, return type, "
    "argument count/order, local variable set, statements, or control flow. PRESERVE every line of the draft "
    "byte-for-byte except the few tokens you must change -- INCLUDING seemingly-redundant or no-op statements (e.g. "
    "`x = x;`, a cast that looks pointless, an extra local): IDO emits real instructions for them and the draft "
    "needs them to match. Rewriting a near-correct draft REGRESSES it (adds missing instructions / breaks structure) "
    "-- if a mismatch cannot be fixed by a small local token change, make NO change at all and leave it for another "
    "pass.\n"
    "The mismatches to fix are: wrong pointer/element TYPES (load/store width, array stride, field offsets), wrong "
    "CONSTANTS, or wrong OPERATORS/signedness. You are ALSO shown OTHER diffs (structure / memory-reloc / "
    "stack-frame) as CONTEXT only: do not make those worse -- other passes handle them.\n"
    "Tools (read-only): TOOL search <TypeName> | TOOL read <header.h> | TOOL grep <regex>. When you are UNSURE of a "
    "struct/field/typedef type, CALL TOOL search BEFORE guessing -- do not deliberate at length about a type you can "
    "look up in one call. If search does not resolve a name: a TYPE name (struct/typedef) that is missing is likely "
    "hallucinated -> use a real type; but a DATA symbol (e.g. D_8xxxxxxx, jtbl_xxx) is NOT covered by the header tool "
    "and is NOT hallucinated -> infer its element width/stride from the diff instead. IDO is C89: declare all locals "
    "at the TOP of the function (no mid-block declarations); a void function returns no value. When ready, reply with "
    "ONLY the full corrected C in one ```c block. Keep reasoning short."
)

LOGIC2_SYS = (
    "ESCALATION stage. A TYPE/width/stride/constant/operator in this draft is wrong, and the earlier strict pass "
    "could NOT fix it without breaking structure -- because applying the CORRECT type REQUIRES adjusting the "
    "surrounding structure too. Typical couplings: an 8-byte array stride (index<<3) combined with a 4-byte store "
    "means the element is a STRUCT (T arr[]; arr[i].field = ...), NOT a scalar f64/s64; a float parameter spilled "
    "to the stack needs the right local/spill; a wider type splits one load/store into several. You MAY now change "
    "types AND structure TOGETHER: add/remove/reorder statements, introduce locals, adjust the declaration -- "
    "whatever applies the correct type cleanly. Use the M2C REFERENCE below for the correct structure.\n"
    "HARD RULE: your result must reduce the OVERALL mismatch. A change that removes the type diff but ADDS missing "
    "instructions is INCOMPLETE -- finish the WHOLE cascade so nothing is left missing, or make NO change. Keep the "
    "parts of the draft that already match byte-for-byte (including redundant-looking statements). "
    "Tools: TOOL m2c | TOOL search <Type> | TOOL read <h> | TOOL grep <re>. IDO is C89: locals at the TOP, void "
    "returns nothing. When ready, reply with ONLY the full corrected C in one ```c block. Keep reasoning short."
)

FOCI = {
    "structure": {"sys": STRUCT_SYS, "in_focus": _struct_in_focus, "count": _struct_count,
                  "label": "structural mismatches (missing/extra/reordered)", "use_m2c": True,
                  "protect": (0,), "tolerate": (), "codegen_skip": _pure_codegen_structure},  # Struktur hoechste
    "logic":     {"sys": LOGIC_SYS, "in_focus": _logic_in_focus, "count": _logic_count,
                  "label": "type/constant/operator mismatches", "use_m2c": False,
                  "protect": (0, 1, 2, 3), "tolerate": (4, 5, 6),  # md/p1/p2/p3 schuetzen; alloc/frame+tier+mm tolerant
                  "tips": _logic_tips, "context": True},
    # STUFE 2 (Eskalation): darf Logik UND Struktur aendern; Gate = NETTO-Metrik-Verbesserung (net_gate) statt
    # protect -> sicher (nie netto-schlechter), muss die ganze Kopplungs-Kaskade landen. m2c fuer Strukturkontext.
    "logic_struct": {"sys": LOGIC2_SYS, "in_focus": _logic_in_focus, "count": _logic_count,
                     "label": "type/constant/operator mismatches (structure-coupled)", "use_m2c": True,
                     "protect": (), "tolerate": (4, 5, 6), "net_gate": True,
                     "tips": _logic_tips, "context": True},
}
FOCI["typ"] = FOCI["logic"]   # Back-compat-Alias (alte Test-Harness-Aufrufe mit focus="typ")


# ----------------------------------------------------------------- Diff-Rendering (fokus-gefiltert)
def focus_diff_lines(ents, in_focus, cap=14):
    out = []
    for e in ents:
        if not in_focus(e): continue
        tg, dr = e.get("target"), e.get("draft")
        if e.get("type") == "Missing in Draft":
            miss = e.get("missing") or []
            txt = " ; ".join(re.sub(r"/\*.*?\*/", "", str(x)).strip() for x in miss)[:160] if miss else str(tg)[:90]
            out.append(f"  [MISSING {len(miss)} instr] {txt}")
        else:
            if isinstance(tg, list): tg = " ; ".join(re.sub(r"/\*.*?\*/", "", str(x)).strip() for x in tg)[:80]
            if isinstance(dr, list): dr = " ; ".join(str(x) for x in dr)[:80]
            out.append(f"  [{e.get('type')}] {e.get('op','')} target=({str(tg)[:60]}) draft=({str(dr)[:60]})")
        if len(out) >= cap: break
    return out


def _mkey(m, tolerate):
    """Metrik fuer den Vergleich mit genullten TOLERATE-Buckets -> ein echter Logic-Fortschritt wird NICHT
    verworfen, nur weil alloc/frame (p4) oder tier/mm als Nebenwirkung wachsen (Lukas: Gate grosszuegig bei
    alloc/frame). protect bleibt die harte Schranke (md/p1/p2/p3 duerfen NICHT regressieren)."""
    if not m or not tolerate:
        return m
    return tuple(0 if i in tolerate else v for i, v in enumerate(m))


def _strip_code(s):
    s = re.sub(r"<think>.*?</think>", "", s or "", flags=re.S); s = re.sub(r"^.*?</think>", "", s, flags=re.S)
    m = re.search(r"```(?:c|C)?\s*\n(.*?)```", s, flags=re.S)
    return m.group(1).strip() if m else None


MODEL_CTX = 32768
_MAX_INSTR = int(os.environ.get("STRUCT_MAX_INSTR", "700"))    # harte Obergrenze; darunter via m2c-Tool + Budget-Skip
_LARGE_INSTR = int(os.environ.get("STRUCT_LARGE_INSTR", "90"))  # darueber: m2c NICHT im Prompt, sondern per TOOL m2c (pull)


def _target_instr_count(sp):
    return sum(1 for ln in open(sp, errors="replace")
               if re.search(r"/\*\s*[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}", ln))


def _compiled_instr_count(code, fn):
    """#Instruktionen, zu denen `code` WIRKLICH kompiliert (objdump). draft_o=0 => genuin leer (Leiter-Ziel);
    draft_o ~ target => volle Funktion (md-hoch ist dann nur Alignment-Artefakt, KEIN Leiter-Fall, z.B. chdippy)."""
    try:
        comp = _ic.compile_code(code, fn)
        o = comp.get("temp_o_path"); n = 0
        if comp.get("success") and o and os.path.exists(o):
            out = ""
            for tool in ("mips-linux-gnu-objdump", "mips64-elf-objdump", "objdump"):
                try:
                    out = subprocess.run([tool, "-d", o], capture_output=True, text=True).stdout
                    if out:
                        break
                except FileNotFoundError:
                    continue
            n = sum(1 for l in out.splitlines() if ":\t" in l)
        _ic.cleanup_temp(comp.get("temp_dir", ""))
        return n
    except Exception:
        return -1


def _est_tokens(messages):
    return sum(len(m.get("content", "")) for m in messages) // 3   # grobe chars->tokens


def _budget(messages, cap=14000, margin=600):
    return min(cap, MODEL_CTX - _est_tokens(messages) - margin)


def chat(messages, max_tokens=14000, timeout=400, retries=3):
    """SSE-Streaming-Chat mit Retry gegen transiente Pod-Fehler (5xx/Connection/Timeout) unter Parallelitaet.
    HTTP 400 (Kontext/Bad Request) ist DETERMINISTISCH -> kein Retry, sofort raise (Budget-Logik faengt es upstream)."""
    body = {"model": MODEL, "messages": messages, "temperature": 0.3, "max_tokens": max_tokens, "stream": True}
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        cp, rp, fin = [], [], None
        try:
            req = urllib.request.Request(BASE + "/chat/completions", data=data,
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}",
                                                  "User-Agent": "curl/8.0", "Accept": "text/event-stream"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"): continue
                    p = line[5:].strip()
                    if p == "[DONE]": break
                    try: ch = json.loads(p)["choices"][0]
                    except Exception: continue
                    d = ch.get("delta") or {}
                    if d.get("content"): cp.append(d["content"])
                    rc = d.get("reasoning_content") or d.get("reasoning")
                    if rc: rp.append(rc)
                    if ch.get("finish_reason"): fin = ch["finish_reason"]
            return {"content": "".join(cp), "reasoning": "".join(rp), "finish": fin}
        except urllib.error.HTTPError as e:
            if e.code == 400:
                raise                      # Kontext/Bad-Request -> deterministisch, kein Retry
            last = e
        except Exception as e:
            last = e
        time.sleep(2 * (attempt + 1))
    raise last


def run_tool(line):
    parts = line.strip().split(None, 2)
    if len(parts) < 3 or parts[0].upper() != "TOOL": return None
    cmd, arg = parts[1].lower(), parts[2].strip()
    if cmd == "search": return json.dumps(ht.search_symbol(arg), ensure_ascii=False)[:1800]
    if cmd == "read":
        r = ht.read_header(arg); return json.dumps({k: r[k] for k in r if k != "text"}, ensure_ascii=False) + "\n" + r.get("text", "")[:1500]
    if cmd == "grep": return "\n".join(f"{f}:{ln}: {tx}" for f, ln, tx in ht.grep_headers(arg))[:1500]
    return f"unknown tool {cmd}"


def run(focus, fn, sp, code0, eval_fn, max_turns=8, quiet=False, code_override=None, deadline_ts=None):
    """focus, fn, sp(=target .s), code0(=aktueller Draft), eval_fn(code,fn,sp)->(metrik, entries).
    Multi-turn Compile-Feedback-Loop. Gibt Ergebnis-Dict (incl. new_code) zurueck.
    deadline_ts: absolute Wall-Clock-Grenze (epoch). Der Agent startet keinen neuen Turn mehr danach (er bekommt
    Zeit fuer einen LAUFENDEN Turn, verharrt aber nicht). None = unbegrenzt (offline-Batch)."""
    _buf = []
    def _p(*a, **k):
        line = " ".join(str(x) for x in a)
        _buf.append(line)
        if not quiet:
            print(line)
    cfg = FOCI[focus]
    tolerate = cfg.get("tolerate", ())
    if code_override is not None:
        code0 = code_override
    tgt = open(sp, errors="replace").read()
    base_m, ent0 = eval_fn(code0, fn, sp)
    f0 = cfg["count"](ent0)
    reason_chars = []; turns_used = 0
    _p(f"=== {fn} | focus={focus} | base metric={base_m} | focus-diffs={f0} ===\n")
    if f0 == 0:           # KEINE Diffs dieser Domaene -> Agent ueberspringen (kein Modell-Turn fuer nichts)
        _p(f"[skip] keine {focus}-Diffs -> uebersprungen")
        return {"fn": fn, "focus": focus, "f0": 0, "fN": 0, "m0": base_m, "mN": base_m,
                "turns": 0, "reason_chars": [], "status": "no-focus-diffs", "new_code": code0,
                "transcript": "\n".join(_buf)}
    n_instr = _target_instr_count(sp)
    if n_instr > _MAX_INSTR:             # zu gross: m2c-Ref+Draft sprengen Kontext/Laufzeit (453 instr=210s, 0 Fortschritt)
        _p(f"[skip] Funktion zu gross ({n_instr} Instr > {_MAX_INSTR}) -> Handoff (Permuter/Splitter)")
        return {"fn": fn, "focus": focus, "f0": f0, "fN": f0, "m0": base_m, "mN": base_m,
                "turns": 0, "reason_chars": [], "status": "too-large", "new_code": code0,
                "transcript": "\n".join(_buf)}
    skip = cfg.get("codegen_skip")
    if skip and skip(ent0):
        _p("[skip] Rest-md ist reines Codegen (b/j/nop) -> Agent uebersprungen, Handoff an Permuter")
        return {"fn": fn, "focus": focus, "f0": f0, "fN": f0, "m0": base_m, "mN": base_m,
                "turns": 0, "reason_chars": [], "status": "codegen-skip", "new_code": code0,
                "transcript": "\n".join(_buf)}

    # Volles Target-ASM nur bei KLEINEN Funktionen mitgeben (bei grossen redundant mit m2c-Ref + sprengt Kontext).
    is_large = n_instr > _LARGE_INSTR
    _m2c_cache = {}

    def get_m2c():
        if "v" not in _m2c_cache:
            ref = eg._m2c_ref(sp, fn)
            if ref:
                try:
                    cleaned = _m2c.clean_m2c(ref)     # M2C_FIELD/M2C_UNK-Makros inlinen -> kompilierbares Ref-C
                    if cleaned: ref = cleaned
                except Exception:
                    pass
            _m2c_cache["v"] = ref or "(m2c reference unavailable)"
        return _m2c_cache["v"]

    parts = [f"Target MIPS:\n{tgt}\n"] if len(tgt) < 4500 else \
            ["(Target assembly is large -- rely on the M2C reference and the diff below.)\n"]
    if cfg["use_m2c"]:
        _ref = get_m2c()
        # m2c INLINE wenn es ins Budget passt (auch bei grossen Fkt -- der Agent ruft das Tool oft nicht von selbst).
        # Nur bei wirklich riesiger m2c -> per Tool (pull). M2C_FIELD/M2C_UNK ggf. uncleanbar -> Agent expandiert sie.
        if _ref and _ref != "(m2c reference unavailable)" and len(_ref) < 11000:
            parts.append("M2C REFERENCE (the COMPLETE correct structure of the target). It may contain placeholders: "
                         "M2C_FIELD(p, T, off) means *(T*)((char*)p + off); replace each M2C_UNK with the correct real "
                         "type (use TOOL search on the structs/symbols involved). Reproduce it FULLY -- every "
                         "statement, do not shorten or summarize:\n" + _ref + "\n")
        else:
            parts.append("The target function is very large: call `TOOL m2c` to fetch the reference structure "
                         "(decompiled target) and reproduce it fully.\n")
    parts.append(f"Current C draft:\n```c\n{code0}\n```\n")
    # GROSS + viel fehlt (md hoch) = fast leerer Draft -> ganze Funktion aus der m2c-Ref bauen (Voll-Gen).
    # GROSS + md klein (viele Mismatch-Bloecke) = meist Codegen/Regalloc -> nicht strukturell (Handoff).
    # Leiter-Trigger = GENUIN LEER (draft kompiliert zu << target), NICHT md (md luegt bei vollen, fehl-
    # ausgerichteten Funktionen wie chdippy: draft_o=230~target, md=155 reines Alignment-Artefakt -> kein Leiter).
    draft_o = _compiled_instr_count(code0, fn) if is_large else n_instr
    mostly_missing = 0 <= draft_o < n_instr * 0.5
    leiter = is_large and mostly_missing       # grosse, GENUIN LEERE Funktion -> aus m2c voll aufbauen (Leiter)
    dcap = 10 if is_large else 14
    if leiter:
        incr_note = ("This is a LARGE, mostly-incomplete function. The M2C REFERENCE above is the COMPLETE structure -- "
                     "reproduce it FULLY (expand M2C_FIELD, resolve M2C_UNK via TOOL search, every statement, IDO C89). "
                     "Do NOT shorten or summarize. The compile feedback lists which target instructions are STILL "
                     "missing -- treat it as a checklist; on the next turn ADD the missing ones (keep your previous "
                     "code) until none remain.\n")
        diffblock = "Target instructions still missing (must all be covered, in order):\n" + \
                    "\n".join("  " + x for x in _missing_window(ent0))
        max_turns = max(max_turns, 14)         # Leiter braucht viele Runden zum Aufbauen
    elif is_large:
        incr_note = ("This is a LARGE function with mismatched regions: fix the listed regions, keep the rest "
                     "byte-for-byte identical. If a region is only register/scheduling differences, leave it.\n")
        diffblock = f"{cfg['label']} to fix (target vs draft):\n" + "\n".join(focus_diff_lines(ent0, cfg["in_focus"], cap=dcap))
    else:
        incr_note = ""
        diffblock = f"{cfg['label']} to fix (target vs draft):\n" + "\n".join(focus_diff_lines(ent0, cfg["in_focus"], cap=dcap))
    parts.append(incr_note + diffblock)
    if cfg.get("tips"):                       # diff-abhaengige Tipps (nur die vorhandenen Sub-Domaenen)
        tips = cfg["tips"](ent0)
        if tips:
            parts.append("\nHints for the mismatches present:\n" + "\n".join("- " + t for t in tips))
    if cfg.get("context"):                    # volles Diff als KONTEXT (nicht verschlechtern)
        ol = _other_diff_lines(ent0, cfg["in_focus"])
        if ol:
            parts.append("\nOTHER diffs (CONTEXT ONLY -- do NOT make these worse, other passes fix them):\n"
                         + "\n".join(ol))
    parts.append("\nUse tools if needed, then return the full corrected C.")
    messages = [{"role": "system", "content": cfg["sys"]}, {"role": "user", "content": "\n".join(parts)}]
    cur_code, cur_m, cur_f = code0, base_m, f0
    # Compile-Fail (cfail) UND semantischer Stillstand (stale) GETRENNT: einen Compile-Fehler zu fixen IST
    # Fortschritt (kein Abbruchgrund). Leiter (grosse Fkt aufbauen) braucht mehr Spielraum.
    stale = 0; cfail = 0
    cfail_max = 4 if leiter else 2
    stale_max = 3 if leiter else 2
    if _budget(messages) < 1500:        # Prompt zu gross fuer Output -> Handoff (nicht crashen)
        _p("[skip] Funktion zu gross fuer das Kontextfenster -> Handoff")
        return {"fn": fn, "focus": focus, "f0": f0, "fN": f0, "m0": base_m, "mN": base_m,
                "turns": 0, "reason_chars": [], "status": "too-large", "new_code": code0,
                "transcript": "\n".join(_buf)}

    for turn in range(1, max_turns + 1):
        turns_used = turn
        if deadline_ts and time.time() >= deadline_ts:
            _p("[deadline] Zeitbudget erschoepft -> Abbruch (Handoff)"); break
        _p(f"\n########## TURN {turn} ({focus}) ##########")
        room = _budget(messages)
        if room < 1200:
            _p("[skip] Kontext erschoepft (Konversation zu lang) -> Abbruch/Handoff"); break
        r = chat(messages, max_tokens=room)
        reason_chars.append(len(r["reasoning"]))
        _p(f"--- REASONING ({len(r['reasoning'])} chars) ---\n{r['reasoning'][:4000]}\n")
        messages.append({"role": "assistant", "content": r["content"]})
        cand = _strip_code(r["content"])
        if cand and os.environ.get("DUMP_CAND"):
            open(os.environ["DUMP_CAND"], "w", encoding="utf-8").write(cand)
        tool_lines = [l for l in (r["content"] or "").splitlines() if l.strip().upper().startswith("TOOL ")]
        if cand:
            _p(f"--- PROPOSED C ---\n{cand[:1000]}\n")
            m, ent = eval_fn(cand, fn, sp)
            if m is None:
                err = _compile_err(cand, fn)
                _p(f"[eval] COMPILE FAIL:\n{err}\n[/COMPILE FAIL]")
                cfail += 1
                if cfail >= cfail_max:
                    _p(f"[eval] {cfail_max}x Compile-Fail -> Abbruch (Handoff)"); break
                fb = (f"compile FAILED with IDO error:\n{err}\nFixes: IDO is C89 (locals at top, no mid-block decls, "
                      f"void returns nothing). M2C often declares a POINTER as s32/int -- if a value is dereferenced "
                      f"or indexed (*(x+..) / x[..]), give it a POINTER type. Return the full corrected C.")
            else:
                cfail = 0                  # kompiliert -> Compile-Fail-Strecke zurueckgesetzt
                regress = any(m[i] > base_m[i] for i in cfg["protect"])
                nf = cfg["count"](ent)
                if cfg.get("net_gate"):    # STUFE 2: nur NETTO-Metrik-Verbesserung zaehlt (Teil-Kaskade=netto-schlechter=raus)
                    improved = _mkey(m, tolerate) < _mkey(cur_m, tolerate)
                else:
                    improved = (not regress) and (nf < cur_f or _mkey(m, tolerate) < _mkey(cur_m, tolerate))
                _p(f"[eval] metric {base_m}->{m} | focus {f0}->{nf} | regress={regress} | improved={improved}")
                if improved: cur_code, cur_m, cur_f = cand, m, nf
                if nf == 0 and improved:    # Focus geloest UND uebernommen -> fertig (sonst weiterprobieren)
                    _p("[eval] *** focus-diffs == 0 (kept) -> GOAL ***"); break
                stale = 0 if improved else stale + 1
                if stale >= stale_max:
                    _p(f"[eval] {stale_max}x kein Fortschritt -> Abbruch (Handoff an Ping-Pong)"); break
                if leiter:
                    breaker = ("" if improved else "Your last code did NOT reduce the missing instructions. ADD C for "
                               "the next ones below, KEEP your previous code, do not stop early.\n")
                    fb = (breaker + f"compile OK. md={m[0]} target-instr still missing ({nf} structural diffs, was {f0}).\n"
                          + "Target instructions STILL missing (add the next ones, in order):\n"
                          + "\n".join("  " + x for x in _missing_window(ent)) + "\nReturn the FULL C so far.")
                else:
                    breaker = ("Your last attempt did NOT improve the result -- do NOT repeat it. Try a STRUCTURALLY "
                               "DIFFERENT approach (e.g. explicit local variables for each load/store, or different "
                               "argument structure / number of dereferences).\n" if not improved else "")
                    fb = (breaker + f"compile OK. {cfg['label']} remaining: {nf} (was {f0}); metric {m[:5]} "
                          f"{'OK' if not regress else 'REGRESSED protected bucket'}.\nStill:\n"
                          + "\n".join(focus_diff_lines(ent, cfg["in_focus"], cap=dcap)) + "\nRefine; tools allowed; return full C.")
            messages.append({"role": "user", "content": fb})
        elif tool_lines:
            res = []
            for tl in tool_lines[:4]:
                if tl.strip().lower().split()[:2] == ["tool", "m2c"]:
                    out = "M2C REFERENCE (reference structure, plain C):\n" + get_m2c()
                else:
                    out = run_tool(tl)
                _p(f"--- {tl.strip()} ---\n{(out or '')[:600]}\n"); res.append(f"{tl.strip()} ->\n{out}")
            messages.append({"role": "user", "content": "\n\n".join(res) + "\n\nContinue (more tools or final ```c)."})
        else:
            messages.append({"role": "user", "content": "Reply with TOOL lines or the full corrected C in a ```c block."})

    if cur_m is not None and cur_m[-1] == 0:
        status = "MATCH"
    elif cur_f == 0:
        status = "focus-solved"
    elif cur_f < f0 or (cur_m is not None and _mkey(cur_m, tolerate) < _mkey(base_m, tolerate)):
        status = "handoff"          # Fortschritt (Focus ODER Metrik gefallen, tolerate-bewusst) -> Ping-Pong
    else:
        status = "stuck"
    _p(f"\n=== RESULT {fn} ({focus}): metric {base_m}->{cur_m} | focus {f0}->{cur_f} | {status} ===")
    return {"fn": fn, "focus": focus, "f0": f0, "fN": cur_f, "m0": base_m, "mN": cur_m,
            "turns": turns_used, "reason_chars": reason_chars, "status": status, "new_code": cur_code,
            "transcript": "\n".join(_buf)}


def solve_recursive(focus, fn, sp, code0, eval_fn, max_turns=6, max_stages=3, quiet=True, deadline_ts=None):
    """REKURSIV (kein best-of-N): jede Stufe arbeitet auf dem BESTEN bisherigen Code weiter (kompoundiert
    Teilfortschritt); bei Stillstand (keine Code-Aenderung) wird die naechste Stufe FRISCH (code0) gewuerfelt
    (gegen Sample-Varianz). ABBRUCH sobald geloest/codegen-skip -> ressourcensparend. Gibt bestes Ergebnis + #Stufen.
    deadline_ts: absolute Wall-Clock-Grenze (epoch) -> keine neue Stufe danach + an run() durchgereicht."""
    best = None; warm = None; stages = 0; transcripts = []
    for s in range(max_stages):
        if deadline_ts and time.time() >= deadline_ts:
            break
        res = run(focus, fn, sp, code0, eval_fn, max_turns=max_turns, quiet=quiet, code_override=warm,
                  deadline_ts=deadline_ts)
        stages += 1
        transcripts.append(f"--- STAGE {s+1} ---\n" + res.get("transcript", ""))
        if best is None or (res["fN"], res["mN"] or (9,) * 7) < (best["fN"], best["mN"] or (9,) * 7):
            best = res
        if res["status"] in ("MATCH", "focus-solved", "codegen-skip"):
            break
        # Naechste Stufe NUR wenn DIESE Stufe verbessert hat -> warm weiter (Restluecken schliessen). Bei
        # Stillstand: KEIN Blind-Re-Roll mehr, sofort raus (Handoff) -- die frische 2. Stufe verdoppelte sonst
        # die Wall-Clock OHNE Fortschritt (siehe two_phase_pipeline: 5000s-Leerlauf). Kaskade liefert weitere Wuerfe.
        if res["new_code"] and res["fN"] < res["f0"]:
            warm = res["new_code"]
        else:
            break
    best = dict(best); best["stages"] = stages; best["transcript"] = "\n".join(transcripts)
    return best
