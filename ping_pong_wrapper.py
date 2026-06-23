"""Ping-Pong-Dispatcher: orchestriert die deterministischen Experten (+ optional Phase-0
Similar-Transplant) bis der Diff leer ist (Match) oder kein Experte mehr verbessert (Stall).

ARCHITEKTUR:
  Phase 0  Similar-Transplant (micro_permuter Rescue-Stufen) -> Match? fertig; sonst SEED.
  Phase 1  Diff-geroutetes Experten-Ping-Pong:
             jede Runde: Diff -> nach Symptom-Typen die relevanten Experten laufen lassen ->
             besten (orakel-verifiziert, niedrigster Diff) uebernehmen. Best-Stand, nie Regress.
           Gate: Diff==0 (Match) ODER keine Runde verbessert (Stall).
  Phase 2  AI-Synthese (missing-block/instr-block) -- NOCH NICHT verdrahtet (Stub).

Alles orakel-gegated ueber modules/ido_compiler (PC-Freeze-Regel). Deterministisch, kein AI.
Batch:  python3 ping_pong_wrapper.py [N] [SEED]
"""
import os, sys, json, types, glob, random, re, time, threading
from functools import lru_cache

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT); sys.path.insert(0, ROOT)
# openai-Stub (kein AI in der det. Orchestrierung)
_f = types.ModuleType("openai"); _f.OpenAI = object
class _E(Exception): pass
_f.APIConnectionError = _f.APITimeoutError = _f.APIError = _E
sys.modules.setdefault("openai", _f)

from modules import ido_compiler, ido_comparison, diff_generator as dg
from modules import (expert_immediate as e_imm, expert_memory_offset as e_mo,
                     expert_instr_width as e_iw, expert_instr_op as e_io,
                     expert_stack_frame as e_sf, expert_extra_code as e_ec,
                     expert_reorder as e_ro, expert_register_order as e_rg,
                     expert_missing_block as e_mb, expert_frankenstein as e_fk)
try:
    from modules import micro_permuter as mp
except Exception:
    mp = None


# ---------------------------------------------------------------- Orakel-Helfer
def _diff(code, fn, sp):
    """-> (total, entries) oder (None, None) bei Compile-Fehler."""
    comp = ido_compiler.compile_code(code, fn)
    if not comp.get("success"):
        ido_compiler.cleanup_temp(comp.get("temp_dir", "")); return None, None
    dg.DIFF_GROUP_THRESHOLD = 10 ** 9
    try:
        flat = json.loads(dg.create_json_diff(comp["temp_o_path"], sp, focus="all"))
    except Exception:
        ido_compiler.cleanup_temp(comp.get("temp_dir", "")); return None, None
    ido_compiler.cleanup_temp(comp.get("temp_dir", ""))
    ents = [e for e in flat if e.get("type") not in ("Summary", "Note")]
    return len(ents), ents

def _total(code, fn, sp):
    t, _ = _diff(code, fn, sp); return t


# --- missing_ops: Opcode-Klassen, die das TARGET hat und der Draft NICHT erzeugt (alignment-immun,
#     register/reorder/immediate-blind). DAS ist die EIGENE Metrik des instr-block-Experten -- nicht
#     der Gesamt-Diff. Ein korrekter Operations-Fix senkt missing_ops, auch wenn er neue Register-/
#     Reorder-Diffs (-> andere Experten, Ping-Pong) erzeugt und damit den Gesamt-Diff ERHOEHT.
def _opclass(h):
    w = int(h, 16); op = (w >> 26) & 0x3F
    if op == 0:  return f"R{w & 0x3F}"
    if op == 17: return f"C1_{(w >> 21) & 0x1F}"
    return f"O{op}"

def _missing_ops(code, fn, sp):
    """-> Anzahl Target-Opcode-Klassen, die der kompilierte Draft nicht erzeugt; None bei Compile-Fehler.
    Compile-Pflicht eingebaut: bricht der KI-Edit das Kompilieren, ist das Ergebnis None -> Gate verwirft.
    NUR fuer Messung -- alignment-BLIND (Multiset). Zum GATEN _synth_count benutzen (positionell)."""
    from collections import Counter
    comp = ido_compiler.compile_code(code, fn)
    if not comp.get("success"):
        ido_compiler.cleanup_temp(comp.get("temp_dir", "")); return None
    miss = None
    try:
        dw = ido_comparison.extract_words_from_objdump(comp["temp_o_path"])
        tw = ido_comparison.extract_words_from_original_s(sp)
        while len(dw) > 1 and dw[-1][0] == "00000000":
            dw.pop()
        dm = Counter(_opclass(h) for h, _ in dw if h and h != "00000000")
        tm = Counter(_opclass(h) for h, _ in tw if h and h != "00000000")
        miss = sum((tm - dm).values())
    except Exception:
        miss = None
    ido_compiler.cleanup_temp(comp.get("temp_dir", "")); return miss


# SYNTHESE-TYPEN: nur diese duerfen NICHT wachsen (kein Downstream-Experte synthetisiert Code).
# Alle anderen (Register/Immediate, Reordered, Memory Access, Stack, Extra in Draft, Address Load)
# haben einen Empfaenger-Experten -> Wachstum = gewollter Hand-off (Lukas' Regel + Ping-Pong).
_SYNTH_DIFF_TYPES = ("Instruction Mismatch", "Missing in Draft")

def _synth_types(code, fn, sp):
    """-> (instr_mismatch, missing_in_draft) Entry-Anzahl, oder None bei Compile-Fehler."""
    t, ents = _diff(code, fn, sp)
    if ents is None:
        return None
    im = sum(1 for e in ents if e.get("type") == "Instruction Mismatch")
    md = sum(1 for e in ents if e.get("type") == "Missing in Draft")
    return im, md

def make_synth_gate(base_code, fn, sp):
    """count_fn-Fabrik mit PER-TYP-Monotonie (Lukas' Regel: WEDER instr-block NOCH missing-block darf
    gegenueber dem Ausgangs-Draft wachsen -- kein blosses Summen-Kriterium, das Taeusche IM<->MD zulaesst).
    Liefert gate(code,fn,sp) -> Fortschritts-Skalar (im+md, kleiner=besser) ODER None (= Gate verwirft),
    wenn EINER der beiden Synthese-Typen die Ausgangszahl ueberschreitet oder Compile bricht. Register/
    Reorder/Memory/Extra/Stack/Address duerfen wachsen (Hand-off an deren Experten)."""
    base = _synth_types(base_code, fn, sp)
    bim, bmd = base if base else (10 ** 9, 10 ** 9)
    def gate(code, f, s):
        cur = _synth_types(code, f, s)
        if cur is None:
            return None                      # Compile-Fehler -> verwerfen
        im, md = cur
        if im > bim or md > bmd:
            return None                      # ein Synthese-Typ waechst -> verwerfen (Regel)
        return im + md
    return gate


# ----------------------------------------------------- Phase A: Compile-Fix (det.)
_LHS = r"([A-Za-z_]\w*(?:\[[^\]]*\])?|\([^()]*\))"
_UNK_ARROW = re.compile(_LHS + r"\s*->\s*unk([0-9A-Fa-f]+)")
_UNK_DEREF_DOT = re.compile(r"\(\s*\*\s*" + _LHS + r"\s*\)\s*\.\s*unk([0-9A-Fa-f]+)")

def _unk_rewrite(code):
    """unkNN-Feldzugriffe -> expliziter Offset-Cast (m2c-Konvention, kein Struct noetig):
       `X->unkNN`        -> `(*(s32*)((char*)(X)+0xNN))`
       `(*X).unkNN`      -> `(*(s32*)((char*)(X)+0xNN))`"""
    prev = None
    while prev != code:
        prev = code
        code = _UNK_DEREF_DOT.sub(lambda m: f"(*(s32 *)((char *)({m.group(1)}) + 0x{m.group(2)}))", code)
        code = _UNK_ARROW.sub(lambda m: f"(*(s32 *)((char *)({m.group(1)}) + 0x{m.group(2)}))", code)
    return code

def _declare_undef(code, log):
    """Undefinierte Symbole extern deklarieren. Form aus der NUTZUNG: als Zuweisungsziel/Skalar
    `extern s32 X;`, sonst (Adresse/Index) `extern u8 X[];` (lvalue-Fehler vermeiden)."""
    add = ""
    for s in set(re.findall(r"'([A-Za-z_]\w*)' undefined", log)):
        if re.search(r"\bextern\b[^;]*\b" + re.escape(s) + r"\b", code):
            continue
        rs = re.escape(s)
        scalar = re.search(rf"\b{rs}\s*=(?!=)", code) and not re.search(rf"\b{rs}\s*\[", code) \
            and not re.search(rf"&\s*{rs}\b", code)
        add += (f"extern s32 {s};\n" if scalar else f"extern u8 {s}[];\n")
    return add + code if add else code

def _dedup_cases(code):
    """Doppelte `case X:` im selben Block entfernen (haeufiges KI-Artefakt in grossen switches)."""
    seen = set(); out = []
    for ln in code.splitlines():
        m = re.match(r"\s*case\s+([0-9xXA-Fa-f]+)\s*:\s*$", ln)
        if m:
            key = m.group(1).lower()
            if key in seen:
                continue
            seen.add(key)
        out.append(ln)
    return "\n".join(out)

def compile_fix(code, fn, rounds=8):
    """Deterministische Phase A: Draft kompilierbar machen. Returns (code, ok)."""
    import modules.ido_compiler as _ic
    for _ in range(rounds):
        comp = _ic.compile_code(code, fn)
        if comp.get("success"):
            _ic.cleanup_temp(comp.get("temp_dir", "")); return code, True
        log = comp.get("error_log", ""); _ic.cleanup_temp(comp.get("temp_dir", ""))
        nc = _declare_undef(_dedup_cases(_unk_rewrite(code)), log)
        if nc == code:
            break
        code = nc
    comp = _ic.compile_code(code, fn); ok = comp.get("success")
    _ic.cleanup_temp(comp.get("temp_dir", "")); return code, ok

def _is_match(code, fn, sp):
    """Echter Byte-Match (perfect) ueber .o-vs-.o."""
    comp = ido_compiler.compile_code(code, fn)
    if not comp.get("success"):
        ido_compiler.cleanup_temp(comp.get("temp_dir", "")); return False
    o = comp["temp_o_path"]; to = ido_comparison._assemble_target_o(sp)
    ok = bool(to) and (ido_comparison.extract_words_from_objdump(o) ==
                       ido_comparison.extract_words_from_objdump(to))
    ido_compiler.cleanup_temp(comp.get("temp_dir", "")); return ok


# ------------------------------------------------- per-Experte eval_fn-Factory
# OBJEKTIV-AUSGERICHTETE Fortschrittsmetrik (tier, mm): tier = 0 perfect (mm==0) / 1 permuter_finishable
# (nur Reg-Alloc/Scheduling-Diffs -> externer Permuter schliesst es) / 2 sonst; mm = BYTE-Mismatches
# (autoritativ, == perfect-Routing). Lexikografisch kleiner = besser. DAS optimiert direkt das ROUTING:
# - Hand-off (op/width-Fix, der mm hebt) wird NUR akzeptiert, wenn er tier 1 (finishable) ERREICHT --
#   dann finisht der Permuter; erreicht er es nicht, zaehlt mm -> KEINE Byte-Regression (kein Korpus-Schaden).
# - mm-Reduktionen werden immer akzeptiert. Streng fallend -> Terminierung garantiert (LOOP-SCHUTZ).
# WICHTIG: sekundaer ist mm (BYTES), NICHT die diff_generator-Entry-Anzahl (die divergiert von Bytes:
# extra-code senkt Entries, hebt Bytes). tier subsumiert "synth-frei" (finishable = keine IM/MD-Diffs).
_SYNTH_TYPES = ("Instruction Mismatch", "Missing in Draft")

@lru_cache(maxsize=8192)
def _target_words(sp):
    """Target-Worte (Hex+Reloc) pro .s EINMAL cachen -- aendern sich nie, sparen die Objdump-Extraktion
    bei JEDEM Kandidaten-Eval (vorher der halbe Eval-Aufwand bei Tausenden Evals/Grossfunktion)."""
    tgt_o = ido_comparison._assemble_target_o(sp)
    return tuple(ido_comparison._words_with_reloc(tgt_o)) if tgt_o else None

def _eval_full(code, fn, sp):
    """EIN Compile -> ((tier, mm), all_entries). mm + finishable wie evaluate_match (Hex+Reloc-Vergleich
    gegen assembliertes Target). None-Metrik bei compile-/Vergleichs-Fehler."""
    comp = ido_compiler.compile_code(code, fn)
    if not comp.get("success"):
        ido_compiler.cleanup_temp(comp.get("temp_dir", "")); return None, []
    o = comp["temp_o_path"]; metric = None; ents = []
    try:
        dg.DIFF_GROUP_THRESHOLD = 10 ** 9
        flat = json.loads(dg.create_json_diff(o, sp, focus="all"))
        ents = [e for e in flat if e.get("type") not in ("Summary", "Note")]
        tw = _target_words(sp)
        if tw is not None:
            dw = ido_comparison._words_with_reloc(o); tw = list(tw)
            while len(dw) > len(tw) and dw and dw[-1][0] == "00000000":
                dw.pop()
            mx, mn = max(len(dw), len(tw)), min(len(dw), len(tw))
            ex = sum(1 for i in range(mn)
                     if dw[i][0] == tw[i][0] and (dw[i][1] or "") == (tw[i][1] or ""))
            mm = mx - ex
            if mm == 0:
                tier = 0
            elif ido_comparison.permuter_finishable(o, ido_comparison._assemble_target_o(sp)):
                tier = 1                                   # nur Reg/Scheduling -> Permuter finisht (_assemble cached)
            else:
                tier = 2
            metric = (tier, mm)
    except Exception:
        metric, ents = None, []
    ido_compiler.cleanup_temp(comp.get("temp_dir", ""))
    return metric, ents

# ---- SYMPTOM-PRIORITAET (Lukas 2026-06-18): Diff-Symptome nach Haerte/Root-Cause ordnen, Frame & reines
# Register ZULETZT. Metrik = (md, p1, p2, p3, p4, tier, mm). md=Vorbedingung (fehlende Instr -- Block-Fills
# muessen persistieren). p1=Root-Logik (Instruction Mismatch + falscher Immediate-WERT). p2=Struktur (Extra,
# Reordered). p3=Datenreferenz (Memory-Access-Struct-Offset + Address-Load-SYMBOL). p4=Allokation/Layout (reines
# Register + Stack-Frame + sp-Offsets). tier/mm = Tiebreak (Permuter-Routing/Byte-Naehe). So akzeptiert das Gate
# echte Root-Cause-Fixes AUCH wenn sie den Frame perturbieren (Frame zaehlt erst in p4) -> behebt "Haengen am Frame".
# Der Typ "Register/Immediate"/"Address Load" ist MEHRDEUTIG -> Operanden-Split: nicht-Register-Diff (Immediate/
# Symbol) = Root/Datenref, NUR-Register-Diff = Allokation. mm bleibt LETZTES Element -> Experten mit [-1] sicher.
_REG = re.compile(r"^\$?(zero|at|v[01]|a[0-3]|t\d|s[0-8]|k[01]|gp|sp|fp|ra|f\d+|hi|lo)$")


def _nonreg_differs(e, default=True):
    """True, wenn ein NICHT-Register-Operand (Immediate/Symbol/Wert) zwischen target/draft differiert (=Logik/
    Datenref). False, wenn NUR Register differieren (=Allokation). default, wenn keine Operanden-Details da."""
    tg, dr = str(e.get("target") or "").strip(), str(e.get("draft") or "").strip()
    if not tg or not dr or tg in ("None", "[]") or dr in ("None", "[]"):
        return default
    a = [x for x in re.split(r"[,\s]+", tg) if x]
    b = [x for x in re.split(r"[,\s]+", dr) if x]
    if len(a) != len(b):
        return True
    return any(x != y and not (_REG.match(x) and _REG.match(y)) for x, y in zip(a, b))


def _has_sp_fp(e):
    """True, wenn ein Operand sp/fp ist -> der abweichende Wert ist ein STACK-OFFSET (Frame), kein C-Literal."""
    for side in ("target", "draft"):
        s = str(e.get(side) or "")
        if any(x.strip() in ("sp", "fp", "$sp", "$fp") for x in re.split(r"[,\s]+", s)):
            return True
    return False


def _entry_prio(e):
    """Symptom-Prioritaet eines Diff-Eintrags: 1=Root-Logik, 2=Struktur, 3=Datenref, 4=Allokation/Frame; None=ignor."""
    t = e.get("type")
    if t == "Instruction Mismatch": return 1
    # sp/fp-Offset-Diff = Frame (p4), NICHT Root-Logik: ein abweichender addiu-reg,sp,OFFSET ist Stack-Layout,
    # kein editierbares C-Literal (Voll-Scan 2026-06-19: 66/476 p1-Eintraege waren sp-Offsets = aufgeblaeht).
    if t == "Register/Immediate":   return 4 if _has_sp_fp(e) else (1 if _nonreg_differs(e) else 4)
    if t in ("Extra in Draft", "Reordered"): return 2
    if t == "Memory Access":        return 3
    if t == "Address Load":         return 4 if _has_sp_fp(e) else (3 if _nonreg_differs(e) else 4)
    if t in ("Stack Frame Mismatch", "Memory Access (stack)"): return 4
    return None


def _priority_counts(ents):
    p = [0, 0, 0, 0]
    for e in ents:
        b = _entry_prio(e)
        if b:
            p[b - 1] += 1
    return tuple(p)


def _md_im(ents):
    # md zaehlt fehlende INSTRUKTIONEN (Lukas' "Luecke fuellen"-Prinzip): ein Block von 4->1 fehlende Instr ist
    # ECHTER Fortschritt. im (Instruction-Mismatch) bleibt als Hilfsgroesse (steckt jetzt in p1). Beide monoton.
    md = sum(len(e.get("missing", [])) for e in ents if e.get("type") == "Missing in Draft")
    im = sum(1 for e in ents if e.get("type") == "Instruction Mismatch")
    return md, im


def _synth_metric(m, ents):
    """(md, p1, p2, p3, p4, tier, mm) -- Symptom-PRIORITAET, Frame/Register zuletzt (p4). m=(tier,mm)."""
    md, _ = _md_im(ents)
    p1, p2, p3, p4 = _priority_counts(ents)
    return (md, p1, p2, p3, p4, m[0], m[1])


def _ev(types_):
    # Metrik = priorisiert (_synth_metric); filtert die ZURUECKGEGEBENEN Entries auf types_ (die der Experte
    # fuer seine Logik braucht), die METRIK bleibt aber voll-priorisiert (sonst Gate prioritaets-blind).
    def f(code, fn, sp):
        # ZENTRALE DET-BREMSE (mm-getriebene Kandidaten-Explosion): nach Ablauf des det-Budgets gibt jede Eval
        # None zurueck -> der Experte sieht "alle Kandidaten scheitern" -> bricht ab -> Zeit geht an die Agenten.
        # Greift NUR fuer die det-Experten (diese _ev-Closures); _EV_SYNTH/_EV_MB (Orchestrator/Agenten) unberuehrt.
        _dl = getattr(_det_tl, "deadline", 0.0)
        if _dl and time.time() > _dl:
            return None, []
        m, e = _eval_full(code, fn, sp)
        if m is None:
            return None, []
        return _synth_metric(m, e), [x for x in e if x.get("type") in types_]
    return f
_EV_RI = _ev(("Register/Immediate", "Address Load"))
_EV_MEM = _ev(("Memory Access",))
_EV_IM = _ev(("Instruction Mismatch",))
def _EV_TOT(code, fn, sp):
    return _eval_full(code, fn, sp)[0]


def _EV_SYNTH(code, fn, sp):
    """Priorisierte Orchestrator-Metrik (md, p1, p2, p3, p4, tier, mm) -- s.o. None bei compile-fail."""
    m, ents = _eval_full(code, fn, sp)
    if m is None:
        return None
    return _synth_metric(m, ents)


def _EV_MB(code, fn, sp):
    """eval_fn fuer den missing-block-Experten: priorisierte Metrik + ALLE Entries (braucht die Missing-Bloecke)."""
    m, ents = _eval_full(code, fn, sp)
    if m is None:
        return None, []
    return _synth_metric(m, ents), ents


# ---- PERMUTER-GATE (Muster wie reorder_expert verdict='permuter') ----------------
# WICHTIG (Lukas' Einwand): NUR routen, wenn KEINE permuter-HINDERNDEN Diffs da sind. Diff-TYP-Labels
# (Register/Immediate, Address Load, Memory Access (stack)) sind MEHRDEUTIG -- koennen ein Konstanten-/
# Symbol-/Offset-Diff sein, den der Permuter NICHT loesen kann. Daher INSTRUKTIONS-Level-Check
# (ido_comparison.permuter_finishable_codegen): gleiche Laenge, Reloc + register-maskierte Operanden an
# JEDER Position identisch (immediates/symbole/offsets/arg-regs exakt, nur t*/s*/f* frei), Mnemonic gleich
# ODER addu<->subu-Codegen. So zaehlt JEDER permuter-hindernde Diff (Konstante/Symbol/Offset/Missing/Extra/
# Laenge/nicht-codegen-Mnemonic) als NICHT permuter-pending.
def permuter_pending(code, fn, sp):
    """True, wenn der Rest (nach den Experten) NUR aus reiner Register-Allokation + op-Codegen (addu/subu)
    besteht -> der externe Permuter erledigt das -> struct. SICHER gegen Konstanten/Symbole/Offsets/Missing/
    Extra (= permuter-hindernd). Heuristisch (codegen) ueber der garantierten reinen Reg-Alloc."""
    comp = ido_compiler.compile_code(code, fn)
    if not comp.get("success"):
        ido_compiler.cleanup_temp(comp.get("temp_dir", "")); return False
    res = False
    try:
        tgt_o = ido_comparison._assemble_target_o(sp)
        if tgt_o:
            res = ido_comparison.permuter_finishable_codegen(comp["temp_o_path"], tgt_o)
    finally:
        ido_compiler.cleanup_temp(comp.get("temp_dir", ""))
    return res


# STUFE-2-KI fuer width/op (orchestrate setzt es aus seinem ai-Param; gleicher Call fuer alle Funktionen).
# None -> bit-identischer rein-deterministischer Pfad (no-AI-Lauf). Modul-Global: ProcessPool=per-Prozess,
# Thread=geteilt aber identischer Wert -> sicher.
_AI_CALL = None
_det_tl = threading.local()   # THREAD-LOKALE det-Deadline (epoch); pro Worker eigener Wert -> kein Clobbern unter ThreadPool
_MB_USE_M2C = True       # missing-block Schicht 2a (m2c-Ganzfunktion-Kandidat). Langsam -> abschaltbar.
_MB_CONTEXT = None       # context_c fuer m2c (Callee-Signaturen); verbessert m2c-Compile-Erfolg.
_MB_TECHENV = None       # optionaler TechEnv-Kontext im missing-mode-Prompt (Test-Parameter; kann Prompt aufblaehen).
_MB_AI_RETRIES = int(os.environ.get("MB_AI_RETRIES", "4"))  # KI-Graft-Versuche/Aufruf (v2/v3-Alternierung+Varianz); nur mit ai_call.
_MB_CONSEC = int(os.environ.get("MB_CONSEC", "3"))         # EXTRA konsekutive missing-block-Paesse bei md>0-Stall (Lukas: mehrfach/hintereinander; Varianz+Kompoundierung).
# AI-SPARER: missing-block-AI ueberspringen, wenn das "Fehlende" eine jump-table ist (Target hat jtbl_-Symbol).
# Die Sprungtabellen-Arme kann KEINE KI-Graft synthetisieren (struktureller switch->jtbl-Fall) -> AI-Call futil.
# Befund 2026-06-17: 4/25 'Missing in Draft'-Stalls sind jtbl, davon 2 mit md=56/73 = je ~12 AI-Retries auf
# grossen langsamen Funktionen = groesster Einzel-AI-Verschwender. Greift NUR mit KI (no-AI bleibt bit-identisch).
_MB_SKIP_JTBL = os.environ.get("MB_SKIP_JTBL", "1") == "1"
_JTBL_CACHE = {}                                            # sp -> bool (Target hat jump-table); Target-.s nur 1x lesen
# FRANKENSTEIN (Phase 0): Multi-Donor-Transplant aus der Fragment-DB. Self-gating
# (low-coverage/no-donor -> sofort raus, keine Evals); orakel-gegated auf OBJEKTIV
# (tier, mm) -> nimmt NUR strikte Diff-Senkungen, nie schlechter. Per FRANK_ENABLE=0
# abschaltbar; FRANK_MAX_EVALS begrenzt das Orakel-Budget des Phase-0-Vorlaufs.
_FRANK_ENABLE = os.environ.get("FRANK_ENABLE", "1") == "1"
# Frankenstein-Gate: default _EV_TOT (tier,mm). NICHT md-primaer per Default -- A/B (2026-06-17, 50 Stalls):
# md-primaer = WASH/leicht negativ (2 besser/45 gleich/3 schlechter, ALLE 5 mit hoeherem mm). Grund: Frankenstein
# laeuft in Phase 0c ZUERST als aggressives Transplant; md-Blindheit dort ist NUETZLICH (die nachgelagerte md-
# primaere Schleife holt md zurueck + erreicht niedrigeres mm). Per FRANK_MD_PRIMARY=1 testweise umschaltbar.
_FRANK_MD_PRIMARY = os.environ.get("FRANK_MD_PRIMARY", "0") == "1"
_FRANK_MAX_EVALS = int(os.environ.get("FRANK_MAX_EVALS", "250"))
_GARBAGE_ENABLE = os.environ.get("GARBAGE_ENABLE", "1") == "1"   # universeller Stall-Fallback (nur mit KI)
_GARBAGE_MAX = int(os.environ.get("GARBAGE_MAX", "3"))           # max. Garbage-Aufrufe pro Funktion (Compute-Bremse)
# STRUKTUR-AGENT (modules/agent_engine, focus=structure): agentischer Compile-Feedback-Loop NACH dem Garbage bei
# strukturellem Stall (md>0). OFF by default (heavy, multi-turn) -> per STRUCT_AGENT=1 aktivieren. Nur mit KI;
# no-AI bleibt bit-identisch (Gate _AI_CALL is not None).
_STRUCT_AGENT = os.environ.get("STRUCT_AGENT", "0") == "1"
_STRUCT_AGENT_MAX = int(os.environ.get("STRUCT_AGENT_MAX", "1"))     # max. Aufrufe/Funktion
_STRUCT_AGENT_TURNS = int(os.environ.get("STRUCT_AGENT_TURNS", "8")) # Turns/Stufe (Leiter braucht mehr)
_STRUCT_AGENT_STAGES = int(os.environ.get("STRUCT_AGENT_STAGES", "1"))  # IN-PIPELINE 1 (Kosten!); offline-Batch 3
# LOGIC-CLOSER (modules/agent_engine, focus=logic): Domaenen-Schliesser Typ/Konst/Op. Zweistufig: Stufe 1 strict
# (protect Struktur/md) -> bei Rest-Logic-Diffs Stufe 2 logic_struct (darf Struktur, net-gated, sicher). NACH dem
# Struktur-Agent, OFF by default (heavy) -> LOGIC_CLOSER=1. Nur mit KI; no-AI bleibt bit-identisch. Validiert
# offline: Stufe1 63% + Stufe2 +14pp = 77% sicher (0 Regress). Gate _EV_SYNTH<cur_m reicht (Logic-Diffs=p1/2/3
# dominieren p4); tolerate-Logik steckt intern in der Engine.
_LOGIC_CLOSER = os.environ.get("LOGIC_CLOSER", "0") == "1"
_LOGIC_MAX = int(os.environ.get("LOGIC_MAX", "1"))                   # max. Closer-Aufrufe/Funktion
_LOGIC_TURNS = int(os.environ.get("LOGIC_TURNS", "5"))              # Turns/Stufe
_LOGIC_STAGES = int(os.environ.get("LOGIC_STAGES", "2"))           # rekursive Stufen je Closer-Stufe
_LOGIC2_ENABLE = os.environ.get("LOGIC2_ENABLE", "1") == "1"       # Eskalation Stufe 2 (logic_struct) an/aus
_LOGIC_P3_MAX = int(os.environ.get("LOGIC_P3_MAX", "2"))           # Closer NICHT auf dataref-schweren Fkt feuern (p3>max)
# STUFENABHAENGIGE DEADLINE (Lukas 2026-06-20): deterministische (tool-lose) Experten bekommen ein KLEINES Budget,
# bei Ablauf KEIN Funktions-Abbruch, sondern WEITERREICHEN an die agentischen Stufen. Die tool-nutzenden Agenten
# bekommen je ein eigenes, groesseres Budget (deadline_ts=now+_AGENT_BUDGET, intern geprueft -> kein Ueberschuss).
# deadline_s bleibt der GROSSZUEGIGE Funktions-Backstop (echter Abbruch nur ganz am Ende).
_DET_BUDGET = int(os.environ.get("DET_BUDGET", "120"))            # det-Experten/Runde (Kandidaten-Explosion) -> dann an Agenten
_AGENT_BUDGET = int(os.environ.get("AGENT_BUDGET", "330"))        # je agentische Stufe (struct-agent / logic-closer)
_AGENT_DUMP = os.environ.get("AGENT_DUMP_DIR", "")               # !="" -> Agent-Transkripte (Reasoning) je Fkt/Stufe dumpen


def _dump_agent(fn, tag, txt):
    """Agent-Transkript (Reasoning+eval) der IN-PIPELINE-Calls dumpen -> stichprobenweise lesbar. Thread-safe
    (eigener Dateiname je Fkt/Stufe). Nur wenn AGENT_DUMP_DIR gesetzt."""
    if not _AGENT_DUMP or not txt:
        return
    try:
        os.makedirs(_AGENT_DUMP, exist_ok=True)
        open(f"{_AGENT_DUMP}/{fn}__{tag}.txt", "w", encoding="utf-8").write(txt)
    except Exception:
        pass
_LOOSEN = os.environ.get("ORCH_LOOSEN", "0") == "1"              # EXPERIMENT: Handoff-Trades am Stall annehmen (md gleich, mm runter)


# Experten-Runner: (Name, benoetigte Symptom-Typen, Funktion code->new_code)
def _run_imm(c, fn, sp): return e_imm.immediate_expert(c, fn, sp, eval_fn=_EV_RI)["new_code"]
def _run_mo(c, fn, sp):  return e_mo.memory_offset_expert(c, fn, sp, eval_fn=_EV_MEM)["new_code"]
def _run_iw(c, fn, sp):  return e_iw.instr_width_expert(c, fn, sp, eval_fn=_EV_IM, ai_call=_AI_CALL)["new_code"]
def _run_io(c, fn, sp):  return e_io.instr_op_expert(c, fn, sp, eval_fn=_EV_IM, ai_call=_AI_CALL)["new_code"]
# count_fn=_EV_SYNTH (md, im, tier, mm) statt _EV_TOT (tier, mm): das INTERNE Ranking dieser Experten ist sonst
# md/im-BLIND -> sie koennen einen md/im-erhoehenden Kandidaten (niedrigeres mm) als "best" zurueckgeben, den der
# Loop dann verwirft, waehrend der saubere md-erhaltende Kandidat nie zum Loop kommt. md-primaer -> sie waehlen
# den md-erhaltenden Best -> Loop akzeptiert. (extra-code "vermehrte" so md; Lukas-Befund 2026-06-16.)
def _run_sf(c, fn, sp):  return e_sf.stack_frame_expert(c, fn, sp, count_fn=_EV_SYNTH, diff_fn=_diff)["new_code"]
def _run_ec(c, fn, sp):  return e_ec.extra_code_expert(c, fn, sp, count_fn=_EV_SYNTH)["new_code"]
def _run_ro(c, fn, sp):  return e_ro.reorder_expert(c, fn, sp, count_fn=_EV_SYNTH)["new_code"]
def _run_rg(c, fn, sp):  return e_rg.register_order_expert(c, fn, sp, count_fn=_EV_SYNTH)["new_code"]
def _target_has_jtbl(sp):
    """True, wenn das Target eine jump-table hat (jtbl_-Symbol). Gecacht (Target-.s nur 1x lesen)."""
    if sp in _JTBL_CACHE:
        return _JTBL_CACHE[sp]
    try:
        r = bool(re.search(r"jtbl_[0-9A-Fa-f]+", open(sp, errors="replace").read()))
    except Exception:
        r = False
    _JTBL_CACHE[sp] = r
    return r
def _run_mb(c, fn, sp):
    # AI-SPARER: jump-table-Fehlen kann die KI nicht grafen -> AI-Call sparen (nur mit KI; no-AI unveraendert).
    if _MB_SKIP_JTBL and _AI_CALL is not None and _target_has_jtbl(sp):
        return c
    return e_mb.missing_block_expert(c, fn, sp, eval_fn=_EV_MB, ai_call=_AI_CALL, use_m2c=_MB_USE_M2C, context_c=_MB_CONTEXT, mode="missing", tech_env=_MB_TECHENV, ai_retries=_MB_AI_RETRIES)["new_code"]
# REROUTE: instr-width/op-Garbage (Instruction Mismatch, den single-instr nicht loest) -> missing-block
# im GARBAGE-Modus ("hier ist Muell, das ist der Diff, fix es"). Fallback NACH width/op in der Route.
def _run_mb_garbage(c, fn, sp): return e_mb.missing_block_expert(c, fn, sp, eval_fn=_EV_MB, ai_call=_AI_CALL, use_m2c=False, context_c=_MB_CONTEXT, mode="garbage", ai_retries=_MB_AI_RETRIES)["new_code"]

# Routing: Symptom-Typ -> relevante Experten-Runner
_ROUTING = [
    (("Register/Immediate", "Address Load"), [("immediate", _run_imm), ("register-order", _run_rg)]),
    (("Memory Access",),                     [("memory-offset", _run_mo)]),
    (("Instruction Mismatch",),              [("instr-width", _run_iw), ("instr-op", _run_io),
                                              ("mb-garbage", _run_mb_garbage)]),
    (("Missing in Draft",),                  [("missing-block", _run_mb)]),
    # stack-frame Stage 1 = gezielte Local-Transforms/Padding; Stage 2 = extra-codes ERSCHOEPFENDES Trimmen
    # (totes/redundantes Statement entfernen -> Frame schrumpft, md/im sinken). Validiert: trim findet 4/8
    # frame-Reduktionen, die der Regex verpasst (Lukas-Idee 2026-06-17). extra-code ist md-primaer gegatet.
    (("Stack Frame Mismatch", "Memory Access (stack)"), [("stack-frame", _run_sf), ("extra-code", _run_ec)]),
    (("Extra in Draft",),                    [("extra-code", _run_ec)]),
    (("Reordered",),                         [("reorder", _run_ro)]),
]


def _relevant_experts(types_present):
    seen = set(); out = []
    for trig, experts in _ROUTING:
        if set(trig) & types_present:
            for name, fn in experts:
                if name not in seen:
                    seen.add(name); out.append((name, fn))
    return out


# --------------------------------------------------------- Phase 0: Transplant
def phase0_similar(code, fn, sp, similar_ref):
    """micro_permuter Similar-Rescue-Stufen -> bester orakel-verifizierter Transplant oder code."""
    if not (mp and similar_ref):
        return code, None
    base = _EV_TOT(code, fn, sp)               # (tier, mm) statt Entry-Count
    best, best_t, best_lbl = code, base, None
    for pass_fn in (getattr(mp, "_pass_similar_transplant", None),
                    getattr(mp, "_pass_similar_unit_remap", None),
                    getattr(mp, "_pass_similar_sig_skeleton", None),
                    getattr(mp, "_pass_similar_types", None),
                    getattr(mp, "_pass_similar_decl_order", None)):
        if pass_fn is None:
            continue
        try:
            cands = pass_fn(code, func_name=fn, similar_ref=similar_ref)
        except TypeError:
            try: cands = pass_fn(code, similar_ref=similar_ref)
            except Exception: cands = []
        except Exception:
            cands = []
        for lbl, cand in (cands or []):
            t = _EV_TOT(cand, fn, sp)
            if t is not None and (best_t is None or t < best_t):
                best, best_t, best_lbl = cand, t, lbl
    return best, best_lbl


# ------------------------------------------------------- Orchestrierung (Kern)
def _resolve_deadline(deadline_s, ai):
    """KI-ABHAENGIGER Wall-Clock-Deadline (Lukas): OHNE KI schnell (ORCH_DEADLINE, Default 20s -- det-Lauf,
    Kandidaten-Explosion bremsen); MIT KI hoch (ORCH_DEADLINE_AI, Default 240s -- der missing-block-Experte
    braucht m2c + MB_AI_RETRIES + MB_CONSEC konsekutive Paesse = oft Minuten/Funktion; bei zu kurzem Deadline
    wuerde er mitten in der Arbeit abgeschnitten -> revert auf Original). Explizit uebergebener deadline_s
    gewinnt IMMER (hoechste Praezedenz)."""
    if deadline_s is not None:
        return float(deadline_s)
    if ai is not None:
        return float(os.environ.get("ORCH_DEADLINE_AI", "240"))
    return float(os.environ.get("ORCH_DEADLINE", "20"))


def orchestrate(code, fn, sp, similar_ref=None, max_rounds=14, compile_ai=None, tech_env="",
                use_m2c=True, context_c=None, deadline_s=None, ai=None, mb_m2c=True):
    """deadline_s: WALL-CLOCK-Obergrenze pro Funktion. None -> KI-abhaengig via _resolve_deadline (ohne KI
    ORCH_DEADLINE=20s, mit KI ORCH_DEADLINE_AI=240s). Bei grossen Funktionen (hohe mm -> Kandidaten-Explosion:
    1000+ Evals) bzw. langsamen KI-Experten wird sonst ein Worker blockiert. Bei Ablauf: bestes Zwischen-
    ergebnis zurueck (status 'timeout'). Bremst NICHT die Korrektheit, nur die Tiefe der Exploration.

    ai: Modell-Call fuer die STUFE-2-KI der width/op-Experten (oneshot). None -> rein deterministisch
    (no-AI-Lauf, bit-identisch). Wird ins Modul-Global _AI_CALL gesetzt (die Runner lesen es)."""
    global _AI_CALL, _MB_USE_M2C, _MB_CONTEXT, _MB_TECHENV
    deadline_s = _resolve_deadline(deadline_s, ai)   # KI-abhaengig (s.o.)
    _AI_CALL = ai
    # mb_m2c steuert NUR den missing-block-Experten (m2c-Whole-Transplant-Schicht). use_m2c steuert den
    # compile-error-Experten (Phase 0b) UNABHAENGIG -> --no-m2c (mb_m2c=False) laesst den Compile-Experten
    # m2c weiter nutzen, unterbindet aber die m2c-Volltransplantate des missing-block-Experten.
    _MB_USE_M2C = mb_m2c
    _MB_CONTEXT = context_c
    _MB_TECHENV = tech_env or None   # TechEnv in den missing-mode-Prompt (Test-Parameter); "" -> None
    log = []
    cur = code
    _t_start = time.time()
    # Phase 0b: Compiler-Error-Experte (self-contained: det -> halluc-KI iterativ -> m2c-Rescue)
    # falls Draft nicht kompiliert. m2c ist jetzt die LETZTE Experten-Stufe (nicht mehr Phase 0c).
    if _total(cur, fn, sp) is None:
        from modules import expert_compile_fix as ecf
        r = ecf.compile_error_expert(cur, fn, ai_call=compile_ai, tech_env=tech_env,
                                     target_s_path=sp, context_c=context_c, use_m2c=use_m2c)
        cur, ok = r["code"], r["ok"]
        if ok:
            log.append(("phase0b", f"compile-fix:{r['stage']}", _total(cur, fn, sp)))
        if not ok:
            return {"code": cur, "status": "compile-fail", "diffs_before": None,
                    "diffs_after": None, "match": False, "log": log + [("phase0b", "compile-fix-failed", None)]}
    _bm = _EV_SYNTH(cur, fn, sp)
    base_t = _bm[-1] if _bm else None         # Reporting in mm (Bytes); Metrik = (md,p1,p2,p3,p4,tier,mm)
    # Phase 0
    if similar_ref:
        seeded, lbl = phase0_similar(cur, fn, sp, similar_ref)
        if lbl:
            cur = seeded; log.append(("phase0", lbl, _total(cur, fn, sp)))
    # Phase 0c: FRANKENSTEIN -- Multi-Donor-Transplant aus der Fragment-DB. Self-gating
    # (lohnt nur bei union_coverage>=0.5 mit DB-Eintrag, sonst sofort raus). md-PRIMAER
    # gegated (_EV_SYNTH = md,im,tier,mm) -> nimmt nur strikte md-primaere Verbesserungen, NIE
    # schlechter. (War _EV_TOT = tier,mm = md-BLIND -> transplantierte mm-Senkungen, die md
    # ANHEBEN; gleiche Gate-Bug-Klasse wie 2026-06-16, dort fuer alle anderen Experten gefixt.
    # Direkter Beweis: bsdrone-Partial md=0 -> md=3 mit altem Gate. Per FRANK_MD_PRIMARY=0
    # auf das alte Verhalten umschaltbar.) Laeuft NACH similar. Budget via FRANK_MAX_EVALS.
    if _FRANK_ENABLE and _total(cur, fn, sp) is not None:
        try:
            _frank_ev = _EV_SYNTH if _FRANK_MD_PRIMARY else _EV_TOT
            fr = e_fk.frankenstein_expert(cur, fn, sp, _frank_ev, max_evals=_FRANK_MAX_EVALS)
            if fr.get("applied"):
                cur = fr["new_code"]
                # Label bare "frankenstein" (Konvention wie andere Experten) -> erscheint
                # in der Session-Log-chain UND in der "Experten-Nutzung"-Statistik als
                # frankenstein=N. (Nur applied wird geloggt -> verdict immer 'improved'.)
                log.append(("phase0c", "frankenstein", _total(cur, fn, sp)))
        except Exception:
            pass
    # Phase 1: Ping-Pong
    status = "stall"
    cur_m = _EV_SYNTH(cur, fn, sp)            # synth-bewusste Metrik (md, im, tier, mm)
    # EXPLORATION laeuft md-primaer (Loop akzeptiert strikt-lexikografisch -> missing-block-Fills + Ketten
    # werden uebernommen). RUECKGABE jetzt EBENFALLS md-primaer (md, tier, mm) [Lukas 2026-06-16]: ein gefuellter
    # Block (md kleiner) ist STRUKTURELL naeher am Ziel (mm=0 erfordert md=0) und der EINZIGE resume-Stand, von
    # dem die Folge-Experten (op/width/garbage/permuter) ueberhaupt weiterkommen -- md>0 = fehlender Code =
    # Sackgasse fuer alle anderen. Die alte (tier,mm)-Rueckgabe verwarf md-Fills mit mm-Anstieg (38/57 Faelle!)
    # -> ewiger Stall ohne Kompoundierung. (md, tier, mm): md zuerst (Fill persistieren), dann tier (permuter-
    # ready), dann mm (Byte-Naehe). Match=mm0 und permuter-ready=tier1 bleiben korrekt bevorzugt.
    best_obj_code, best_obj = cur, cur_m      # objektiv-bester = volle Prioritaets-Metrik (md,p1..p4,tier,mm)
    seen = set()                              # LOOP-SCHUTZ: besuchte Zustaende
    _gb_used = 0                              # Garbage-Aufrufe pro Funktion (Compute-Bremse)
    _sa_used = 0                              # Struktur-Agent-Aufrufe pro Funktion
    _lc_used = 0                              # Logic-Closer-Aufrufe pro Funktion
    _timings = {"det": 0.0, "garbage": 0.0, "struct": 0.0, "closer": 0.0}  # Wall-Clock je Stufe (Budget-Kalibrierung)
    rejects = []                              # GATE-DIAGNOSE: (name, cur_m, nm) verworfener Experten-Vorschlaege
    last_diag, last_ents = [], []             # WARUM-kein-Experte: Outcomes + Entries der letzten Runde
    for rnd in range(max_rounds):
        if cur_m is None:
            status = "compile-fail"; break
        if best_obj is None or cur_m < best_obj:   # objektiv-besten (volle Prioritaets-Metrik) Stand mitfuehren
            best_obj_code, best_obj = cur, cur_m
        if cur_m[-1] == 0:                          # mm==0 -> Match
            status = "match"; break
        if time.time() - _t_start > deadline_s:        # WALL-CLOCK-DEADLINE (Runden-Grenze)
            status = "timeout"; break
        h = hash(cur)
        if h in seen:                         # Oszillation erkannt -> abbrechen (zusaetzl. zur Metrik-Schranke)
            status = "cycle"; break
        seen.add(h)
        _, ents = _diff(cur, fn, sp)
        types_present = set(e.get("type") for e in ents)
        best_code, best_m, best_name = None, cur_m, None
        loosen_code, loosen_m, loosen_name = None, None, None   # bester HANDOFF-Vorschlag (md gleich, mm runter)
        round_diag = []                          # WARUM-kein-Experte: (name, 'rej'/'noop'/'err') pro geroutetem Experten
        _det_t0 = time.time()                     # Budget der deterministischen Experten DIESER Runde
        _det_tl.deadline = (_det_t0 + _DET_BUDGET) if _DET_BUDGET > 0 else 0.0  # <=0 = UNBEGRENZT (reine det-Passes, Phase 1/Re-Pass)
        for name, run in _relevant_experts(types_present):
            if time.time() - _t_start > deadline_s:    # FUNKTIONS-BACKSTOP (echter Abbruch, grosszuegig)
                status = "timeout"; break
            if time.time() - _det_t0 > _DET_BUDGET:    # det-Budget erschoepft -> KEIN Abbruch, an Agenten WEITERREICHEN
                break
            try:
                nc = run(cur, fn, sp); _err = False
            except Exception:
                nc = cur; _err = True
            round_diag.append((name, "err" if _err else ("rej" if (nc and nc != cur) else "noop")))
            if nc and nc != cur:
                nm = _EV_SYNTH(nc, fn, sp)
                if nm is not None and nm < best_m:   # LEXIKOGRAFISCH besser: synth_count zuerst, dann tier, mm
                    best_code, best_m, best_name = nc, nm, name
                if nm is not None and not (nm < cur_m):   # GATE-DIAGNOSE: Experte schlug etwas vor, das die
                    rejects.append((name, cur_m, nm))     # strikte Loop-Regel verwirft (nicht besser als Rundenstart)
                # HANDOFF-Kandidat (nur fuer ORCH_LOOSEN-Experiment): md HEILIG/unveraendert, aber mm sinkt
                # (im/tier duerfen steigen = Hand-off an op/width/permuter). Bester = niedrigstes mm.
                if (_LOOSEN and nm is not None and nm[0] == cur_m[0] and nm[-1] < cur_m[-1]
                        and (loosen_m is None or nm[-1] < loosen_m[-1])):
                    loosen_code, loosen_m, loosen_name = nc, nm, name
        last_diag, last_ents = round_diag, ents       # WARUM-kein-Experte: letzte Runde merken
        _det_tl.deadline = 0.0                         # Bremse aus (mb-consec/garbage/agenten unbeeinflusst)
        _timings["det"] += time.time() - _det_t0       # det-Stufe Wall-Clock (akkumuliert ueber Runden)
        if status == "timeout":
            if best_code is not None:                  # bis hierhin Bestes uebernehmen, dann Schluss
                cur, cur_m = best_code, best_m
                log.append((f"r{rnd}", best_name, best_m))
            break
        if best_code is None:
            # "STUCK ist relativ" (Lukas): missing-block ist VARIANZ-behaftet + KOMPOUNDIEREND. Solange noch
            # md>0 (fehlende Bloecke) UND KI da, NICHT sofort stallen -> bis zu _MB_CONSEC EXTRA KONSEKUTIVE
            # missing-block-Paesse (frische Samples; v2/v3-Alternierung intern). Jede lexikografische (md-
            # primaere) Verbesserung wird uebernommen, dann normale Runde fortsetzen.
            if (_AI_CALL is not None and cur_m is not None and cur_m[0] > 0
                    and not (_MB_SKIP_JTBL and _target_has_jtbl(sp))):   # jtbl: AI-Call futil -> sparen
                got = False
                for _k in range(_MB_CONSEC):
                    if time.time() - _t_start > deadline_s:
                        status = "timeout"; break
                    try:
                        nc = _run_mb(cur, fn, sp)
                    except Exception:
                        nc = cur
                    if nc and nc != cur:
                        nm = _EV_SYNTH(nc, fn, sp)
                        if nm is not None and nm < cur_m:
                            cur, cur_m = nc, nm
                            log.append((f"r{rnd}", "mb-consec", cur_m))
                            got = True; break
                if status == "timeout":
                    break
                if got:
                    continue                  # neue Runde mit verbessertem Stand
            # GATE-LOCKERUNG (ORCH_LOOSEN, experimentell): kein strikter Fortschritt, aber ein Experte bot einen
            # HANDOFF an (md unveraendert, mm runter, im/tier evtl. hoch) -> annehmen + Ping-Pong (op/width/permuter
            # bereinigen die weicheren Diffs). Loop-Schutz (seen/max_rounds/deadline) + objektiver Return sichern ab.
            if _LOOSEN and loosen_code is not None:
                cur, cur_m = loosen_code, loosen_m
                log.append((f"r{rnd}", f"loosen:{loosen_name}", cur_m))
                continue
            # GARBAGE-EXPERTE (universeller Stall-Fallback, Ping-Pong-Philosophie): kein anderer Experte zuendet
            # und mm>0 -> EIN anker-gestuetzter KI-Versuch (3 Wege min/rw/asm + Hunk-Mix), der die Sackgasse in
            # einen DETERMINISTISCH-loesbaren Zustand eintauscht (Escape-Gate = orchestrate(ai=None) intern;
            # keine Rekursion, da Garbage nur bei _AI_CALL!=None zuendet). Akzeptiert auch eine kurze Metrik-
            # Talfahrt, SOLANGE der Downstream-Zustand besser als der Eingang ist. Danach NORMALE Runde -> die
            # anderen Experten greifen wieder (Ping-Pong); Multi-Spot loest sich durch wiederholtes Zuenden.
            if (_GARBAGE_ENABLE and _AI_CALL is not None and cur_m is not None and cur_m[-1]
                    and _gb_used < _GARBAGE_MAX and time.time() - _t_start <= deadline_s):
                _gb_used += 1
                _gt = time.time()
                # resolve ruft orchestrate(ai=None) VERSCHACHTELT -> ueberschreibt die Modul-Globals
                # (_AI_CALL->None etc.). Sichern + nach dem Garbage-Aufruf wiederherstellen, sonst sind im
                # aeusseren Loop danach die KI-Experten aus.
                _save = (_AI_CALL, _MB_USE_M2C, _MB_CONTEXT, _MB_TECHENV)
                try:
                    from modules import expert_garbage as _egb
                    from modules import expert_missing_block as _emb_cr
                    _cr = lambda c: _emb_cr._compile_repair(c, fn, sp, _EV_SYNTH)
                    _rs = lambda c: orchestrate(c, fn, sp, ai=None, mb_m2c=_save[1],
                                                context_c=_save[2], tech_env=_save[3] or "")["code"]
                    ngc = _egb.garbage_expert(cur, fn, sp, _save[0], lambda c: _EV_SYNTH(c, fn, sp),
                                              _diff, resolve=_rs, compile_repair=_cr)
                except Exception:
                    ngc = None
                finally:
                    _AI_CALL, _MB_USE_M2C, _MB_CONTEXT, _MB_TECHENV = _save   # Globals wiederherstellen
                _timings["garbage"] += time.time() - _gt
                if ngc and ngc != cur:
                    nm = _EV_SYNTH(ngc, fn, sp)
                    if nm is not None:
                        cur, cur_m = ngc, nm
                        log.append((f"r{rnd}", "garbage", cur_m))
                        continue              # Ping-Pong: andere Experten erneut auf den neuen Stand
            # STRUKTUR-AGENT (agentischer Compile-Feedback-Loop) NACH dem Garbage: struktureller Stall (md>0),
            # AI an, OFF by default (STRUCT_AGENT=1). Akzeptiert nur bei lexikografischer Verbesserung (md-primaer);
            # Struktur-Fix darf Frame/Register perturbieren (Handoff). Danach Ping-Pong (continue).
            if (_STRUCT_AGENT and _AI_CALL is not None and cur_m is not None and cur_m[0] > 0
                    and _sa_used < _STRUCT_AGENT_MAX and time.time() - _t_start <= deadline_s):
                _sa_used += 1
                _st = time.time()
                try:
                    from modules import agent_engine as _ae
                    _r = _ae.solve_recursive("structure", fn, sp, cur, _EV_MB,
                                             max_turns=_STRUCT_AGENT_TURNS, max_stages=_STRUCT_AGENT_STAGES,
                                             quiet=True, deadline_ts=time.time() + _AGENT_BUDGET)
                    nsc = _r.get("new_code")
                    _dump_agent(fn, "struct", _r.get("transcript", ""))
                except Exception:
                    nsc = None
                _timings["struct"] += time.time() - _st
                if nsc and nsc != cur:
                    nm = _EV_SYNTH(nsc, fn, sp)
                    if nm is not None and nm < cur_m:
                        cur, cur_m = nsc, nm
                        log.append((f"r{rnd}", "struct-agent", cur_m))
                        continue
            # LOGIC-CLOSER (Domaenen-Schliesser Typ/Konst/Op) NACH dem Struktur-Agent: feuert bei offenen Logic-
            # Diffs. Stufe 1 strict (protect Struktur/md); raeumt sie die Logic-Diffs NICHT, eskaliert Stufe 2
            # (logic_struct, darf Struktur, net-gated=sicher). Akzeptiert nur bei lexikografischer Verbesserung.
            # GATE (kalibriert): nur wenn Struktur fertig (md==0) UND nicht dataref-schwer (p3<=max) -- sonst sind
            # die "Logic"-Diffs (oft Memory-Access-Offsets) gekoppelt -> Closer verbrennt ~600s ohne Gewinn.
            if (_LOGIC_CLOSER and _AI_CALL is not None and cur_m is not None
                    and cur_m[0] == 0 and cur_m[3] <= _LOGIC_P3_MAX
                    and _lc_used < _LOGIC_MAX and time.time() - _t_start <= deadline_s):
                nlc = None
                _ct = time.time()
                try:
                    from modules import agent_engine as _ae
                    _, _lent = _diff(cur, fn, sp)
                    if _ae._logic_count(_lent) > 0:        # nur feuern, wenn es Typ/Konst/Op-Diffs gibt (AI-spar)
                        _lc_used += 1
                        _dts = time.time() + _AGENT_BUDGET
                        _r = _ae.solve_recursive("logic", fn, sp, cur, _EV_MB,
                                                 max_turns=_LOGIC_TURNS, max_stages=_LOGIC_STAGES, quiet=True,
                                                 deadline_ts=_dts)
                        nlc = _r.get("new_code") or cur
                        _dump_agent(fn, "logic", _r.get("transcript", ""))
                        # Stufe 2 (Eskalation) NUR wenn Stufe 1 Logic-Diffs UEBRIG laesst (gekoppelte Faelle)
                        if _LOGIC2_ENABLE and (time.time() < _dts):
                            _, _lent2 = _diff(nlc, fn, sp)
                            if _ae._logic_count(_lent2) > 0:
                                base2 = nlc if _EV_SYNTH(nlc, fn, sp) and _EV_SYNTH(nlc, fn, sp) <= cur_m else cur
                                _r2 = _ae.solve_recursive("logic_struct", fn, sp, base2, _EV_MB,
                                                          max_turns=_LOGIC_TURNS, max_stages=_LOGIC_STAGES,
                                                          quiet=True, deadline_ts=_dts)
                                nlc = _r2.get("new_code") or nlc
                                _dump_agent(fn, "logic2", _r2.get("transcript", ""))
                except Exception:
                    nlc = None
                _timings["closer"] += time.time() - _ct
                if nlc and nlc != cur:
                    nm = _EV_SYNTH(nlc, fn, sp)
                    if nm is not None and nm < cur_m:
                        cur, cur_m = nlc, nm
                        log.append((f"r{rnd}", "logic-closer", cur_m))
                        continue
            status = "stall"; break          # Gate: keine lexikografische Verbesserung (Hand-off erlaubt)
        cur, cur_m = best_code, best_m        # synth_count faellt STRENG -> Terminierung garantiert
        log.append((f"r{rnd}", best_name, best_m))
    # Rueckgabe = OBJEKTIV-bester (tier, mm) Stand (Match/permuter-ready/niedrigstes mm). Letzten cur noch
    # gegenpruefen (falls die letzte Runde objektiv besser war als das bisher beste).
    _fin = _EV_SYNTH(cur, fn, sp)
    if _fin is not None and (best_obj is None or _fin < best_obj):   # volle Prioritaets-Metrik
        best_obj_code, best_obj = cur, _fin
    cur = best_obj_code
    final_m = _EV_SYNTH(cur, fn, sp)
    final_t = final_m[-1] if final_m else None        # mm (Bytes)
    final_tier = final_m[-2] if final_m else None     # 0 perfect / 1 permuter-finishable / 2 sonst
    # PERMUTER-READY (analog width/reorder verdict='permuter'): kein voller Match (mm>0), aber der Rest ist
    # NUR reine Register-Allokation/Codegen -> der externe Permuter finisht -> als struct sortierbar (in
    # 0_wrapper). Genau "nur noch dieser Fehler" (Lukas): tier==1.
    permuter_ready = (final_t is not None and final_t > 0 and final_tier == 1)
    # WARUM-kein-Experte: Bottleneck-Klasse des RUECKGABE-Stands + Outcomes der gerouteten Experten der letzten
    # Runde (rej=Vorschlag verworfen, noop=nichts gefunden, err=Exception). 'none-routed' = kein Experte fuer
    # die Diff-Typen. Nur fuer nicht-Match (Bottleneck-Analyse), guenstig (eine Klassifikation am Ende).
    stall_reason = None
    if final_t not in (0, None):
        try:
            from modules import expert_garbage as _egb
            _, fent = _diff(cur, fn, sp)
            rc = _egb.residual_class(sp, fent)
        except Exception:
            rc = "?"
        routed = ",".join(f"{n}={o}" for n, o in last_diag) or "none-routed"
        stall_reason = f"{rc} | routed: {routed}"
    return {"code": cur, "status": status, "diffs_before": base_t, "diffs_after": final_t,
            "match": (final_t == 0), "tier": final_tier, "permuter_ready": permuter_ready, "log": log,
            "rejects": rejects, "stall_reason": stall_reason, "timings": {k: round(v, 1) for k, v in _timings.items()}}


# ---------------------------------------------------------------------- Batch
def _batch(n, seed):
    idx = {}
    for r, _d, fs in os.walk("data/target_asm"):
        for x in fs:
            if x.endswith(".s"): idx[x[:-2]] = os.path.join(r, x)
    names = list(idx.keys()); random.seed(seed); random.shuffle(names)
    checked = matched = improved = stalled = cfail = 0
    big = []
    for nm in names:
        if checked >= n: break
        cps = glob.glob(f"data/c_input/**/{nm}.c", recursive=True)
        if not cps: continue
        code = open(cps[0], encoding="utf-8", errors="replace").read()
        b = _total(code, nm, idx[nm])
        if b is None or b == 0: continue
        checked += 1
        res = orchestrate(code, nm, idx[nm])
        if res["match"]: matched += 1
        elif res["diffs_after"] is not None and res["diffs_after"] < b:
            improved += 1
            if b - (res["diffs_after"] or 0) >= 4: big.append((nm, b, res["diffs_after"], res["log"][-3:]))
        elif res["status"] == "compile-fail": cfail += 1
        else: stalled += 1
    print(f"\n=== Ping-Pong Batch: {checked} Funktionen ===")
    print(f"  MATCH (Diff->0):     {matched} ({100*matched//max(checked,1)}%)")
    print(f"  verbessert (Diff<):  {improved} ({100*improved//max(checked,1)}%)")
    print(f"  Stall (unmatched):   {stalled}  <- AI/Synthese-Kandidaten")
    print(f"  Compile-Fail:        {cfail}")
    for nm, b, a, lg in big[:12]:
        print(f"    {nm}: {b}->{a}  {lg}")


# ---------------------------------------------------------------- SPEICHER-BATCH (Stufe 4: Integration)
# Vollstaendiger, SPEICHERNDER Lauf der Ping-Pong-Pipeline (im Ggs. zu _batch = nur Statistik). Routing in
# experten/-Unterordner: match (mm=0) -> perfect_matches | permuter_ready (tier-1, nur Reg-Alloc-Rest) ->
# struct_matches | sonst Fortschritt -> partial_matches (Seed fuer den naechsten Lauf, Kompoundierung).
# ai=None -> rein deterministisch (bit-identisch). Output-Dir isolierbar (outroot) fuer sicheres Testen.
_SUB = "experten"
_BATCH_AI = None
_BATCH_SIDX = {}
_BATCH_DIDX = {}
_BATCH_SIM = None
_BATCH_OUTROOT = "output"
_BATCH_MB_M2C = True            # missing-block m2c-Whole-Transplant-Schicht (per --no-m2c abschaltbar)


def _src_indices():
    sidx, didx = {}, {}
    for r, _d, fs in os.walk("data/target_asm"):
        for x in fs:
            if x.endswith(".s"): sidx[x[:-2]] = os.path.join(r, x)
    for r, _d, fs in os.walk("data/c_input"):
        for x in fs:
            if x.endswith(".c"): didx[x[:-2]] = os.path.join(r, x)
    return sidx, didx


def _relpath_for(fn):
    if fn in _BATCH_SIDX:
        return os.path.relpath(os.path.dirname(_BATCH_SIDX[fn]), "data/target_asm")
    if fn in _BATCH_DIDX:
        return os.path.relpath(os.path.dirname(_BATCH_DIDX[fn]), "data/c_input")
    return ""


def _outpath(kind, fn, rel):
    from pathlib import Path
    return Path(_BATCH_OUTROOT) / kind / _SUB / rel / f"{fn}.c"


def _clean_m2c_safe(code, fn):
    """SAVE-BOUNDARY-FILTER: strippt M2C-Header/Makros (clean_m2c) bevor IRGENDWAS gespeichert wird. Der
    M2C-Header ist FATAL fuer die Similar-Pipeline (keine #define/Kommentare) -> darf NIE in den Daten landen.
    Ein Choke-Point deckt ALLE Leck-Pfade ab (compile-fix-m2c-Rescue, Passthrough von m2c-c_input-Drafts, ...).
    ORAKEL-GEGATET + NO-REGRESS: nur saeubern, wenn die gesaeuberte Version dieselbe Metrik kompiliert (clean_m2c
    ist semantik-erhaltend -> .o identisch). Bei Regress/uncleanable: Original behalten (sehr selten). Determi-
    nistisch -> no-AI bleibt reproduzierbar."""
    if "M2C_" not in code:
        return code
    try:
        from modules import m2c_donor
        cleaned = m2c_donor.clean_m2c(code)
        if cleaned is None:
            return code
        sidx = globals().get("_BATCH_SIDX") or {}
        sp = sidx.get(fn) or next(iter(glob.glob(f"data/target_asm/**/{fn}.s", recursive=True)), None)
        if sp and _EV_SYNTH(cleaned, fn, sp) == _EV_SYNTH(code, fn, sp):
            return cleaned
        return code
    except Exception:
        return code


def _save_result(kind, fn, code, rel):
    code = _clean_m2c_safe(code, fn)                 # M2C-Header NIE speichern (Similar-fatal), no-regress
    p = _outpath(kind, fn, rel); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(code, encoding="utf-8")


def _instr_count(sp):
    try:
        return sum(1 for l in open(sp, encoding="utf-8", errors="replace") if l.lstrip().startswith("/*"))
    except Exception:
        return 0


def _size_bucket(sz):
    return "<25" if sz < 25 else "25-49" if sz < 50 else "50-99" if sz < 100 else ">=100"


def _batch_work(fn):
    sp = _BATCH_SIDX[fn]; rel = _relpath_for(fn)
    seed = _outpath("partial_matches", fn, rel)              # Kompoundierung: bester bisheriger Stand
    src = str(seed) if seed.exists() else _BATCH_DIDX.get(fn)
    if not src or not os.path.exists(src):
        return {"fn": fn, "result": "no-src"}
    code = open(src, encoding="utf-8", errors="replace").read()
    sref = _BATCH_SIM.get(fn) if _BATCH_SIM else None
    try:
        r = orchestrate(code, fn, sp, similar_ref=sref, ai=_BATCH_AI, mb_m2c=_BATCH_MB_M2C)
    except Exception as e:
        return {"fn": fn, "result": "error", "err": repr(e)[:100]}
    best = r["code"]
    if r.get("match"):
        kind = "perfect_matches"; res = "perfect"
    elif r.get("permuter_ready"):
        kind = "struct_matches"; res = "struct"
    else:
        kind = "partial_matches"; res = "partial"
    _save_result(kind, fn, best, rel)
    if res != "partial" and seed.exists():                    # Seed entfernen, wenn promoted (kein Stale)
        try: seed.unlink()
        except OSError: pass
    # chain = Experten-Kette (in->out): (Experte, Ergebnis-Metrik) pro angewandtem Schritt -> Session-Log
    chain = [(lg[1], lg[2]) for lg in r.get("log", [])]
    return {"fn": fn, "result": res, "sz": _instr_count(sp), "tier": r.get("tier"),
            "before": r.get("diffs_before"), "after": r.get("diffs_after"), "status": r.get("status"),
            "chain": chain, "rejects": r.get("rejects", []), "stall_reason": r.get("stall_reason")}


def _strip_code(s):
    """KI-Antwort -> reiner C-Code (Markdown-Fences + <think>-Bloecke entfernen)."""
    if not s:
        return s
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    s = re.sub(r"^.*?</think>", "", s, flags=re.S)
    m = re.search(r"```(?:c|C)?\s*\n(.*?)```", s, flags=re.S)
    if m:
        s = m.group(1)
    return s.strip()


def make_pod_ai():
    """Self-contained Oneshot-KI-Call (urllib, KEIN openai-Dep) aus den TERMINAL-Env-Vars LOCAL_API_BASE /
    LOCAL_API_KEY (+ optional LOCAL_API_MODEL). Gibt None zurueck, wenn LOCAL_API_BASE nicht gesetzt ist ->
    dann laeuft die Pipeline deterministisch (no-AI). So aktiviert man den Pod per `export LOCAL_API_BASE=...`."""
    base = os.environ.get("LOCAL_API_BASE", "").rstrip("/")
    if not base:
        return None
    import urllib.request
    key = os.environ.get("LOCAL_API_KEY", "EMPTY")
    model = os.environ.get("LOCAL_API_MODEL", "Qwen/Qwen3.6-35B-A3B")

    def ai(system, user, think=False):
        body = {"model": model, "temperature": 0.2, "max_tokens": 8192,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if not think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        data = json.dumps(body).encode()
        last = None
        for _ in range(3):
            try:
                req = urllib.request.Request(base + "/chat/completions", data=data,
                                             headers={"Content-Type": "application/json",
                                                      "Authorization": f"Bearer {key}", "User-Agent": "curl/8.0"})
                with urllib.request.urlopen(req, timeout=300) as resp:
                    d = json.loads(resp.read())
                return _strip_code(d["choices"][0]["message"].get("content") or "")
            except Exception as e:
                last = e; time.sleep(2)
        raise last
    return ai


def run_batch(limit=0, ai=None, workers=None, use_similar=True, outroot="output", shuffle_seed=None,
              no_m2c=False):
    """Speichernder Voll-/Teil-Lauf. ai=None -> deterministisch (bit-identisch). Routet match/permuter_ready/
    sonst nach perfect/struct/partial (experten/-Unterordner). Resume: ueberspringt bereits geloeste
    (perfect+struct). no_m2c -> missing-block-Experte OHNE m2c-Whole-Transplant (compile-Experte unberuehrt).
    Schreibt ein Session-Log (Funktion + Experten-Kette). ProcessPool (fork). Gibt (counts, results) zurueck."""
    global _BATCH_AI, _BATCH_SIDX, _BATCH_DIDX, _BATCH_SIM, _BATCH_OUTROOT, _BATCH_MB_M2C
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from collections import Counter
    from pathlib import Path
    _BATCH_SIDX, _BATCH_DIDX = _src_indices()
    _BATCH_AI = ai
    _BATCH_OUTROOT = outroot
    _BATCH_MB_M2C = not no_m2c
    _BATCH_SIM = None
    if use_similar:
        try:
            from modules import cheatsheet, mother_similar
            db = cheatsheet.load_similar_db(); thr = mother_similar.SIMILAR_SCORE_THRESHOLD
            _BATCH_SIM = {f: e.get("c_code") for f, e in db.items()
                          if e and e.get("struct_score", 0) >= thr and e.get("c_code")}
        except Exception:
            _BATCH_SIM = None
    done = set()
    for kind in ("perfect_matches", "struct_matches"):
        d = Path(outroot) / kind
        if d.exists():
            done |= {p.stem for p in d.rglob("*.c")}
    todo = []
    for fn in _BATCH_SIDX:
        if fn in done:
            continue
        if fn in _BATCH_DIDX or _outpath("partial_matches", fn, _relpath_for(fn)).exists():
            todo.append(fn)
    todo.sort()
    if shuffle_seed is not None:
        random.seed(shuffle_seed); random.shuffle(todo)
    if limit:
        todo = todo[:limit]
    workers = workers or int(os.environ.get("MATCH_WORKERS", "8"))
    print(f"run_batch: todo {len(todo)} | bereits geloest(perfect+struct) {len(done)} | "
          f"ai={'JA' if ai else 'NEIN'} | similar={'JA' if _BATCH_SIM else 'NEIN'} | workers {workers} | "
          f"out {outroot}/", flush=True)
    counts = Counter(); bysize = {}; results = []; chain_use = Counter()
    reject_use = Counter(); reject_handoff = Counter()    # GATE-DIAGNOSE: verworfene Experten-Vorschlaege
    stall_use = Counter(); expert_err = Counter()         # BOTTLENECK-Klassen + Experten-Exceptions (err)
    total = len(todo); t0 = time.time(); last = 0.0; err_shown = 0
    tty = sys.stdout.isatty(); send = "\r" if tty else "\n"; iv = 1.5 if tty else 20.0
    # SESSION-LOG: pro Funktion die Experten-Kette (in->out). Liegt unter <outroot>/logs/.
    import datetime as _dt
    _logdir = Path(outroot) / "logs"; _logdir.mkdir(parents=True, exist_ok=True)
    _logpath = _logdir / f"ppw_{_dt.datetime.now():%Y%m%d_%H%M%S}.log"
    _logf = open(_logpath, "w", encoding="utf-8")
    _logf.write(f"# Ping-Pong Session | todo {total} | ai={'JA' if ai else 'NEIN'} | no_m2c={no_m2c} | "
                f"workers {workers} | out {outroot}/\n# fn | sz | result | mm before->after | tier | "
                f"chain (Experte=Metrik, in->out pro Schritt)\n")
    _logf.flush()
    print(f"Session-Log: {_logpath}", flush=True)

    def _hms(s):
        s = int(s)
        return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"

    def _logline(d):
        if d["result"] in ("error", "no-src"):
            return f"{d['fn']} | {d['result'].upper()} | {d.get('err', '')}"
        ch = " -> ".join(f"{nm}={m}" for nm, m in d.get("chain", []))
        sr = d.get("stall_reason")
        # WARUM-kein-Experte: leere Kette -> Bottleneck-Klasse + geroutete Experten-Outcomes; sonst Kette + Rest-
        # Bottleneck anhaengen (zeigt, woran der partial-Stand am Ende haengt).
        if not ch:
            ch = f"STALL[{sr}]" if sr else "(kein Experte griff)"
        elif d["result"] == "partial" and sr:
            ch = f"{ch}  || STALL[{sr}]"
        return (f"{d['fn']} | {d.get('sz', '?')}i | {d['result']} | mm {d.get('before')}->{d.get('after')} "
                f"| tier{d.get('tier')} | {ch}")

    def _status(n):
        el = time.time() - t0; rate = n / el if el > 0 else 0
        eta = _hms((total - n) / rate) if rate > 0 and n < total else "0m00s"
        nerr = counts["error"] + counts.get("no-src", 0)
        print(f"[{n}/{total}] {rate:4.1f}/s ETA {eta} | perfect {counts['perfect']} struct {counts['struct']} "
              f"partial {counts['partial']} err {nerr}   ", end=send, flush=True)

    ctx = multiprocessing.get_context("fork")
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futs = {ex.submit(_batch_work, fn): fn for fn in todo}
            for fu in as_completed(futs):
                try: d = fu.result()
                except Exception as e: d = {"fn": futs[fu], "result": "error", "err": repr(e)[:80]}
                counts[d["result"]] += 1; results.append(d)
                for _nm, _m in d.get("chain", []):
                    chain_use[_nm] += 1
                _sr = d.get("stall_reason")                 # Bottleneck-Klasse (vor ' | routed') tallyen
                if _sr:
                    stall_use[_sr.split(" | ")[0]] += 1
                    for _nm, _o in (seg.split("=") for seg in _sr.split("routed:")[-1].split(",") if "=" in seg):
                        if _o.strip() == "err":
                            expert_err[_nm.strip()] += 1
                _logf.write(_logline(d) + "\n")
                _rj = d.get("rejects", [])                 # GATE-DIAGNOSE-Zeile: verworfene Vorschlaege je Funktion
                if _rj:
                    parts = []
                    for nm_, cur_, got_ in _rj:
                        reject_use[nm_] += 1
                        # handoff-foermig (KANDIDAT fuer lockereres Gate): md (heilig, Index 0) UNVERAENDERT,
                        # aber eine weichere Komponente (im/tier/mm) verbessert -> reiner Soft-Trade.
                        if (cur_ and got_ and got_[0] == cur_[0]
                                and any(got_[i] < cur_[i] for i in range(1, len(cur_)))):
                            reject_handoff[nm_] += 1
                        parts.append(f"{nm_}:{cur_}>{got_}")
                    _logf.write(f"# REJECT {d['fn']} | " + " | ".join(parts) + "\n")
                _logf.flush()
                if d.get("sz") is not None:
                    key = (_size_bucket(d["sz"]), d["result"]); bysize[key] = bysize.get(key, 0) + 1
                if d["result"] in ("error", "no-src") and err_shown < 12:   # Fehler KONKRET, aber gedeckelt
                    print(f"{'  ' if tty else ''}! {d['result']} {d['fn']}: {d.get('err', '')}", flush=True)
                    err_shown += 1
                    if err_shown == 12:
                        print("! (weitere Fehler werden nur noch gezaehlt)", flush=True)
                n = sum(counts.values())
                now = time.time()
                if now - last >= iv or n == total:        # ZEIT-gedrosselt -> kein Zumuellen, ETA live
                    last = now; _status(n)
    except KeyboardInterrupt:
        print("\n[STOP] Abbruch -- gespeicherte .c sind gueltig, Lauf resumable.", flush=True)
    if tty:
        print(flush=True)                                  # \r-Zeile sauber abschliessen
    print(f"==== run_batch fertig in {_hms(time.time() - t0)}: {dict(counts)} ====", flush=True)
    _errs = [d for d in results if d["result"] in ("error", "no-src")]
    if _errs:
        print(f"Fehler gesamt: {len(_errs)} (erste {min(5, len(_errs))}):", flush=True)
        for d in _errs[:5]:
            print(f"  {d['fn']}: {d.get('err', '')}", flush=True)
    # Groessen-Auswertung: ueberwindet die Pipeline die <50-Instr-Schwaeche der ersten Pipeline?
    print("Routing nach Funktionsgroesse (Instruktionen):", flush=True)
    print(f"  {'Bucket':8} {'perfect':>8} {'struct':>8} {'partial':>8}", flush=True)
    for b in ("<25", "25-49", "50-99", ">=100"):
        row = [bysize.get((b, k), 0) for k in ("perfect", "struct", "partial")]
        if any(row) or b in ("50-99", ">=100"):
            print(f"  {b:8} {row[0]:>8} {row[1]:>8} {row[2]:>8}", flush=True)
    _logf.write(f"\n# Routing: {dict(counts)}\n# Experten-Nutzung (Schritte gesamt): {dict(chain_use)}\n")
    # GATE-DIAGNOSE-Footer: pro Experte verworfene Vorschlaege + davon handoff-foermige (niedrigere
    # Komponente verbessert, hoehere verschlechtert) -> Kandidaten fuer GELOCKERTE Gate-Regeln.
    _logf.write(f"# Gate-Rejects (Vorschlag verworfen): {dict(reject_use)}\n")
    _logf.write(f"# Gate-Rejects HANDOFF-foermig (Trade -> evtl. lockern): {dict(reject_handoff)}\n")
    # BOTTLENECK-Histogramm: welche Residual-Klasse blockiert wie viele partials (woran haengt der Lauf) +
    # Experten, die mit Exception (err) abbrechen (latente Bugs).
    _logf.write(f"# Bottlenecks (partial-Residual-Klassen): {dict(stall_use.most_common())}\n")
    _logf.write(f"# Experten-Errors (Exception): {dict(expert_err)}\n")
    _logf.close()
    print(f"Experten-Nutzung: {dict(chain_use)}", flush=True)
    if reject_use:
        print(f"Gate-Rejects: {dict(reject_use)}", flush=True)
        print(f"  davon HANDOFF-foermig (Kandidaten fuer lockerere Gates): {dict(reject_handoff)}", flush=True)
    if stall_use:
        print(f"BOTTLENECKS (partial-Klassen): {dict(stall_use.most_common())}", flush=True)
    if expert_err:
        print(f"!! Experten-Errors (Exception): {dict(expert_err)}", flush=True)
    print(f">>> Session-Log gespeichert: {_logpath}", flush=True)
    return counts, results


if __name__ == "__main__":
    # SPEICHERNDER Lauf ist DEFAULT: python3 ping_pong_wrapper.py [limit] [-j N] [--no-ai]
    #   KI-Lauf = DEFAULT, sobald LOCAL_API_BASE im Terminal gesetzt ist (export LOCAL_API_BASE=...,
    #   LOCAL_API_KEY=...). --no-ai erzwingt deterministisch. -j N = Worker (ProcessPool). limit optional.
    #   'save' wird als Alias akzeptiert (Rueckwaerts-Kompat). Alter Benchmark: `... bench [N] [SEED]`.
    args = sys.argv[1:]
    if args and args[0] == "bench":
        N = int(args[1]) if len(args) > 1 else 120
        SEED = int(args[2]) if len(args) > 2 else 2
        _batch(N, SEED)
    else:
        if args and args[0] == "save":          # optionales Alias
            args = args[1:]
        no_ai = "--no-ai" in args
        no_m2c = "--no-m2c" in args             # NUR missing-block-Experte ohne m2c-Whole-Transplant
        args = [a for a in args if a not in ("--no-ai", "--no-m2c")]
        workers = None
        if "-j" in args:
            i = args.index("-j"); workers = int(args[i + 1]); del args[i:i + 2]
        limit = int(args[0]) if args and args[0].lstrip("-").isdigit() else 0
        outroot = os.environ.get("PPW_OUTROOT", "output")
        ai = None if no_ai else make_pod_ai()
        if no_ai:
            print(">>> Modus: --no-ai (deterministisch, bit-identisch)", flush=True)
        elif ai is not None:
            print(f">>> Modus: KI-Lauf ueber LOCAL_API_BASE={os.environ.get('LOCAL_API_BASE')}", flush=True)
        else:
            print(">>> LOCAL_API_BASE nicht gesetzt -> deterministischer Lauf. Fuer KI: "
                  "export LOCAL_API_BASE=... (+ LOCAL_API_KEY); oder --no-ai zum bewussten Erzwingen.", flush=True)
        if no_m2c:
            print(">>> --no-m2c: missing-block-Experte OHNE m2c-Whole-Transplant (Compile-Experte unberuehrt)", flush=True)
        run_batch(limit=limit, ai=ai, workers=workers, outroot=outroot, no_m2c=no_m2c)
