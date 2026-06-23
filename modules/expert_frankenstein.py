"""Frankenstein-Experte (Stufe 1 / MVP) -- Multi-Donor-Transplant.

Plan: analysis/experts/frankenstein-expert.md

Fuer GROSSE Problemfunktionen, die kein einzelner Similar loest: liest die
Fragment-DB (data/fragment_asm_db.jsonl, gebaut von build_fragment_asm_db.py) und
fuellt die Luecken des Drafts aus MEHREREN Donor-Fragmenten zusammen
("best-of-N in eine Funktion"). Voll ORAKEL-GEGATED: jeder Transplant-Versuch wird
kompiliert + gemessen (count_fn), nur STRIKTE Verbesserungen bleiben -> NIE schlechter
als der Eingabe-Draft (sicher fuer Phase-0-Einsatz).

Baut auf dem getesteten Graft-Kern modules/expert_synthesis_graft auf
(_body / _top_statements / _sig_params / graft_candidates). Verallgemeinert ihn von
EINEM Donor (m2c) auf N region-spezifische Donoren aus der DB.

MVP (Ansatz A aus dem Plan): Donor-Statements grob als Kandidaten, Orakel als Wahrheit.
Das block-praezise C-Mapping (Ansatz B) und das block-t_start-Positions-Fenster sind
als Stufe 2/3 markiert (TODO).

GATE-Konvention (wie alle Experten): count_fn(c, fn, sp) -> metric (lexikografisches
Tupel, KLEINER = besser) ODER None (compile-/Vergleichs-Fehler). Im Wrapper ist das
_EV_SYNTH (md, im, tier, mm) bzw. _EV_TOT (tier, mm).
"""
import os
import re
import json
import logging
from pathlib import Path

from modules.expert_synthesis_graft import (
    _body, _top_statements, _sig_params, graft_candidates,
)

log = logging.getLogger("expert.frankenstein")

_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH_DEFAULT = _ROOT / "data" / "fragment_asm_db.jsonl"

# Routing-Schwelle: unter dieser Union-Coverage gilt das Target als "genuin neu"
# -> Frankenstein lohnt nicht (Plan, Abschnitt 1).
MIN_UNION_COVERAGE = 0.5

# Sicherheits-/Effizienz-Grenzen.
MAX_DONORS = 6          # wie viele DB-Donoren (Coverage-absteigend) anzapfen
MAX_EVALS = 600         # Orakel-Budget pro Funktion (Compile = 90-95% der Zeit)
MAX_ITER = 4            # iterative Graft-Runden (mehrere Luecken hintereinander)


# ──────────────────────────────────────────────────────────── DB laden (gecacht)
_DB_CACHE = None
_DB_CACHE_PATH = None


def load_fragment_db(path=None):
    """Laedt fragment_asm_db.jsonl -> dict {target_asm: entry}. Gecacht pro Pfad."""
    global _DB_CACHE, _DB_CACHE_PATH
    p = str(path or _DB_PATH_DEFAULT)
    if _DB_CACHE is not None and _DB_CACHE_PATH == p:
        return _DB_CACHE
    db = {}
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    db[e["target_asm"]] = e
                except (json.JSONDecodeError, KeyError):
                    pass
    else:
        log.warning(f"Fragment-DB nicht gefunden: {p}")
    _DB_CACHE, _DB_CACHE_PATH = db, p
    return db


# ──────────────────────────────────────────────────────── Donor-Vorbereitung
def _read_donor(ref_path):
    """Donor-C aus ref_path (relativ zur Pipeline-Wurzel) lesen. None bei Fehler."""
    fp = _ROOT / ref_path
    try:
        return fp.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _remap_params_positional(donor_c, donor_fn, draft_c, draft_fn):
    """Donor-Parameternamen positionsweise auf die Draft-Parameter abbilden.
    (Donoren sind echte C-Funktionen mit eigenen Param-Namen -- KEIN m2c arg<i>.)
    Globals/Callees sind via Target/Header identisch -> unveraendert."""
    dp = _sig_params(donor_c, donor_fn)
    tp = _sig_params(draft_c, draft_fn)
    out = donor_c
    for i, dname in enumerate(dp):
        if i < len(tp) and dname and tp[i] and dname != tp[i]:
            out = re.sub(r"\b" + re.escape(dname) + r"\b", tp[i], out)
    return out


_ASM_NOISE = {
    "nop", "move", "addiu", "addu", "subu", "sll", "srl", "sra", "lui", "ori",
    "andi", "xori", "slti", "sltiu", "and", "or", "xor", "nor", "slt", "sltu",
    "mult", "multu", "div", "divu", "mflo", "mfhi", "jr", "jal", "jalr", "beq",
    "bne", "blez", "bgtz", "bltz", "bgez", "beqz", "bnez", "lw", "sw", "lh",
    "lhu", "lb", "lbu", "sh", "sb", "lwc1", "swc1", "ldc1", "sdc1", "mtc1",
    "mfc1", "cvt", "trunc", "add", "sub", "mul", "neg", "abs", "zero", "at",
    "glabel", "endlabel", "func", "null",
}


def _target_symbols(target_s_path):
    """Symbol-Namen, die im Target-.s vorkommen (Callees + %hi/%lo-Globals +
    benannte Operanden). Dient NUR dem Vorfilter der Donor-Statements; das Orakel
    entscheidet final. Leeres Set -> kein Filter (nicht ueberfiltern)."""
    syms = set()
    try:
        raw = open(target_s_path, encoding="utf-8", errors="replace").read()
    except Exception:
        return syms
    for line in raw.splitlines():
        s = re.sub(r"/\*.*?\*/", "", line).strip()
        if not s or s.startswith((".", "/*")):
            continue
        for m in re.finditer(r"%(?:hi|lo|gp_rel|call16|got)\((\w+)\)", s):
            syms.add(m.group(1))
        # benannte Operanden (Callees, Globals): Identifier mit Buchstabe, len>=4,
        # keine Register ($..) -- der Mnemonic-Teil wird durch _ASM_NOISE gefiltert.
        for tok in re.findall(r"(?<![\$.%])\b([A-Za-z_]\w{3,})\b", s):
            if tok.lower() not in _ASM_NOISE:
                syms.add(tok)
    return syms


def _relevant(stmt, target_syms):
    """Donor-Statement ist Kandidat, wenn es ein Symbol referenziert, das im Target
    vorkommt (Call/Global/Feldname). Ohne Target-Symbole: alles zulassen."""
    if not target_syms:
        return True
    toks = set(re.findall(r"[A-Za-z_]\w*", stmt))
    return bool(toks & target_syms)


# ─────────────────────────── SYMBOL-REMAP via Block-Alignment (Ansatz B)
# Die DB matcht das MASKIERTE Skelett (jal-Ziel + %hi/%lo maskiert) -> struktur-
# gleiche Donoren rufen ANDERE Funktionen auf. Der block-Eintrag [t_start, n,
# r_start] alignt aber Donor-Instruktion r_start+k mit Target-Instruktion t_start+k.
# An aligned jal/%hi/%lo-Positionen koennen wir Donor-Symbol -> RICHTIGES Target-
# Symbol ablesen und den Donor-Code in die Vokabular des Targets uebersetzen.

def _extract_sym(op_text):
    """Referenziertes Symbol einer Instruktions-Operanden-Zeile (jal-Ziel / %hi/%lo-
    Global) oder None. .L-Labels (lokale Branches) sind KEINE Symbole."""
    mh = re.search(r"%(?:hi|lo|call16|got|gp_rel)\((\w+)\)", op_text)
    if mh:
        return mh.group(1)
    mj = re.match(r"(?:jal|j)\s+([A-Za-z_]\w*)", op_text)
    if mj and not mj.group(1).startswith(".L"):
        return mj.group(1)
    return None


def _symbols_from_s(s_path):
    """Pro .text-Instruktion das referenzierte Symbol (oder None) -- INDEX-DECKUNGS-
    GLEICH mit ido_comparison.extract_words_from_original_s (gleiche Iteration), damit
    die DB-Block-Indizes passen."""
    out = []
    if not s_path or not os.path.exists(s_path):
        return out
    in_text = False
    with open(s_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith(".section"):
                in_text = ".text" in s
                continue
            if s.startswith("glabel"):
                in_text = True
                continue
            if s.startswith("endlabel"):
                in_text = False
                continue
            if not in_text:
                continue
            if not s or s.startswith((".L", ".", "#", "dlabel", "enddlabel")):
                continue
            m = re.search(r"/\*.*?([0-9a-fA-F]{8})\s*\*/\s+([a-z].*)$", s)
            if not m:
                m = re.match(r"([0-9a-fA-F]{8})\s+([a-z].*)$", s)
            if m:
                out.append(_extract_sym(m.group(2)))
    return out


def _donor_s_path(donor):
    """Pfad zur Donor-.s (fuer Symbol-Alignment). Nur perfect_matches-Donoren haben
    eine spiegelnde .s unter data/target_asm; refdata-Donoren (compiliert) nicht."""
    if donor.get("source") != "perfect_matches":
        return None
    rp = donor.get("ref_path", "")
    cand = rp.replace("output/perfect_matches/nonmatchings/", "data/target_asm/nonmatchings/")
    cand = cand[:-2] + ".s" if cand.endswith(".c") else cand
    p = _ROOT / cand
    return str(p) if p.exists() else None


# Per-Instruktions-Symbole eines objdump -dr Outputs (parallel zu
# ido_comparison.extract_words_from_objdump -> index-deckungsgleich mit dem
# refdata-Skelett der DB).
_OBJ_INSN_RE = re.compile(r"\s*[0-9a-fA-F]+:\s+([0-9a-fA-F]{8})\s+(\S+)")
_OBJ_RELOC_RE = re.compile(r"R_MIPS_\w+\s+(\S+)")

_REFDATA_SYM_CACHE = {}    # ref_path -> list[symbol|None] (gecacht; Compile ist teuer)


def _symbols_from_objdump_text(dis_text):
    """Pro Instruktion das Reloc-Symbol (jal-Callee/Global) oder None, aus objdump
    -dr Text. Reloc-Folgezeile (R_MIPS_* SYM) markiert die zuletzt gesehene Instr."""
    out = []
    for line in dis_text.splitlines():
        if _OBJ_INSN_RE.match(line):
            out.append(None)
            continue
        rm = _OBJ_RELOC_RE.search(line)
        if rm and out:
            out[-1] = rm.group(1).split("+")[0]   # '+0x..' Offset abstreifen
    return out


def _refdata_symbols(ref_path):
    """refdata-Donor kompilieren (orakel-sicher, wie der DB-Bau) + objdump -dr ->
    per-Instruktions-Symbole, index-deckungsgleich mit dem DB-Skelett. Gecacht."""
    if ref_path in _REFDATA_SYM_CACHE:
        return _REFDATA_SYM_CACHE[ref_path]
    syms = []
    try:
        import subprocess
        import shutil as _sh
        import build_fragment_asm_db as bfa
        o_file, temp_dir = bfa.compile_refdata_to_o(_ROOT / ref_path)
        try:
            if o_file:
                r = subprocess.run([bfa.OBJDUMP, "-dr", "-z", o_file],
                                   capture_output=True, text=True, timeout=10)
                syms = _symbols_from_objdump_text(r.stdout)
        finally:
            _sh.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        syms = []
    _REFDATA_SYM_CACHE[ref_path] = syms
    return syms


def _build_symbol_map(blocks, tsyms, dsyms):
    """Donor-Symbol -> Target-Symbol via Block-Alignment (Mehrheitsentscheid). tsyms/
    dsyms = per-Instruktions-Symbollisten (index-deckungsgleich mit den DB-Bloecken).
    Leer, wenn eine Liste fehlt oder kein aligned Symbol-Paar existiert."""
    from collections import Counter
    if not tsyms or not dsyms:
        return {}
    votes = {}
    for blk in blocks:
        t_start, n, r_start = blk[0], blk[1], blk[2]
        for k in range(n):
            ti, ri = t_start + k, r_start + k
            if ti >= len(tsyms) or ri >= len(dsyms):
                continue
            ds, ts = dsyms[ri], tsyms[ti]
            if ds and ts and ds != ts:
                votes.setdefault(ds, Counter())[ts] += 1
    return {ds: c.most_common(1)[0][0] for ds, c in votes.items()}


def _apply_symbol_map(donor_c, smap):
    """Simultane wortgrenz-genaue Ersetzung Donor-Symbol -> Target-Symbol (keine
    Kaskaden: alles in EINEM Pass)."""
    if not smap:
        return donor_c
    pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in smap) + r")\b")
    return pat.sub(lambda m: smap[m.group(1)], donor_c)


def _is_decl_stmt(stmt):
    """reine lokale Deklaration (kein Transplant-Wert)."""
    return bool(re.match(r"^[A-Za-z_]\w*\s*\**\s*[A-Za-z_]\w*\s*;$", stmt))


def _donor_statement_pool(entry, draft_c, draft_fn, target_syms, target_s_path, max_donors):
    """Top-Donoren der DB vorbereiten (symbol- + param-remapped). Liefert (pool, whole_cands):
      pool        = relevante Donor-Statements (flach, dedupliziert; fuer den Graft)
      whole_cands = [(label, full_code)] -- Draft-Signatur + Donor-Body (Voll-Transplant).

    HINWEIS: Ein block-GEFUEHRTER Graft (Donor-Region via proportionalem Alignment an die
    Soll-Position) wurde getestet und gegen den erschoepfenden Brute-Force-Graft A/B-gemessen:
    Delta = 0 (kein Gewinn). Grund: die blocks sind im Instruktionsraum praezise, aber die
    proportionale Instruktion->C-Statement-Abbildung ist zu grob; der Brute-Force erreicht
    dieselben Multi-Region-Montagen erschoepfend + besser. Verworfen. Eine PRAEZISE Variante
    braeuchte echtes Compile-Zeilen-Mapping (objdump-Zeile -> C-Quellzeile)."""
    pool = []
    whole_cands = []
    seen = set()
    tsyms = _symbols_from_s(target_s_path)        # einmal: per-Instr Target-Symbole
    d_pre, _d_body, d_suf = _body(draft_c, draft_fn)   # Draft-Signatur/-Rahmen
    for d in entry.get("similars", [])[:max_donors]:
        donor_c = _read_donor(d.get("ref_path", ""))
        if not donor_c:
            continue
        donor_fn = d.get("ref", "")
        # SYMBOL-REMAP (Ansatz B): Donor-Calls/Globals via Block-Alignment auf die
        # RICHTIGEN Target-Symbole uebersetzen -> strukturgleiche Donoren werden
        # transplantierbar (sonst disjunkte Symbole = nutzlos). Donor-Symbole aus
        # der .s (perfect_matches) ODER per Compile+objdump (refdata).
        ds_path = _donor_s_path(d)
        if ds_path:
            dsyms = _symbols_from_s(ds_path)
        elif d.get("source") == "refdata_original":
            dsyms = _refdata_symbols(d.get("ref_path", ""))
        else:
            dsyms = []
        if dsyms:
            smap = _build_symbol_map(d.get("blocks", []), tsyms, dsyms)
            donor_c = _apply_symbol_map(donor_c, smap)
        # Parameter positionsweise (Donor-Param -> Draft-Param); disjunkt zu Symbolen.
        donor_c = _remap_params_positional(donor_c, donor_fn, draft_c, draft_fn)
        pre, body, suf = _body(donor_c, donor_fn)
        if body is None:
            continue
        # VOLL-TRANSPLANT: Draft-Signatur (korrekte Typen/Name) + remappter Donor-Body
        # (richtige Struktur). Ein Kandidat pro Donor -- bei near-identischen Cluster-
        # Funktionen kann das einen VOLLEN Match liefern, nicht nur eine mm-Senkung.
        if d_pre is not None:
            whole_cands.append((f"whole:{donor_fn}", d_pre + body + d_suf))
        for st in _top_statements(body):
            stmt = st if st.endswith((";", "}")) else st + ";"
            if _relevant(stmt, target_syms) and not _is_decl_stmt(stmt) and stmt not in seen:
                seen.add(stmt)
                pool.append(stmt)
    return pool, whole_cands


# ─────────────────────────────────────────────────────────────── der Experte
def frankenstein_expert(c_code, func_name, target_s_path, count_fn,
                        db_entry=None, db_path=None,
                        min_union_coverage=MIN_UNION_COVERAGE,
                        max_donors=MAX_DONORS, max_evals=MAX_EVALS,
                        max_iter=MAX_ITER):
    """Multi-Donor-Transplant, orakel-gegated.

    count_fn(c, fn, sp) -> metric (kleiner=besser) ODER None (compile-fail).
    Rueckgabe-Dict: applied, new_code, diffs_before, diffs_after, steps, verdict, evals.
      verdict: 'low-coverage' | 'no-donor' | 'no-candidates' | 'compile-fail' |
               'improved' | 'unchanged'
    """
    result = {"applied": False, "new_code": c_code, "diffs_before": None,
              "diffs_after": None, "steps": [], "verdict": "unchanged", "evals": 0}

    base = count_fn(c_code, func_name, target_s_path)
    result["diffs_before"] = result["diffs_after"] = base
    if base is None:
        result["verdict"] = "compile-fail"   # Phase 0 sollte das vorher abfangen
        return result

    entry = db_entry if db_entry is not None else load_fragment_db(db_path).get(func_name)
    if not entry or not entry.get("similars"):
        result["verdict"] = "no-donor"
        return result
    if entry.get("union_coverage", 0.0) < min_union_coverage:
        result["verdict"] = "low-coverage"
        return result

    target_syms = _target_symbols(target_s_path)
    pool, whole_cands = _donor_statement_pool(
        entry, c_code, func_name, target_syms, target_s_path, max_donors)
    if not pool and not whole_cands:
        result["verdict"] = "no-candidates"
        return result

    cur, cur_m = c_code, base
    steps = []
    evals = 0

    def _try(cands):
        """Beste strikte Verbesserung aus cands waehlen (orakel-gegated)."""
        nonlocal evals
        best = None
        for lbl, cand in cands:
            if evals >= max_evals:
                break
            evals += 1
            m = count_fn(cand, func_name, target_s_path)
            if m is not None and m < cur_m and (best is None or m < best[0]):
                best = (m, cand, lbl)
        return best

    # STUFE 1: VOLL-TRANSPLANT -- ganzer remappter Donor (Draft-Signatur + Donor-Body)
    # als EIN Kandidat je Donor. Billig (1 Eval), kann einen vollen Match liefern.
    best = _try(whole_cands)
    if best is not None:
        cur_m, cur = best[0], best[1]
        steps.append((best[2], cur_m))

    # STUFE 2: STATEMENT-GRAFT -- relevante Donor-Statements (Multi-Donor-Pool) an jeder
    # Position, erschoepfend + iterativ. Macht implizit Multi-Region (Fragmente aus
    # mehreren Donoren an passenden Stellen) und findet empirisch die groessten Gewinne.
    for _ in range(max_iter):
        if evals >= max_evals or not pool:
            break
        best = _try(graft_candidates(cur, pool, func_name))
        if best is None:
            break
        cur_m, cur = best[0], best[1]
        steps.append((best[2], cur_m))

    result.update({
        "applied": cur != c_code,
        "new_code": cur,
        "diffs_after": cur_m,
        "steps": steps,
        "evals": evals,
        "verdict": "improved" if cur != c_code else "unchanged",
    })
    return result
