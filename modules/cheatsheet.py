# modules/cheatsheet.py
# Analysiert perfect_matches und erzeugt Kontext fuer den KI-Uebersetzer.
#
# Zwei Nutzungsarten:
#   1. Full rebuild:  full_rebuild(pm_dir) -> results
#   2. Inkrementell:  incremental_update(results, c_file_path) -> results
#
# Der Wrapper ruft incremental_update() bei jedem neuen Perfect Match auf.
# generate_cheatsheet.py (CLI) ruft full_rebuild() auf.

import re
import json
import logging
import threading
from pathlib import Path
from collections import defaultdict, Counter

_log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# REGEX / CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

RE_FUNC_SIG = re.compile(
    r'^(?:extern\s+)?'
    r'([\w\s\*]+?)\s+'         # return type
    r'(\w+)'                    # function name
    r'\s*\(([^)]*)\)',          # params
    re.MULTILINE
)
RE_CALL = re.compile(r'\b(\w+)\s*\(')
RE_EXTERN = re.compile(r'extern\s+([\w\s\*]+?)\s+(\w+)\s*[\[;]')
RE_ENTRYPOINT = re.compile(r'_entrypoint_(\d+)')

CACHE_PATH = "data/cheatsheet_cache.json"
SIMILAR_DB_PATH = "data/similar_asm_db.jsonl"

# Schwelle: ab diesem struct_score wird der Referenz-Code direkt injiziert.
# Darunter greifen nur Cheatsheet-Patterns (Entrypoint/Overlay), falls vorhanden.
SIMILAR_SCORE_THRESHOLD = 70.0

# Thread-Safety: Cache-Save darf nicht parallel laufen
_save_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════════
# PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_c_file(path: Path):
    """Parst eine C-Datei und extrahiert strukturierte Infos."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None

    info = {
        "filename": path.name,
        "func_name": path.stem,
        "text": text,
        "lines": text.count("\n") + 1,
        "return_type": None,
        "params": [],
        "calls": [],
        "externs": [],
        "is_entrypoint": False,
        "entrypoint_idx": None,
        "overlay_prefix": None,
        "overlay_name": None,
    }

    # Entrypoint-Info
    ep_match = RE_ENTRYPOINT.search(path.stem)
    if ep_match:
        info["is_entrypoint"] = True
        info["entrypoint_idx"] = int(ep_match.group(1))

    # Overlay-Info aus Pfad (z.B. overlays/bs/btrot -> prefix=bs, name=btrot)
    parts = path.parts
    try:
        ov_idx = parts.index("overlays")
        if ov_idx + 1 < len(parts):
            info["overlay_prefix"] = parts[ov_idx + 1]
        if ov_idx + 2 < len(parts):
            info["overlay_name"] = parts[ov_idx + 2]
    except ValueError:
        pass

    # Signatur parsen — suche die Definition die den Funktionsnamen enthaelt
    func_name = info["func_name"]
    sig = None
    for m in RE_FUNC_SIG.finditer(text):
        if m.group(2) == func_name:
            sig = m
            break
    if not sig:
        sig = RE_FUNC_SIG.search(text)

    if sig:
        info["return_type"] = sig.group(1).strip()
        raw_params = sig.group(3).strip()
        if raw_params and raw_params != "void":
            info["params"] = [p.strip() for p in raw_params.split(",")]
        info["func_body"] = text[sig.start():]

    # Body-Aufrufe extrahieren (alles nach der ersten {)
    brace = text.find("{")
    if brace >= 0:
        body = text[brace:]
        calls = RE_CALL.findall(body)
        skip = {"if", "for", "while", "switch", "return", "sizeof", "else",
                info["func_name"]}
        info["calls"] = [c for c in calls if c not in skip]

    # Externs
    for m in RE_EXTERN.finditer(text):
        info["externs"].append({"type": m.group(1).strip(), "name": m.group(2)})

    return info


def parse_c_code(func_name: str, code: str, rel_path: str = ""):
    """Parst C-Code direkt (ohne Datei auf Disk).

    Wird von incremental_update() genutzt um den Code zu analysieren
    ohne ihn nochmal von Disk lesen zu muessen.
    """
    text = code.strip()
    if not text:
        return None

    info = {
        "filename": f"{func_name}.c",
        "func_name": func_name,
        "text": text,
        "lines": text.count("\n") + 1,
        "return_type": None,
        "params": [],
        "calls": [],
        "externs": [],
        "is_entrypoint": False,
        "entrypoint_idx": None,
        "overlay_prefix": None,
        "overlay_name": None,
        "_path": rel_path,
    }

    ep_match = RE_ENTRYPOINT.search(func_name)
    if ep_match:
        info["is_entrypoint"] = True
        info["entrypoint_idx"] = int(ep_match.group(1))

    # Overlay-Info aus rel_path
    if rel_path:
        parts = Path(rel_path).parts
        try:
            ov_idx = parts.index("overlays")
            if ov_idx + 1 < len(parts):
                info["overlay_prefix"] = parts[ov_idx + 1]
            if ov_idx + 2 < len(parts):
                info["overlay_name"] = parts[ov_idx + 2]
        except ValueError:
            pass

    sig = None
    for m in RE_FUNC_SIG.finditer(text):
        if m.group(2) == func_name:
            sig = m
            break
    if not sig:
        sig = RE_FUNC_SIG.search(text)

    if sig:
        info["return_type"] = sig.group(1).strip()
        raw_params = sig.group(3).strip()
        if raw_params and raw_params != "void":
            info["params"] = [p.strip() for p in raw_params.split(",")]
        info["func_body"] = text[sig.start():]

    brace = text.find("{")
    if brace >= 0:
        body = text[brace:]
        calls = RE_CALL.findall(body)
        skip = {"if", "for", "while", "switch", "return", "sizeof", "else", func_name}
        info["calls"] = [c for c in calls if c not in skip]

    for m in RE_EXTERN.finditer(text):
        info["externs"].append({"type": m.group(1).strip(), "name": m.group(2)})

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# KLASSIFIKATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_entrypoint_pattern(info):
    """Klassifiziert ein Entrypoint-Codemuster."""
    text = info["text"]
    body_start = text.find("{")
    if body_start < 0:
        return "unknown"
    body = text[body_start:].strip()

    if re.search(r'return\s+\w+\[', body) and info["lines"] <= 6:
        return "array_lookup"
    if any(k in body for k in ("baflag_clear", "baflag_set", "func_8009E474")):
        return "init_setup"
    if "bs_setState" in body:
        return "state_handler"
    if "bainput_enable" in body:
        return "input_handler"
    if len(info["calls"]) == 1 and info["lines"] <= 5:
        return "simple_delegator"
    if info["lines"] <= 8:
        return "short_logic"
    return "complex"


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSE (full rebuild)
# ═══════════════════════════════════════════════════════════════════════════════

def _empty_results():
    """Erzeugt eine leere Ergebnis-Struktur."""
    return {
        "total_files": 0,
        "entrypoint_slots": defaultdict(lambda: {"count": 0, "patterns": Counter(),
                                                  "examples": []}),
        "known_signatures": {},
        "overlay_api_usage": defaultdict(Counter),
        "common_calls": Counter(),
        "extern_vars": {},
        "category_stats": Counter(),
        "overlay_prefix_stats": Counter(),
        "type_usage": Counter(),
        "_known_funcs": set(),  # Tracking: welche Funktionen schon im Cache sind
    }


def _integrate_info(results, info):
    """Integriert ein einzelnes parse-Ergebnis in die results-Struktur.

    Wird sowohl von full_rebuild als auch von incremental_update genutzt.
    """
    fn = info["func_name"]

    # Duplikat-Check
    if fn in results.get("_known_funcs", set()):
        return
    results.setdefault("_known_funcs", set()).add(fn)

    results["total_files"] += 1

    # Kategorie
    path_str = info.get("_path", "")
    if path_str:
        parts = Path(path_str).parts
        try:
            nm_idx = parts.index("nonmatchings")
            cat = parts[nm_idx + 1] if nm_idx + 1 < len(parts) else "unknown"
        except ValueError:
            cat = "unknown"
        results["category_stats"][cat] += 1

    if info["overlay_prefix"]:
        results["overlay_prefix_stats"][info["overlay_prefix"]] += 1

    for call in info["calls"]:
        results["common_calls"][call] += 1

    if info["return_type"]:
        results["type_usage"][info["return_type"]] += 1

    for ext in info["externs"]:
        results["extern_vars"][ext["name"]] = ext["type"]

    if fn not in results["known_signatures"]:
        results["known_signatures"][fn] = {
            "return_type": info["return_type"],
            "params": info["params"],
            "count": 1,
        }
    else:
        results["known_signatures"][fn]["count"] += 1

    if info["is_entrypoint"]:
        idx = info["entrypoint_idx"]
        slot = results["entrypoint_slots"][idx]
        slot["count"] += 1
        pattern = classify_entrypoint_pattern(info)
        slot["patterns"][pattern] += 1
        if len(slot["examples"]) < 2:
            slot["examples"].append({
                "name": fn,
                "code": info.get("func_body", info["text"]).strip(),
                "pattern": pattern,
            })

    if info["overlay_prefix"]:
        for call in info["calls"]:
            results["overlay_api_usage"][info["overlay_prefix"]][call] += 1


def full_rebuild(pm_dir="output/perfect_matches/nonmatchings"):
    """Scannt alle perfect_matches und baut die Analyse komplett neu auf."""
    pm = Path(pm_dir)
    if not pm.exists():
        _log.warning(f"Cheatsheet: {pm} nicht gefunden.")
        return _empty_results()

    c_files = sorted(pm.rglob("*.c"))
    results = _empty_results()
    errors = 0

    for cf in c_files:
        try:
            info = parse_c_file(cf)
            if info:
                info["_path"] = str(cf)
                _integrate_info(results, info)
        except Exception:
            errors += 1

    if errors:
        _log.info(f"Cheatsheet full_rebuild: {errors} Dateien mit Parse-Fehlern uebersprungen")

    _log.info(f"Cheatsheet full_rebuild: {results['total_files']} Dateien analysiert")
    return results


def sync_cache(results, pm_dir="output/perfect_matches/nonmatchings"):
    """Gleicht den Cache mit dem aktuellen perfect_matches-Verzeichnis ab.

    Findet Dateien die seit dem letzten Cache-Save hinzugekommen sind
    und fuegt sie inkrementell hinzu. Deutlich schneller als full_rebuild
    weil nur die Differenz geparst wird.

    Args:
        results:  Geladene Cache-Ergebnisse (wird in-place mutiert)
        pm_dir:   Pfad zum perfect_matches/nonmatchings Verzeichnis

    Returns:
        (results, added_count) — aktualisierte results und Anzahl neuer Eintraege
    """
    pm = Path(pm_dir)
    if not pm.exists():
        return results, 0

    known = results.get("_known_funcs", set())
    added = 0

    for cf in sorted(pm.rglob("*.c")):
        if cf.stem in known:
            continue
        try:
            info = parse_c_file(cf)
            if info:
                info["_path"] = str(cf)
                _integrate_info(results, info)
                added += 1
        except Exception:
            pass

    if added > 0:
        save_cache(results)
        _log.info(f"Cheatsheet sync: {added} neue Dateien nachgetragen")

    return results, added


# ═══════════════════════════════════════════════════════════════════════════════
# INKREMENTELLES UPDATE
# ═══════════════════════════════════════════════════════════════════════════════

def incremental_update(results, func_name, c_code, rel_path="",
                       auto_save=True):
    """Fuegt ein neues Perfect Match in die bestehenden Results ein.

    Args:
        results:    Bestehende Analyse-Ergebnisse (wird in-place mutiert)
        func_name:  Funktionsname (z.B. "bsbtrot_entrypoint_3")
        c_code:     Der C-Code der Funktion
        rel_path:   Relativer Pfad (z.B. "overlays/bs/btrot/bsbtrot_entrypoint_3.c")
        auto_save:  Ob der Cache automatisch gespeichert werden soll

    Returns:
        Die aktualisierten results (selbes Objekt, in-place mutiert)
    """
    if results is None:
        results = _empty_results()

    info = parse_c_code(func_name, c_code, rel_path)
    if not info:
        return results

    _integrate_info(results, info)

    if auto_save:
        _save_cache_async(results)

    return results


# Async-Save: laeuft im Hintergrund, blockiert den Worker nicht
def _save_cache_async(results):
    """Speichert den Cache thread-safe im Hintergrund."""
    def _do_save():
        with _save_lock:
            try:
                save_cache(results)
            except Exception as e:
                _log.debug(f"Cheatsheet cache save fehlgeschlagen: {e}")
    t = threading.Thread(target=_do_save, daemon=True)
    t.start()


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE (load / save)
# ═══════════════════════════════════════════════════════════════════════════════

def load_cache(cache_path=CACHE_PATH):
    """Laedt gecachte Analyse-Ergebnisse.

    Gibt None zurueck wenn kein Cache existiert.
    """
    p = Path(cache_path)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        results = {
            "total_files": raw["total_files"],
            "entrypoint_slots": {},
            "known_signatures": raw.get("known_signatures", {}),
            "overlay_api_usage": defaultdict(Counter),
            "common_calls": Counter(raw.get("common_calls", {})),
            "extern_vars": raw.get("extern_vars", {}),
            "category_stats": Counter(raw.get("category_stats", {})),
            "overlay_prefix_stats": Counter(raw.get("overlay_prefix_stats", {})),
            "type_usage": Counter(raw.get("type_usage", {})),
            "_known_funcs": set(raw.get("_known_funcs", [])),
        }
        for idx_str, slot_data in raw.get("entrypoint_slots", {}).items():
            results["entrypoint_slots"][int(idx_str)] = {
                "count": slot_data["count"],
                "patterns": Counter(slot_data["patterns"]),
                "examples": slot_data.get("examples", []),
            }
        for prefix, calls in raw.get("overlay_api_usage", {}).items():
            results["overlay_api_usage"][prefix] = Counter(calls)
        return results
    except Exception:
        return None


def save_cache(results, cache_path=CACHE_PATH):
    """Speichert Analyse-Ergebnisse als JSON-Cache."""
    serializable = {
        "total_files": results["total_files"],
        "entrypoint_slots": {
            str(idx): {
                "count": slot["count"],
                "patterns": dict(slot["patterns"]),
                "examples": slot["examples"],
            }
            for idx, slot in results["entrypoint_slots"].items()
        },
        "known_signatures": results["known_signatures"],
        "overlay_api_usage": {
            prefix: dict(counter)
            for prefix, counter in results["overlay_api_usage"].items()
        },
        "common_calls": dict(results["common_calls"]),
        "extern_vars": results["extern_vars"],
        "category_stats": dict(results["category_stats"]),
        "overlay_prefix_stats": dict(results["overlay_prefix_stats"]),
        "type_usage": dict(results["type_usage"]),
        "_known_funcs": sorted(results.get("_known_funcs", set())),
    }
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(serializable, indent=1), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# SIMILAR ASM DB
# ═══════════════════════════════════════════════════════════════════════════════

def load_similar_db(db_path=SIMILAR_DB_PATH):
    """Laedt die similar_asm_db.jsonl als dict: target_asm -> entry.

    Wird einmalig beim Pipeline-Start aufgerufen (in run_batch).
    Gibt ein leeres dict zurueck wenn die DB nicht existiert.
    """
    p = Path(db_path)
    if not p.exists():
        _log.info(f"similar_asm_db nicht gefunden: {p}")
        return {}
    db = {}
    try:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                key = entry.get("target_asm", "")
                if key:
                    db[key] = entry
        _log.info(f"similar_asm_db geladen: {len(db)} Eintraege")
    except Exception as e:
        _log.warning(f"similar_asm_db Ladefehler: {e}")
    return db


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT-KONTEXT (fuer KI)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_prompt_context(results, func_name="", overlay_prefix="",
                            similar_entry=None, current_struct=0.0):
    """Generiert Kontext fuer den KI-Prompt. Gibt ein Dict zurueck:

    {
        "similar": str   — Separater Similar-Block (eigene Prompt-Sektion)
        "cheatsheet": str — Cheatsheet-Kontext (geht in tech_env)
    }

    Stufe 1 — Hoher Similar-Score (>= SIMILAR_SCORE_THRESHOLD):
        Der C-Code der aehnlichsten Referenz wird als eigener Block zurueckgegeben.
        Cheatsheet wird dann NICHT angehaengt (kein Rauschen).

    Stufe 2 — Kein hoher Similar-Score (oder Similar unterdrueckt):
        Nur entrypoint-/overlay-spezifischer Kontext, keine generischen Listen.
    """
    similar_lines = []
    cheatsheet_lines = []

    # ── Stufe 1: Similar-ASM Referenz ────────────────────────────────────────
    has_similar = False
    if similar_entry:
        score = similar_entry.get("struct_score", 0)
        ref_code = similar_entry.get("c_code", "").strip()
        ref_name = similar_entry.get("best_ref", "?")
        ref_src = similar_entry.get("best_ref_source", "?")
        ref_hex = similar_entry.get("ref_hex_count", 0)
        target_hex = similar_entry.get("target_hex_count", 0)

        if score >= SIMILAR_SCORE_THRESHOLD and ref_code and current_struct < score:
            has_similar = True
            w = similar_lines.append
            w(f"Reference: {ref_name} ({ref_hex} instructions, {score:.0f}% structural match)")
            w(f"Target: {func_name} ({target_hex} instructions)")
            w("")
            w("Reproduce the reference's exact code structure:")
            w("- Same array dimensions (e.g. 2D array stays 2D, not flattened)")
            w("- Same temporary/intermediate variables (e.g. pointer-to-row)")
            w("- Same expression shapes (e.g. '^ 0' trick, cast patterns)")
            w("- Same control flow nesting (if/else depth, loop type)")
            w("Only change identifiers and constants to match the TARGET assembly.")
            w("")
            w(ref_code)

    # ── Stufe 2: Cheatsheet-Patterns ─────────────────────────────────────────
    # Override aufgehoben: Cheatsheet wird IMMER berechnet (sofern results da sind),
    # auch bei starkem Similar. Die Anti-Drowning-Entscheidung trifft jetzt die
    # KONSUM-Schicht: Single-Calls (generate_fix) unterdruecken den Cheatsheet bei
    # vorhandenem Similar; der Batch-Fahrplan staffelt ihn (Slot 'full' = Similar +
    # Cheatsheet zusammen, spaetere Slots ohne). So bleibt der "beides"-Slot moeglich.
    if results is not None:
        has_specific = False
        w = cheatsheet_lines.append

        w("=== DECOMPILATION CONTEXT (from perfect matches) ===")
        w("")

        # Entrypoint-spezifischer Kontext
        ep_match = RE_ENTRYPOINT.search(func_name)
        if ep_match:
            idx = int(ep_match.group(1))
            slot = results["entrypoint_slots"].get(idx)
            if slot:
                has_specific = True
                dom = slot["patterns"].most_common(1)
                dom_name = dom[0][0] if dom else "unknown"
                dom_pct = (dom[0][1] / slot["count"] * 100) if dom else 0
                w(f"TARGET is entrypoint slot {idx} ({slot['count']}x seen, "
                  f"{dom_pct:.0f}% are {dom_name}).")

                if dom_name == "array_lookup":
                    w("PATTERN: Simple array getter — return extern_array[param_0];")
                    w("Use: extern s32 D_XXXXXXXX_name[]; and return it indexed.")
                elif dom_name == "init_setup":
                    w("PATTERN: Init/setup — call setup funcs, clear flags with baflag_clear.")
                elif dom_name == "state_handler":
                    w("PATTERN: State handler — check anim, call bs_setState.")
                elif dom_name == "input_handler":
                    w("PATTERN: Input handler — call bainput_enable, then update+setState.")
                elif dom_name == "short_logic":
                    w("PATTERN: Short function — usually <=8 lines, simple logic.")

                if slot["examples"]:
                    ex = slot["examples"][0]
                    w(f"Example from {ex['name']}:")
                    for cl in ex["code"].split("\n"):
                        w(f"  {cl}")
                w("")

        # Overlay-spezifischer Kontext
        if overlay_prefix and overlay_prefix in results["overlay_api_usage"]:
            api = results["overlay_api_usage"][overlay_prefix]
            top_calls = api.most_common(20)
            named = [(c, n) for c, n in top_calls
                     if not c.startswith("func_80") and n >= 2]
            if named:
                has_specific = True
                w(f"Known API for '{overlay_prefix}' overlays:")
                for call, cnt in named:
                    w(f"  {call} ({cnt}x)")
                w("")

        if not has_specific:
            cheatsheet_lines.clear()

    return {
        "similar": "\n".join(similar_lines),
        "cheatsheet": "\n".join(cheatsheet_lines),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN CHEATSHEET (fuer Menschen)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_cheatsheet_md(results):
    """Generiert das volle Markdown-Cheatsheet (fuer manuelles Lesen)."""
    lines = []
    w = lines.append

    w("# Banjo-Tooie Decompilation Cheatsheet")
    w(f"# Auto-generated from {results['total_files']} perfect matches")
    w("")

    w("## 1. ENTRYPOINT SLOT PATTERNS")
    w("Each overlay/DLL exports functions via a jump table. The number is the")
    w("slot index. Same slot index across DLLs usually means same role:")
    w("")

    for idx in sorted(results["entrypoint_slots"]):
        slot = results["entrypoint_slots"][idx]
        dominant = slot["patterns"].most_common(1)
        dom_name = dominant[0][0] if dominant else "unknown"
        dom_pct = (dominant[0][1] / slot["count"] * 100) if dominant else 0
        w(f"### Slot {idx} ({slot['count']}x, {dom_pct:.0f}% {dom_name})")
        for pat, cnt in slot["patterns"].most_common():
            w(f"  - {pat}: {cnt}x")
        if slot["examples"]:
            ex = slot["examples"][0]
            w(f"  Example ({ex['name']}):")
            for code_line in ex["code"].split("\n"):
                w(f"    {code_line}")
        w("")

    w("## 2. OVERLAY PREFIX CONTEXT")
    w("")
    for prefix, count in results["overlay_prefix_stats"].most_common():
        api_calls = results["overlay_api_usage"].get(prefix, Counter())
        top_calls = api_calls.most_common(15)
        if not top_calls:
            continue
        w(f"### {prefix} ({count} perfect matches)")
        framework_calls = [(c, n) for c, n in top_calls
                           if not c.startswith("func_80") and "_entrypoint_" not in c]
        entrypoint_calls = [(c, n) for c, n in top_calls if "_entrypoint_" in c]
        if framework_calls:
            w("  Known API functions:")
            for call, cnt in framework_calls:
                w(f"    - {call} ({cnt}x)")
        if entrypoint_calls:
            w("  Cross-DLL calls:")
            for call, cnt in entrypoint_calls:
                w(f"    - {call} ({cnt}x)")
        w("")

    w("## 3. KNOWN FUNCTION SIGNATURES")
    w("")
    named_sigs = {
        name: sig for name, sig in results["known_signatures"].items()
        if not name.startswith("func_80") and sig["return_type"]
    }
    for name, sig in sorted(named_sigs.items()):
        params_str = ", ".join(sig["params"]) if sig["params"] else "void"
        w(f"  {sig['return_type']} {name}({params_str});")
    w("")

    w("## 4. MOST COMMON FRAMEWORK CALLS")
    w("")
    for call, cnt in results["common_calls"].most_common(80):
        if call.startswith("func_80"):
            continue
        if cnt < 3:
            break
        w(f"  {call} ({cnt}x)")
    w("")

    w("## 5. FREQUENTLY CALLED INTERNAL FUNCTIONS")
    w("")
    for call, cnt in results["common_calls"].most_common(60):
        if not call.startswith("func_80") or cnt < 5:
            continue
        w(f"  {call} ({cnt}x)")
    w("")

    w("## 6. RETURN TYPES DISTRIBUTION")
    w("")
    for typ, cnt in results["type_usage"].most_common(20):
        w(f"  {typ}: {cnt}x")
    w("")

    return "\n".join(lines)
