# modules/session.py
# Persistenz-Modul: Speichert und laedt den Zustand pro Funktion.
# Ueberlebt Pipeline-Neustarts.
# Trackt: bester Code, bester Score, Historie der KI-Aenderungen.

import os
import json
import hashlib
import logging
from pathlib import Path
from difflib import unified_diff

log = logging.getLogger(__name__)

SESSION_DIR = os.path.join("data", "session")


def _session_path(func_name: str) -> str:
    os.makedirs(SESSION_DIR, exist_ok=True)
    return os.path.join(SESSION_DIR, f"{func_name}.json")


def load_session(func_name: str) -> dict:
    """
    Laedt den gespeicherten Zustand einer Funktion.
    Returns: dict mit best_code, best_score, best_struct, history, seen_hashes.
    Falls keine Session existiert: leeres Default-Dict.
    """
    path = _session_path(func_name)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            log.info(f"[{func_name}] Session geladen: Score={data.get('best_score', 0)}% "
                     f"({len(data.get('history', []))} vorherige Versuche)")
            return data
        except (json.JSONDecodeError, KeyError) as e:
            log.warning(f"[{func_name}] Session korrupt, starte neu: {e}")

    return {
        "baseline_code": "",    # Originaler Input — wird NIEMALS veraendert
        "best_code": "",
        "best_score": 0.0,
        "best_struct": 0.0,
        "history": [],
        "seen_hashes": [],
        "permuter_report": "",  # Letzer Permuter-Report (ueberlebt Neustarts)
    }


def save_session(func_name: str, session: dict):
    """Speichert den aktuellen Zustand persistent."""
    path = _session_path(func_name)
    # History begrenzen: max 50 Eintraege (alle gegen dieselbe Baseline)
    hist = session.get("history", [])
    if len(hist) > 50:
        session["history"] = hist[-50:]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=1)


def code_hash(c_code: str) -> str:
    """Berechnet einen kurzen Hash des C-Codes fuer Duplicate-Detection."""
    return hashlib.md5(c_code.strip().encode()).hexdigest()[:12]


def is_duplicate(session: dict, c_code: str) -> bool:
    """Prueft ob dieser Code schon mal generiert wurde."""
    h = code_hash(c_code)
    return h in session.get("seen_hashes", [])


def record_attempt(session: dict, c_code: str, score: float, struct_score: float,
                   temperature: float, phase: str, prev_score: float = 0.0,
                   prev_code: str = ""):
    """
    Zeichnet einen KI-Versuch auf.
    Diff wird IMMER gegen die unveraenderliche Baseline berechnet.
    """
    h = code_hash(c_code)
    if h not in session.get("seen_hashes", []):
        session.setdefault("seen_hashes", []).append(h)

    # Diff gegen die UNVERAENDERLICHE Baseline
    baseline = session.get("baseline_code", "")
    diff_vs_baseline = ""
    if baseline and baseline.strip() != c_code.strip():
        diff_lines = list(unified_diff(
            baseline.splitlines(keepends=True),
            c_code.splitlines(keepends=True),
            lineterm="",
            n=1,
        ))
        changes = [l.rstrip() for l in diff_lines
                   if l.startswith(("+", "-", " ")) and not l.startswith(("+++", "---"))]
        diff_vs_baseline = "\n".join(changes[:30])

    # Ergebnis-Bewertung
    if phase == "syntax_broken":
        result = "FAILED (broke compilation)"
    elif score > prev_score:
        result = f"IMPROVED (+{score - prev_score:.1f}%)"
    elif score == prev_score:
        result = "NO CHANGE"
    else:
        result = f"WORSE ({score - prev_score:+.1f}%)"

    entry = {
        "score": round(score, 2),
        "struct": round(struct_score, 2),
        "temp": round(temperature, 2),
        "phase": phase,
        "result": result,
        "diff_vs_baseline": diff_vs_baseline,
    }
    session.setdefault("history", []).append(entry)

    # Best-Code updaten (separat von Baseline!)
    if score > session.get("best_score", 0.0):
        session["best_score"] = round(score, 2)
        session["best_struct"] = round(struct_score, 2)
        session["best_code"] = c_code


def build_history_context(session: dict, max_entries: int = 10) -> str:
    """
    Baut den History-Kontext fuer den KI-Prompt.
    
    Struktur:
    1. BASELINE CODE — der originale Input (unveraenderlich, alle Diffs gehen hiervon aus)
    2. CURRENT BEST  — der bisher beste Code (vollstaendig)
    3. HISTORY        — alle Versuche als Diffs gegen die Baseline
    """
    history = session.get("history", [])
    baseline = session.get("baseline_code", "")
    best_code = session.get("best_code", "")
    best_score = session.get("best_score", 0)

    if not baseline and not history:
        return ""

    lines = ["<previous_attempts>"]

    # 1. Baseline (unveraenderlich — der Ausgangspunkt)
    if baseline:
        lines.append(f"BASELINE CODE (original input):")
        lines.append("```c")
        lines.append(baseline.strip())
        lines.append("```")
        lines.append("")

    # 2. Current Best (der bisher beste Stand)
    if best_code and best_code.strip() != baseline.strip():
        lines.append(f"CURRENT BEST CODE (score: {best_score}%):")
        lines.append("```c")
        lines.append(best_code.strip())
        lines.append("```")
        lines.append("")

    # 3. History — alle Versuche als Diffs gegen Baseline
    if history:
        lines.append(f"ATTEMPT HISTORY ({len(history)} total, showing last {min(len(history), max_entries)}):")
        lines.append("All diffs are relative to the BASELINE CODE above.")
        lines.append("")

        recent = history[-max_entries:]
        for i, entry in enumerate(recent):
            n = len(history) - len(recent) + i + 1
            result = entry.get("result", "?")
            diff = entry.get("diff_vs_baseline", "")

            lines.append(f"  #{n}: score={entry['score']}% {result}")
            if diff:
                for dl in diff.split("\n")[:15]:
                    lines.append(f"    {dl}")
            else:
                lines.append(f"    (identical to baseline)")
            lines.append("")

    lines.append("CRITICAL: Do NOT repeat any approach that resulted in 'NO CHANGE' or 'WORSE'.")
    lines.append("The baseline and all diffs above show exactly what was already tried.")
    lines.append("</previous_attempts>")
    return "\n".join(lines)