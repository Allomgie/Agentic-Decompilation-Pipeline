# modules/guided_weights.py
# Diff-gesteuerte Pass-Gewichtung fuer den --guided Modus von batch_permuter.py.
#
# Der externe decomp-permuter waehlt seine Randomization-Passes GEWICHTET-ZUFAELLIG
# (siehe randomizer.py: random_weighted(self.methods)). Es gibt keine Moeglichkeit,
# einen einzelnen Pass deterministisch "aufzurufen" — aber die Verteilung laesst sich
# ueber [weight_overrides] in der settings.toml des Projekts steuern (main.py liest sie
# und merged sie ueber die Default-Gewichte).
#
# Dieses Modul ist REINE LOGIK (keine Seiteneffekte, kein Permuter-Aufruf): es nimmt
# den JSON-Diff aus modules/diff_generator.py, klassifiziert die Problemzonen und
# liefert ein weight_overrides-Dict. STRATEGIE: relevante Passes BOOSTEN, irrelevante
# leicht DAEMPFEN (nie 0). Damit bleibt der Suchraum vollstaendig erreichbar (jede
# Pass-Sequenz bleibt moeglich), nur die Wahrscheinlichkeitsverteilung verschiebt sich.
#
# Zusaetzlich eine KONSERVATIVE Unsolvable-Klassifikation: Signale, die mit KEINEM
# semantik-erhaltenden Pass reparierbar sind (literale Konstanten-Diffs, falsche
# Nicht-Stack-Struct-Offsets). Diese wird im guided-Modus zunaechst NUR berichtet,
# nie zum Skippen benutzt — bis die Offline-Simulation die False-Positive-Rate auf
# bekannt-loesbaren Funktionen (struct_matches) belegt.

import json
import re
from pathlib import Path

try:
    import toml
except ImportError:  # pragma: no cover
    toml = None


# ---------------------------------------------------------------------------
# Diff-Kategorie  ->  zu boostende Passes
# ---------------------------------------------------------------------------
# Begruendung pro Eintrag (welche AST-Transformation erzeugt typischerweise diese
# Art ASM-Aenderung). Bewusst KONSERVATIV: lieber ein paar Passes zu viel boosten
# als einen noetigen Zwischenschritt unterdruecken (das passiert ohnehin nie, da
# wir nicht prunen, sondern nur daempfen).
_GUIDED_PASS_MAP = {
    # WURZEL: systematischer Register-Tausch (diff_generator _group_systematic) ->
    # Register-Vergabe haengt bei IDO (priority-based coloring, Chow&Hennessy) an
    # Deklarationsreihenfolge + Live-Ranges -> Passes, die genau das umwuerfeln.
    "ROOT: Register order": ["perm_temp_for_expr", "perm_reorder_decls",
                             "perm_split_assignment", "perm_add_self_assignment",
                             "perm_dummy_comma_expr"],
    # WURZEL: Frame-Shift (konsumiert "Stack Frame Mismatch" + verschobene
    # "Memory Access (stack)"-Eintraege) -> Frame-Layout-Hebel.
    "ROOT: Stack frame": ["perm_temp_for_expr", "perm_reorder_decls",
                          "perm_pad_var_decl"],
    # Gleiche Instruktion, andere Position -> Statement/Decl-Umordnung
    "Reordered": ["perm_reorder_stmts", "perm_reorder_decls", "perm_sameline"],
    # Register/Immediate-Abweichung -> Maskierung, Cast, kommutative Umformung.
    # Der Typ deckt AUCH unsystematische reine Register-Diffs ab (Residuen, die
    # nicht in eine ROOT-Mapping passen) -> temp_for_expr/reorder_decls mit dabei,
    # damit der Workhorse-Pass hier nie faelschlich gedaempft wird.
    "Register/Immediate": ["perm_add_mask", "perm_cast_simple", "perm_xor_zero",
                           "perm_commutative", "perm_refer_to_var",
                           "perm_temp_for_expr", "perm_reorder_decls"],
    # Fehlende Instruktion(en) im Draft -> Temp/Expand/Block einfuehren, inlinen
    "Missing in Draft": ["perm_temp_for_expr", "perm_expand_expr", "perm_ins_block",
                         "perm_inline"],
    # Ueberzaehlige Instruktion(en) -> Temp/Expand/AST-Entfernung
    "Extra in Draft": ["perm_temp_for_expr", "perm_expand_expr", "perm_remove_ast"],
    # Speicherzugriff (nicht-Stack) -> Struct-Ref-Form, Variablen-Referenz
    "Memory Access": ["perm_struct_ref", "perm_refer_to_var"],
    # Stack-Speicherzugriff -> Frame-Layout: Temp/Decl-Umordnung/Padding
    "Memory Access (stack)": ["perm_temp_for_expr", "perm_reorder_decls",
                              "perm_pad_var_decl"],
    # lui/Adress-Load -> Float-Literal-Form, Maskierung
    "Address Load": ["perm_float_literal", "perm_add_mask"],
    # Anderer Opcode-Block -> kommutativ, Ungleichungen, add/sub, Faktorisierung
    "Instruction Mismatch": ["perm_commutative", "perm_inequalities", "perm_add_sub",
                             "perm_compound_assignment", "perm_factor_mult",
                             "perm_factor_shift"],
    # Stack-Frame-Groesse -> Frame-Layout
    "Stack Frame Mismatch": ["perm_temp_for_expr", "perm_reorder_decls",
                             "perm_pad_var_decl"],
}

# Diff-Typen, die KEINE Pass-Boosts ausloesen (Meta/Infozeilen).
_META_TYPES = {"Summary", "Note", "Error"}

# Default-Boost/Damp (im --guided Modus per Flag uebersteuerbar).
DEFAULT_BOOST = 5.0
DEFAULT_DAMP = 0.3
# Untergrenze, damit ein gedaempfter Pass NIE auf 0 faellt (Suchraum bleibt erreichbar).
_MIN_WEIGHT = 0.1


# ---------------------------------------------------------------------------
# Default-Gewichte laden
# ---------------------------------------------------------------------------
def load_ido_default_weights(permuter_root):
    """Liest decomp-permuter/default_weights.toml und merged [base] mit [ido],
    exakt wie get_default_randomization_weights('ido') im Permuter. Rueckgabe:
    dict[pass_name -> float]."""
    if toml is None:
        raise RuntimeError("python-toml nicht verfuegbar")
    path = Path(permuter_root) / "default_weights.toml"
    obj = toml.load(open(path, encoding="utf-8"))
    base = {k: float(v) for k, v in obj.get("base", {}).items()}
    for k, v in obj.get("ido", {}).items():
        base[k] = float(v)
    return base


# ---------------------------------------------------------------------------
# Diff-Parsing-Helfer
# ---------------------------------------------------------------------------
def _entry_instruction_weight(e):
    """Gewicht eines Diff-Eintrags fuer das Gate (= "ungeklaerte Distanz").

    ROOT-Wurzeln zaehlen BEWUSST als 1, nicht als e['symptoms']: das Gate soll
    Funktionen aussortieren, bei denen die Diff->Pass-Abbildung Rauschen ist —
    eine Wurzel ist aber per Konstruktion ein SYSTEMATISCHES Hochsicherheits-
    Muster mit EINER Ursache (egal wie viele Symptome). Wuerde man symptoms
    zaehlen, fielen genau die besten Kandidaten (grosse Register-Schwaerme =
    struct_matches) faelschlich ins Gate."""
    t = e.get("type")
    if t == "Missing in Draft":
        return max(1, len(e.get("missing", [1])))
    if t == "Extra in Draft":
        return max(1, len(e.get("extra", [1])))
    if t == "Instruction Mismatch":
        return max(len(e.get("target", [])), len(e.get("draft", [])), 1)
    return 1


def _last_imm(operands):
    """Letztes literales Immediate eines normalisierten Operandenstrings (oder None).
    z.B. 't6,a0,-64' -> -64 ; 'a0,RELOC@x' -> None ; 't6,t7' -> None."""
    if not operands or "RELOC" in operands:
        return None
    m = re.search(r",(-?\d+)$", operands)
    return int(m.group(1)) if m else None


def _mem_offset(operands):
    """Literaler Speicher-Offset 'IMM(reg)' (oder None bei RELOC/keinem)."""
    if not operands or "RELOC" in operands:
        return None
    m = re.search(r"(-?\d+)\(", operands)
    return int(m.group(1)) if m else None


def _is_stack(operands):
    return bool(operands) and "sp" in operands


def _mask_last_imm(operands):
    """Ersetzt das letzte literale Immediate durch IMM (Rest unveraendert)."""
    return re.sub(r"(-?\d+)$", "IMM", operands)


def _mask_mem_off(operands):
    """Ersetzt den literalen Speicher-Offset IMM(reg) durch IMM (Rest unveraendert)."""
    return re.sub(r"-?\d+(?=\()", "IMM", operands)


# ---------------------------------------------------------------------------
# Unsolvable-Klassifikation (konservativ, NUR Hochsicherheits-Signale)
# ---------------------------------------------------------------------------
def _unsolvable_reasons(entries):
    """Sammelt Signale, die mit KEINEM semantik-erhaltenden Pass reparierbar sind.
    Konservativ: nur literale Konstanten-Diffs und falsche Nicht-Stack-Offsets.
    Reine Register-Diffs (loesbar via Reg-Alloc-Umordnung) loesen NICHTS aus.
    Rueckgabe: Liste von (confidence, reason)-Tupeln, confidence in {'high','low'}."""
    reasons = []
    for e in entries:
        t = e.get("type")
        op = e.get("op", "")
        tgt = e.get("target", "")
        drf = e.get("draft", "")
        if not isinstance(tgt, str) or not isinstance(drf, str):
            continue  # Block-Eintraege (Listen) hier nicht betrachten

        if t in ("Register/Immediate", "Address Load"):
            ti, di = _last_imm(tgt), _last_imm(drf)
            # STRENG: nur flaggen, wenn ALLES ausser dem letzten Immediate identisch ist
            # (gleiche Register). Abweichende Register -> evtl. Reg-Alloc -> NICHT flaggen.
            if (ti is not None and di is not None and ti != di and not _is_stack(tgt)
                    and _mask_last_imm(tgt) == _mask_last_imm(drf)):
                reasons.append(("high",
                    f"literaler Konstanten-Diff ({op}: {ti} vs {di}) — kein Pass "
                    f"erfindet eine andere Konstante"))

        elif t == "Memory Access":  # NICHT "Memory Access (stack)"
            to, do = _mem_offset(tgt), _mem_offset(drf)
            # STRENG: nur flaggen, wenn Ziel- UND Basisregister identisch sind und sich
            # NUR der Offset unterscheidet (= echtes falsches Feld, keine Umadressierung).
            if (to is not None and do is not None and to != do
                    and _mask_mem_off(tgt) == _mask_mem_off(drf)):
                reasons.append(("high",
                    f"Nicht-Stack-Offset-Diff ({op}: {to} vs {do}) — vermutlich "
                    f"falsches Struct-Feld, kein Pass erfindet ein Feld"))

        elif t == "Instruction Mismatch":
            # Opcode-Wechsel kann durch kommutativ/add_sub/factor erreichbar sein
            # -> nur als LOW-confidence Hinweis melden, NIE als hartes Signal.
            reasons.append(("low",
                "Opcode-Block-Diff — evtl. ausserhalb der Pass-Reichweite (pruefen)"))
    return reasons


# ---------------------------------------------------------------------------
# Hauptentscheidung
# ---------------------------------------------------------------------------
def compute_overrides(present_types, defaults, boost=DEFAULT_BOOST, damp=DEFAULT_DAMP):
    """Aus den vorhandenen Diff-Typen ein weight_overrides-Dict bauen.
    Geboostete Passes: default*boost. Alle uebrigen: default*damp (>= _MIN_WEIGHT).
    Gibt nur Keys zurueck, die vom Default abweichen. Leer, wenn nichts zu boosten."""
    boost_passes = set()
    for t in present_types:
        boost_passes.update(_GUIDED_PASS_MAP.get(t, []))
    boost_passes &= set(defaults)  # nur real existierende Passes
    if not boost_passes:
        return {}, set()

    overrides = {}
    for name, w in defaults.items():
        if name in boost_passes:
            nw = round(w * boost, 3)
        else:
            nw = round(max(_MIN_WEIGHT, w * damp), 3)
        if abs(nw - w) > 1e-9:
            overrides[name] = nw
    return overrides, boost_passes


def guided_decision(diff_json, defaults, threshold=6,
                    boost=DEFAULT_BOOST, damp=DEFAULT_DAMP):
    """Komplette guided-Entscheidung fuer eine Funktion.

    diff_json: JSON-String aus diff_generator.create_json_diff(..).
    defaults:  IDO-Default-Gewichte (load_ido_default_weights).
    threshold: max. Mismatch-Anzahl, ab der NICHT mehr fokussiert wird (Gate).

    Rueckgabe-Dict:
      ok               -> Diff parsebar
      mismatch_total   -> geschaetzte Anzahl abweichender Instruktionen
      present_types    -> sortierte Liste der vorhandenen Diff-Kategorien
      gated            -> True, wenn ueber Schwelle (=> keine Overrides)
      guidance         -> True, wenn Overrides erzeugt wurden
      overrides        -> dict (leer, wenn keine Guidance)
      boosted_passes   -> sortierte Liste der geboosteten Passes
      unsolvable       -> True bei >=1 high-confidence Unsolvable-Signal
      unsolvable_reasons -> Liste (confidence, text)
    """
    res = {"ok": False, "mismatch_total": 0, "present_types": [], "gated": False,
           "guidance": False, "overrides": {}, "boosted_passes": [],
           "unsolvable": False, "unsolvable_reasons": []}
    try:
        entries = json.loads(diff_json)
    except (ValueError, TypeError):
        return res
    if not isinstance(entries, list):
        return res
    res["ok"] = True

    if any(e.get("type") == "Error" for e in entries):
        res["ok"] = False
        return res

    real = [e for e in entries if e.get("type") not in _META_TYPES]
    present = {e.get("type") for e in real}
    res["present_types"] = sorted(present)

    # Mismatch-Anzahl (inkl. versteckter, aus Note-Zeilen)
    total = sum(_entry_instruction_weight(e) for e in real)
    for e in entries:
        if e.get("type") == "Note":
            total += int(e.get("remaining", 0)) + int(e.get("other_mismatches_hidden", 0))
    res["mismatch_total"] = total

    # Unsolvable-Klassifikation (immer berechnen, nur berichten)
    reasons = _unsolvable_reasons(real)
    res["unsolvable_reasons"] = reasons
    res["unsolvable"] = any(c == "high" for c, _ in reasons)

    if total == 0:
        return res  # nichts abweichend -> keine Guidance noetig
    if total > threshold:
        res["gated"] = True
        return res  # zu weit weg -> Default-Gewichte (Altverhalten)

    overrides, boosted = compute_overrides(present, defaults, boost, damp)
    if overrides:
        res["guidance"] = True
        res["overrides"] = overrides
        res["boosted_passes"] = sorted(boosted)
    return res


def render_weight_overrides_toml(overrides):
    """Erzeugt einen [weight_overrides]-TOML-Block (nur fuer Debug/Anzeige).
    NICHT zum Anhaengen an die vom Import erzeugte settings.toml benutzen — die hat
    bereits eine (auskommentierte) [weight_overrides]-Tabelle, ein zweiter Header
    waere doppelt und damit ein TOML-Parse-Fehler. Zum Schreiben apply_weight_overrides()."""
    if not overrides:
        return ""
    lines = ["", "# --- von batch_permuter --guided erzeugt (Diff-fokussierte Gewichte) ---",
             "[weight_overrides]"]
    for name in sorted(overrides):
        lines.append(f"{name} = {overrides[name]}")
    return "\n".join(lines) + "\n"


def apply_weight_overrides(settings_path, overrides):
    """Schreibt overrides ROBUST in eine bestehende settings.toml: parst die Datei,
    mergt in die (evtl. leere) weight_overrides-Tabelle und schreibt zurueck. Vermeidet
    den doppelten [weight_overrides]-Header (= TOML-Fehler), den ein blindes Anhaengen
    erzeugen wuerde. Rueckgabe True bei Erfolg, sonst False."""
    if toml is None or not overrides:
        return False
    try:
        with open(settings_path, encoding="utf-8") as f:
            data = toml.load(f)
        wo = data.get("weight_overrides") or {}
        wo.update(overrides)
        data["weight_overrides"] = wo
        with open(settings_path, "w", encoding="utf-8") as f:
            toml.dump(data, f)
        return True
    except (OSError, ValueError, toml.TomlDecodeError):
        return False
