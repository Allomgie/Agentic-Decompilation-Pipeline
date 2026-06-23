# modules/diff_generator.py
# Erzeugt ein strukturiertes JSON-Diff aus kompilierter .o und originaler .s Datei.
# Vermeidet Kaskaden-Diffs durch isoliertes Opcode-Alignment.

import os
import re
import json
import subprocess
from difflib import SequenceMatcher

OBJDUMP = "mips-linux-gnu-objdump"

# --- Wurzel-Gruppierung fuer GROSSE Funktionen -----------------------------------
# Ab so vielen Gesamt-Mismatches bekommt eine Funktion das verdichtete Wurzel-Format
# (statt Cap/Focus-Anzeige). Kleine/mittlere Funktionen (<= Schwelle) behalten den
# bisherigen Diff (sie funktionieren). Per Env uebersteuerbar.
DIFF_GROUP_THRESHOLD = int(os.environ.get("DIFF_GROUP_THRESHOLD", "12"))

_REG_ONE = re.compile(r"(?:zero|at|v[01]|a[0-3]|t[0-9]|s[0-8]|k[01]|gp|sp|fp|ra|f\d+)")
_REG_TOK = re.compile(r"\b(?:zero|at|v[01]|a[0-3]|t[0-9]|s[0-8]|k[01]|gp|sp|fp|ra|f\d+)\b")

def _mask_regs(s):
    return _REG_TOK.sub("<r>", str(s))

def _stack_off(s):
    m = re.search(r"(-?\d+)\(sp\)", str(s))
    return int(m.group(1)) if m else None

def _first_reg(s):
    for tok in str(s).split(","):
        tok = tok.strip()
        if _REG_ONE.fullmatch(tok):
            return tok
    return "?"

def _group_systematic(entries):
    """Verdichtet systematische Symptom-Schwaerme zu WURZEL-Aussagen — OHNE Information
    zu verlieren: jeder Mismatch ist entweder als Einzel-Eintrag gezeigt ODER in einer
    Wurzel vollstaendig erfasst (Wurzel traegt die komplette Zuordnung + 'symptoms').
    Wurzeln: 'Register order' (gleiche Werte, andere Register) + 'Stack frame' (Frame-
    Shift). Alles andere (Missing/Extra/Instruction Mismatch/echt verschiedene Register)
    bleibt als Einzel-Eintrag, ungekappt."""
    from collections import Counter
    REG_TYPES = ("Address Load", "Register/Immediate")
    reg, stack, frame, rest = [], [], [], []
    for e in entries:
        t = e.get("type")
        if t in REG_TYPES and isinstance(e.get("target"), str) and isinstance(e.get("draft"), str):
            reg.append(e)
        elif (t == "Memory Access (stack)" and isinstance(e.get("target"), str)
              and _stack_off(e.get("target")) is not None
              and _stack_off(e.get("draft")) is not None
              and _stack_off(e.get("target")) != _stack_off(e.get("draft"))):
            stack.append(e)
        elif t == "Stack Frame Mismatch":
            frame.append(e)
        else:
            rest.append(e)

    roots = []
    # --- WURZEL 1: Register order (gleiche Werte, andere Register) ---
    if reg:
        tvals = Counter(_mask_regs(e["target"]) for e in reg)
        dvals = Counter(_mask_regs(e["draft"]) for e in reg)
        explained, residual = [], []
        for e in reg:
            if _mask_regs(e["target"]) in dvals and _mask_regs(e["draft"]) in tvals:
                explained.append(e)
            else:
                residual.append(e)
        if len(explained) >= 2:
            tmap, dmap = {}, {}
            for e in reg:
                tmap.setdefault(_mask_regs(e["target"]), _first_reg(e["target"]))
                dmap.setdefault(_mask_regs(e["draft"]), _first_reg(e["draft"]))
            mapping = []
            _seen_map = set()
            for v in tmap:
                if v in dmap and tmap[v] != dmap[v]:
                    val = v.replace("<r>,", "").replace("<r>", "").strip(", ")
                    line = f"{val}: target {tmap[v]} / draft {dmap[v]}"
                    if line not in _seen_map:
                        _seen_map.add(line)
                        mapping.append(line)
            roots.append({
                "type": "ROOT: Register order",
                "symptoms": len(explained),
                "hint": ("Same values loaded into the WRONG registers — the register-allocation "
                         "order differs (usually your local variable declaration/use order). "
                         "Reorder so each value lands in the target register; all of these snap "
                         "together at once."),
                "mapping": mapping,
            })
            rest.extend(residual)
        else:
            rest.extend(reg)
    # --- WURZEL 2: Stack frame (Frame-Shift) ---
    if frame or stack:
        deltas = Counter()
        slots = []
        for e in stack:
            deltas[_stack_off(e["target"]) - _stack_off(e["draft"])] += 1
            slots.append(f'{e.get("draft")}  ->  {e.get("target")}')
        dom = max(deltas, key=deltas.get) if deltas else 0
        # WICHTIG: target = Ziel (soll), draft = aktuell (ist). Nicht vertauschen!
        fr = (f'your current frame is {frame[0].get("draft","?")}, the TARGET frame is '
              f'{frame[0].get("target","?")}; ' if frame else "")
        roots.append({
            "type": "ROOT: Stack frame",
            "symptoms": len(frame) + len(stack),
            "hint": (f"{fr}{len(stack)} stack slots shifted by {dom:+d} bytes. Fix the frame size "
                     f"/ local layout — every slot snaps back together."),
            "slots": slots,
        })

    pr = {"Stack Frame Mismatch": 0, "Missing in Draft": 1, "Extra in Draft": 2, "Reordered": 2,
          "Instruction Mismatch": 3, "Memory Access (stack)": 4, "Memory Access": 5,
          "Address Load": 6, "Register/Immediate": 7}
    rest.sort(key=lambda e: pr.get(e.get("type", ""), 99))
    return roots + rest

_I_TYPE_OPCODES = {
    0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    0x20, 0x21, 0x23, 0x24, 0x25, 0x28, 0x29, 0x2B, 0x31, 0x35, 0x39, 0x3D,
}


def _is_reloc_only_diff(hex_a: str, hex_b: str) -> bool:
    """True wenn sich zwei Hex-Codes nur im relocatable Teil unterscheiden."""
    if not hex_a or not hex_b or len(hex_a) != 8 or len(hex_b) != 8:
        return False
    if hex_a == hex_b:
        return True
    top = int(hex_a[0:2], 16)
    opcode = top >> 2
    if opcode in (0x02, 0x03):
        return hex_a[0:2] == hex_b[0:2]
    if opcode in _I_TYPE_OPCODES:
        return hex_a[:4] == hex_b[:4]
    return False


def _merge_reordered(entries: list) -> list:
    """Fasst 'Missing in Draft'- + 'Extra in Draft'-Instruktionen mit IDENTISCHEM
    Hex zu 'Reordered'-Eintraegen zusammen: dieselbe Instruktion, nur an anderer
    Position — statt sie doppelt (fehlt hier / extra dort) zu zaehlen. Uebrige
    bleiben als Missing/Extra. Andere Eintragstypen unveraendert."""
    miss, extr, others = [], [], []
    for e in entries:
        if e["type"] == "Missing in Draft":
            for h, r, p in e.get("_items", []):
                miss.append([h, r, p, False])
        elif e["type"] == "Extra in Draft":
            for h, r, p in e.get("_items", []):
                extr.append([h, r, p, False])
        else:
            others.append(e)
    reordered = []
    for m in miss:
        if not m[0]:
            continue
        for x in extr:
            if not x[3] and m[0] == x[0]:
                m[3] = x[3] = True
                reordered.append({"type": "Reordered", "instr": m[1],
                                  "target_pos": m[2], "draft_pos": x[2]})
                break
    out = others + reordered
    rem_m = [m[1] for m in miss if not m[3]]
    if rem_m:
        e = {"type": "Missing in Draft", "missing": rem_m}
        hint = _likely_permuter(rem_m)
        if hint:
            e["likely_permuter"] = hint
        out.append(e)
    rem_x = [x[1] for x in extr if not x[3]]
    if rem_x:
        e = {"type": "Extra in Draft", "extra": rem_x}
        hint = _likely_permuter(rem_x)
        if hint:
            e["likely_permuter"] = hint
        out.append(e)
    return out


def _raw_mn_ops(raw: str):
    """(mnemonic, operands) aus einer Roh-Instruktionszeile (objdump/spimdisasm)."""
    s = re.sub(r"/\*.*?\*/", "", str(raw))
    s = re.sub(r"^\s*[0-9a-fA-F]+:\s*[0-9a-fA-F]{8}\s*", "", s)   # objdump 'addr:\thex'
    s = s.replace("\t", " ").strip().replace("$", "")
    m = re.match(r"([a-z][a-z0-9.]*)\s*(.*)$", s)
    if not m:
        return None, None
    return m.group(1), m.group(2).replace(" ", "")


def _likely_permuter(raws: list):
    """Konservative, NICHT-destruktive Klassifikation auf ROH-Strings: besteht das gesamte
    Segment NUR aus eindeutig EMERGENTEN Artefakten (kein entfernbarer/fehlender C-Eingriff)?
    -> Grund-String, sonst None. Bewusst KONSERVATIV (nur unmissverstaendliche Faelle), damit
    genuine Faelle wie `move v0,zero` aus `return 0;` NICHT faelschlich markiert werden:
      - ori rX,rX,imm           = untere Haelfte einer lui+ori-Konstante (Float/Adresse)
      - andi rX,rY,0xffff|0xff  = Int-Promotion-Maske (Guide: 'x & 0xffff')
    Nur ein Hinweis-FELD am Eintrag — Zaehler/Typ unveraendert (regressionssicher)."""
    reasons = set()
    for raw in raws:
        mn, ops = _raw_mn_ops(raw)
        if mn == "ori":
            parts = ops.split(",")
            if len(parts) == 3 and parts[0] == parts[1]:
                reasons.add("constant-load half (lui+ori)")
                continue
            return None
        if mn == "andi" and re.search(r",(0xffff|65535|0xff|255)$", ops):
            reasons.add("int-promotion mask (& 0xffff)")
            continue
        return None  # nicht-eindeutig-emergente Instruktion -> kein Hinweis
    return ", ".join(sorted(reasons)) if reasons else None


def create_json_diff(draft_o_path: str, target_s_path: str, focus: str = "all") -> str:
    """
    Wandelt kompilierte .o und originale .s Dateien in ein LLM-freundliches JSON-Diff um.
    
    focus:
        "structure" — nur Stack Frame, Missing/Extra Instructions (erste Runden)
        "detail"    — nur Register, Memory, Immediate Mismatches (spaetere Runden)
        "all"       — alles (Fallback)
    """
    draft_insns = _parse_objdump(draft_o_path)
    target_insns = _parse_s_file(target_s_path)

    if not draft_insns or not target_insns:
        return json.dumps([{"type": "Error", "message": "Failed to parse assembly files."}])

    # Strip alignment padding: trailing NOPs in draft beyond target length
    while (len(draft_insns) > len(target_insns)
           and draft_insns and draft_insns[-1].get("hex") == "00000000"):
        draft_insns.pop()

    summary = {
        "type": "Summary",
        "focus": focus,
        "target_count": len(target_insns),
        "draft_count": len(draft_insns),
        "diff": len(draft_insns) - len(target_insns),
    }

    # Opcode-basiertes Alignment
    draft_opcodes = [ins["opcode"] for ins in draft_insns]
    target_opcodes = [ins["opcode"] for ins in target_insns]

    # autojunk=False ZWINGEND: bei Sequenzen >=200 markiert difflibs autojunk-Heuristik haeufige Opcodes
    # (lw/addiu/sw/nop kommen dutzendfach vor = >1%) als "junk" und ignoriert sie -> zerstoertes Alignment
    # grosser Funktionen -> stark aufgeblaehtes/falsches Missing(md). Bug 2026-06-20 (chdippy md 155 statt 23,
    # chtoothyfish 0 statt 54). Kleine Funktionen (<200) waren unberuehrt.
    sm = SequenceMatcher(None, target_opcodes, draft_opcodes, autojunk=False)
    all_entries = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for idx in range(i2 - i1):
                t_ins = target_insns[i1 + idx]
                d_ins = draft_insns[j1 + idx]
                # Identische Bytes = identische Instruktion -> NIE ein Diff (faengt
                # Disassembler-Pseudonyme wie li/ori, move/or generisch ab).
                if t_ins.get("hex") and t_ins["hex"] == d_ins.get("hex"):
                    continue
                # Reloc-Felder sind auf BEIDEN Seiten zu RELOC normalisiert
                # (Target: %hi/%lo; Draft: via -dr Reloc-Tabelle). Echte Immediate-/
                # Offset-Diffs bleiben erhalten -> der Operanden-Vergleich ist
                # autoritativ. (Frueher: _is_reloc_only_diff-Hex-Trick verbarg ALLE
                # I-Type-Immediate-Diffs faelschlich als 'reloc-only'.)
                if t_ins["operands"] != d_ins["operands"]:
                    all_entries.append(_analyze_operand_mismatch(t_ins, d_ins, i1 + idx))

        elif tag == "replace":
            # 1:1-Block mit identischen Bytes = dieselbe Instruktion (Disassembler-
            # Pseudonym, z.B. li=ori) -> KEIN Diff.
            if (i2 - i1 == 1 and j2 - j1 == 1
                    and target_insns[i1].get("hex")
                    and target_insns[i1]["hex"] == draft_insns[j1].get("hex")):
                continue
            all_entries.append({
                "type": "Instruction Mismatch",
                "pos": f"target[{i1}:{i2}] vs draft[{j1}:{j2}]",
                "target": [ins["raw"] for ins in target_insns[i1:i2]],
                "draft": [ins["raw"] for ins in draft_insns[j1:j2]],
            })

        elif tag == "delete":
            seg = target_insns[i1:i2]
            # nop-only = Delay-Slot/Scheduling-Artefakt (kein C-Eingriff -> Permuter). Nicht als
            # entfernbares/fehlendes Symptom melden, sonst jagt die KI ein Phantom + _diff_is_empty
            # bleibt faelschlich offen. (match_rate ist davon unberuehrt, eigene .o-Quelle.)
            if all(ins.get("opcode") == "nop" for ins in seg):
                continue
            all_entries.append({
                "type": "Missing in Draft",
                "pos": f"target[{i1}:{i2}]",
                "missing": [ins["raw"] for ins in seg],
                "_items": [(ins.get("hex", ""), ins["raw"], i1 + k)
                           for k, ins in enumerate(seg)],
            })

        elif tag == "insert":
            seg = draft_insns[j1:j2]
            if all(ins.get("opcode") == "nop" for ins in seg):
                continue
            all_entries.append({
                "type": "Extra in Draft",
                "pos": f"draft[{j1}:{j2}]",
                "extra": [ins["raw"] for ins in seg],
                "_items": [(ins.get("hex", ""), ins["raw"], j1 + k)
                           for k, ins in enumerate(seg)],
            })

    # Umsortierte Instruktionen (Missing+Extra DERSELBEN Instruktion) -> 'Reordered'
    all_entries = _merge_reordered(all_entries)

    # Focus-Filter
    STRUCTURE_TYPES = {"Stack Frame Mismatch", "Missing in Draft", "Extra in Draft",
                       "Instruction Mismatch", "Reordered"}
    DETAIL_TYPES = {"Memory Access (stack)", "Memory Access", "Register/Immediate", "Address Load"}

    if focus == "structure":
        filtered = [e for e in all_entries if e.get("type") in STRUCTURE_TYPES]
        # Wenn keine strukturellen Probleme mehr: automatisch zu detail wechseln
        if not filtered:
            filtered = [e for e in all_entries if e.get("type") in DETAIL_TYPES]
            summary["focus"] = "detail (auto — no structural issues left)"
    elif focus == "detail":
        filtered = [e for e in all_entries if e.get("type") in DETAIL_TYPES]
        if not filtered:
            filtered = all_entries  # Fallback: alles zeigen
    else:
        filtered = all_entries

    # Prioritaets-Sortierung
    priority = {"Stack Frame Mismatch": 0, "Missing in Draft": 1, "Extra in Draft": 2,
                "Reordered": 2, "Instruction Mismatch": 3, "Memory Access (stack)": 4,
                "Memory Access": 5, "Address Load": 6, "Register/Immediate": 7}
    filtered.sort(key=lambda e: priority.get(e.get("type", ""), 99))

    # GROSSE Funktion (> Schwelle Gesamt-Mismatches): Wurzel-Format ueber ALLE Eintraege.
    # KLEINE Funktion (<= Schwelle): ALLE Eintraege flach (prioritaets-sortiert).
    # In BEIDEN Faellen wird NICHTS vor der KI versteckt — der Focus-Filter (structure/detail)
    # diente nur dem Kappen und wird hier bewusst ignoriert (er versteckte sonst echte Mismatches,
    # z.B. Instruction-Mismatches bei focus=detail). Vollstaendig, immer.
    diff_log = [summary]
    if len(all_entries) > DIFF_GROUP_THRESHOLD:
        summary["mode"] = "grouped (large function — complete, roots first)"
        diff_log.extend(_group_systematic(all_entries))
    else:
        diff_log.extend(sorted(all_entries, key=lambda e: priority.get(e.get("type", ""), 99)))

    return json.dumps(diff_log, indent=2)


_ALIAS_EXPAND = {
    "move":   ("or",    "append_zero"),
    "not":    ("nor",   "append_zero"),
    "li":     ("addiu", "insert_zero"),
    "negu":   ("subu",  "insert_zero"),
    "neg":    ("sub",   "insert_zero"),
    "bnez":   ("bne",   "insert_zero"),
    "beqz":   ("beq",   "insert_zero"),
    "bnezl":  ("bnel",  "insert_zero"),
    "beqzl":  ("beql",  "insert_zero"),
}


def _canonicalize_alias(opcode: str, operands: str) -> tuple:
    if opcode not in _ALIAS_EXPAND:
        return opcode, operands
    canonical, transform = _ALIAS_EXPAND[opcode]
    parts = operands.split(",") if operands else []
    if len(parts) != 2:
        return opcode, operands
    if transform == "append_zero":
        return canonical, f"{parts[0]},{parts[1]},zero"
    else:
        return canonical, f"{parts[0]},zero,{parts[1]}"


def _analyze_operand_mismatch(t_ins: dict, d_ins: dict, position: int) -> dict:
    """Kategorisiert Operanden-Abweichungen. Keine Implications — nur rohe Daten."""
    opcode = t_ins["opcode"]
    t_ops = t_ins["operands"]
    d_ops = d_ins["operands"]

    # Kategorisierung
    if opcode in ("addiu", "addi") and "sp,sp" in t_ops and "sp,sp" in d_ops:
        mtype = "Stack Frame Mismatch"
    elif opcode in ("lw", "sw", "lh", "sh", "lb", "sb", "lhu", "lbu", "lwc1", "swc1"):
        mtype = "Memory Access" + (" (stack)" if "sp" in t_ops else "")
    elif opcode in ("lui",):
        mtype = "Address Load"
    else:
        mtype = "Register/Immediate"

    return {
        "type": mtype,
        "pos": position,
        "op": opcode,
        "target": t_ops,
        "draft": d_ops,
    }


def _parse_objdump(obj_path: str) -> list:
    """Disassembliert .o Datei und extrahiert normalisierte Opcodes/Operanden."""
    insns = []
    if not obj_path or not os.path.exists(obj_path):
        return insns

    try:
        result = subprocess.run(
            [OBJDUMP, "-dr", "-z", obj_path],   # -dr: Relocations mitnehmen
            capture_output=True, text=True, timeout=10
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return insns

    reloc_re = re.compile(r"R_MIPS_\w+")
    for line in result.stdout.splitlines():
        # Format: "   14:   27bdffe8    addiu   sp,sp,-24"
        match = re.search(
            r"^\s*[0-9a-fA-F]+:\s+([0-9a-fA-F]{8})\s+([a-z][a-z0-9_.]*)\s*(.*)?",
            line,
        )
        if match:
            ins = _normalize_instruction(match.group(2), match.group(3), line,
                                         hex_code=match.group(1))
            if ins:
                insns.append(ins)
            continue
        # Reloc-Folgezeile (z.B. "  4: R_MIPS_LO16  D_80079AC0") markiert das
        # zuletzt geparste Wort als relocatable und maskiert sein Immediate-Feld zu
        # RELOC — exakt wie die Target-Seite (%hi/%lo). So verschwinden REINE
        # Reloc-Unterschiede, waehrend echte Immediate/Offset-Diffs erhalten bleiben.
        if reloc_re.search(line) and insns:
            insns[-1]["is_reloc"] = True
            rm = re.search(r"R_MIPS_\w+\s+(\S+)", line)
            # .lower() fuer Konsistenz mit der Target-Seite (_normalize lowercased
            # die Operanden -> sonst Case-Mismatch desselben Symbols = False Positive).
            sym = rm.group(1).split("+")[0].lower() if rm else None
            insns[-1]["reloc_sym"] = sym
            insns[-1]["operands"] = _mask_reloc_operand(insns[-1]["operands"], sym)

    return insns


def _mask_reloc_operand(operands: str, sym: str = None) -> str:
    """Maskiert das relocatable Immediate-Feld zu RELOC@<sym> (symbol-bewusst,
    passend zur Target-Seite). Greift nur, wenn noch kein RELOC vorhanden ist."""
    if "RELOC" in operands:
        return operands
    tok = f"RELOC@{sym}" if sym else "RELOC"
    # Load/Store: IMM(reg) -> RELOC@sym(reg)
    new = re.sub(r"-?\d+(?=\()", tok, operands, count=1)
    if new != operands:
        return new
    # addiu/ori/lui etc.: letztes ,IMM -> ,RELOC@sym
    return re.sub(r",-?\d+$", f",{tok}", operands)


def _parse_s_file(s_path: str) -> list:
    """
    Liest originale .s Datei (spimdisasm-Format) und extrahiert Instruktionen.
    Nur aus dem .text-Bereich.
    
    spimdisasm Format:
        /* ROM VRAM HEX */ opcode operands
        /* 1F70078 80800248 248EFFC0 */  addiu  $t6, $a0, -0x40
    """
    insns = []
    if not s_path or not os.path.exists(s_path):
        return insns

    in_text = False

    with open(s_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            # Sektions-Tracking
            if stripped.startswith(".section"):
                in_text = ".text" in stripped
                continue
            if stripped.startswith("glabel"):
                in_text = True
                continue
            if stripped.startswith("endlabel"):
                in_text = False
                continue

            if not in_text:
                continue

            # Skip Labels, Direktiven
            if not stripped or stripped.startswith((".L", ".", "#", "dlabel", "enddlabel")):
                continue

            # spimdisasm: /* ... HEX8 */  opcode  operands
            # Wir matchen das letzte 8-Zeichen-Hex im Kommentar + den Opcode danach
            m = re.search(
                r"/\*.*?([0-9a-fA-F]{8})\s*\*/\s+([a-z][a-z0-9_.]*)\s*(.*)",
                stripped,
            )
            if m:
                ins = _normalize_instruction(m.group(2), m.group(3), stripped,
                                             hex_code=m.group(1))
                if ins:
                    insns.append(ins)
                continue

            # Fallback: raw format
            m = re.match(r"([0-9a-fA-F]{8})\s+([a-z][a-z0-9_.]*)\s*(.*)", stripped)
            if m:
                ins = _normalize_instruction(m.group(2), m.group(3), stripped,
                                             hex_code=m.group(1))
                if ins:
                    insns.append(ins)

    return insns


def _normalize_instruction(opcode: str, operands: str, raw_line: str,
                           hex_code: str = "") -> dict:
    """
    Normalisiert Opcodes und Operanden fuer den Vergleich.
    Alle numerischen Werte werden in dezimal normalisiert (kein hex vs decimal Mismatch).
    Pseudo-Opcodes werden auf ihre kanonische Form gebracht (move->or, li->addiu, etc.).
    """
    opcode = opcode.lower().strip()

    if not operands:
        operands = ""

    # Normalisierung: $a0 -> a0, Leerzeichen weg
    operands = operands.split("#")[0]
    operands = operands.lower().replace(" ", "").replace("$", "").strip()

    # Entferne objdump-Annotations wie "<func_800123bc+0x20>" und "<func_800123bc>"
    operands = re.sub(r"<[^>]+>", "", operands)

    # Reloc-Symbol (Daten) merken BEVOR maskiert wird.
    reloc_sym = None
    msym = re.search(r"%\w+\(\s*([A-Za-z_]\w*)", operands)
    if msym:
        reloc_sym = msym.group(1)

    # %got/%call16/%lo/%hi SYMBOL-BEWUSST maskieren: RELOC@<symbol>. So bleiben
    # ECHTE Symbol-Unterschiede (draft D_X vs target D_Y) im Vergleich sichtbar,
    # waehrend gleiche Symbole matchen. (Frueher: generisch RELOC -> Symbol-Diffs
    # versteckt -> leere Diffs / Unter-Berichtung.)
    operands = re.sub(r"%\w+\(\s*([A-Za-z_]\w*)[^)]*\)",
                      lambda m: f"RELOC@{m.group(1)}", operands)
    operands = re.sub(r"%\w+\([^)]*\)", "RELOC", operands)  # ohne klaren Symbolnamen

    # Hex zu Dezimal normalisieren: 0x18 -> 24, -0x18 -> -24
    def _hex_to_dec(m):
        val = m.group(0)
        try:
            if val.startswith("-"):
                return str(-int(val[1:], 16))
            return str(int(val, 16))
        except ValueError:
            return val
    operands = re.sub(r"-?0x[0-9a-f]+", _hex_to_dec, operands)

    # spimdisasm computed Konstanten AUSWERTEN (nicht maskieren): (V>>16), (V&N)
    # -> literaler Wert, damit er direkt mit dem Draft-Literal (lui/ori) vergleichbar
    # ist. So matchen gleiche Float-/Adress-Konstanten und ECHTE Konstanten-Diffs
    # bleiben sichtbar. (Laeuft NACH hex->dezimal, damit auch (0x..>>16) erfasst wird.)
    operands = re.sub(r"\((\d+)>>16\)", lambda m: str(int(m.group(1)) >> 16), operands)
    operands = re.sub(r"\((\d+)\s*&\s*(\d+)\)",
                      lambda m: str(int(m.group(1)) & int(m.group(2))), operands)

    # Pseudo-Opcode Kanonisierung (move->or, li->addiu, bnezl->bnel, etc.)
    opcode, operands = _canonicalize_alias(opcode, operands)

    # Branch: NUR Target (= letzter Operand) maskieren, Bedingungsregister behalten.
    # Target kann dezimal, HEX (objdump zeigt z.B. ',7c'), Label (.l..) oder Symbol
    # sein — alle zu ADDR. (Frueher nur dezimal -> Hex-Targets = False Positives.)
    if opcode.startswith("b") and opcode != "break":
        if "," in operands:
            operands = re.sub(r",[^,]+$", ",ADDR", operands)
        elif operands:
            operands = "ADDR"
    elif opcode.startswith("j"):
        if opcode in ("jal", "j"):
            operands = "ADDR"
        operands = re.sub(r"\.l\w+", "ADDR", operands)

    # KEINE pauschale lui-Maskierung mehr: relozierte lui werden ueber %hi (Target)
    # bzw. die Reloc-Tabelle (Draft, _parse_objdump) zu RELOC@<sym>; literale lui-
    # Konstanten bleiben erhalten und werden korrekt als Diff gezeigt, wenn sie
    # abweichen. (Frueher unterdrueckte das pauschale Masking echte Konstanten.)

    return {
        "opcode": opcode,
        "operands": operands,
        "raw": raw_line.strip()[:120],
        "hex": hex_code.lower() if hex_code else "",
        "reloc_sym": reloc_sym,
    }