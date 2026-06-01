# modules/diff_generator.py
# Erzeugt ein strukturiertes JSON-Diff aus kompilierter .o und originaler .s Datei.
# Vermeidet Kaskaden-Diffs durch isoliertes Opcode-Alignment.

import os
import re
import json
import subprocess
from difflib import SequenceMatcher

OBJDUMP = "mips-linux-gnu-objdump"

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

    sm = SequenceMatcher(None, target_opcodes, draft_opcodes)
    all_entries = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for idx in range(i2 - i1):
                t_ins = target_insns[i1 + idx]
                d_ins = draft_insns[j1 + idx]
                t_hex = t_ins.get("hex", "")
                d_hex = d_ins.get("hex", "")
                # Hex-Match oder Relocation-Only-Diff -> kein realer Diff
                if t_hex and d_hex and _is_reloc_only_diff(t_hex, d_hex):
                    continue
                if t_ins["operands"] != d_ins["operands"]:
                    all_entries.append(_analyze_operand_mismatch(t_ins, d_ins, i1 + idx))

        elif tag == "replace":
            all_entries.append({
                "type": "Instruction Mismatch",
                "pos": f"target[{i1}:{i2}] vs draft[{j1}:{j2}]",
                "target": [ins["raw"] for ins in target_insns[i1:i2]],
                "draft": [ins["raw"] for ins in draft_insns[j1:j2]],
            })

        elif tag == "delete":
            all_entries.append({
                "type": "Missing in Draft",
                "pos": f"target[{i1}:{i2}]",
                "missing": [ins["raw"] for ins in target_insns[i1:i2]],
            })

        elif tag == "insert":
            all_entries.append({
                "type": "Extra in Draft",
                "pos": f"draft[{j1}:{j2}]",
                "extra": [ins["raw"] for ins in draft_insns[j1:j2]],
            })

    # Focus-Filter
    STRUCTURE_TYPES = {"Stack Frame Mismatch", "Missing in Draft", "Extra in Draft", "Instruction Mismatch"}
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
                "Instruction Mismatch": 3, "Memory Access (stack)": 4, "Memory Access": 5,
                "Address Load": 6, "Register/Immediate": 7}
    filtered.sort(key=lambda e: priority.get(e.get("type", ""), 99))

    # Limitieren
    max_entries = 8
    diff_log = [summary]
    if len(filtered) > max_entries:
        remaining = len(filtered) - max_entries
        diff_log.extend(filtered[:max_entries])
        diff_log.append({"type": "Note", "remaining": remaining,
                         "total_mismatches": len(all_entries)})
    else:
        diff_log.extend(filtered)
        if len(all_entries) > len(filtered):
            diff_log.append({"type": "Note",
                             "other_mismatches_hidden": len(all_entries) - len(filtered)})

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
            [OBJDUMP, "-d", "-z", obj_path],
            capture_output=True, text=True, timeout=10
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return insns

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

    return insns


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

    # %got/%call16/%lo/%hi Relocations maskieren (Linker-Artefakte)
    operands = re.sub(r"%\w+\([^)]*\)", "RELOC", operands)

    # spimdisasm computed relocations: (VALUE>>16), (VALUE&65535)
    operands = re.sub(r"\(\d+>>16\)", "RELOC", operands)
    operands = re.sub(r"\(\d+&\d+\)", "RELOC", operands)

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

    # Pseudo-Opcode Kanonisierung (move->or, li->addiu, bnezl->bnel, etc.)
    opcode, operands = _canonicalize_alias(opcode, operands)

    # Branch: NUR Target maskieren, Bedingungsregister behalten
    if opcode.startswith("b"):
        operands = re.sub(r",\d{2,}$", ",ADDR", operands)
        operands = re.sub(r",\.l\w+", ",ADDR", operands)
        operands = re.sub(r",func_\w+", ",ADDR", operands)
    elif opcode.startswith("j"):
        if opcode in ("jal", "j"):
            operands = "ADDR"
        operands = re.sub(r"\.l\w+", "ADDR", operands)

    # lui: Immediate maskieren — RELOC (gleicher Token wie %hi/%lo)
    if opcode == "lui" and "sp" not in operands:
        operands = re.sub(r",\d+$", ",RELOC", operands)

    return {
        "opcode": opcode,
        "operands": operands,
        "raw": raw_line.strip()[:120],
        "hex": hex_code.lower() if hex_code else "",
    }