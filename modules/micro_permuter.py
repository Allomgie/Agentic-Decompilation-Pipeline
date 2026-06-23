# modules/micro_permuter.py
# Pass-basierter Micro-Permuter v2.
# Passes: Regex-Heuristiken + AST-Permutation (pycparser).
# CompileCache, Permuter-Report fuer KI-Feedback.

import re
import os
import hashlib
import logging
from difflib import SequenceMatcher

from modules import ido_compiler
from modules import ido_comparison

log = logging.getLogger(__name__)

# Versuche pycparser zu laden (optional — AST-Pass nur wenn verfuegbar)
try:
    import pycparser
    HAS_PYCPARSER = True
except ImportError:
    HAS_PYCPARSER = False


# =====================================================================
# COMPILE CACHE
# =====================================================================

class _CompileCache:
    def __init__(self):
        self._results = {}
        self.stats = {"cache_hits": 0, "ido_calls": 0, "ido_ok": 0, "ido_fail": 0}

    def _hash(self, code):
        return hashlib.md5(code.strip().encode()).hexdigest()[:16]

    def get_score(self, code, func_name, target_skel, label=""):
        h = self._hash(code)
        if h in self._results:
            self.stats["cache_hits"] += 1
            return self._results[h][0]

        self.stats["ido_calls"] += 1
        comp = ido_compiler.compile_code(code, func_name)
        if not comp["success"]:
            self.stats["ido_fail"] += 1
            self._results[h] = (-1.0, label)
            ido_compiler.cleanup_temp(comp.get("temp_dir", ""))
            return -1.0

        try:
            gen_words = ido_comparison.extract_words_from_objdump(comp["temp_o_path"])
            if not gen_words:
                self._results[h] = (-1.0, label)
                return -1.0
            gen_skel = ido_comparison.build_skeleton(gen_words)
            score = SequenceMatcher(None, target_skel, gen_skel, autojunk=False).ratio() * 100.0  # autojunk=False: >=200 Worte sonst verfaelscht
            self.stats["ido_ok"] += 1
            self._results[h] = (score, label)
            return score
        finally:
            ido_compiler.cleanup_temp(comp.get("temp_dir", ""))


# =====================================================================
# HELPER: Sicheres Argument-Parsing (kein naives split(","))
# =====================================================================

def _split_c_args(args_str):
    """
    Splittet C-Funktionsargumente sicher am Komma.
    Respektiert verschachtelte Klammern: void (*cb)(s32, s32) bleibt intakt.
    """
    args = []
    depth = 0
    current = []
    for ch in args_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            args.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append(''.join(current).strip())
    return args


# =====================================================================
# PASS DEFINITIONS
# =====================================================================

def _pass_return_type(c_code, func_name, **_):
    """Return-Typ Variationen: void/int/s32."""
    variants = []
    for old_t in ["int", "s32", "void"]:
        if f"{old_t} {func_name}" not in c_code:
            continue
        for new_t in ["void", "s32", "int"]:
            if new_t == old_t:
                continue
            v = c_code.replace(f"{old_t} {func_name}", f"{new_t} {func_name}", 1)
            if new_t == "void":
                v = re.sub(r"return\s+[^;]+;", "return;", v)
            variants.append((f"ret_{old_t}_to_{new_t}", v))
    return variants


def _pass_decl_reorder(c_code, **_):
    """
    Adaptiver Deklarations-Reorder (Greedy, Score-guided).
    
    Statt n! Permutationen (720 fuer 6 Vars, 3.6M fuer 10):
    Greedy-Ansatz mit n*(n-1) Varianten pro Runde.
    
    Strategie: Generiere alle paarweisen Swaps (nicht nur benachbarte).
    Der aeussere Loop in run_permuter iteriert, bis kein Swap mehr hilft.
    Das findet die optimale Reihenfolge in O(n^2 * rounds) statt O(n!).
    
    Ignoriert Leerzeilen und Kommentare zwischen Deklarationen.
    Keine Obergrenze fuer die Anzahl der Variablen.
    """
    variants = []
    lines = c_code.split("\n")

    decl_pattern = re.compile(r"^\s+(s32|u32|s16|u16|s8|u8|f32|int|float|void\s*\*)\s+\w+")
    skip_pattern = re.compile(r"^\s*$|^\s*//|^\s*/\*.*\*/\s*$")
    decl_indices = [i for i, l in enumerate(lines) if decl_pattern.match(l)]

    if len(decl_indices) < 2:
        return variants

    # Finde zusammenhaengende Bloecke
    contiguous = [decl_indices[0]]
    for k in range(1, len(decl_indices)):
        gap_ok = True
        for g in range(decl_indices[k - 1] + 1, decl_indices[k]):
            if not skip_pattern.match(lines[g]):
                gap_ok = False
                break
        if gap_ok:
            contiguous.append(decl_indices[k])
        else:
            break

    if len(contiguous) < 2:
        return variants

    # Generiere ALLE paarweisen Swaps (nicht nur benachbarte)
    # Bei n Variablen = n*(n-1)/2 Varianten — fuer 10 Vars nur 45 statt 3.6M
    for a in range(len(contiguous)):
        for b in range(a + 1, len(contiguous)):
            idx_a, idx_b = contiguous[a], contiguous[b]
            sw = list(lines)
            sw[idx_a], sw[idx_b] = sw[idx_b], sw[idx_a]
            # Label zeigt welche Variablen getauscht wurden
            name_a = lines[idx_a].strip().split()[-1].rstrip(";") if lines[idx_a].strip() else f"L{idx_a}"
            name_b = lines[idx_b].strip().split()[-1].rstrip(";") if lines[idx_b].strip() else f"L{idx_b}"
            variants.append((f"decl_swap_{name_a}_{name_b}", "\n".join(sw)))

    return variants


def _pass_type_change(c_code, asm, **_):
    """Datentyp-Aenderungen basierend auf ASM-Opcodes."""
    variants = []
    syms = set(re.findall(r"\b(local_\d+|D_[0-9A-Fa-f]{8})\b", c_code))
    has_byte = bool(re.search(r"\b(sb|lbu|lb)\b", asm))
    has_half = bool(re.search(r"\b(sh|lhu|lh)\b", asm))
    has_float = bool(re.search(r"\b(lwc1|swc1|add\.s|mul\.s|div\.s|cvt\.)\b", asm))

    for sym in syms:
        if f"s32 {sym}" not in c_code:
            continue
        if has_byte:
            variants.append((f"u8_{sym}", c_code.replace(f"s32 {sym}", f"u8 {sym}")))
        if has_half:
            variants.append((f"s16_{sym}", c_code.replace(f"s32 {sym}", f"s16 {sym}")))
            variants.append((f"u16_{sym}", c_code.replace(f"s32 {sym}", f"u16 {sym}")))
        if has_float:
            variants.append((f"f32_{sym}", c_code.replace(f"s32 {sym}", f"f32 {sym}")))
    return variants


def _pass_cond_swap(c_code, **_):
    """
    Bedingungs-Tausch. Nutzt Klammer-Matching statt einfacher Regex
    fuer robustere Erkennung.
    """
    variants = []
    # Finde "if (" und matche die aeusseren Klammern
    for m in re.finditer(r"\bif\s*\(", c_code):
        start = m.end() - 1  # Position der oeffnenden Klammer
        depth = 0
        end = start
        for i in range(start, len(c_code)):
            if c_code[i] == '(':
                depth += 1
            elif c_code[i] == ')':
                depth -= 1
                if depth == 0:
                    end = i
                    break

        cond = c_code[start + 1:end].strip()

        # Suche den Vergleichsoperator auf der aeussersten Ebene
        flip = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "==": "==", "!=": "!="}
        d = 0
        for op in ["==", "!=", "<=", ">=", "<", ">"]:
            # Finde op auf Tiefe 0
            idx = -1
            d = 0
            for i in range(len(cond)):
                if cond[i] == '(':
                    d += 1
                elif cond[i] == ')':
                    d -= 1
                elif d == 0 and cond[i:i + len(op)] == op:
                    # Pruefe dass es nicht Teil eines laengeren Ops ist
                    before = cond[i - 1] if i > 0 else ""
                    after = cond[i + len(op)] if i + len(op) < len(cond) else ""
                    if before not in "=!<>" and after not in "=":
                        idx = i
                        break
            if idx >= 0:
                left = cond[:idx].strip()
                right = cond[idx + len(op):].strip()
                new_cond = f"if ({right} {flip[op]} {left})"
                variants.append(("cond_swap", c_code[:m.start()] + new_cond + c_code[end + 1:]))
                break

    return variants


def _pass_volatile_args(c_code, func_name, **_):
    """Volatile Parameter-Spill mit sicherem Argument-Parsing."""
    variants = []
    m = re.search(rf"{re.escape(func_name)}\(([^)]*)\)\s*{{", c_code)
    if not m:
        return variants
    args_str = m.group(1)
    if not args_str.strip() or args_str.strip() == "void":
        return variants

    args = _split_c_args(args_str)
    for j, arg in enumerate(args):
        if "volatile" in arg:
            continue
        # Fuege volatile vor dem letzten Token (dem Parameternamen) ein
        tokens = arg.strip().rsplit(None, 1)
        if len(tokens) == 2:
            new_arg = f"{tokens[0]} volatile {tokens[1]}"
            new_args = list(args)
            new_args[j] = new_arg
            new_args_str = ", ".join(new_args)
            variants.append((f"volatile_arg{j}", c_code.replace(args_str, new_args_str, 1)))
    return variants


def _pass_extern_type_variants(c_code, asm, **_):
    """
    Extern-Variablen-Typ-Variationen.
    Ein 'extern s32 D_XXX' das eigentlich ein 'extern u8* D_XXX' sein sollte
    erzeugt komplett andere Load-Instruktionen (lw vs lbu).
    """
    variants = []
    for m in re.finditer(r"extern\s+(s32|u32|int)\s+(D_[0-9A-Fa-f]+)\s*;", c_code):
        old_type = m.group(1)
        sym = m.group(2)
        for new_type in ["u8", "s16", "u16", "f32", "u8*", "s32*", "void*"]:
            if new_type == old_type:
                continue
            variants.append((f"ext_{sym}_{new_type}",
                             c_code.replace(f"extern {old_type} {sym};", f"extern {new_type} {sym};")))
    return variants


def _pass_call_reorder(c_code, **_):
    """
    Tausche benachbarte Funktionsaufrufe (standalone statements).
    IDO alloziert Register basierend auf der Aufruf-Reihenfolge.
    """
    variants = []
    lines = c_code.split("\n")
    call_pattern = re.compile(r"^\s+\w+\s*\([^)]*\)\s*;\s*$")

    call_indices = [i for i, l in enumerate(lines) if call_pattern.match(l)]

    for k in range(len(call_indices) - 1):
        i, j = call_indices[k], call_indices[k + 1]
        if j == i + 1:  # Direkt benachbart
            sw = list(lines)
            sw[i], sw[j] = sw[j], sw[i]
            variants.append((f"call_swap_{k}", "\n".join(sw)))
    return variants


def _pass_pointer_deref(c_code, **_):
    """
    Pointer-Dereferenzierungs-Varianten.
    *ptr vs ptr[0] vs *(type*)ptr erzeugen unterschiedlichen Code bei IDO.
    """
    variants = []
    # *var -> var[0]
    for m in re.finditer(r"\*(\b[a-zA-Z_]\w*\b)", c_code):
        var = m.group(1)
        if var in ("void", "const", "volatile", "unsigned", "signed"):
            continue
        new = c_code[:m.start()] + f"{var}[0]" + c_code[m.end():]
        variants.append((f"deref_to_idx_{var}", new))

    # var[0] -> *var
    for m in re.finditer(r"(\b[a-zA-Z_]\w*)\[0\]", c_code):
        var = m.group(1)
        new = c_code[:m.start()] + f"*{var}" + c_code[m.end():]
        variants.append((f"idx_to_deref_{var}", new))

    return variants


def _pass_cond_negate(c_code, **_):
    """
    Bedingungs-Negation: if(a){X}else{Y} <-> if(!a){Y}else{X}
    Gleiche Semantik, aber IDO generiert unterschiedliche Branch-Instruktionen.
    """
    variants = []
    for m in re.finditer(r"\bif\s*\((!?)([^)]+)\)\s*\{", c_code):
        neg = m.group(1)
        cond = m.group(2)
        if neg:
            new_cond = f"if ({cond}) {{"
        else:
            if "&&" not in cond and "||" not in cond:
                new_cond = f"if (!({cond})) {{"
            else:
                continue
        variants.append(("cond_negate", c_code[:m.start()] + new_cond + c_code[m.end():]))
    return variants


def _pass_arithmetic_equiv(c_code, **_):
    """
    Arithmetische Aequivalenzen die bei IDO unterschiedlichen Code erzeugen.
    """
    variants = []

    # x * 2 -> x << 1 und umgekehrt
    for m in re.finditer(r"(\b[a-zA-Z_]\w*)\s*\*\s*2\b", c_code):
        variants.append(("mul2_to_shl1", c_code[:m.start()] + f"{m.group(1)} << 1" + c_code[m.end():]))
    for m in re.finditer(r"(\b[a-zA-Z_]\w*)\s*<<\s*1\b", c_code):
        variants.append(("shl1_to_mul2", c_code[:m.start()] + f"{m.group(1)} * 2" + c_code[m.end():]))

    # x / 2 -> x >> 1 (nur fuer unsigned)
    for m in re.finditer(r"(\b[a-zA-Z_]\w*)\s*/\s*2\b", c_code):
        variants.append(("div2_to_shr1", c_code[:m.start()] + f"{m.group(1)} >> 1" + c_code[m.end():]))

    # x / 4 -> x >> 2
    for m in re.finditer(r"(\b[a-zA-Z_]\w*)\s*/\s*4\b", c_code):
        variants.append(("div4_to_shr2", c_code[:m.start()] + f"{m.group(1)} >> 2" + c_code[m.end():]))

    # x + (-y) -> x - y und umgekehrt (selten aber IDO-relevant)
    for m in re.finditer(r"(\b[a-zA-Z_]\w*)\s*\+\s*\(-(\b[a-zA-Z_]\w*)\)", c_code):
        variants.append(("add_neg_to_sub",
                         c_code[:m.start()] + f"{m.group(1)} - {m.group(2)}" + c_code[m.end():]))

    return variants


def _pass_condition_negate(c_code, **_):
    """
    Bedingungs-Negation: if(a){X}else{Y} <-> if(!a){Y}else{X}
    Gleiche Semantik, aber IDO generiert unterschiedliche Branch-Instruktionen.
    """
    variants = []

    # Suche if-else Bloecke und tausche die Arme
    # Einfache Heuristik: if (COND) { ... } else { ... }
    for m in re.finditer(r"\bif\s*\((!?)([^)]+)\)\s*\{", c_code):
        neg = m.group(1)
        cond = m.group(2)
        if neg:
            # if (!cond) -> if (cond)
            new_cond = f"if ({cond}) {{"
        else:
            # if (cond) -> if (!({cond}))
            # Nur wenn die Bedingung einfach genug ist
            if "&&" not in cond and "||" not in cond:
                new_cond = f"if (!({cond})) {{"
            else:
                continue
        variants.append(("cond_negate", c_code[:m.start()] + new_cond + c_code[m.end():]))

    return variants


def _pass_temp_var_eliminate(c_code, **_):
    """
    Temporaere Variablen eliminieren oder einfuehren.
    'temp = func(); use(temp);' vs 'use(func());'
    IDO alloziert Register unterschiedlich.
    """
    variants = []
    lines = c_code.split("\n")

    # Finde: var = func_call(...); ... var ... (nur einmal benutzt)
    for i, line in enumerate(lines):
        m = re.match(r"(\s+)(\w+)\s*=\s*(\w+\([^)]*\))\s*;", line)
        if not m:
            continue
        indent = m.group(1)
        var = m.group(2)
        call = m.group(3)

        # Pruefe ob var nur in der naechsten Zeile benutzt wird
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            # Zaehle Vorkommen von var in restlichen Zeilen
            rest = "\n".join(lines[i + 1:])
            count = len(re.findall(rf"\b{re.escape(var)}\b", rest))
            if count == 1 and re.search(rf"\b{re.escape(var)}\b", next_line):
                # Inline: ersetze var durch den Call
                new_next = re.sub(rf"\b{re.escape(var)}\b", call, next_line, count=1)
                new_lines = list(lines)
                new_lines[i] = ""  # Zeile entfernen
                new_lines[i + 1] = new_next
                # Leere Zeilen bereinigen
                result = "\n".join(l for l in new_lines if l.strip() or not l)
                variants.append((f"inline_{var}", result))

    return variants


def _pass_stack_padding(c_code, func_name, asm, **_):
    """
    Stack-Frame-Groesse anpassen: Eine Variable hinzufuegen oder eine unbenutzte entfernen.
    Nur 2 Varianten (statt 6) um IDO-Calls zu sparen.
    """
    variants = []
    lines = c_code.split("\n")

    # Finde die oeffnende Klammer der Funktion
    brace_idx = -1
    for i, line in enumerate(lines):
        if "{" in line and func_name in "\n".join(lines[max(0, i - 2):i + 1]):
            brace_idx = i
            break
    if brace_idx < 0:
        return variants

    # Insert-Position: nach der letzten Deklaration
    decl_pattern = re.compile(r"^\s+(s32|u32|s16|u16|s8|u8|f32|int|float)\s+(\w+)")
    last_decl = brace_idx
    for i in range(brace_idx + 1, len(lines)):
        if decl_pattern.match(lines[i]):
            last_decl = i
        elif lines[i].strip() and not lines[i].strip().startswith("//"):
            break

    # Variante 1: Eine s32 Padding-Variable hinzufuegen
    new_lines = list(lines)
    new_lines.insert(last_decl + 1, "  s32 pad0;")
    variants.append(("stack_pad_add", "\n".join(new_lines)))

    # Variante 2: Letzte unbenutzte Variable entfernen
    for i in range(last_decl, brace_idx, -1):
        m = decl_pattern.match(lines[i])
        if m:
            var_name = m.group(2)
            rest = "\n".join(lines[i + 1:])
            if var_name not in rest:
                new_lines = list(lines)
                del new_lines[i]
                variants.append((f"stack_pad_rm_{var_name}", "\n".join(new_lines)))
            break

    return variants


def _pass_signed_unsigned(c_code, asm, **_):
    """
    Signed/Unsigned Vergleichsvarianten.
    'if (a < 0)' vs 'if ((s32)a < 0)' erzeugen bltz vs slt bei IDO.
    """
    variants = []

    # Fuege explizite Casts bei Vergleichen mit 0 hinzu
    for m in re.finditer(r"(\b[a-zA-Z_]\w*)\s*(>=|<=|>|<)\s*0\b", c_code):
        var = m.group(1)
        op = m.group(2)
        # (s32) Cast
        new = f"(s32){var} {op} 0"
        variants.append((f"s32_cmp_{var}", c_code[:m.start()] + new + c_code[m.end():]))
        # (u32) Cast
        new = f"(u32){var} {op} 0"
        variants.append((f"u32_cmp_{var}", c_code[:m.start()] + new + c_code[m.end():]))

    return variants


def _pass_cast_fixes(c_code, **_):
    """Cast-Korrekturen."""
    variants = []
    if "<<" in c_code:
        fix = re.sub(r"(\b[a-zA-Z_]\w*)\s*<<", r"((u32)\1) <<", c_code)
        if fix != c_code:
            variants.append(("u32_lshift", fix))
    for m in re.finditer(r"&([a-zA-Z0-9_]+)\[0\]", c_code):
        variants.append(("ptr_simplify", c_code[:m.start()] + m.group(1) + c_code[m.end():]))
    return variants


def _pass_fall_off(c_code, func_name, **_):
    """Fall-off-the-end Trick."""
    variants = []
    v = c_code
    if f"void {func_name}" in v:
        v = v.replace(f"void {func_name}", f"s32 {func_name}", 1)
    v = re.sub(r"return\s+([a-zA-Z0-9_]+\([^;]+\));", r"\1;", v)
    v = re.sub(r"return\s*;\s*}", "}", v)
    if v != c_code:
        variants.append(("fall_off", v))
    return variants


def _pass_or_zero(c_code, **_):
    """x | 0 IDO-Tricks fuer Register-Allokation."""
    variants = []
    for m in re.finditer(r"(\w+)\(([^)]+)\)", c_code):
        call = m.group(1)
        if call in ("if", "while", "for", "switch", "return"):
            continue
        args = _split_c_args(m.group(2))
        for i, arg in enumerate(args):
            if "| 0" not in arg and re.match(r"^[a-zA-Z_]\w*$", arg.strip()):
                new_args = list(args)
                new_args[i] = f"{arg.strip()} | 0"
                new_call = f"{call}({', '.join(new_args)})"
                variants.append((f"or0_{call}_a{i}", c_code.replace(m.group(0), new_call, 1)))
    return variants


# =====================================================================
# AST PERMUTATION PASS (pycparser)
# =====================================================================

def _pass_ast_stmt_reorder(c_code, func_name, **_):
    """
    AST-basierte Statement-Permutation innerhalb von Basisblöcken.
    Tauscht benachbarte unabhängige Statements (keine Kontrollfluss-Aenderung).
    Braucht pycparser.
    """
    if not HAS_PYCPARSER:
        return []

    variants = []
    lines = c_code.split("\n")

    # Finde "einfache" Statement-Zeilen (Zuweisungen, Funktionsaufrufe)
    # Keine Kontrollfluss-Statements (if, while, for, switch, return)
    stmt_pattern = re.compile(
        r"^\s+([a-zA-Z_]\w*(?:\[[^\]]*\])?(?:\.[a-zA-Z_]\w*)*"  # LHS
        r"\s*(?:=|\+=|-=|\|=|&=)\s*"  # Assignment op
        r".+;)\s*$"  # Rest + semicolon
    )
    call_pattern = re.compile(r"^\s+[a-zA-Z_]\w*\s*\([^)]*\)\s*;\s*$")

    stmt_indices = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith(("if", "while", "for", "switch", "return", "else", "{", "}", "//")):
            continue
        if stmt_pattern.match(line) or call_pattern.match(line):
            stmt_indices.append(i)

    # Tausche benachbarte Statements
    for k in range(len(stmt_indices) - 1):
        i, j = stmt_indices[k], stmt_indices[k + 1]
        if j == i + 1:
            sw = list(lines)
            sw[i], sw[j] = sw[j], sw[i]
            variants.append((f"stmt_swap_{k}", "\n".join(sw)))

    return variants


# =====================================================================
# SIMILAR-GUIDED PASSES (nutzen eine strukturell aehnliche Referenz-Funktion)
# =====================================================================
# WICHTIG: Diese Passes sind ADDITIV und durch das Score-Gate abgesichert. Die
# Referenz liefert nur Vorschlaege; jede Variante wird gegen das TARGET-Skelett
# gescored. Eine irrefuehrende Referenz verliert einfach den Wettbewerb — anders
# als im KI-Prompt (wo similar die Generierung verankert) kann sie hier nicht
# in die Irre fuehren.

_C_KEYWORDS = {"if", "while", "for", "switch", "return", "sizeof", "void",
               "const", "volatile", "unsigned", "signed", "do", "else", "case"}

# Lokale-Deklaration: optionaler unsigned/signed, ein Basistyp, optionale *, Name
_DECL_RE = re.compile(
    r"^\s*((?:unsigned\s+|signed\s+)?"
    r"(?:s32|u32|s16|u16|s8|u8|f32|f64|int|float|char|short|long|void)\s*\**)"
    r"\s+(\w+)\s*;\s*$")


def _extract_decls(code):
    """Liste (name, type) der lokalen Deklarationen, in Quelltext-Reihenfolge."""
    out = []
    for line in code.split("\n"):
        m = _DECL_RE.match(line)
        if m:
            out.append((m.group(2), m.group(1).strip()))
    return out


def _ordered_unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _defined_func_name(code):
    """Name der in diesem Code DEFINIERTEN Funktion (erste Signatur mit Body)."""
    m = re.search(r"\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*\{", code)
    return m.group(1) if m else None


def _callee_names(code, exclude=None):
    """Aufgerufene Funktionsnamen (ohne Keywords, ohne die def. Funktion)."""
    out = []
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", code):
        nm = m.group(1)
        if nm in _C_KEYWORDS or nm == exclude:
            continue
        out.append(nm)
    return _ordered_unique(out)


def _global_syms(code):
    """Globale Daten-/Rodata-Symbole. Erfasst auch Overlay-Symbole mit Suffix
    (z.B. D_80800D74_subaddietaxi) und rodata-Labels (R_...): nach dem Hex-Lauf
    darf ein optionaler _<name>-Suffix folgen. Frueher: r'\\bD_[0-9A-Fa-f]+\\b'
    -> verfehlte JEDES Suffix-Symbol (nach den Hex-Ziffern steht '_', ein
    Wortzeichen -> kein \\b), also alle Overlay-Globals."""
    return _ordered_unique(
        re.findall(r"\b[DR]_[0-9A-Fa-f]+(?:_\w+)?\b", code))


def _pass_similar_types(c_code, similar_ref=None, **_):
    """Vorschlag 1: Typ einer lokalen Variable auf den Typ aus der Referenz
    aendern (gleicher Name = gleicher Stack-Slot bei m2c-Namenskonvention)."""
    if not similar_ref:
        return []
    ref_types = dict(_extract_decls(similar_ref))
    if not ref_types:
        return []
    variants = []
    for name, cur_typ in _extract_decls(c_code):
        ref_typ = ref_types.get(name)
        if ref_typ and ref_typ != cur_typ:
            pat = re.compile(rf"(^\s*){re.escape(cur_typ)}(\s+{re.escape(name)}\s*;)", re.M)
            new = pat.sub(rf"\g<1>{ref_typ}\g<2>", c_code, count=1)
            if new != c_code:
                tag = ref_typ.replace(" ", "").replace("*", "p")
                variants.append((f"simtype_{name}_{tag}", new))
    return variants


def _pass_similar_decl_order(c_code, similar_ref=None, **_):
    """Vorschlag 2: Die lokalen Deklarationen in die Reihenfolge der Referenz
    bringen (eine gezielte Variante statt blinder O(n^2)-Swaps)."""
    if not similar_ref:
        return []
    ref_order = [nm for nm, _ in _extract_decls(similar_ref)]
    if len(ref_order) < 2:
        return []

    lines = c_code.split("\n")
    decl_idx = [i for i, l in enumerate(lines) if _DECL_RE.match(l)]
    if len(decl_idx) < 2:
        return []
    # zusammenhaengenden Block ab erster Decl (nur Leerzeilen dazwischen erlaubt)
    block = [decl_idx[0]]
    for k in range(1, len(decl_idx)):
        if all(not lines[g].strip() for g in range(decl_idx[k - 1] + 1, decl_idx[k])):
            block.append(decl_idx[k])
        else:
            break
    if len(block) < 2:
        return []

    cur = [(_DECL_RE.match(lines[i]).group(2), lines[i]) for i in block]
    rank = {nm: r for r, nm in enumerate(ref_order)}
    # stabil sortieren: bekannte Namen nach Ref-Rang, unbekannte ans Ende
    ordered = sorted(range(len(cur)),
                     key=lambda k: (rank.get(cur[k][0], len(ref_order) + k),))
    new_block = [cur[k][1] for k in ordered]
    if new_block == [ln for _, ln in cur]:
        return []  # bereits in Referenz-Reihenfolge
    new_lines = list(lines)
    for pos, i in enumerate(block):
        new_lines[i] = new_block[pos]
    return [("simorder_decls", "\n".join(new_lines))]


def _func_sig_match(text, fname):
    """re.Match der Funktionssignatur (bis inkl. oeffnender '{') von fname."""
    return re.search(
        rf"[A-Za-z_][\w\s\*]*\b{re.escape(fname)}\s*\([^;{{]*\)\s*\{{", text)


def _ref_function_only(similar_ref, ref_fname):
    """Nur die Referenz-FUNKTION (Signatur + Body), ohne ihre eigenen
    externs/includes — diese wuerden mit dem Preamble der aktuellen Datei
    kollidieren."""
    m = _func_sig_match(similar_ref, ref_fname)
    return similar_ref[m.start():] if m else similar_ref


def _braced_body(func_text):
    """Inhalt zwischen den aeussersten geschweiften Klammern (ohne die Klammern)."""
    i = func_text.find("{")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(func_text)):
        if func_text[j] == "{":
            depth += 1
        elif func_text[j] == "}":
            depth -= 1
            if depth == 0:
                return func_text[i + 1:j]
    return func_text[i + 1:]


def _map_namewise(ref_syms, cur_syms, mapping):
    """Namensbewusstes Mapping ref->cur. Symbole, die in BEIDEN identisch
    vorkommen, bleiben sich selbst zugeordnet (kein Eintrag). Nur die jeweils
    NICHT gemeinsamen Symbole werden — positionell, in Quelltext-Reihenfolge —
    gepaart. Verhindert die Fehler des reinen zip():
      - destruktive Swaps (ref [A,B] / cur [B,A] -> A->B UND B->A),
      - Fehlmapping eines eigentlich identischen Symbols durch Positionsversatz."""
    cur_set = set(cur_syms)
    ref_set = set(ref_syms)
    ref_left = [r for r in ref_syms if r not in cur_set]
    cur_left = [c for c in cur_syms if c not in ref_set]
    for r, c in zip(ref_left, cur_left):
        if r != c:
            mapping[r] = c


def _symbol_mapping(c_code, func_name, ref_func, ref_fname):
    """dict ref_symbol -> cur_symbol fuer Funktionsname, Callees und Globals.
    Namensbewusst (siehe _map_namewise): gemeinsame Symbole bleiben unveraendert,
    nur abweichende werden positionell gepaart."""
    mapping = {}
    if ref_fname and ref_fname != func_name:
        mapping[ref_fname] = func_name
    _map_namewise(_callee_names(ref_func, exclude=ref_fname),
                  _callee_names(c_code, exclude=func_name), mapping)
    _map_namewise(_global_syms(ref_func), _global_syms(c_code), mapping)
    return mapping


def _similar_prep(c_code, func_name, similar_ref):
    """Gemeinsame Vorbereitung fuer alle Similar-Strategien.
    Returns (ref_fname, ref_func, mapping) oder None."""
    if not similar_ref:
        return None
    ref_fname = _defined_func_name(similar_ref)
    if not ref_fname:
        return None
    ref_func = _ref_function_only(similar_ref, ref_fname)
    mapping = _symbol_mapping(c_code, func_name, ref_func, ref_fname)
    return ref_fname, ref_func, mapping


def _apply_mapping(text, mapping):
    """Zwei-Phasen-Ersetzung ueber Platzhalter (verhindert Ketten-Substitution,
    z.B. A->B und B->C duerfen sich nicht ueberlagern)."""
    out = text
    ph = {}
    for i, (r, c) in enumerate(mapping.items()):
        token = f"\x00PH{i}\x00"
        ph[token] = c
        out = re.sub(rf"\b{re.escape(r)}\b", token, out)
    for token, c in ph.items():
        out = out.replace(token, c)
    return out


def _pass_similar_transplant(c_code, func_name, similar_ref=None, **_):
    """Vorschlag 3: Skelett-Transplantat. Die Struktur der Referenz uebernehmen,
    aber deren Symbole (Funktionsname, Callees, Globals) positionsweise auf die
    Symbole der AKTUELLEN Funktion umschreiben. Erzeugt EINEN ganzen Kandidaten;
    das Preamble (externs/includes) der aktuellen Datei bleibt erhalten, damit es
    kompiliert. Score-Gate verwirft Fehlschlaege."""
    prep = _similar_prep(c_code, func_name, similar_ref)
    if not prep:
        return []
    ref_fname, ref_func, mapping = prep
    transplant = _rescue_stage1(c_code, func_name, similar_ref, ref_fname, ref_func, mapping)
    if not transplant or transplant.strip() == c_code.strip():
        return []
    return [("sim_transplant", transplant)]


def _pass_similar_unit_remap(c_code, func_name, similar_ref=None, **_):
    """Strategie wie Rescue-Stufe 2: ganze Similar-Einheit (inkl. eigener externs)
    umgemappt. Self-contained, kompiliert garantiert wenn das Similar kompiliert.
    Score-gated — wird nur uebernommen wenn es das Diff verbessert."""
    prep = _similar_prep(c_code, func_name, similar_ref)
    if not prep:
        return []
    ref_fname, ref_func, mapping = prep
    cand = _rescue_stage2(c_code, func_name, similar_ref, ref_fname, ref_func, mapping)
    if not cand or cand.strip() == c_code.strip():
        return []
    return [("sim_unit_remap", cand)]


def _pass_similar_sig_skeleton(c_code, func_name, similar_ref=None, **_):
    """Strategie wie Rescue-Stufe 3: aktuelle Signatur/Preamble behalten, nur den
    Referenz-RUMPF transplantieren. Score-gated."""
    prep = _similar_prep(c_code, func_name, similar_ref)
    if not prep:
        return []
    ref_fname, ref_func, mapping = prep
    cand = _rescue_stage3(c_code, func_name, similar_ref, ref_fname, ref_func, mapping)
    if not cand or cand.strip() == c_code.strip():
        return []
    return [("sim_sig_skeleton", cand)]


# =====================================================================
# PHASE-A RESCUE: kompilierbare Funktion aus dem Similar erbauen
# =====================================================================
# Anwendungsfall: Die aktuelle Funktion scheitert beim Kompilieren (Phase A),
# es existiert aber ein Similar. Das Similar IST eine kompilierbare Funktion —
# sie hat nur die falschen globalen Namen und evtl. Typabweichungen. Wir bauen
# drei Stufen unterschiedlicher "Fidelity" und nehmen die, die das beste Diff
# (bzw. ueberhaupt) kompiliert. Ziel ist NICHT 100%, sondern ein kompilierbarer
# Startpunkt, den der (Micro-/externe) Permuter weitertreiben kann.

def _rescue_stage1(c_code, func_name, similar_ref, ref_fname, ref_func, mapping):
    """Stufe 1 (hoechste Fidelity): Voll-Transplantat. Preamble der AKTUELLEN
    Datei (echte externs/Typen) + Referenz-Funktion mit komplett umgemappten
    Symbolen. Bestes Diff, kann aber scheitern wenn das Preamble nicht alle
    benoetigten Symbole deklariert."""
    remapped = _apply_mapping(ref_func, mapping)
    sig = _func_sig_match(c_code, func_name)
    preamble = c_code[:sig.start()] if sig else ""
    return (preamble + remapped) if preamble else remapped


def _rescue_stage2(c_code, func_name, similar_ref, ref_fname, ref_func, mapping):
    """Stufe 2 (mittlere Fidelity): Ganze Similar-Einheit (inkl. ihrer eigenen
    externs) umgemappt. Da die externs mit umbenannt werden, bleibt die Einheit
    self-contained und kompiliert garantiert (sofern das Similar kompiliert).
    Im Prinzip 'das Similar, nur mit den richtigen globalen Namen'."""
    return _apply_mapping(similar_ref, mapping)


def _rescue_stage3(c_code, func_name, similar_ref, ref_fname, ref_func, mapping):
    """Stufe 3 (niedrigste Fidelity, hoechste Compile-Sicherheit): Signatur-
    Skelett. Signatur + Preamble der AKTUELLEN Funktion bleiben unveraendert
    (korrektes Interface), nur der RUMPF der Referenz wird transplantiert und
    seine Symbole umgemappt. Hilft, wenn ref- und cur-Signatur abweichen."""
    body = _braced_body(ref_func)
    if body is None:
        return None
    # Funktionsname nicht im Body mappen (Signatur kommt von cur)
    body_map = {r: c for r, c in mapping.items() if r != ref_fname}
    body_remapped = _apply_mapping(body, body_map)
    sig = _func_sig_match(c_code, func_name)
    if not sig:
        return None
    head = c_code[:sig.end()]  # Preamble + Signatur inkl. '{'
    return f"{head}{body_remapped}\n}}\n"


def run_rescue(c_code, func_name, target_s_path, similar_ref):
    """Baut aus dem Similar eine kompilierbare Funktion. Erzeugt drei Stufen,
    kompiliert/bewertet jede gegen das Target-Skelett und waehlt das beste Diff
    unter den kompilierenden (Tie-Break: hoehere Fidelity Stufe1>2>3).

    Returns dict:
        compiled (bool), best_c_code (str), match_rate (float = struct score),
        stage (int|None), report (str).
    """
    result = {"compiled": False, "best_c_code": c_code, "match_rate": 0.0,
              "stage": None, "report": ""}
    prep = _similar_prep(c_code, func_name, similar_ref)
    if not prep:
        return result
    ref_fname, ref_func, mapping = prep

    target_words = ido_comparison.extract_words_from_original_s(target_s_path)
    target_skel = ido_comparison.build_skeleton(target_words) if target_words else []

    cache = _CompileCache()
    # Reihenfolge invertiert (3 -> 2 -> 1): erst die sichersten/niedrigsten,
    # dann die hochwertigeren. Report folgt dieser Reihenfolge.
    builders = [(3, _rescue_stage3), (2, _rescue_stage2), (1, _rescue_stage1)]
    report_lines = ["<rescue_report>",
                    f"Phase-A Rescue fuer {func_name} aus Similar {ref_fname}:", ""]
    best = None  # (score, stage, code)
    for stage_no, builder in builders:
        try:
            cand = builder(c_code, func_name, similar_ref, ref_fname, ref_func, mapping)
        except Exception as e:
            report_lines.append(f"  Stufe {stage_no}: BUILD-FEHLER ({e})")
            continue
        if not cand or not cand.strip():
            report_lines.append(f"  Stufe {stage_no}: leer/uebersprungen")
            continue
        score = cache.get_score(cand, func_name, target_skel, f"rescue_stage{stage_no}") \
            if target_skel else (-1.0)
        if score < 0:
            report_lines.append(f"  Stufe {stage_no}: COMPILE FAIL")
            continue
        report_lines.append(f"  Stufe {stage_no}: kompiliert, Struct={score:.1f}%")
        # Tie-Break: bei gleichem Score gewinnt die spaeter getestete = hoehere
        # Fidelity (Stufe1 zuletzt). Strikt groesser uebernimmt sowieso.
        if best is None or score >= best[0]:
            best = (score, stage_no, cand)

    if best is not None:
        result["compiled"] = True
        result["match_rate"] = round(best[0], 2)
        result["stage"] = best[1]
        result["best_c_code"] = best[2]
        report_lines.append("")
        report_lines.append(f"  => Gewaehlt: Stufe {best[1]} (Struct={best[0]:.1f}%)")
    else:
        report_lines.append("")
        report_lines.append("  => Keine Stufe kompilierte.")
    report_lines.append("</rescue_report>")
    result["report"] = "\n".join(report_lines)
    return result


# =====================================================================
# TARGET-GEFUEHRTE SYMBOL-/OFFSET-REPARATUR
# =====================================================================
# Anwendungsfall: Eine Funktion ist "struct-aehnlich" aber NICHT permuter-
# finishable, weil sie das FALSCHE Global-Symbol oder den falschen Offset
# referenziert (haeufig bei m2c-Drafts: Setter/Getter auf ein falsch geratenes
# D_-Symbol). Solche Fehler kann der Permuter NICHT beheben (er erfindet keine
# Symbole/Konstanten) — aber die assemblierte Target-.o KENNT die richtigen
# Werte. Wir lesen Symbol + Offset + Zugriffstyp aus der Target-.o und schreiben
# den C-Code darauf um. Verifiziert via evaluate_match -> nur ein Kandidat, der
# wirklich 100% (oder permuter-finishable) trifft, wird zurueckgegeben.

_LS_TYPE = {
    "lb": "s8", "lbu": "u8", "sb": "u8", "lh": "s16", "lhu": "u16", "sh": "u16",
    "lw": "s32", "sw": "s32", "lwc1": "f32", "swc1": "f32",
    "ldc1": "f64", "sdc1": "f64",
}
_IMM_PAREN = re.compile(r",\s*(-?0x[0-9A-Fa-f]+|-?\d+)\s*\(")
_IMM_TAIL = re.compile(r"(-?0x[0-9A-Fa-f]+|-?\d+)\s*$")


def _ls_immediate(ops):
    """Numerisches Immediate (Offset/Addend) aus objdump-Operanden ziehen:
    'reg, 0x24($at)' -> 0x24 ; 'reg, 0' -> 0."""
    m = _IMM_PAREN.search(ops)
    if m:
        return int(m.group(1), 0)
    tail = ops.rsplit(",", 1)
    if len(tail) == 2:
        m = _IMM_TAIL.match(tail[1].strip())
        if m:
            return int(m.group(1), 0)
    return 0


def _ls_refs(insns):
    """Geordnete Load/Store-Reloc-Referenzen (symbol, offset, c_type). Nur
    Instruktionen, die das LO16-Reloc UND den Offset tragen (die haeufige Form
    'lui hi; <load/store> lo(reg)'). HI16/jal/reine addiu-Adressberechnung werden
    ausgelassen -> v1 deckt skalare Globals ab."""
    out = []
    for w in insns:
        if not w["reloc"]:
            continue
        parts = w["reloc"].split()
        if len(parts) < 2 or parts[0] != "R_MIPS_LO16":
            continue
        if w["mnem"] not in _LS_TYPE:
            continue
        out.append((parts[1].split("+")[0], _ls_immediate(w["ops"]), _LS_TYPE[w["mnem"]]))
    return out


def _jal_refs(insns):
    """Geordnete Liste der aufgerufenen Funktionssymbole (R_MIPS_26 / jal)."""
    out = []
    for w in insns:
        if not w["reloc"]:
            continue
        parts = w["reloc"].split()
        if len(parts) >= 2 and parts[0] == "R_MIPS_26":
            out.append(parts[1].split("+")[0])
    return out


# Eine Deklarationszeile (kein Statement!): optional extern, Typ-Tokens, Symbol,
# optional LEERE oder numerische [] (Array-Decl), Semikolon. Statement-Keywords
# werden ausgeschlossen, damit z.B. 'return D_X[param_0];' NICHT als Deklaration
# missgedeutet wird; nicht-numerische Klammern (Index) ebenfalls nicht.
_DECL_LINE = (r"^[ \t]*(?!(?:return|if|while|for|switch|else|do|goto)\b)"
              r"(?:extern[ \t]+)?[A-Za-z_][\w \t\*]*\b{sym}\b"
              r"[ \t]*(?:\[[ \t]*\d*[ \t]*\])?[ \t]*;[ \t]*$")


def run_symbol_repair(c_code, func_name, target_s_path):
    """Liest aus der Target-.o das korrekte Symbol+Offset+Typ jeder Load/Store-
    Reloc-Referenz und schreibt den C-Code um. Verifiziert mit evaluate_match.

    Returns dict: compiled(bool), fixed(bool=100%/finishable), best_c_code,
    match_rate, struct_score, report.
    """
    res = {"compiled": False, "fixed": False, "best_c_code": c_code,
           "match_rate": 0.0, "struct_score": 0.0, "report": ""}

    comp = ido_compiler.compile_code(c_code, func_name)
    if not comp.get("success"):
        return res
    tgt_o = ido_comparison._assemble_target_o(target_s_path)
    if not tgt_o:
        ido_compiler.cleanup_temp(comp["temp_dir"])
        return res
    d_ins = ido_comparison._objdump_insns(comp["temp_o_path"])
    t_ins = ido_comparison._objdump_insns(tgt_o)
    d_refs, t_refs = _ls_refs(d_ins), _ls_refs(t_ins)
    d_jals, t_jals = _jal_refs(d_ins), _jal_refs(t_ins)
    ido_compiler.cleanup_temp(comp["temp_dir"])

    from collections import Counter

    # --- Daten-Load/Store-Refs (Symbol + Offset + Typ) ---
    # Nur wenn sich das (Symbol,Offset)-MULTISET unterscheidet — bei reinem
    # Reordering (gleiches Multiset) ist es Scheduling (permuter-finishable),
    # NICHT reparieren.
    mapping = {}
    if (d_refs and len(d_refs) == len(t_refs)
            and Counter((s, o) for s, o, _ in d_refs)
            != Counter((s, o) for s, o, _ in t_refs)):
        cnt = Counter(sd for sd, _, _ in d_refs)
        for (sd, ad, _), (st, at, tt) in zip(d_refs, t_refs):
            if (sd, ad) == (st, at):
                continue
            if cnt[sd] > 1 or (sd in mapping and mapping[sd] != (st, at, tt)):
                return res  # mehrdeutig
            mapping[sd] = (st, at, tt)

    # --- jal-Callees (B1): falscher Funktionsaufruf -> Callee positionell
    # umbenennen. NUR wenn das Callee-Multiset abweicht (echt andere Funktion);
    # bei gleichem Multiset = bloss umsortiert = Scheduling -> nicht anfassen.
    # Zwei-Phasen-Substitution loest Ketten/Swaps; verify-gegated.
    call_map = {}
    if (d_jals and len(d_jals) == len(t_jals)
            and Counter(d_jals) != Counter(t_jals)):
        jcnt = Counter(d_jals)
        for cd, ct in zip(d_jals, t_jals):
            if cd == ct:
                continue
            if jcnt[cd] > 1 or (cd in call_map and call_map[cd] != ct):
                return res  # mehrdeutig
            call_map[cd] = ct

    if not mapping and not call_map:
        return res

    # Rewrite: Deklarationszeilen der betroffenen Symbole entfernen, dann je nach
    # Verwendung umschreiben:
    #   - Array  S_d[i]  (Ziel-Offset = k*Elementgroesse) -> S_t[(i) + k]
    #   - Skalar S_d      -> (*(T*)((u8*)&S_t + A_t))
    # externs fuer die Ziel-Symbole ergaenzen.
    _ELEM = {"s8": 1, "u8": 1, "s16": 2, "u16": 2, "s32": 4, "f32": 4, "f64": 8}
    fixed = c_code
    involved = set(mapping) | {st for st, _, _ in mapping.values()}
    for sym in involved:
        fixed = re.sub(_DECL_LINE.format(sym=re.escape(sym)) + r"\n?", "", fixed, flags=re.M)

    ext_kind = {}   # S_t -> ("array", tt) | ("scalar", None)
    ph = {}
    for i, (sd, (st, at, tt)) in enumerate(mapping.items()):
        is_array = re.search(rf"\b{re.escape(sd)}\s*\[", fixed) is not None
        if is_array:
            elem = _ELEM.get(tt, 0)
            if not elem or at % elem != 0:
                return res  # Byte-Offset nicht index-darstellbar -> Fallback
            k = at // elem

            def _idx(m, st=st, k=k):
                inner = m.group(1).strip()
                return f"{st}[({inner}) + {k}]" if k else f"{st}[{inner}]"
            fixed = re.sub(rf"\b{re.escape(sd)}\s*\[([^\]\[]*)\]", _idx, fixed)
            ext_kind[st] = ("array", tt)
        else:
            expr = (f"(*({tt} *)((u8 *)&{st} + {hex(at)}))" if at
                    else f"(*({tt} *)&{st})")
            token = f"\x00SR{i}\x00"
            ph[token] = expr
            fixed = re.sub(rf"\b{re.escape(sd)}\b", token, fixed)
            ext_kind.setdefault(st, ("scalar", None))

    # jal-Callee-Renames (B1): Token -> Ziel-Funktionsname, ebenfalls zwei-phasig
    # (verhindert Ketten-Substitution bei Swaps wie A->B, B->C).
    for j, (cd, ct) in enumerate(call_map.items()):
        token = f"\x00JR{j}\x00"
        ph[token] = ct
        fixed = re.sub(rf"\b{re.escape(cd)}\b", token, fixed)

    for token, expr in ph.items():
        fixed = fixed.replace(token, expr)

    externs = ""
    for st in sorted(ext_kind):
        kind, tt = ext_kind[st]
        externs += f"extern {tt} {st}[];\n" if kind == "array" else f"extern u8 {st};\n"
    fixed = externs + fixed

    vcomp = ido_compiler.compile_code(fixed, func_name)
    if not vcomp.get("success"):
        res["report"] = "Symbol-Repair: Rewrite kompiliert nicht."
        return res
    rank = ido_comparison.evaluate_match(func_name, fixed, vcomp["temp_o_path"],
                                         target_s_path, update_leaderboard=False)
    ido_compiler.cleanup_temp(vcomp["temp_dir"])
    res.update({
        "compiled": True, "best_c_code": fixed,
        "match_rate": rank["match_rate"], "struct_score": rank["struct_score"],
        "fixed": rank["match_rate"] >= 100.0 or bool(rank.get("permuter_finishable")),
        "report": (f"Symbol-Repair: {len(mapping)} Daten-Ref + {len(call_map)} "
                   f"Callee korrigiert -> match={rank['match_rate']}% "
                   f"struct={rank['struct_score']}%"),
    })
    return res


# =====================================================================
# (C) EINZEL-KONSTANTEN-REPARATUR — streng & verify-gegated
# =====================================================================
# NUR der eindeutige Fall: Draft und Target sind instruktionsgleich BIS AUF
# genau EINE Konstanten-Änderung (gleiche Mnemonic/Register/Reloc, nur eine
# Zahl anders), UND dieser Wert kommt als eindeutiges Integer-Literal im C vor.
# Dann den Wert aus der Target-.o einsetzen. Skalierte Offsets/Indizes (Immediate
# != C-Literal) und Pointer-Typen haben keinen direkten Anker -> werden korrekt
# uebersprungen. Kann nichts kaputt machen (verify-gegated gegen die Target-.o).

_NUM_TOKEN = re.compile(r"-?\b0[xX][0-9a-fA-F]+\b|-?\b\d+\b")


def _mask_nums(ops):
    return _NUM_TOKEN.sub("#", ops)


def _nums(ops):
    return [int(t, 0) for t in _NUM_TOKEN.findall(ops)]


def run_const_repair(c_code, func_name, target_s_path):
    """Siehe Modul-Kommentar oben. Returns dict wie run_symbol_repair."""
    res = {"compiled": False, "fixed": False, "best_c_code": c_code,
           "match_rate": 0.0, "struct_score": 0.0, "report": ""}
    comp = ido_compiler.compile_code(c_code, func_name)
    if not comp.get("success"):
        return res
    tgt_o = ido_comparison._assemble_target_o(target_s_path)
    if not tgt_o:
        ido_compiler.cleanup_temp(comp["temp_dir"])
        return res
    di = ido_comparison._objdump_insns(comp["temp_o_path"])
    ti = ido_comparison._objdump_insns(tgt_o)
    ido_compiler.cleanup_temp(comp["temp_dir"])
    if not di or len(di) != len(ti):
        return res

    pairs = set()
    for a, b in zip(di, ti):
        if a["hex"] == b["hex"]:
            continue
        # Nur reiner Immediate-Diff: gleiche Mnemonic + Reloc + identische
        # Operanden-Struktur (Zahlen maskiert). Sonst -> struktureller Diff -> bail.
        if (a["mnem"] != b["mnem"] or (a["reloc"] or "") != (b["reloc"] or "")
                or _mask_nums(a["ops"]) != _mask_nums(b["ops"])):
            return res
        for vd, vt in zip(_nums(a["ops"]), _nums(b["ops"])):
            if vd != vt:
                pairs.add((vd, vt))
    if len(pairs) != 1:
        return res  # genau EINE Konstanten-Änderung

    vd, vt = next(iter(pairs))
    lits = [m for m in re.finditer(r"\b0[xX][0-9a-fA-F]+\b|\b\d+\b", c_code)
            if int(m.group(0), 0) == vd]
    if len(lits) != 1:
        return res  # Wert nicht eindeutig als C-Literal -> Anker fehlt -> bail
    m = lits[0]
    repl = hex(vt) if m.group(0).lower().startswith("0x") else str(vt)
    fixed = c_code[:m.start()] + repl + c_code[m.end():]

    vcomp = ido_compiler.compile_code(fixed, func_name)
    if not vcomp.get("success"):
        return res
    rank = ido_comparison.evaluate_match(func_name, fixed, vcomp["temp_o_path"],
                                         target_s_path, update_leaderboard=False)
    ido_compiler.cleanup_temp(vcomp["temp_dir"])
    res.update({
        "compiled": True, "best_c_code": fixed,
        "match_rate": rank["match_rate"], "struct_score": rank["struct_score"],
        "fixed": rank["match_rate"] >= 100.0,
        "report": f"Const-Repair: {vd}->{vt} -> match={rank['match_rate']}% "
                  f"struct={rank['struct_score']}%",
    })
    return res


# =====================================================================
# PHASES
# =====================================================================

PHASES = [
    ("Phase 1: Types", [_pass_return_type, _pass_type_change, _pass_extern_type_variants], 2),
    ("Phase 2: Stack", [_pass_stack_padding, _pass_decl_reorder], 1),
    ("Phase 3: Structure", [_pass_cond_swap, _pass_cond_negate, _pass_ast_stmt_reorder, _pass_call_reorder], 2),
    ("Phase 4: Expressions", [_pass_pointer_deref, _pass_arithmetic_equiv, _pass_temp_var_eliminate, _pass_signed_unsigned], 2),
    ("Phase 5: IDO Tricks", [_pass_volatile_args, _pass_cast_fixes, _pass_or_zero, _pass_fall_off], 2),
]


# =====================================================================
# MAIN
# =====================================================================

_SIMILAR_PASSES = [_pass_similar_transplant, _pass_similar_unit_remap,
                   _pass_similar_sig_skeleton, _pass_similar_types,
                   _pass_similar_decl_order]


def _all_passes(similar_ref):
    """Alle Varianten-Generatoren (dedupliziert). Similar-Passes zuerst, wenn
    eine Referenz vorliegt."""
    out = []
    seen = set()
    pool = (_SIMILAR_PASSES if similar_ref else []) + [
        fn for _, fns, _ in PHASES for fn in fns]
    for fn in pool:
        if fn not in seen:
            seen.add(fn)
            out.append(fn)
    return out


def _run_beam(c_code, func_name, asm, target_skel, cache, similar_ref,
              tested, beam_width, max_rounds):
    """Vorschlag 4: Beam-Search statt rein greedy. Behaelt die Top-K Kandidaten
    und expandiert ALLE pro Runde — so entstehen Kombinationen aus mehreren
    Passes (z.B. Typ-Aenderung + Decl-Reorder gleichzeitig), die der greedy
    Hill-Climber nie findet."""
    passes = _all_passes(similar_ref)
    base = cache.get_score(c_code, func_name, target_skel, "baseline")
    if base < 0:
        return -1.0, c_code, "baseline"
    beam = [(base, c_code)]
    best = (base, c_code)
    best_label = "baseline"
    seen = {cache._hash(c_code)}

    for _ in range(max_rounds):
        candidates = list(beam)
        new_variants = 0  # neue (ungesehene) Kandidaten dieser Runde
        for sc, code in beam:
            for pass_fn in passes:
                for label, variant in pass_fn(c_code=code, func_name=func_name,
                                              asm=asm, similar_ref=similar_ref):
                    h = cache._hash(variant)
                    if h in seen:
                        continue
                    seen.add(h)
                    new_variants += 1
                    vs = cache.get_score(variant, func_name, target_skel, label)
                    if vs < 0:
                        tested.append((label, 0, "COMPILE FAIL"))
                        continue
                    tested.append((label, vs - sc, "IMPROVED" if vs > sc else "NO HELP"))
                    candidates.append((vs, variant))
                    if vs > best[0]:
                        best = (vs, variant)
                        best_label = label
                        if vs >= 100.0:
                            return best[0], best[1], best_label
        # auf Top-K kuerzen (dedupliziert)
        candidates.sort(key=lambda x: x[0], reverse=True)
        new_beam = []
        bseen = set()
        for sc, code in candidates:
            h = cache._hash(code)
            if h in bseen:
                continue
            bseen.add(h)
            new_beam.append((sc, code))
            if len(new_beam) >= beam_width:
                break
        beam = new_beam
        # Fixpunkt: keine NEUEN Kandidaten mehr generiert -> abbrechen. (Nicht
        # 'kein Improvement' — Beam soll neutrale Zwischenschritte weiter
        # expandieren, um 2-Schritt-Kombinationen zu finden.)
        if new_variants == 0:
            break
    return best[0], best[1], best_label


def _run_greedy(c_code, func_name, asm, target_skel, cache, similar_ref, tested):
    """Bisheriger greedy Hill-Climber. Wenn eine Referenz vorliegt, laeuft eine
    vorgeschaltete 'Phase 0: Similar' (Transplantat + referenz-informierte Passes)."""
    best_code = c_code
    best_score = cache.get_score(c_code, func_name, target_skel, "baseline")
    best_label = "baseline"
    if best_score < 0:
        return -1.0, c_code, best_label

    phases = list(PHASES)
    if similar_ref:
        phases = [("Phase 0: Similar", _SIMILAR_PASSES, 1)] + phases

    for phase_name, pass_fns, max_rounds in phases:
        improved = True
        rnd = 0
        while improved and rnd < max_rounds:
            improved = False
            rnd += 1
            for pass_fn in pass_fns:
                variants = pass_fn(c_code=best_code, func_name=func_name,
                                   asm=asm, similar_ref=similar_ref)
                for label, variant in variants:
                    if variant == best_code:
                        continue
                    score = cache.get_score(variant, func_name, target_skel, label)
                    delta = score - best_score if score >= 0 else -999
                    if score > best_score:
                        tested.append((label, delta, "IMPROVED"))
                        best_score = score
                        best_code = variant
                        best_label = label
                        improved = True
                        if best_score >= 100.0:
                            return best_score, best_code, best_label
                    elif score >= 0:
                        tested.append((label, delta, "NO HELP"))
                    else:
                        tested.append((label, 0, "COMPILE FAIL"))
    return best_score, best_code, best_label


def run_permuter(c_code, func_name, target_s_path, similar_ref=None,
                 strategy="greedy", beam_width=4, beam_rounds=5):
    target_words = ido_comparison.extract_words_from_original_s(target_s_path)
    if not target_words:
        return {"best_c_code": c_code, "match_rate": 0.0, "report": ""}

    target_skel = ido_comparison.build_skeleton(target_words)

    asm = ""
    if os.path.exists(target_s_path):
        with open(target_s_path, "r", encoding="utf-8") as f:
            asm = f.read()

    cache = _CompileCache()
    tested = []

    if strategy == "beam":
        best_score, best_code, best_label = _run_beam(
            c_code, func_name, asm, target_skel, cache, similar_ref,
            tested, beam_width, beam_rounds)
    else:
        best_score, best_code, best_label = _run_greedy(
            c_code, func_name, asm, target_skel, cache, similar_ref, tested)

    if best_score < 0:
        return {"best_c_code": c_code, "match_rate": 0.0, "report": ""}

    # Deterministische Repairs auf dem Beam-Ergebnis: der Permuter hat die Struktur
    # (Reg-Alloc/Scheduling) optimiert; jetzt verbleibende Symbol-/Konstanten-Fehler
    # aus der Target-.o fixen. Verify-gegated -> nur ein echtes byte-100% wird
    # uebernommen. Greift bei JEDEM run_permuter-Aufruf (initial + Fast-Lane). Die
    # Repairs self-gaten billig (kein Mismatch -> sofort zurueck), daher kaum
    # Overhead bei Funktionen ohne Symbol/Const-Fehler.
    for _repair in (run_symbol_repair, run_const_repair):
        try:
            rr = _repair(best_code, func_name, target_s_path)
        except Exception as e:
            log.warning(f"[{func_name}] Repair-Fehler: {e}")
            continue
        if rr.get("match_rate", 0) >= 100.0:
            log.info(f"[{func_name}] Permuter+Repair: byte-100% via {rr['report']}")
            return {"best_c_code": rr["best_c_code"], "match_rate": 100.0,
                    "report": _build_report(tested, cache.stats) + "\n  " + rr["report"]}

    if best_score >= 100.0:
        return {"best_c_code": best_code, "match_rate": 100.0,
                "report": _build_report(tested, cache.stats)}

    report = _build_report(tested, cache.stats)
    log.info(f"[{func_name}] Permuter[{strategy}]: {cache.stats['ido_calls']} IDO calls, "
             f"{cache.stats['cache_hits']} cache, score {best_score:.1f}% via {best_label}"
             f"{' (similar)' if similar_ref else ''}")

    return {"best_c_code": best_code, "match_rate": round(best_score, 2), "report": report}


def _build_report(tested, stats):
    if not tested:
        return ""

    lines = ["<permuter_report>",
             "The micro-permuter already tested these changes deterministically:", ""]

    improved = [(l, d) for l, d, r in tested if r == "IMPROVED"]
    no_help = [(l, d) for l, d, r in tested if r == "NO HELP"]
    failed = [(l, d) for l, d, r in tested if r == "COMPILE FAIL"]

    if improved:
        lines.append("APPLIED TO CURRENT CODE (these worked):")
        for label, delta in improved:
            lines.append(f"  + {label} (+{delta:.1f}%)")
        lines.append("")

    if no_help:
        lines.append("TESTED BUT DID NOT HELP (do NOT repeat):")
        for label, delta in no_help[:20]:
            lines.append(f"  - {label} ({delta:+.1f}%)")
        if len(no_help) > 20:
            lines.append(f"  ... and {len(no_help) - 20} more")
        lines.append("")

    if failed:
        lines.append(f"CAUSED COMPILE ERRORS ({len(failed)}):")
        for label, _ in failed[:10]:
            lines.append(f"  x {label}")
        lines.append("")

    lines.append(f"Stats: {stats['ido_calls']} compilations, {stats['cache_hits']} cache hits")
    lines.append("</permuter_report>")
    return "\n".join(lines)