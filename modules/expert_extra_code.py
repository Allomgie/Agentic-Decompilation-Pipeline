"""extra-code Experte (deterministisch, orakel-verifiziert, KEIN AI).

"Extra in Draft" = der Draft erzeugt Instruktionen, die das Target NICHT hat = die KI macht
MEHRARBEIT (Richtungs-Befund 2026-06). Drei deterministische Transform-Klassen, alle
orakel-gegated (nur Verbesserung behalten -> sicher per Konstruktion):

- REMOVE-STATEMENT: ueberfluessige Anweisung/Zuweisung/Berechnung entfernen (iterativ, da oft
  mehrere). Faengt redundante Arbeit generisch.
- MERGE-DUPLICATE-CALL: `a = g(args); ... b = g(args);` -> 2. Aufruf durch a ersetzen
  (doppelter jal). Faengt die Datenfluss-Doppelung (func_800F6C88-Klasse).
- PASSTHROUGH-PARAM: Target laesst ein Arg-Register UNGESETZT -> der entsprechende Call-Arg ist
  Durchreiche -> als Funktions-Parameter deklarieren (bsbeemain-Klasse). Target-ASM-getrieben.

Iterativer Reduzierer: wende wiederholt den besten verbessernden Kandidaten an, bis kein
Fortschritt mehr. Verifikation IMMER ueber modules/ido_compiler (PC-Freeze-Regel).
Die Restmenge (was KEINE dieser Transformen loest) = der echte AI/Pod-Bedarf.
"""
import re

_AREG_ORDER = ["a0", "a1", "a2", "a3"]

# ---- STUFE 3: pycparser-AST (optional). Findet TOTE Variablen (zugewiesen, aber NIE gelesen) und entfernt
# Deklaration + ALLE Zuweisungen ATOMAR -- was der Regex (ein Statement/Schritt) nicht kann. AST nur zum
# IDENTIFIZIEREN; Entfernung am ORIGINALTEXT (pycparser+c_generator-Roundtrip wuerde Typen/Makros verlieren).
try:
    from pycparser import c_parser as _c_parser, c_ast as _c_ast
    _HAS_PYC = True
    _PYC = _c_parser.CParser()
except Exception:
    _HAS_PYC = False

_PYC_PRELUDE = ("typedef signed char s8; typedef unsigned char u8; typedef short s16; typedef unsigned short u16;"
                "typedef int s32; typedef unsigned int u32; typedef long long s64; typedef unsigned long long u64;"
                "typedef float f32; typedef double f64; typedef int M2C_UNK; typedef unsigned int size_t;")
_PYC_STD = {"s8", "u8", "s16", "u16", "s32", "u32", "s64", "u64", "f32", "f64", "M2C_UNK", "size_t", "int",
            "char", "short", "long", "float", "double", "void", "unsigned", "signed", "const", "volatile",
            "struct", "union", "enum"}


def _pyc_strip(code):
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    return "\n".join(l for l in code.splitlines() if not l.lstrip().startswith("#"))


def _pyc_cand_types(code):
    cand = set(re.findall(r"\b([A-Za-z_]\w*)\s*\*", code)) | \
        set(re.findall(r"(?m)^\s*([A-Za-z_]\w*)\s+\w+\s*[;=]", code)) | \
        set(re.findall(r"\(\s*([A-Za-z_]\w*)\s*\)", code))
    return [t for t in cand if t not in _PYC_STD]


def _pyc_parse_func(code):
    """-> FuncDef der Funktion (oder None). Typedef-Stubbing (Kandidaten vorab + iterativ aus Fehlern)."""
    if not _HAS_PYC:
        return None
    body = _pyc_strip(code)
    stubs = _pyc_cand_types(body)
    for _ in range(60):
        src = _PYC_PRELUDE + "".join(f"typedef int {t};" for t in stubs) + "\n" + body
        try:
            ast = _PYC.parse(src)
            fd = next((e for e in reversed(ast.ext) if isinstance(e, _c_ast.FuncDef)), None)
            return fd
        except Exception as e:
            m = re.search(r"before:\s*([A-Za-z_]\w*)", str(e))
            if m and m.group(1) not in _PYC_STD and m.group(1) not in stubs:
                stubs.append(m.group(1)); continue
            return None
    return None


class _ReadCounter(_c_ast.NodeVisitor if _HAS_PYC else object):
    """zaehlt LESE-Zugriffe je Name; sammelt adress-genommene (&x) Namen (= nicht tot)."""
    def __init__(self):
        self.reads = {}; self.addr = set()

    def visit_Assignment(self, node):
        if not isinstance(node.lvalue, _c_ast.ID):    # a[i]=/a->f= : Base ist LESE -> normal besuchen
            self.visit(node.lvalue)
        self.visit(node.rvalue)

    def visit_Decl(self, node):
        if node.init is not None:
            self.visit(node.init)

    def visit_UnaryOp(self, node):
        if node.op == "&" and isinstance(node.expr, _c_ast.ID):
            self.addr.add(node.expr.name)
        else:
            self.visit(node.expr)

    def visit_ID(self, node):
        self.reads[node.name] = self.reads.get(node.name, 0) + 1


def ast_dead_var_candidates(c_code):
    """TOTE lokale Variablen (zugewiesen/deklariert, aber NIE gelesen, nicht adress-genommen) -> je Kandidat
    Deklaration + alle reinen Zuweisungs-Statements der Var aus dem ORIGINALTEXT entfernen. Orakel-gegated."""
    fd = _pyc_parse_func(c_code)
    if fd is None or not getattr(fd.body, "block_items", None):
        return []
    local_decls = [it.name for it in fd.body.block_items
                   if isinstance(it, _c_ast.Decl) and it.name]
    if not local_decls:
        return []
    rc = _ReadCounter(); rc.visit(fd.body)
    dead = [n for n in local_decls if rc.reads.get(n, 0) == 0 and n not in rc.addr]
    out = []
    for v in dead:
        ev = re.escape(v)
        # Zeilen entfernen: `T v;` / `T v = ...;` (Decl) und `v = ...;` / `v OP= ...;` (reine Zuweisung)
        pat = re.compile(rf"^\s*(?:[A-Za-z_][\w \*]*\s+)?{ev}\s*(?:=[^;]*)?;\s*$|^\s*{ev}\s*[-+|&^]?=\s*[^;]*;\s*$")
        new = "\n".join(l for l in c_code.splitlines() if not pat.match(l))
        if new != c_code:
            out.append((f"ast-deadvar:{v}", new))
    return out


def _body_lines(c_code):
    """(prefix, body_lines, suffix) — body = Zeilen zwischen erstem { und letztem }."""
    i = c_code.find("{"); j = c_code.rfind("}")
    if i < 0 or j < 0 or j < i:
        return None
    return c_code[:i + 1], c_code[i + 1:j].splitlines(), c_code[j:]


def _is_removable_stmt(line):
    s = line.strip()
    if not s or s in ("{", "}"):
        return False
    if s.startswith(("return", "if", "else", "for", "while", "switch", "case", "default", "goto", "{", "}")):
        return False
    if s.endswith(("{", "}", ":")):
        return False
    # einfache Anweisung/Zuweisung/Call, endet mit ;
    return s.endswith(";")


def _is_decl(line):
    return bool(re.match(r"^\s*([A-Za-z_][\w ]*?\**)\s+([A-Za-z_]\w*)\s*(=|;)", line)) and \
           not line.strip().startswith(("return", "if", "else"))


def remove_statement_candidates(c_code):
    """Je eine Variante mit einer entfernten Anweisung."""
    parts = _body_lines(c_code)
    if not parts:
        return []
    pre, body, suf = parts
    out = []
    for k, ln in enumerate(body):
        if _is_removable_stmt(ln):
            nb = body[:k] + body[k + 1:]
            out.append((f"remove:{ln.strip()[:30]}", pre + "\n".join(nb) + "\n" + suf))
    return out


def merge_duplicate_call_candidates(c_code):
    """`X = g(ARGS); ... Y = g(ARGS);` (gleiche Args) -> 2. durch X ersetzen."""
    out = []
    # finde Zuweisungen mit Call rechts
    calls = list(re.finditer(r"([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\(([^;]*)\)\s*;", c_code))
    by_sig = {}
    for m in calls:
        lhs, fn, args = m.group(1), m.group(2), re.sub(r"\s+", "", m.group(3))
        by_sig.setdefault((fn, args), []).append((lhs, m.span()))
    for (fn, args), occ in by_sig.items():
        if len(occ) >= 2:
            first_lhs = occ[0][0]
            # ersetze die 2. Vorkommens-Zuweisung: `Y = g(ARGS);` -> `Y = first_lhs;`
            second_lhs, span = occ[1]
            new = c_code[:span[0]] + f"{second_lhs} = {first_lhs};" + c_code[span[1]:]
            out.append((f"merge_call:{fn}", new))
    return out


def _parse_s_min(target_s_path):
    out = []
    for ln in open(target_s_path, encoding="utf-8", errors="replace"):
        s = re.sub(r"/\*.*?\*/", "", ln).strip()
        if not s or s.startswith((".", "glabel", "endlabel")) or s.endswith(":"):
            continue
        m = re.match(r"([a-z][a-z0-9.]*)\s+(.*)", s)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def _unset_arg_regs(target_s_path):
    """Welche a0-a3 werden VOR dem ersten jal NICHT als Ziel geschrieben? (= Durchreiche)."""
    insns = _parse_s_min(target_s_path)
    written = set(); first_jal = len(insns)
    for i, (mn, ops) in enumerate(insns):
        if mn in ("jal", "jalr"):
            first_jal = i; break
    for mn, ops in insns[:first_jal]:
        d = re.match(r"\$(\w+)", ops)
        if d and d.group(1) in _AREG_ORDER:
            written.add(d.group(1))
    # auch im Delay-Slot (Zeile nach jal) zaehlt noch als vor-Call-Setup
    if first_jal < len(insns):
        d = re.match(r"\$(\w+)", insns[first_jal][1] if False else "")
    used_args = [a for a in _AREG_ORDER]
    return [a for a in used_args if a not in written]


def passthrough_param_candidates(c_code, target_s_path):
    """Target laesst Arg-Register i ungesetzt -> mach den i-ten Call-Arg zum Funktions-Parameter.
    Greift fuer die Form mit GENAU einem (dominanten) Call und parameterloser Funktion."""
    out = []
    m = re.search(r"([A-Za-z_][\w \*]*?\b\w+)\s*\(\s*(void|)\s*\)\s*\{", c_code)
    if not m:
        return []  # Funktion hat schon Parameter -> v1 ueberspringt
    sig_head = m.group(1)  # "int func_..."
    # finde den (ersten) Call mit Argumenten
    call = re.search(r"\b([A-Za-z_]\w*)\s*\(([^;]+)\)\s*;", c_code)
    if not call:
        return []
    fn_called, arglist = call.group(1), call.group(2)
    args = [a.strip() for a in _split_args(arglist)]
    unset = _unset_arg_regs(target_s_path)
    for a in unset:
        idx = _AREG_ORDER.index(a)
        if idx >= len(args):
            continue
        pname = f"arg{idx}"
        new_args = args[:]
        new_args[idx] = pname
        new_call = f"{fn_called}({', '.join(new_args)});"
        # neue Signatur mit Parameter
        new = c_code
        new = re.sub(re.escape(sig_head) + r"\s*\(\s*(?:void|)\s*\)",
                     f"{sig_head}(int {pname})", new, count=1)
        new = re.sub(re.escape(call.group(0)), new_call, new, count=1)
        out.append((f"passthrough:a{idx}", new))
    return out


def _split_args(s):
    out = []; depth = 0; cur = ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def reroute_discarded_call_candidates(c_code):
    """Muster: verworfener Call `g(args);` (kein LHS) + `return h(...) OP K;` -> der Compiler
    nutzt g()s Ergebnis fuer den Vergleich. Kandidat: `return g(args) OP K;`, verworfenen
    Call entfernen. Faengt die Doppel-Call-Datenfluss-Klasse (func_800F6C88/_800F6DA0)."""
    out = []
    parts = _body_lines(c_code)
    if not parts:
        return []
    pre, body, suf = parts
    # verworfene Calls (Zeile = `g(...);`, kein '=')
    disc = []
    for k, ln in enumerate(body):
        s = ln.strip()
        if re.match(r"^[A-Za-z_]\w*\s*\(.*\)\s*;$", s) and "=" not in s.split("(")[0]:
            disc.append((k, s[:-1]))  # ohne ';'
    # return mit Vergleich gegen Konstante: `return <expr> OP K;`
    m = None; ridx = None
    for k, ln in enumerate(body):
        mm = re.match(r"\s*return\s+(.+?)\s*(==|!=|<=|>=|<|>)\s*([^;]+);", ln)
        if mm:
            m = mm; ridx = k
    if m is None:
        return []
    op, rhs = m.group(2), m.group(3).strip()
    for k, call in disc:
        nb = []
        for i, l in enumerate(body):
            if i == k:
                continue  # verworfenen Call entfernen
            if re.match(r"\s*return\s+.+?\s*(==|!=|<=|>=|<|>)\s*[^;]+;", l):
                nb.append(f"  return {call} {op} {rhs};")  # return auf Call-Ergebnis umrouten
            else:
                nb.append(l)
        out.append((f"reroute:{call[:24]}", pre + "\n".join(nb) + "\n" + suf))
    return out


def _top_statements(body):
    """Body (String) in Top-Level-Statements (brace-depth 0) zerlegen -- robust gegen
    mehrzeilige Statements und {}-Bloecke (anders als die zeilen-basierte remove_statement)."""
    out = []; depth = 0; cur = ""
    for ch in body:
        cur += ch
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out.append(cur.strip()); cur = ""
        elif ch == ";" and depth == 0:
            out.append(cur.strip()); cur = ""
    if cur.strip():
        out.append(cur.strip())
    return [s for s in out if s]


def trim_candidates(c_code):
    """STUFE 2 (erschoepfender Trim): je eine Variante mit GENAU EINEM entfernten Top-Level-
    Statement -- inkl. solcher, die remove_statement bewusst auslaesst (z.B. `return 0;`, toter
    Code nach return). Orakel-gegated vom Aufrufer (nur Verbesserung bleibt -> sicher)."""
    parts = _body_lines(c_code)
    if not parts:
        return []
    pre, body_lines, suf = parts
    body = "\n".join(body_lines)
    stmts = _top_statements(body)
    out = []
    for i in range(len(stmts)):
        ns = stmts[:i] + stmts[i + 1:]
        cand = pre + "\n  " + "\n  ".join(ns) + "\n" + suf
        if cand != c_code:
            out.append((f"trim:{stmts[i][:28]}", cand))
    return out


def extra_code_candidates(c_code, target_s_path=None):
    cands = (remove_statement_candidates(c_code) + merge_duplicate_call_candidates(c_code)
             + reroute_discarded_call_candidates(c_code))
    if target_s_path:
        cands += passthrough_param_candidates(c_code, target_s_path)
    seen = set(); out = []
    for lbl, code in cands:
        if code != c_code and code not in seen:
            seen.add(code); out.append((lbl, code))
    return out


def _reduce_loop(cur, cur_d, cand_fn, func_name, target_s_path, count_fn, max_iter, steps):
    """Gemeinsamer iterativer orakel-gegateter Reduzierer: nimm pro Runde den besten
    verbessernden Kandidaten (von cand_fn), bis kein Fortschritt. -> (cur, cur_d)."""
    for _ in range(max_iter):
        if cur_d == 0:
            break
        best_code, best_d, best_lbl = None, cur_d, None
        for lbl, cand in cand_fn(cur):
            d = count_fn(cand, func_name, target_s_path)
            if d is not None and (best_d is None or d < best_d):
                best_code, best_d, best_lbl = cand, d, lbl
        if best_lbl is None:
            break
        cur, cur_d = best_code, best_d
        steps.append((best_lbl, best_d))
    return cur, cur_d


def extra_code_expert(c_code, func_name, target_s_path, count_fn=None, max_iter=8):
    """Zwei Stufen, orakel-gegated (nur Verbesserung bleibt -> sicher per Konstruktion):
      STUFE 1  Muster-Transforms (remove-statement/merge-call/reroute/passthrough), iterativ.
      STUFE 2  ERSCHOEPFENDER TRIM (Lukas' Idee): jedes Top-Level-Statement testweise entfernen
               -- inkl. der von Stufe 1 ausgelassenen (return/toter Code). Greift nur wenn Stufe 1
               nicht auf 0 kam. Loest additive Garbage, die die Muster-Regeln nicht fassen.
    Returns {applied, new_code, diffs_before, diffs_after, steps}."""
    before = count_fn(c_code, func_name, target_s_path)
    steps = []
    # Stufe 1: Muster-Transforms
    cur, cur_d = _reduce_loop(c_code, before, lambda cc: extra_code_candidates(cc, target_s_path),
                              func_name, target_s_path, count_fn, max_iter, steps)
    # Stufe 2: erschoepfender Trim (nur falls noch Rest)
    if cur_d not in (0, None):
        cur, cur_d = _reduce_loop(cur, cur_d, trim_candidates,
                                  func_name, target_s_path, count_fn, max_iter, steps)
    # Stufe 3: pycparser-AST tote-Variablen (atomar Decl+alle Zuweisungen), nur falls noch Rest + pycparser da
    if _HAS_PYC and cur_d not in (0, None):
        cur, cur_d = _reduce_loop(cur, cur_d, ast_dead_var_candidates,
                                  func_name, target_s_path, count_fn, max_iter, steps)
    return {"applied": cur != c_code, "new_code": cur, "diffs_before": before,
            "diffs_after": cur_d, "steps": steps}
