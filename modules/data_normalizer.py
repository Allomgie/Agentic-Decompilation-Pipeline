"""Daten-Normalisierung (ORAKEL-gegated, NO-REGRESS) -- saeubert gespeicherte Funktionen auf unsere Konvention.

ZWECK (Lukas): falls irgendwo ein "Leck" entsteht (m2c-Header oder m2c-Stil-Symbolnamen landen in den Daten),
diesen Sweep ueber die betroffenen Verzeichnisse laufen lassen. Macht zwei Dinge, jeweils nur wenn es die
kompilierte Metrik NICHT verschlechtert (sonst Original behalten):
  1. M2C-HEADER/MAKROS strippen via m2c_donor.clean_m2c (semantik-erhaltend -> .o identisch). Der Makro-Header
     ist FATAL fuer die Similar-Pipeline (#define/Kommentare) -> darf NIE in den Daten liegen.
  2. SYMBOLNAMEN angleichen: m2c-Stil-Lokale (temp_<reg>/var_<reg>/sp<hex>/phi_ + Funktionszeiger-Locals)
     -> local_N (Deklarations-Reihenfolge); Parameter argN -> param_N (unsere Konvention, nicht m2c).
     Reine LOKAL/Param-Renames -> .o-NEUTRAL -> no-regress garantiert (wird trotzdem orakel-verifiziert).
     Globals/Callees werden NICHT angefasst (das Orakel faengt ein Versehen ab -> Regress -> revert).

Das Orakel wird per eval_fn INJIZIERT (Konvention wie die Experten): eval_fn(code, fn, target_s_path) -> Metrik
(kleiner=besser, == zum Vergleichen) ODER None (compile-fail). KEINE harte ping_pong_wrapper-Abhaengigkeit im Kern.
CLI unten verdrahtet ppw._EV_SYNTH. IDO-Kompile laeuft IMMER ueber modules/ido_compiler (via eval_fn) -> PC-sicher.
"""
import re, os, glob, shutil, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
try:                                                       # als Paket importiert (Pipeline)
    from modules import m2c_donor
except ModuleNotFoundError:                                # oder direkt als Skript ausgefuehrt
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from modules import m2c_donor

_M2C_LOCAL = re.compile(r"^(?:temp_[a-z]\w*|var_[a-z]\w*|sp[0-9A-Fa-f]{1,4}|phi_\w+)$")
_DECL = re.compile(r"(?m)^[ \t]*[A-Za-z_][\w \t\*]*?\b([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*[;=]")
_FUNCPTR = re.compile(r"\(\s*\*\s*([A-Za-z_]\w*)\s*\)\s*\(")


def align_symbols(code):
    """m2c-Stil-Lokale -> local_N, argN -> param_N. .o-NEUTRAL. -> (new_code, changed:bool)."""
    body = code[code.find("{"):] if "{" in code else code
    names = []
    for m in _DECL.finditer(body):
        n = m.group(1)
        if _M2C_LOCAL.match(n) and n not in names:
            names.append(n)
    for n in _FUNCPTR.findall(body):                       # Funktionszeiger-Locals: T (*name)(args)
        if _M2C_LOCAL.match(n) and n not in names:
            names.append(n)
    used = set(int(x) for x in re.findall(r"\blocal_(\d+)\b", code))
    nxt, mapping = 0, {}
    for n in names:
        while nxt in used:
            nxt += 1
        mapping[n] = f"local_{nxt}"; used.add(nxt); nxt += 1
    new = code
    for old, rep in mapping.items():
        new = re.sub(r"\b" + re.escape(old) + r"\b", rep, new)
    new = re.sub(r"\barg(\d+)\b", r"param_\1", new)        # Param-Konvention (unser param_N, nicht m2cs argN)
    return new, (new != code)


def normalize(code, fn, target_s_path, eval_fn):
    """clean_m2c + align_symbols, jeder Schritt EINZELN orakel-gegatet (no-regress).
    -> (new_code, status). status: 'clean' | 'unchanged' | 'uncleanable' (M2C-Rest, Header bleibt) |
    'regress-blocked' (Aenderung haette verschlechtert -> verworfen) | 'no-target'."""
    if not target_s_path:
        return code, "no-target"
    base = eval_fn(code, fn, target_s_path)
    cur, status = code, "unchanged"

    def keep(cand):                                        # nur uebernehmen, wenn no-regress vs base
        m = eval_fn(cand, fn, target_s_path)
        return m is not None and (base is None or m == base)

    if "M2C_" in cur:                                      # 1) M2C-Header/Makros strippen
        c = m2c_donor.clean_m2c(cur)
        if c is None:
            status = "uncleanable"                         # exotische Makros -> Header bleibt (selten, flaggen)
        elif keep(c):
            cur, status = c, "clean"
        else:
            status = "regress-blocked"
    aligned, changed = align_symbols(cur)                  # 2) Symbolnamen angleichen
    if changed:
        if keep(aligned):
            cur = aligned
            if status in ("unchanged", "clean"):
                status = "clean"
        elif status == "unchanged":
            status = "regress-blocked"
    return cur, status


def _target_index(target_root="data/target_asm"):
    idx = {}
    for r, _d, fs in os.walk(target_root):
        for x in fs:
            if x.endswith(".s"):
                idx[x[:-2]] = os.path.join(r, x)
    return idx


def sweep(dirs, eval_fn, sidx=None, apply=False, workers=8, backup_root=None):
    """Batch-Normalisierung ueber dirs (nur Dateien mit M2C oder m2c-Stil-Namen). -> dict dir->Counter(status).
    apply=False = Dry-Run (nichts geschrieben). Bei apply: Originale nach backup_root sichern."""
    sidx = sidx if sidx is not None else _target_index()
    backup_root = backup_root or f"analysis/_normalize_bak_{time.strftime('%Y%m%d_%H%M%S')}"
    trigger = re.compile(r"M2C_|\btemp_[a-z]|\bvar_[a-z]|\bsp[0-9A-Fa-f]{2}|\bphi_|\barg\d")
    out = {}
    for d in dirs:
        files = [f for f in glob.glob(d + "/**/*.c", recursive=True)
                 if trigger.search(open(f, errors="replace").read())]

        def work(f):
            fn = os.path.basename(f)[:-2]
            code = open(f, errors="replace").read()
            new, st = normalize(code, fn, sidx.get(fn), eval_fn)
            if apply and new != code and st in ("clean",):
                bp = os.path.join(backup_root, f)
                os.makedirs(os.path.dirname(bp), exist_ok=True); shutil.copy2(f, bp)
                open(f, "w").write(new)
            return st
        cnt = Counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fu in as_completed({ex.submit(work, f): f for f in files}):
                cnt[fu.result()] += 1
        out[d] = cnt
    return out, (backup_root if apply else None)


if __name__ == "__main__":
    import sys, types
    sys.path.insert(0, ".")
    _o = types.ModuleType("openai"); _o.OpenAI = object
    class _E(Exception): pass
    _o.APIConnectionError = _o.APITimeoutError = _o.APIError = _E; sys.modules["openai"] = _o
    import importlib.util
    _s = importlib.util.spec_from_file_location("ppw", "ping_pong_wrapper.py")
    ppw = importlib.util.module_from_spec(_s); _s.loader.exec_module(ppw)

    apply = len(sys.argv) > 1 and sys.argv[1] == "apply"
    dirs = sys.argv[2:] if len(sys.argv) > 2 else [
        "output/perfect_matches", "output/partial_matches", "output/struct_matches", "data/c_input"]
    res, bak = sweep(dirs, ppw._EV_SYNTH, apply=apply)
    print(f"Daten-Normalisierung [{'APPLY' if apply else 'DRY-RUN'}]")
    for d, cnt in res.items():
        print(f"  {d:28} {dict(cnt)}")
    print("Backup:", bak) if bak else print("DRY-RUN -- nichts geschrieben. Apply: python3 modules/data_normalizer.py apply")
