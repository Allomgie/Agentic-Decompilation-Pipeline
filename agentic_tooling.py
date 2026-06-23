"""ZWEI-PHASEN-PIPELINE / agentic_tooling (Architektur-Pivot 2026-06-20, memory two_phase_pipeline).
Det. und agentischer Zweig ENTKOPPELT. JEDE Phase/Stufe ist EINZELN aufrufbar; der Zustand wird ueber Platte
verkettet (PIPELINE_WORK), damit Phase 1 (deterministisch, OHNE Pod) und Phase 2 (mit Pod) getrennt -- auch in
verschiedenen Terminals / ueber Nacht -- laufen koennen.

ZUSTAND (PIPELINE_WORK, default analysis/_pipeline_work):
  code/<fn>.c    -- aktueller bester Code
  meta/<fn>.json -- {fn, sp, m0, m, tier, mm, history}
  pod.json       -- einmal eingelesene Pod-Config (LOCAL_API_BASE/MODEL/KEY) -> in Folge-Sessions wiederverwendet

BEFEHLE (PYTHONPATH=. python3 agentic_tooling.py ...):
  phase1 [-j N] [limit]                  -- det. Ping-Pong ueber ALLE data/c_input-Fkt (orchestrate ai=None,
                                            UNBEGRENZT), KEIN Pod. (--from digest|@file|<fn>... zum Einschraenken)
  phase2 [-j N] [limit]                   -- KOMPLETT-AUTOMATISCHE Kaskade auf die OFFENEN (tier==2):
                                            garbage->det->struct->det->logic->det->logic2->det (BRAUCHT Pod).
                                            Fuer unbeaufsichtigte Laeufe. Manuelle Kontrolle -> 'stage'.
  stage <det|garbage|struct|logic|logic2> [-j N]  -- GENAU EINE Stufe (ATOMAR, kein automatisches det danach).
                                            Du komponierst selbst, z.B.: stage garbage ; stage det ; stage struct ;
                                            stage det ; stage logic ; ... (dazwischen Permuter moeglich -- die
                                            naechste Stufe greift dessen Gewinne via _best_initial auf).
  status                                  -- Tier-Zusammenfassung.
WORKER: -j N = feste Worker-Zahl. --max-j N = ADAPTIVER Ramp: startet niedrig, steigt schrittweise bis N solange
  die CPU Luft hat (loadavg<0.7*Kerne), drosselt bei Last>1.1*Kerne. Bei det-Phasen findet er so das CPU-Maximum,
  bei agentischen Phasen rampt er bis N (=Pod-Kapazitaet). Status-Zeile zeigt 'workers target/max load'.
  Tunebar: RAMP_START/RAMP_STEP/RAMP_INTERVAL/RAMP_LOW/RAMP_HIGH.
RESUME: --skip N ueberspringt die ersten N Fkt der Liste (Reihenfolge wie gehabt, glob-stabil) -> dort
  weitermachen, wo aufgehoert. limit greift NACH skip (--skip 700 500 = Fkt 700..1200). Lauf druckt erste/letzte Fkt.
POD einmal einlesen (wird in pod.json gespeichert): export LOCAL_API_KEY=... ; export LOCAL_API_BASE=...
Sauberer Abbruch: Ctrl-C -> keine neuen Tasks, laufende beenden sich (beschraenkt), kein verwaister Worker.
"""
import os, sys, time, json, glob, re, threading, signal
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import ping_pong_wrapper as ppw
from modules import agent_engine as ae, expert_garbage as egb, expert_missing_block as emb

# --- Entkopplung: In-Loop-Agenten/garbage AUS (Phase 2 ruft Agenten EXPLIZIT als Stufen). det-Budget wird
#     pro Befehl gesetzt (phase1/stage det = unbegrenzt; phase2/agent-stage = P2_DET_BUDGET, sonst re-Explosion). ---
ppw._STRUCT_AGENT = False
ppw._LOGIC_CLOSER = False
ppw._GARBAGE_ENABLE = False

WORK = os.environ.get("PIPELINE_WORK", "analysis/_pipeline_work")
DET_DEADLINE = int(os.environ.get("P2_DET_DEADLINE", "100000"))   # "ohne Limit" (Funktions-Backstop)
P2_DET_BUDGET = int(os.environ.get("P2_DET_BUDGET", "120"))       # det-Bremse/Runde in Phase 2 (gegen Re-Explosion)
# Stufen = Wuerfelrunden in solve_recursive. Stufe 2 greift jetzt NUR noch, wenn Stufe 1 verbessert hat (warm
# weiter, Restluecken schliessen) -- bei Stillstand bricht solve_recursive sofort ab (kein Blind-Re-Roll mehr).
# Daher koennen struct/logic wieder stages=2 haben, ohne die teure Leerlauf-Stufe. Alles env-ueberschreibbar.
ST_TURNS  = int(os.environ.get("P2_ST_TURNS", "8"));   ST_STAGES  = int(os.environ.get("P2_ST_STAGES", "2"))
LG_TURNS  = int(os.environ.get("P2_LG_TURNS", "5"));   LG_STAGES  = int(os.environ.get("P2_LG_STAGES", "2"))
LG2_TURNS = int(os.environ.get("P2_LG2_TURNS", "5"));  LG2_STAGES = int(os.environ.get("P2_LG2_STAGES", "2"))

_STOP = threading.Event()            # sauberer Abbruch
_active = {}                         # fn -> aktuelle Stufe (fuer Live-Anzeige)
_active_lock = threading.Lock()
_counts = Counter()                 # MATCH / pmatch / improved / open / aborted / nofile
_t_start = time.time()
_ramp = {"target": 0, "max": 0, "load": 0.0}   # adaptiver Worker-Ramp (Status-Anzeige)


# ----------------------------------------------------------------- Pod-Config (einmal einlesen, persistent)
def _detect_model(base, key):
    """Servierten Model-Namen aus /v1/models holen -> kein Name-Mismatch (404) mehr."""
    try:
        import urllib.request
        req = urllib.request.Request(base.rstrip("/") + "/models",
                                     headers={"Authorization": f"Bearer {key}", "User-Agent": "curl/8.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())["data"][0]["id"]
    except Exception:
        return None


def _resolve_pod():
    cfgp = os.path.join(WORK, "pod.json")
    base = os.environ.get("LOCAL_API_BASE") or os.environ.get("POD_BASE")
    if base:
        base = base.rstrip("/")
        key = os.environ.get("LOCAL_API_KEY") or os.environ.get("POD_KEY") or "EMPTY"
        # Model: explizit gesetzt gewinnt; sonst AUTO aus /v1/models (verhindert 404 bei Modellwechsel am Pod).
        model = os.environ.get("LOCAL_API_MODEL") or os.environ.get("POD_MODEL") or _detect_model(base, key) or ae.MODEL
        cfg = {"base": base, "model": model, "key": key}
        os.makedirs(WORK, exist_ok=True); json.dump(cfg, open(cfgp, "w"))
    elif os.path.exists(cfgp):
        cfg = json.load(open(cfgp))
    else:
        return None
    os.environ["LOCAL_API_BASE"] = cfg["base"]; os.environ["LOCAL_API_MODEL"] = cfg["model"]
    os.environ["LOCAL_API_KEY"] = cfg["key"]
    ae.BASE, ae.MODEL, ae.KEY = cfg["base"], cfg["model"], cfg["key"]   # agent_engine las env beim Import -> hier spiegeln
    return cfg


# ----------------------------------------------------------------- Stufen (code,fn,sp,ai)->new_code|None
def _stage_det(code, fn, sp, ai):
    try:
        return ppw.orchestrate(code, fn, sp, ai=None, deadline_s=DET_DEADLINE)["code"]
    except Exception:
        return None


def _stage_garbage(code, fn, sp, ai):
    cr = lambda c: emb._compile_repair(c, fn, sp, ppw._EV_SYNTH, ai_call=ai)
    return egb.garbage_expert(code, fn, sp, ai, lambda x: ppw._EV_SYNTH(x, fn, sp), ppw._diff,
                              resolve=lambda x: _stage_det(x, fn, sp, None) or x, compile_repair=cr)


def _stage_struct(code, fn, sp, ai):
    return ae.solve_recursive("structure", fn, sp, code, ppw._EV_MB,
                              max_turns=ST_TURNS, max_stages=ST_STAGES, quiet=True).get("new_code")


def _stage_logic(code, fn, sp, ai):
    return ae.solve_recursive("logic", fn, sp, code, ppw._EV_MB,
                              max_turns=LG_TURNS, max_stages=LG_STAGES, quiet=True).get("new_code")


def _stage_logic2(code, fn, sp, ai):
    return ae.solve_recursive("logic_struct", fn, sp, code, ppw._EV_MB,
                              max_turns=LG2_TURNS, max_stages=LG2_STAGES, quiet=True).get("new_code")


STAGES = {"det": (False, _stage_det), "garbage": (True, _stage_garbage), "struct": (True, _stage_struct),
          "logic": (True, _stage_logic), "logic2": (True, _stage_logic2)}
# KEIN fuehrendes det0: Phase 1 hat det bereits maximal ausgereizt (Eingabe via _best_initial ist det-maxed) ->
# det0 waere redundant + re-explodiert auf hoch-mm Fkt VOR dem ersten Pod-Call. Die det-Re-Passes ZWISCHEN den
# agentischen Stufen bleiben (raeumen auf, was der Agent aufschliesst) -- in Phase 2 gebremst (P2_DET_BUDGET).
CASCADE = ["garbage", "det", "struct", "det", "logic", "det", "logic2", "det"]


# ----------------------------------------------------------------- Zustand (Platte)
def _cp(fn): return os.path.join(WORK, "code", f"{fn}.c")
def _mp(fn): return os.path.join(WORK, "meta", f"{fn}.json")
def _ensure_dirs():
    for d in ("code", "meta"):
        os.makedirs(os.path.join(WORK, d), exist_ok=True)
def _tier_mm(m): return (m[-2], m[-1]) if m else (None, None)
def _load_meta(fn):
    try: return json.load(open(_mp(fn)))
    except Exception: return None
def _save_state(fn, sp, code, m, m0, note):
    open(_cp(fn), "w", encoding="utf-8").write(code)
    meta = _load_meta(fn) or {"fn": fn, "sp": sp, "m0": list(m0) if m0 else None, "history": []}
    tier, mm = _tier_mm(m)
    meta.update({"sp": sp, "m": list(m) if m else None, "tier": tier, "mm": mm})
    meta["history"].append({"note": note, "m": list(m) if m else None})
    json.dump(meta, open(_mp(fn), "w"), ensure_ascii=False)


# ----------------------------------------------------------------- Writeback in den Match-Kreislauf (PROMOTE)
PROMOTE = os.environ.get("PROMOTE", "1") == "1"   # =0 zum Testen (nichts nach output/ schreiben)


def _rel_from_sp(sp):
    key = "data/target_asm/"
    if sp and key in sp:
        return sp.split(key, 1)[1][:-2]           # 'nonmatchings/<overlay>/<addr>/fn'  (ohne .s)
    return None


def _writeback(fn, sp, code, m, improved):
    """perfect(mm==0)->perfect_matches; permuter-ready(tier==1)->struct_matches (VERSCHOBEN, raus aus partials);
    tier2-verbessert->partials aktualisieren. Alle Buckets nutzen output/<bucket>/experten/<rel>.c."""
    if not PROMOTE or not m or not sp:
        return None
    rel = _rel_from_sp(sp)
    if not rel:
        return None
    tier, mm = m[-2], m[-1]

    def _put(bucket):
        tp = os.path.join("output", bucket, "experten", rel + ".c")
        os.makedirs(os.path.dirname(tp), exist_ok=True)
        open(tp, "w", encoding="utf-8").write(code)
        return bucket

    def _drop(bucket):
        for p in glob.glob(f"output/{bucket}/**/{fn}.c", recursive=True):
            try: os.remove(p)
            except Exception: pass

    if mm == 0:
        _drop("partial_matches"); _drop("struct_matches"); return _put("perfect_matches")
    if tier == 1:
        _drop("partial_matches"); return _put("struct_matches")     # zum Permuter; faellt er, mischt Lukas zurueck
    if improved:
        return _put("partial_matches")
    return None


# ----------------------------------------------------------------- Auswahl
def _digest_rows():
    if not os.path.exists("analysis/_stall_digest_reclass.md"):
        return {}
    t = open("analysis/_stall_digest_reclass.md").read(); rows = {}
    for b in re.split(r"(?m)^## ", t)[1:]:
        fn = b.split("\n", 1)[0].strip()
        dc = re.search(r"draft_c: `([^`]+)`", b); ts = re.search(r"target_s: `([^`]+)`", b)
        if fn and dc and ts: rows[fn] = (dc.group(1), ts.group(1))
    return rows
_SP_INDEX = None
_idx_lock = threading.Lock()
def _build_sp_index():
    """EINMAL data/target_asm walken -> {fn: sp-Pfad}. Danach O(1)-Lookup statt rekursivem Glob pro Funktion
    (der war bei hoher Parallelitaet ein Filesystem-Sturm)."""
    global _SP_INDEX
    with _idx_lock:
        if _SP_INDEX is None:
            idx = {}
            for root, _d, files in os.walk("data/target_asm"):
                for f in files:
                    if f.endswith(".s"): idx[f[:-2]] = os.path.join(root, f)
            _SP_INDEX = idx
    return _SP_INDEX
def _resolve_sp(fn): return _build_sp_index().get(fn)
def _all_cinput_fns():
    return sorted({os.path.basename(p)[:-2] for p in glob.glob("data/c_input/**/*.c", recursive=True)})
_SOLVED = None
def _solved_set():
    """Funktionen, die bereits in output/perfect_matches/ liegen (analog alte Pipeline) -> ueberspringen."""
    global _SOLVED
    if _SOLVED is None:
        _SOLVED = {os.path.basename(p)[:-2] for p in glob.glob("output/perfect_matches/**/*.c", recursive=True)}
    return _SOLVED
def _best_initial(fn, sp):
    """METRIK-BASIERTER Daten-Abgleich (in JEDER Phase): nimm aus ALLEN Quellen die BESTE Datei (niedrigste
    Prioritaets-Metrik), nicht nur per Reihenfolge. Quellen: PIPELINE_WORK/code (resume), output/struct_matches
    + output/partial_matches (auch Permuter-Verbesserungen zwischen Phasen!), data/c_input (Rohentwurf).
    Content-dedupe -> identische Dateien werden nur EINMAL kompiliert. Gibt (code, quelle) zurueck."""
    # ABGELEITETE Pfade statt rekursivem Glob (c_input/partials/struct spiegeln alle nonmatchings/<rel> wie
    # target_asm) -> O(1) exists-Check, KEIN FS-Sturm. Abgleich bleibt erhalten (partials gehen nicht verloren).
    rel = _rel_from_sp(sp)
    paths = [_cp(fn)]
    if rel:
        paths += ["output/struct_matches/experten/" + rel + ".c",
                  "output/partial_matches/experten/" + rel + ".c",
                  "output/partial_matches/" + rel + ".c",
                  "data/c_input/" + rel + ".c"]
    seen = set(); cands = []
    for p in paths:
        try:
            if not os.path.exists(p): continue
            code = open(p).read()
        except Exception: continue
        h = hash(code)
        if h in seen: continue
        seen.add(h)
        m, ents = ppw._EV_MB(code, fn, sp)        # Metrik + Entries (fuer Vorab-Routing der Stage)
        cands.append((m, code, ents))
    if not cands:
        return None, None, None
    cands.sort(key=lambda c: (c[0] is None, tuple(c[0]) if c[0] else (9,) * 7))   # beste (niedrigste) Metrik zuerst
    return cands[0][1], cands[0][0], cands[0][2]   # code, metrik, entries


def _targets(spec, rows, default):
    """default: 'all' (phase1) | 'tier2' (phase2/stage). spec ueberschreibt: @file / <fn> / 'digest' / 'all' / 'tier2'."""
    explicit, mode = [], None
    for a in spec:
        if a in ("digest", "all", "tier2"): mode = a
        elif a.startswith("@"): explicit += [l.strip() for l in open(a[1:]) if l.strip()]
        else: explicit.append(a)
    if explicit:
        fns = explicit
    else:
        mode = mode or default
        if mode == "digest":
            fns = list(rows.keys())
        elif mode == "all":
            fns = _all_cinput_fns()
        else:   # tier2: offene aus dem Zustand
            fns = [m["fn"] for m in (json.load(open(p)) for p in glob.glob(os.path.join(WORK, "meta", "*.json")))
                   if m.get("tier") not in (0, 1)]
            if not fns:
                fns = list(rows.keys())
    # perfect_matches rausfiltern (bereits geloest -- analog alte Pipeline)
    solved = _solved_set()
    keep = [f for f in fns if f not in solved]
    sk = len(fns) - len(keep)
    if sk:
        print(f"  perfect_matches gefiltert: {sk} bereits geloest uebersprungen ({len(keep)} verbleiben)", flush=True)
    return keep


# ----------------------------------------------------------------- Ausfuehrung
def _accept(cur, cur_m, cand, fn, sp):
    if cand and cand != cur:
        m = ppw._EV_SYNTH(cand, fn, sp)
        if m is not None and (cur_m is None or m < cur_m): return cand, m, True
    return cur, cur_m, False


def run_steps(fn, sp, code0, ai, steps, m0=None):
    t0 = time.time(); cur = code0
    cur_m = m0 if m0 is not None else ppw._EV_SYNTH(cur, fn, sp)   # m0 vorberechnet (aus _best_initial) -> 1 Compile gespart
    m0 = cur_m; log = []
    for st in steps:
        if _STOP.is_set() or (cur_m is not None and cur_m[-1] == 0): break
        needs_ai, fnc = STAGES[st]
        if needs_ai and ai is None: log.append((st, "skip-noai")); continue
        # KEINE eigene garbage-Schranke: garbage_expert self-gated nativ via _kept_spots (kein AI-Call ohne
        # verankerbare Stellen) -> universeller Fallback, wie vorgesehen. struct/logic self-skippen via run() (f0==0).
        with _active_lock: _active[fn] = st
        ts = time.time()
        try: cand = fnc(cur, fn, sp, ai)
        except Exception as e: log.append((st, "ERR", str(e)[:60])); continue
        cur, cur_m, ok = _accept(cur, cur_m, cand, fn, sp)
        log.append((st, "kept" if ok else "noop", round(time.time() - ts, 1)))
    return {"fn": fn, "sp": sp, "m0": list(m0) if m0 else None, "m": list(cur_m) if cur_m else None,
            "new_code": cur, "t": round(time.time() - t0, 1), "log": log,
            "improved": bool(cur_m and m0 and tuple(cur_m) < tuple(m0))}


def _tag(m, improved):
    if not isinstance(m, list): return "compile-fail"
    if m[-1] == 0: return "MATCH"
    if m[-2] == 1: return "pmatch"          # permuter-finishable
    return "improved" if improved else "open"


_IS_TTY = sys.stdout.isatty()
_outlock = threading.Lock()


def _fmt_dur(s):
    s = int(max(0, s))
    return f"{s//3600}h{(s%3600)//60:02d}m" if s >= 3600 else f"{s//60}m{s%60:02d}s"


def _status_text(total):
    done = sum(_counts.values())
    el = time.time() - _t_start
    eta = "--" if done <= 0 else _fmt_dur((total - done) * (el / done))
    with _active_lock: hist = Counter(_active.values())
    act = " ".join(f"{k}={v}" for k, v in sorted(hist.items())) or "-"
    w = f" | workers {_ramp['target']}/{_ramp['max']} load {_ramp['load']:.1f}" if _ramp["max"] else ""
    return (f"[{done}/{total} | {_fmt_dur(el)} | ETA {eta}] "
            f"MATCH={_counts['MATCH']} pmatch={_counts['pmatch']} improved={_counts['improved']} "
            f"open={_counts['open']} | aktiv: {act}{w}")


def _emit_line(text, total):
    """Scrollende Pro-Funktion-Zeile; im TTY danach die FIXIERTE Status-Zeile drunter neu setzen."""
    with _outlock:
        if _IS_TTY:
            sys.stdout.write("\r\033[K" + text + "\n" + _status_text(total))
        else:
            sys.stdout.write(text + "\n")
        sys.stdout.flush()


def _printer(total):
    """Haelt die FIXIERTE Status-Zeile aktuell. TTY: alle 2s in-place (\\r); Logdatei: alle 20s eine Klartext-Zeile."""
    while not _STOP.is_set():
        time.sleep(2 if _IS_TTY else 20)
        if sum(_counts.values()) >= total: break
        with _outlock:
            sys.stdout.write(("\r\033[K" + _status_text(total)) if _IS_TTY else (_status_text(total) + "\n"))
            sys.stdout.flush()


def _run_batch(cmd, steps, spec, workers, limit, need_ai, default_sel, max_j=None, skip=0):
    global _t_start
    _ensure_dirs()
    _t_start = time.time(); _counts.clear()        # ETA/Statistik pro Lauf frisch
    cfg = _resolve_pod() if need_ai else None
    rows = _digest_rows()
    fns = _targets(spec, rows, default_sel)            # Reihenfolge wie gehabt (glob ist auf dem FS stabil) ->
    if skip: fns = fns[skip:]                          # --skip passt zum vorherigen Lauf; erste N ueberspringen
    if limit: fns = fns[:limit]
    ai = ppw.make_pod_ai() if need_ai else None
    print(f"[{cmd}] {len(fns)} Fkt | {workers} parallel | steps={steps} | WORK={WORK}"
          + (f" | skip={skip}" if skip else "")
          + (f" | POD={cfg['base'] if cfg else 'KEINER'}" if need_ai else ""), flush=True)
    if fns:
        print(f"  erste Fkt: {fns[0]}  ...  letzte: {fns[-1]}", flush=True)   # zum Abgleich des Wiedereinstiegs
    if need_ai and ai is None:
        print("  !! Kein Pod (LOCAL_API_BASE/pod.json) -> agentische Stufen werden uebersprungen.", flush=True)

    def one(fn):
        if _STOP.is_set(): return {"fn": fn, "status": "aborted"}
        sp = _resolve_sp(fn)
        if not sp: return {"fn": fn, "status": "nofile"}
        with _active_lock: _active[fn] = "resolve"        # _best_initial kompiliert Quellen -> sichtbar, nicht "-"
        try:
            code0, m0, ents0 = _best_initial(fn, sp)       # metrik-basierter Abgleich + Entries (fuers Routing)
            if code0 is None: return {"fn": fn, "status": "nofile"}
            # VORAB-ROUTING (nur Einzel-Stage): struct nur bei strukturellen Diffs, logic nur bei Logic-Diffs.
            # logic2 (Abschluss-Closer), det, garbage: KEIN Vorab-Check (alle Funktionen).
            if len(steps) == 1 and ents0 is not None:
                st0 = steps[0]
                if (st0 == "struct" and ae._struct_count(ents0) == 0) or \
                   (st0 == "logic" and ae._logic_count(ents0) == 0):
                    return {"fn": fn, "status": "skip-na"}
            r = run_steps(fn, sp, code0, ai, steps, m0=m0)
            _save_state(fn, sp, r["new_code"], r["m"], r["m0"], cmd)
            r["promoted"] = _writeback(fn, sp, r["new_code"], r["m"], r["improved"])
            return r
        finally:
            with _active_lock: _active.pop(fn, None)

    total = len(fns); res = []
    cores = os.cpu_count() or 8
    adaptive = max_j is not None
    cap = max_j if adaptive else workers
    target = min(cap, int(os.environ.get("RAMP_START", str(max(4, cores // 2))))) if adaptive else workers
    step = max(2, int(os.environ.get("RAMP_STEP", str(max(4, cores // 4)))))
    interval = float(os.environ.get("RAMP_INTERVAL", "10"))
    LOW = float(os.environ.get("RAMP_LOW", "0.7")); HIGH = float(os.environ.get("RAMP_HIGH", "1.1"))
    _ramp.update({"target": target, "max": cap if adaptive else 0, "load": 0.0})
    if adaptive:
        print(f"  adaptiver Ramp: start={target} max-j={cap} step={step} (hoch bei load<{cores*LOW:.0f}, "
              f"drossel bei load>{cores*HIGH:.0f}; {cores} Kerne)", flush=True)
    pr = threading.Thread(target=_printer, args=(total,), daemon=True); pr.start()

    def handle(r):
        res.append(r)
        st = r.get("status")
        if st in ("aborted", "nofile", "skip-na"):     # skip-na = Vorab-Routing: Stage nicht zustaendig (kein Spam)
            _counts[st] += 1; return
        tag = _tag(r.get("m"), r.get("improved")); _counts[tag] += 1
        kept = [s[0] for s in r["log"] if len(s) > 1 and s[1] == "kept"]
        prom = r.get("promoted")
        _emit_line(f"[{len(res)}/{total}] {r['fn']:36} {tag:9} {r['m0']}→{r['m']} t={r['t']}s"
                   + (f" via={kept}" if kept else "") + (f" -> {prom}" if prom else ""), total)

    ex = ThreadPoolExecutor(max_workers=cap)
    it = iter(fns); running = set(); last = time.time()

    def fill():
        while len(running) < target and not _STOP.is_set():
            try: fn = next(it)
            except StopIteration: break
            running.add(ex.submit(one, fn))
    try:
        fill()
        while running:
            done_set, _ = wait(running, timeout=interval, return_when=FIRST_COMPLETED)
            for fut in done_set:
                running.discard(fut); handle(fut.result())
            if adaptive and time.time() - last >= interval and not _STOP.is_set():
                try: load1 = os.getloadavg()[0]
                except Exception: load1 = 0.0
                if load1 < cores * LOW and target < cap:
                    target = min(cap, target + step)
                elif load1 > cores * HIGH:
                    target = max(step, target - step)
                _ramp.update({"target": target, "load": load1}); last = time.time()
            fill()
    except KeyboardInterrupt:
        _STOP.set()
        if _IS_TTY: sys.stdout.write("\n")
        print("^C -> sauberer Abbruch: keine neuen Tasks, laufende beenden sich...", flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        print(f"  Zustand gesichert in {WORK} (resume per erneutem Aufruf).", flush=True)
    else:
        ex.shutdown(wait=True)
    _STOP.set()
    if _IS_TTY: sys.stdout.write("\n")        # fixierte Status-Zeile abschliessen
    ok = [r for r in res if isinstance(r.get("m"), list)]
    mt = sum(1 for r in ok if r["m"][-1] == 0); pm = sum(1 for r in ok if r["m"][-2] == 1 and r["m"][-1] != 0)
    im = sum(1 for r in ok if r.get("improved") and r["m"][-2] != 1 and r["m"][-1] != 0)
    n = len(res); skip = _counts.get("skip-na", 0); proc = n - skip
    rate = (mt + pm) / proc * 100 if proc else 0
    print(f"\n[{cmd}] N={n} | verarbeitet={proc} geroutet-uebersprungen={skip} | MATCH={mt} permuter={pm} "
          f"improved={im} | Match-Rate(perfect+permuter, auf verarbeitete)={rate:.1f}%", flush=True)
    open(os.path.join(WORK, f"_{cmd}_results.jsonl"), "w").write("\n".join(json.dumps(r) for r in res))


def cmd_status():
    _ensure_dirs()
    metas = [json.load(open(p)) for p in glob.glob(os.path.join(WORK, "meta", "*.json"))]
    tiers = Counter(m.get("tier") for m in metas)
    print(f"WORK={WORK} | Fkt im Zustand: {len(metas)}")
    print(f"  tier 0 (perfect match):  {tiers.get(0,0)}")
    print(f"  tier 1 (permuter-ready): {tiers.get(1,0)}")
    print(f"  tier 2 (offen):          {tiers.get(2,0)}")
    print(f"  compile-fail/ohne Metrik:{tiers.get(None,0)}")


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    cmd = sys.argv[1]; args = sys.argv[2:]
    workers = 8
    if "-j" in args:
        i = args.index("-j"); workers = int(args[i + 1]); del args[i:i + 2]
    max_j = None                                            # --max-j N -> adaptiver Ramp bis N (statt fixer -j)
    if "--max-j" in args:
        i = args.index("--max-j"); max_j = int(args[i + 1]); del args[i:i + 2]
    skip = 0                                                 # --skip N -> erste N (sortierte) Fkt ueberspringen
    if "--skip" in args:
        i = args.index("--skip"); skip = int(args[i + 1]); del args[i:i + 2]
    limit = next((int(a) for a in args if a.isdigit()), 0)
    spec = [a for a in args if not a.isdigit() and a not in ("--from", "--only")]
    if cmd == "status":
        cmd_status()
    elif cmd == "phase1":
        ppw._DET_BUDGET = 0                                  # det UNBEGRENZT (einmalige, erschoepfende det-Phase)
        _run_batch("phase1", ["det"], spec, workers, limit, need_ai=False, default_sel="all", max_j=max_j, skip=skip)
    elif cmd == "phase2":
        ppw._DET_BUDGET = P2_DET_BUDGET                      # det-Re-Passes gebremst (keine Re-Explosion)
        _run_batch("phase2", CASCADE, spec, workers, limit, need_ai=True, default_sel="tier2", max_j=max_j, skip=skip)
    elif cmd == "stage":
        st = spec[0] if spec else ""
        if st not in STAGES:
            print(f"unbekannte Stufe '{st}' -- waehle aus {list(STAGES)}"); return
        ppw._DET_BUDGET = 0 if st == "det" else P2_DET_BUDGET   # stage det = voller (unbegrenzter) det-Pass
        _run_batch(f"stage_{st}", [st], spec[1:], workers, limit, need_ai=STAGES[st][0],
                   default_sel="tier2", max_j=max_j, skip=skip)
    else:
        print(f"unbekannter Befehl '{cmd}'"); print(__doc__)


if __name__ == "__main__":
    main()
