# modules/ai_agent.py
# Modulare LLM-Schnittstelle.
# Modi: "oneshot" (klassisch) und "thinking" (Qwen3.6 Multi-Turn mit Thinking Preservation)

import re
import os
import json
import logging
import threading
from openai import OpenAI

_log = logging.getLogger(__name__)

_ai_log = logging.getLogger("ai_comms")
_ai_log.setLevel(logging.DEBUG)
if not _ai_log.handlers:
    _ai_fh = logging.FileHandler("ai_communication.log", encoding="utf-8")
    _ai_fh.setFormatter(logging.Formatter("%(asctime)s\n%(message)s\n" + "="*80 + "\n"))
    _ai_log.addHandler(_ai_fh)
_ai_log.propagate = False

AI_BACKEND = os.environ.get("AI_BACKEND", "local")
LOCAL_API_BASE = os.environ.get("LOCAL_API_BASE", "http://localhost:8000/v1")
LOCAL_API_KEY = os.environ.get("LOCAL_API_KEY", "EMPTY")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "Qwen/Qwen3-Coder-Next")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# --- ALIBABA CLOUD (DashScope) CONFIG ---
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.6-plus")
DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# --- MODEL FAMILY DETECTION ---
# Steuert wie Thinking aktiviert wird, wie reasoning extrahiert wird,
# und welche Sampling-Defaults gelten. Wird ueber LOCAL_MODEL-Name
# automatisch erkannt, kann aber per ENV ueberschrieben werden.
def _detect_model_family(model_name: str) -> str:
    name = (model_name or "").lower()
    if "gemma-4" in name or "gemma4" in name:
        return "gemma4"
    # Default: Qwen3-Familie (Qwen3-Coder-Next, qwen3.6-plus, ...)
    return "qwen3"

MODEL_FAMILY = os.environ.get("MODEL_FAMILY", "").lower() or _detect_model_family(LOCAL_MODEL)

# "oneshot": Jeder Call isoliert, history_context im Prompt
# "thinking": Qwen3.6 Multi-Turn, Thinking Preservation, kein history_context noetig
CONV_MODE = os.environ.get("CONV_MODE", "oneshot")
USE_STREAMING = os.environ.get("USE_STREAMING", "") == "1"

# Dynamic-Mode: Tracke welche Funktionen eskaliert wurden.
# - _escalated_funcs:        Funktionen die zu Thinking eskaliert sind (dynamic)
# - _batch_escalated_funcs:  Funktionen die zu Batch eskaliert sind (dynamic_batch)
# - _syntax_thinking_funcs:  Funktionen die im Syntax-Fix-Loop zu Thinking eskaliert sind
#                            (--dynamic-compiler, orthogonal zu CONV_MODE)
_escalated_funcs = set()
_batch_escalated_funcs = set()
_syntax_thinking_funcs = set()
_escalated_lock = threading.Lock()


def escalate_to_thinking(func_name):
    """Markiert eine Funktion fuer EINEN Thinking-Call (danach zurueck zu Oneshot).
    Nur fuer CONV_MODE=dynamic relevant."""
    with _escalated_lock:
        _escalated_funcs.add(func_name)
        _log.info(f"[{func_name}] Eskaliert zu Thinking-Mode (einmalig).")


def de_escalate(func_name):
    """Setzt eine Funktion zurueck auf Oneshot nach dem Thinking-Call."""
    with _escalated_lock:
        _escalated_funcs.discard(func_name)


def escalate_to_batch(func_name):
    """Markiert eine Funktion fuer Batch-Sampling (dynamic_batch).
    Anders als escalate_to_thinking: bleibt aktiv bis explizit zurueckgesetzt,
    weil Batch nicht ein-Call-und-fertig ist sondern bei Stagnation hilft."""
    with _escalated_lock:
        _batch_escalated_funcs.add(func_name)
        _log.info(f"[{func_name}] Eskaliert zu Batch-Mode (dynamic_batch).")


def de_escalate_batch(func_name):
    """Setzt eine Funktion zurueck (z.B. nach Fortschritt)."""
    with _escalated_lock:
        _batch_escalated_funcs.discard(func_name)


def is_batch_escalated(func_name):
    """Vom Wrapper genutzt um zu entscheiden ob Batch- oder Single-Call gemacht wird."""
    with _escalated_lock:
        return func_name in _batch_escalated_funcs


def escalate_syntax_thinking(func_name):
    """Markiert eine Funktion fuer Thinking-Mode im Syntax-Fix-Loop
    (--dynamic-compiler). Bleibt aktiv solange der Syntax-Fix-Loop laeuft,
    wird beim Compile-Erfolg per de_escalate_syntax_thinking zurueckgesetzt.
    Orthogonal zu CONV_MODE — wirkt auch bei oneshot/batch/dynamic_batch."""
    with _escalated_lock:
        if func_name not in _syntax_thinking_funcs:
            _syntax_thinking_funcs.add(func_name)
            _log.info(f"[{func_name}] Eskaliert zu Syntax-Thinking (--dynamic-compiler).")


def de_escalate_syntax_thinking(func_name):
    """Wird vom Wrapper aufgerufen sobald der Compile gruen ist."""
    with _escalated_lock:
        if func_name in _syntax_thinking_funcs:
            _syntax_thinking_funcs.discard(func_name)
            _log.info(f"[{func_name}] Syntax-Thinking deaktiviert (compile ok).")


def is_syntax_thinking_escalated(func_name):
    """Wird in generate_fix abgefragt fuer task_type='syntax_fix'."""
    with _escalated_lock:
        return func_name in _syntax_thinking_funcs


def _effective_mode(func_name):
    """Bestimmt den effektiven Modus fuer eine Funktion (Modell-Verhalten).
    
    dynamic       — eskaliert zu Thinking
    dynamic_batch — bleibt IMMER oneshot auf Modell-Seite, der Wrapper
                    macht statt Single-Call einen Batch-Call wenn eskaliert
    """
    if CONV_MODE == "dynamic":
        with _escalated_lock:
            return "thinking" if func_name in _escalated_funcs else "oneshot"
    if CONV_MODE == "dynamic_batch":
        # Modell-Modus immer Oneshot — Eskalation steuert nur ob Single oder Batch
        return "oneshot"
    return CONV_MODE

_local_client = None
_anthropic_client = None
_dashscope_client = None


def _get_local_client():
    global _local_client
    if _local_client is None:
        _local_client = OpenAI(base_url=LOCAL_API_BASE, api_key=LOCAL_API_KEY, timeout=300.0)
    return _local_client


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY nicht gesetzt.")
        _anthropic_client = OpenAI(
            base_url="https://api.anthropic.com/v1", api_key=ANTHROPIC_API_KEY,
            timeout=120.0, default_headers={"anthropic-version": "2023-06-01"},
        )
    return _anthropic_client


def _get_dashscope_client():
    global _dashscope_client
    if _dashscope_client is None:
        if not DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY nicht gesetzt. Export: DASHSCOPE_API_KEY=sk-xxx")
        _dashscope_client = OpenAI(
            base_url=DASHSCOPE_API_BASE, api_key=DASHSCOPE_API_KEY,
            timeout=300.0,
        )
    return _dashscope_client


# === MULTI-TURN CONVERSATION MANAGER ===

_conversations = {}
_conv_lock = threading.Lock()


def _get_conversation(func_name, system_prompt):
    with _conv_lock:
        if func_name not in _conversations:
            _conversations[func_name] = [{"role": "system", "content": system_prompt}]
        return _conversations[func_name]


def _append_to_conversation(func_name, role, content, reasoning_content=None):
    """Haengt eine Nachricht an die Conversation an.
    
    WICHTIG: reasoning_content wird NICHT in die Nachricht aufgenommen, die
    spaeter an die API zurueckgeschickt wird. Reasoning-Modelle (Qwen3, Gemma4)
    behandeln sichtbare frueheren Reasoning-Content als "Entwurf zu kritisieren",
    nicht als settled state — das fuehrt zu Selbstkorrektur-Loops.
    Reasoning wird stattdessen lokal geloggt (siehe _ai_log Aufruf in generate_fix).
    """
    with _conv_lock:
        if func_name in _conversations:
            # API sieht NUR den finalen Content, nie das Reasoning.
            _conversations[func_name].append({"role": role, "content": content})
            # Lokales Debug-Log: Reasoning separat, geht nicht in den naechsten API-Call
            if reasoning_content:
                _ai_log.debug(
                    f"[REASONING NOT-SENT-TO-API] func={func_name} "
                    f"len={len(reasoning_content)}\n{reasoning_content[:1000]}"
                )


def _trim_conversation(func_name, max_turns=15):
    """Begrenzt die Conversation auf die letzten max_turns User-Assistant-Paare.
    Gibt die evicted Assistant-Contents zurueck fuer die Eviction-Summary.
    """
    with _conv_lock:
        if func_name not in _conversations:
            return []
        conv = _conversations[func_name]
        non_system = conv[1:]

        evicted_codes = []
        if len(non_system) > max_turns * 2:
            evicted = non_system[:-(max_turns * 2)]
            for msg in evicted:
                if msg.get("role") == "assistant" and msg.get("content"):
                    code = msg["content"].strip()
                    if code and len(code) > 10:
                        evicted_codes.append(code)
            non_system = non_system[-(max_turns * 2):]

        # Safety-Net: Sollte irgendein Pfad doch reasoning_content gesetzt haben,
        # raus damit — die API darf es niemals sehen. _append_to_conversation
        # sollte das schon verhindern, aber doppelt haelt besser.
        for msg in non_system:
            msg.pop("reasoning_content", None)

        _conversations[func_name] = [conv[0]] + non_system
        return evicted_codes


def reset_conversation(func_name):
    with _conv_lock:
        if func_name in _conversations:
            turns = len(_conversations[func_name]) - 1  # minus system prompt
            _conversations.pop(func_name)
            _log.debug(f"[{func_name}] Conversation reset ({turns} Turns, {len(_conversations)} aktive Conversations verbleiben)")
        else:
            _log.debug(f"[{func_name}] Conversation reset (war bereits leer)")


def _build_eviction_summary(evicted_codes, baseline_code):
    """
    Baut eine kompakte Diff-Summary aus den C-Codes die aus der Conversation getrimmt wurden.
    Zeigt der KI was in frueheren Runden probiert wurde (ohne den vollen Reasoning-Kontext).
    """
    if not evicted_codes or not baseline_code:
        return ""

    from difflib import unified_diff
    lines = ["<evicted_history>",
             f"The following {len(evicted_codes)} earlier attempts were trimmed from the conversation.",
             "Their C-code diffs vs baseline are shown below. Do NOT repeat these approaches.", ""]

    for i, code in enumerate(evicted_codes[-5:]):  # Max 5 aelteste
        diff_lines = list(unified_diff(
            baseline_code.splitlines(), code.splitlines(),
            lineterm="", n=0,
        ))
        changes = [l for l in diff_lines if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
        if changes:
            lines.append(f"  Earlier attempt {i+1}:")
            for dl in changes[:10]:
                lines.append(f"    {dl}")
            lines.append("")

    lines.append("</evicted_history>")
    return "\n".join(lines)


# === MODEL-FAMILY HELPERS ===

def _sampling_defaults(family: str) -> dict:
    """Sampling-Defaults pro Modell-Familie.
    Gemma 4 empfiehlt temperature=1.0, top_p=0.95, top_k=64.
    Qwen3 empfiehlt top_p=0.95, top_k=20.
    Wir behalten die uebergebene temperature und ueberschreiben nur top_k/top_p.
    """
    if family == "gemma4":
        return {"top_p": 0.95, "top_k": 64}
    return {"top_p": 0.95, "top_k": 20}


def _thinking_extra_body(family: str, enable_thinking: bool) -> dict:
    """Baut den extra_body fuer einen Local-vLLM Call abhaengig von Family.

    Qwen3:  chat_template_kwargs={"enable_thinking": bool, "preserve_thinking": True}
    Gemma4: chat_template_kwargs={"enable_thinking": bool}
            (vLLM nutzt --reasoning-parser gemma4 serverseitig,
             reasoning landet in msg.reasoning_content)
    """
    sampling = _sampling_defaults(family)
    if family == "gemma4":
        ctk = {"enable_thinking": enable_thinking}
    else:
        ctk = {"enable_thinking": enable_thinking}
        if enable_thinking:
            ctk["preserve_thinking"] = True
    return {**sampling, "chat_template_kwargs": ctk}


def _dashscope_extra_body(family: str, enable_thinking: bool) -> dict:
    """DashScope nutzt 'enable_thinking' direkt im extra_body (nicht in
    chat_template_kwargs). Behaelt die historische Logik fuer Qwen bei.
    Fuer Gemma 4 ist DashScope (Stand 2026) nicht der primaere Pfad —
    wir geben aber konsistente Parameter zurueck, falls jemand das nutzt."""
    sampling = _sampling_defaults(family)
    if family == "gemma4":
        return {**sampling, "enable_thinking": enable_thinking}
    body = {**sampling, "enable_thinking": enable_thinking}
    if enable_thinking:
        body["preserve_thinking"] = True
    return body


# Regex fuer Gemma-4 Channel-Bloecke (Thinking-Output).
# Variante 1: <|channel>thought\n...<channel|>   (Standard wenn enable_thinking=True)
# Variante 2: <|channel>thought\n<channel|>      (leerer Block bei enable_thinking=False
#             — Gemma 4 26B A4B / 31B emittieren den trotzdem)
_GEMMA_CHANNEL_RE = re.compile(
    r"<\|channel>\s*thought\s*\n?(.*?)<channel\|>",
    re.DOTALL,
)


def _strip_reasoning_markers(text: str, family: str = None) -> tuple:
    """Entfernt Reasoning/Thinking-Marker aus Modell-Output.
    Returns: (cleaned_text, extracted_reasoning_or_None)

    Funktioniert sowohl fuer Qwen (<think>...</think>) als auch
    Gemma 4 (<|channel>thought\\n...<channel|>). Wir versuchen beides,
    egal welche Family — robuster falls vLLM die Tags nicht serverseitig
    abparst (reasoning_content wird dann leer geliefert)."""
    if not text:
        return "", None

    reasoning = None

    # Gemma 4 Channel-Block (auch der leere Default-Block muss weg!)
    m = _GEMMA_CHANNEL_RE.search(text)
    if m:
        inner = m.group(1).strip()
        if inner:
            reasoning = inner
        text = _GEMMA_CHANNEL_RE.sub("", text, count=0).strip()

    # Qwen <think>...</think>
    tm = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if tm:
        inner = tm.group(1).strip()
        if inner and not reasoning:
            reasoning = inner
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    return text, reasoning


# === CODE EXTRACTION ===

def _extract_c_code(raw_response):
    if not raw_response:
        return ""
    # Entferne sowohl Qwen <think>...</think> als auch
    # Gemma 4 <|channel>thought\n...<channel|> Bloecke (auch leere!).
    cleaned_response, _ = _strip_reasoning_markers(raw_response)
    if not cleaned_response:
        cleaned_response = raw_response

    match_c = re.search(r"```[cC]?\s*\n?(.*?)\s*```", cleaned_response, re.DOTALL)
    if match_c:
        code = match_c.group(1).strip()
        if _looks_like_c(code):
            return code

    cleaned = cleaned_response.strip()
    for prefix in ["Here is the code:", "Here is the fixed version:", "Sure,",
                    "Here's the corrected code:", "The fixed code is:",
                    "Here is the corrected C code:", "Here are the fixes:"]:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()

    lines = cleaned.split("\n")
    cut_idx = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped.startswith(("note:", "explanation:", "changes made:", "i ", "the ", "this ")):
            if i > 2:
                cut_idx = i
                break
    cleaned = "\n".join(lines[:cut_idx]).strip()

    if _looks_like_c(cleaned):
        return cleaned

    _log.warning(f"Code-Extraktion fehlgeschlagen. Raw-Laenge={len(raw_response)}")
    return ""


def _looks_like_c(code):
    if not code or len(code) < 10:
        return False
    if ";" not in code and "{" not in code:
        return False
    words = code.split()
    if words:
        prose = sum(1 for w in words[:5] if w.lower() in
                    ("the", "this", "i", "you", "we", "here", "note", "please"))
        if prose >= 3:
            return False
    return True


# === PROMPT BUILDER ===

def _build_prompts(c_code, context_data, task_type, tech_env,
                   history_context="", confidence=1.0, thinking_mode=False,
                   similar_context="", cheatsheet_context=""):
    # Cheatsheet wird (falls vorhanden) an den TechEnv-Block angehaengt — frueher
    # im Wrapper verklebt, jetzt separat durchgereicht, damit der Kontext-Fahrplan
    # (siehe generate_fix_batch) ihn pro Stufe ein-/ausschalten kann.
    if cheatsheet_context:
        tech_env = (f"{tech_env}\n\n{cheatsheet_context}" if tech_env
                    else cheatsheet_context)
    conf_note = ""
    if confidence < 1.0:
        conf_pct = int(confidence * 100)
        conf_note = (
            f"\nIMPORTANT: Technical environment has ~{conf_pct}% confidence. "
            f"Trust the assembly diff MORE when they conflict.\n"
        )

    thinking_rule = ""
    if thinking_mode:
        thinking_rule = (
            "\n5. Keep your <think> process extremely concise. Limit your internal reasoning "
            "to a maximum of 5 steps. Do not overthink or loop over previous failed attempts. "
            "Arrive at a conclusion quickly and output the code."
        )

    similar_rule = ""
    if similar_context:
        similar_rule = (
            "\n5. A STRUCTURAL REFERENCE is provided. This is your PRIMARY guide:\n"
            "   - Copy its code structure AND its types: array dimensions and element types, "
            "value-vs-pointer-vs-int distinctions, temporary variables, expression shapes "
            "(casts, '^ 0' tricks, pointer intermediates). These outrank the environment's inferred types.\n"
            "   - Take only the correct symbol names and constants from the environment and diff.\n"
            "   - The reference's CODE PATTERN is more reliable than arithmetic guessing.\n"
            "   - Do NOT flatten 2D arrays, remove temp variables, or restructure expressions."
        )

    if task_type == "syntax_fix":
        system_prompt = (
            "You are an expert on the historical SGI MIPS IDO 5.3 C compiler.\n"
            "Fix the compiler error. Use the TECHNICAL ENVIRONMENT to resolve 'undefined' errors.\n"
            "TECHNICAL ENVIRONMENT key fields:\n"
            "- 'env.ext': Header-declared function signatures. These types are AUTHORITATIVE.\n"
            "  If the target function appears here, its return type and param types MUST be used.\n"
            "- 'ret': C type (e.g. 'u8', 's32') if known from header, otherwise register ('v0').\n"
            "- 'args': either ['a0','a1'] (registers only) or [{'reg':'a0','type':'u8'}] (typed from header).\n"
            "RULES:\n"
            "1. NEVER change logical structure or control flow.\n"
            "2. Add missing extern declarations, casts, semicolons.\n"
            "3. On redeclaration error: use the EXACT return type from 'env.ext' or 'ret'.\n"
            "4. Reply with ONLY corrected C code. No explanations, no markdown, no ```c blocks."
            f"{thinking_rule}"
            f"{conf_note}"
        )
        user_message = (
            f"// === TECHNICAL ENVIRONMENT ===\n{tech_env}\n\n"
            f"{history_context}\n"
            f"// === C CODE (has compiler errors) ===\n{c_code}\n\n"
            f"// === COMPILER ERROR ===\n{context_data}"
        )

    elif task_type == "diff_minimize":
        system_prompt = (
            "You are a reverse engineering expert for N64 MIPS decompilation using IDO 5.3.\n"
            "Modify the C code so IDO produces EXACTLY the target MIPS assembly.\n\n"
            "TECHNICAL ENVIRONMENT fields:\n"
            "- 'env.ext': header-declared types — a strong hint, but the assembly diff overrides them.\n"
            "- 'sf': EXACT stack frame size. 'stk.saved': saved registers. 'stk.locals': variable layout.\n"
            "- 'calls': function calls with 'argc'. 'mem': memory access patterns.\n"
            "- 'ret': return type — C type (e.g. 'u8') if from header, else register ('v0').\n"
            "- 'args': parameters — [{'reg':'a0','type':'u8'}] if typed, else ['a0','a1'] (registers only).\n"
            "- 'argc_conf'/'ret_conf': confidence ('high'/'medium'/'low') of the inferred arg-count/return type.\n"
            "- 'br': branch types.\n\n"
            f"{conf_note}"
            "RULES:\n"
            "1. The target assembly is ground truth. Treat symbol names and the stack frame "
            "('sf', saved registers) as hard facts — match them exactly. The types and arg-count "
            "('ret', 'args', 'argc') are INFERENCES (see the '_conf' flags): when the diff implies "
            "a different load/store width, value-vs-address, or argument, follow the diff over the environment.\n"
            "2. Use the JSON diff to fix specific mismatches.\n"
            "3. Do NOT repeat changes that already failed.\n"
            "4. Reply with ONLY corrected C code. No explanations, no markdown, no ```c blocks."
            f"{similar_rule}"
            f"{thinking_rule}"
        )

        if similar_context:
            user_message = (
                f"// === STRUCTURAL REFERENCE (follow this pattern) ===\n{similar_context}\n\n"
                f"// === TECHNICAL ENVIRONMENT ===\n{tech_env}\n\n"
                f"{history_context}\n"
                f"// === CURRENT C CODE (modify this) ===\n{c_code}\n\n"
                f"// === ASSEMBLY DIFF (JSON) ===\n{context_data}"
            )
        else:
            user_message = (
                f"// === TECHNICAL ENVIRONMENT ===\n{tech_env}\n\n"
                f"{history_context}\n"
                f"// === CURRENT C CODE (modify this) ===\n{c_code}\n\n"
                f"// === ASSEMBLY DIFF (JSON) ===\n{context_data}"
            )
    else:
        raise ValueError(f"Unbekannter task_type: {task_type}")

    return system_prompt, user_message


# === KONTEXT-FAHRPLAN (dynamic_batch) ===
#
# Statt im Batch-Tail N nahezu identische Prompts mit Temperatur-Spread zu feuern,
# variieren wir die KONTEXT-ZUSAMMENSETZUNG ueber die Slots. Der Diff bleibt IMMER
# (einzige Wahrheit), ebenso die HARTEN TechEnv-Fakten (Namen/Groessen/sf/Stack/Calls).
# Reduziert werden nur die INFERIERTEN Felder, der Cheatsheet und (zuletzt) das Similar.
#
# Sinn: eine Funktion landet im Batch-Tail GENAU dann, wenn sie mit Vollkontext
# stagniert. Dann ist "weniger Rauschen" die richtige Antwort — die diff-only-Stufe
# befreit z.B. Faelle, in denen ein Similar das falsche Mikro-Muster aufzwingt.

CONTEXT_STAGES = [
    {"name": "full",      "techenv": "full", "cheatsheet": True,  "similar": True},
    {"name": "no_cs",     "techenv": "full", "cheatsheet": False, "similar": True},
    {"name": "hard_env",  "techenv": "hard", "cheatsheet": False, "similar": True},
    {"name": "diff_only", "techenv": "hard", "cheatsheet": False, "similar": False},
]

# Inferierte Felder, die in der "hard"-Stufe entfernt werden. Der Diff liefert
# Typen/Breiten/Branches praeziser; diese Felder fuehren bei Konflikt in die Irre.
_SOFT_FUNC_FIELDS = {"args", "argc_conf", "ret", "ret_conf", "br", "mem"}


def _strip_techenv_soft(tech_env):
    """Entfernt die inferierten Felder, behaelt die harten Fakten
    (Namen, Groessen, sf, Stack-Layout, Calls, Globals). Bei Parse-Fehler:
    Original unveraendert zurueck."""
    if not tech_env:
        return tech_env
    try:
        data = json.loads(tech_env)
    except Exception:
        return tech_env
    funcs = data.get("funcs")
    if isinstance(funcs, list):
        for f in funcs:
            if isinstance(f, dict):
                for k in _SOFT_FUNC_FIELDS:
                    f.pop(k, None)
    env = data.get("env")
    if isinstance(env, dict):
        env.pop("ext", None)
    try:
        return json.dumps(data, indent=2)
    except Exception:
        return tech_env


def _distinct_stages(has_cheatsheet=True, has_similar=True):
    """Liefert die Liste der TATSAECHLICH unterscheidbaren Stufen, gegeben welche
    Zutaten (Cheatsheet/Similar) fuer diese Funktion vorhanden sind. Stufen, die
    sich nur in fehlenden Zutaten unterscheiden, kollabieren zu einer."""
    seen = set()
    distinct = []
    for st in CONTEXT_STAGES:
        key = (st["techenv"],
               st["cheatsheet"] and has_cheatsheet,
               st["similar"] and has_similar)
        if key in seen:
            continue
        seen.add(key)
        distinct.append(st)
    return distinct


def _distinct_stage_count(has_cheatsheet=True, has_similar=True):
    """Anzahl unterscheidbarer Stufen (1..len(CONTEXT_STAGES)). Basis fuer das
    adaptive Batch-Budget: je mehr Kontext-Zutaten vorhanden, desto mehr lohnt
    sich Exploration."""
    return len(_distinct_stages(has_cheatsheet, has_similar))


def _stages_for_batch(n, has_cheatsheet=True, has_similar=True):
    """Liefert n Stufen, round-robin aufgefuellt — ABER zuerst dedupliziert nach
    den TATSAECHLICH vorhandenen Zutaten.

    Cheatsheet UND Similar werden gemeinsam via --cheatsheet geladen; fehlt das
    Flag, sind beide leer. Dann unterscheiden sich Stufen, die sich nur in diesen
    Zutaten trennen, NICHT mehr — z.B. 'full' und 'no_cs' werden identisch. Solche
    Duplikate werden zu EINER Stufe zusammengefasst, damit kein Slot auf einen
    identischen Prompt verschwendet wird. Die verbleibenden distinct-Stufen werden
    dann round-robin (temp-divers) auf n Slots aufgefuellt.

    Beispiel-Distinct-Zahl:
      cs+sim:        4 (full / no_cs / hard_env / diff_only)
      nur cs:        3 (diff_only faellt mit hard_env zusammen)
      nur sim:       3 (no_cs faellt mit full zusammen)
      weder noch:    2 (nur die TechEnv-Varianten full / hard bleiben)"""
    if n <= 1:
        return [CONTEXT_STAGES[0]]
    distinct = _distinct_stages(has_cheatsheet, has_similar)
    return [distinct[i % len(distinct)] for i in range(n)]


def _apply_stage(stage, tech_env, similar_context, cheatsheet_context):
    """Filtert die Kontext-Komponenten gemaess Stufe. History bleibt immer
    erhalten (Anti-Repeat-Signal, kein Rauschen)."""
    te = _strip_techenv_soft(tech_env) if stage["techenv"] == "hard" else tech_env
    cs = cheatsheet_context if stage["cheatsheet"] else ""
    sim = similar_context if stage["similar"] else ""
    return te, sim, cs


# === MAIN API ===

def generate_fix(
    c_code, context_data, task_type, temperature, tech_env,
    backend=None, history_context="", confidence=1.0, func_name="",
    similar_context="", cheatsheet_context="",
):
    """
    Modi:
      CONV_MODE=oneshot         — Klassisch, jeder Call isoliert
      CONV_MODE=thinking        — Multi-Turn mit Thinking
      CONV_MODE=dynamic         — Startet als Oneshot, eskaliert zu Thinking
      CONV_MODE=dynamic_batch   — Startet als Oneshot, eskaliert zu Batch (siehe wrapper)

    Orthogonal:
      --dynamic-compiler (siehe is_syntax_thinking_escalated) — wenn eine
      Funktion im Syntax-Fix-Loop steckt und 5x gescheitert ist, eskaliert der
      Wrapper sie fuer Syntax-Fix-Calls zu Thinking. Gilt AUSSCHLIESSLICH fuer
      task_type='syntax_fix' und ueberschreibt das Standard-"Syntax-Fix = Oneshot".
    """
    use_backend = backend or AI_BACKEND
    mode = _effective_mode(func_name) if func_name else CONV_MODE

    # SYNTAX-FIXES SIND DEFAULT-DETERMINISTISCH — kein Thinking, keine History.
    # Redeclaration/extern-Probleme sind Lookup-Antworten. Reasoning-Modelle
    # gehen hier in 200+ Token Selbstgespraeche ueber Triviales (siehe
    # func_80016B30-Loop). Forciere Oneshot mit leerer Historie — AUSSER
    # --dynamic-compiler hat fuer diese Funktion gerade explizit eskaliert.
    if task_type == "syntax_fix":
        # Override: --dynamic-compiler Track aktiv?
        if func_name and is_syntax_thinking_escalated(func_name):
            # Thinking fuer Syntax-Fix erlaubt. Conversation NICHT reseten —
            # bei laufendem Syntax-Loop will der Wrapper aufeinander aufbauen.
            mode = "thinking"
            _log.info(f"[{func_name}] Syntax-Fix laeuft in Thinking-Mode (--dynamic-compiler).")
            # History bleibt leer (wie unten fuer alle Thinking-Calls)
            history_context = ""
        else:
            # Default: deterministischer Oneshot
            if mode == "thinking" and func_name:
                # Conversation zuruecksetzen damit fruehere Diff-Turns die
                # Syntax-Antwort nicht kontaminieren
                reset_conversation(func_name)
            mode = "oneshot"
            history_context = ""

    # Im Thinking-Modus: History leer lassen — die Conversation
    # enthaelt die Diffs schon als Multi-Turn, der zusaetzliche
    # <previous_attempts>-Block verdoppelt nur den Druck.
    if mode == "thinking":
        hist = ""
    else:
        hist = history_context

    # Anti-Drowning (Single-Call): bei vorhandenem (starkem) Similar den Cheatsheet
    # unterdruecken. similar_context ist nur gesetzt, wenn der Score die Schwelle
    # reisst — entspricht exakt der frueheren Override-Regel, jetzt aber nur fuer
    # Einzel-Calls. Der Batch-Fahrplan staffelt das selbst und ist nicht betroffen.
    if task_type == "diff_minimize" and similar_context and cheatsheet_context:
        cheatsheet_context = ""

    system_prompt, user_message = _build_prompts(
        c_code, context_data, task_type, tech_env,
        history_context=hist, confidence=confidence,
        thinking_mode=(mode == "thinking"),
        similar_context=similar_context,
        cheatsheet_context=cheatsheet_context,
    )

    reasoning = None
    _log.info(f"[{func_name}] AI Call START: {task_type} mode={mode} temp={temperature:.1f}")
    t_call_start = __import__('time').time()

    try:
        # --- CLAUDE ---
        if use_backend == "claude":
            client = _get_anthropic_client()
            model = ANTHROPIC_MODEL
            user_msg_claude = user_message
            _has_ref = "// === STRUCTURAL REFERENCE (follow this pattern) ===" in user_msg_claude
            if _has_ref:
                user_msg_claude = user_msg_claude.replace("// === STRUCTURAL REFERENCE (follow this pattern) ===", "<structural_reference>")
                user_msg_claude = user_msg_claude.replace("// === TECHNICAL ENVIRONMENT", "</structural_reference>\n<tech_env>")
            else:
                user_msg_claude = user_msg_claude.replace("// === TECHNICAL ENVIRONMENT", "<tech_env>")
            user_msg_claude = user_msg_claude.replace("// === CURRENT C CODE (modify this) ===", "</tech_env>\n<current_code>")
            user_msg_claude = user_msg_claude.replace("// === C CODE (has compiler errors) ===", "</tech_env>\n<current_code>")
            user_msg_claude = user_msg_claude.replace("// === ASSEMBLY DIFF (JSON) ===", "</current_code>\n<assembly_diff>")
            user_msg_claude = user_msg_claude.replace("// === COMPILER ERROR ===", "</current_code>\n<compiler_error>")
            if "<assembly_diff>" in user_msg_claude and not user_msg_claude.rstrip().endswith("</assembly_diff>"):
                user_msg_claude += "\n</assembly_diff>"
            if "<compiler_error>" in user_msg_claude and not user_msg_claude.rstrip().endswith("</compiler_error>"):
                user_msg_claude += "\n</compiler_error>"

            func_match = re.search(r'(?:void|int|s32|u32|f32)\s+(\w+)\s*\(', c_code)
            prefill = func_match.group(0) if func_match else ""
            messages = [{"role": "user", "content": user_msg_claude}]
            if prefill:
                messages.append({"role": "assistant", "content": prefill})

            response = client.chat.completions.create(
                model=model, messages=messages, system=system_prompt,
                temperature=temperature, top_p=0.95, max_tokens=8192,
            )
            raw_output = response.choices[0].message.content
            if prefill:
                raw_output = prefill + raw_output

        # --- DASHSCOPE (Alibaba Cloud API) ---
        elif use_backend == "dashscope":
            client = _get_dashscope_client()
            model = DASHSCOPE_MODEL
            # DashScope-Family aus Modelname ableiten (kann von LOCAL abweichen)
            ds_family = _detect_model_family(DASHSCOPE_MODEL)

            if mode == "thinking" and func_name:
                # Multi-Turn Thinking mit DashScope
                conv = _get_conversation(func_name, system_prompt)
                _append_to_conversation(func_name, "user", user_message)
                _trim_conversation(func_name, max_turns=15)
                conv = _get_conversation(func_name, system_prompt)

                response = client.chat.completions.create(
                    model=model,
                    messages=list(conv),
                    temperature=temperature,
                    max_tokens=8192,
                    extra_body=_dashscope_extra_body(ds_family, enable_thinking=True),
                )
                msg_obj = response.choices[0].message
                raw_output = msg_obj.content or ""
                reasoning = getattr(msg_obj, "reasoning_content", None)
                # Sicherheitsnetz: Channel-/Think-Tags entfernen falls der
                # API-Parser sie nicht abgespalten hat (kommt bei Gemma 4 vor)
                if raw_output:
                    stripped, extracted = _strip_reasoning_markers(raw_output)
                    if extracted and not reasoning:
                        reasoning = extracted
                    raw_output = stripped
                if raw_output:
                    _append_to_conversation(func_name, "assistant", raw_output,
                                            reasoning_content=reasoning)
            else:
                # Oneshot mit DashScope
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temperature,
                    max_tokens=8192,
                    extra_body=_dashscope_extra_body(ds_family, enable_thinking=False),
                )
                raw_output = response.choices[0].message.content or ""

        # --- THINKING MODE (Local vLLM Multi-Turn) ---
        elif mode == "thinking" and func_name:
            client = _get_local_client()
            model = LOCAL_MODEL

            conv = _get_conversation(func_name, system_prompt)
            _append_to_conversation(func_name, "user", user_message)
            _trim_conversation(func_name, max_turns=15)
            conv = _get_conversation(func_name, system_prompt)

            if USE_STREAMING:
                from modules.streaming import stream_thinking_response
                raw_output, reasoning, bailed, stream_duration = stream_thinking_response(
                    client=client, model=model, messages=list(conv),
                    temperature=temperature, func_name=func_name,
                )
            else:
                # Normaler Call (schneller, aber 524-Timeout moeglich)
                bailed = False
                try:
                    response = client.chat.completions.create(
                        model=model, messages=list(conv),
                        temperature=temperature, max_tokens=8192,
                        extra_body=_thinking_extra_body(MODEL_FAMILY, enable_thinking=True),
                    )
                    msg_obj = response.choices[0].message
                    raw_output = msg_obj.content or ""
                    reasoning = getattr(msg_obj, "reasoning_content", None)
                except Exception as call_err:
                    if "524" in str(call_err):
                        _log.warning(f"[{func_name}] 524 Timeout bei Thinking-Call. Versuche Streaming-Fallback...")
                        # Einmaliger Streaming-Fallback
                        try:
                            from modules.streaming import stream_thinking_response
                            raw_output, reasoning, bailed, _ = stream_thinking_response(
                                client=client, model=model, messages=list(conv),
                                temperature=temperature, func_name=func_name,
                            )
                        except Exception:
                            _log.error(f"[{func_name}] Streaming-Fallback auch fehlgeschlagen.")
                            return c_code
                    else:
                        raise

            # Fallback: Thinking-Marker aus Content parsen (Qwen <think> ODER
            # Gemma 4 <|channel>thought\n...<channel|>). Wichtig fuer Gemma 4,
            # da der leere Channel-Block auch bei deaktiviertem Thinking auftaucht.
            if raw_output:
                stripped, extracted = _strip_reasoning_markers(raw_output)
                if extracted and not reasoning:
                    reasoning = extracted
                # Channel-/Think-Tags immer aus raw_output entfernen, damit sie
                # NICHT in der Multi-Turn-History landen (siehe Gemma-Modelcard:
                # "No Thinking Content in History").
                raw_output = stripped

            if raw_output:
                _append_to_conversation(func_name, "assistant", raw_output, reasoning_content=reasoning)

            # Dynamic-Mode: Nach dem Thinking-Call zurueck zu Oneshot
            if CONV_MODE == "dynamic":
                de_escalate(func_name)

            if bailed:
                _log.info(f"[{func_name}] KI hat aufgegeben — Stagnation wird gezaehlt.")
                return c_code

        # --- ONESHOT MODE ---
        else:
            client = _get_local_client()
            model = LOCAL_MODEL
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature, max_tokens=8192,
                extra_body=_thinking_extra_body(MODEL_FAMILY, enable_thinking=False),
            )
            raw_output = response.choices[0].message.content

        reasoning_str = f"\n[REASONING]\n{reasoning}" if reasoning else ""
        t_call_end = __import__('time').time()
        call_duration = t_call_end - t_call_start

        _log.info(f"[{func_name}] AI Call END: mode={mode} Tokens~{len(raw_output)//4} Duration={call_duration:.1f}s")
        _ai_log.info(
            f"[FUNC] {task_type} ({func_name})\n"
            f"[BACKEND] {use_backend} mode={mode} temp={temperature} duration={call_duration:.1f}s\n"
            f"[SYSTEM PROMPT]\n{system_prompt}\n"
            f"[USER MESSAGE]\n{user_message}{reasoning_str}\n"
            f"[RAW RESPONSE]\n{raw_output}"
        )

        clean_code = _extract_c_code(raw_output)
        if not clean_code:
            _log.warning(f"[AI] Code-Extraktion fehlgeschlagen. (Raw: {len(raw_output)} chars)")
            return c_code
        return clean_code

    except Exception as e:
        _log.error(f"AI Agent Fehler ({use_backend}): {e}")
        return c_code


def generate_fix_batch(
    c_code, context_data, task_type, temperature, tech_env,
    backend=None, history_context="", confidence=1.0, func_name="",
    batch_size=1, similar_context="", cheatsheet_context="",
):
    """
    Kontext-Fahrplan-Batch: Generiert batch_size Antworten parallel, wobei JEDER
    Slot eine andere KONTEXT-STUFE bekommt (siehe CONTEXT_STAGES) — von Vollkontext
    bis diff-only. Diversitaet kommt aus der Kontext-Zusammensetzung, nicht aus dem
    Temperatur-Spread. Nur im Oneshot-Modus (local/dashscope).
    Returns: Liste von C-Code Strings.
    """
    if batch_size <= 1:
        result = generate_fix(c_code, context_data, task_type, temperature, tech_env,
                              backend=backend, history_context=history_context,
                              confidence=confidence, func_name=func_name,
                              similar_context=similar_context,
                              cheatsheet_context=cheatsheet_context)
        return [result]

    use_backend = backend or AI_BACKEND

    # Der Kontext-Fahrplan gilt NUR fuer diff_minimize. Bei syntax_fix sind die
    # TechEnv-Typen ('env.ext') autoritativ — wir duerfen sie NICHT strippen.
    # Dort bleibt es beim klassischen Best-of-N (gleicher Prompt, Temp-Spread).
    slots = []  # (stage_name, system_prompt, user_message)
    if task_type == "diff_minimize":
        _stages = _stages_for_batch(batch_size, bool(cheatsheet_context), bool(similar_context))
        for st in _stages:
            te_s, sim_s, cs_s = _apply_stage(st, tech_env, similar_context, cheatsheet_context)
            sp, um = _build_prompts(
                c_code, context_data, task_type, te_s,
                history_context=history_context, confidence=confidence,
                thinking_mode=False, similar_context=sim_s, cheatsheet_context=cs_s,
            )
            slots.append((st["name"], sp, um))
    else:
        sp, um = _build_prompts(
            c_code, context_data, task_type, tech_env,
            history_context=history_context, confidence=confidence,
            thinking_mode=False, similar_context=similar_context,
            cheatsheet_context=cheatsheet_context,
        )
        slots = [("full", sp, um) for _ in range(batch_size)]

    # Milder Temperatur-Spread bleibt als sekundaere Diversitaet erhalten.
    temps = []
    for i in range(batch_size):
        offset = (i - batch_size // 2) * 0.03
        t = max(0.05, min(1.0, temperature + offset))
        temps.append(round(t, 2))

    import concurrent.futures
    import time as _time

    def _single_call(args):
        idx, (stage_name, system_prompt, user_message), t = args
        try:
            if use_backend == "dashscope":
                client = _get_dashscope_client()
                model = DASHSCOPE_MODEL
                extra = _dashscope_extra_body(_detect_model_family(DASHSCOPE_MODEL),
                                              enable_thinking=False)
            else:
                client = _get_local_client()
                model = LOCAL_MODEL
                extra = _thinking_extra_body(MODEL_FAMILY, enable_thinking=False)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=t, max_tokens=8192,
                extra_body=extra,
            )
            raw = response.choices[0].message.content or ""
            code = _extract_c_code(raw)
            return (stage_name, system_prompt, user_message, raw, code if code else "")
        except Exception as e:
            _log.debug(f"[{func_name}] Batch-Slot {idx} ({stage_name}) fehlgeschlagen: {e}")
            return (stage_name, system_prompt, user_message, "", "")

    stage_names = [s[0] for s in slots]
    if task_type == "diff_minimize":
        _log.info(f"[{func_name}] Batch AI Call: {batch_size}x stages={stage_names} "
                  f"(cs={bool(cheatsheet_context)} sim={bool(similar_context)}) temps={temps}")
    else:
        _log.info(f"[{func_name}] Batch AI Call: {batch_size}x (syntax_fix, best-of-N) temps={temps}")
    t_start = _time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
        slot_results = list(pool.map(_single_call, zip(range(batch_size), slots, temps)))

    duration = _time.time() - t_start
    results = [r[4] for r in slot_results]
    valid = [r for r in results if r and r != c_code]
    _log.info(f"[{func_name}] Batch AI Call END: {len(valid)}/{batch_size} valide, "
              f"Duration={duration:.1f}s")

    # Pro Slot vollstaendig protokollieren — damit die Analyse sieht, WELCHE
    # Kontext-Stufe welche Antwort produziert hat.
    for idx, (stage_name, sp, um, raw, code) in enumerate(slot_results):
        _ai_log.info(
            f"[FUNC] {task_type} BATCH ({func_name}) SLOT {idx} stage={stage_name} "
            f"temp={temps[idx]} backend={use_backend}\n"
            f"[SYSTEM PROMPT]\n{sp}\n"
            f"[USER MESSAGE]\n{um}\n"
            f"[RAW RESPONSE]\n{raw}"
        )

    return results if results else [c_code]