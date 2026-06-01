# modules/streaming.py
# Streaming-Handler fuer Thinking-Modelle (Qwen3.x, Gemma 4).
# Debug: --debug Flag -> data/debug/{func_name}.txt pro Funktion

import os
import time
import logging

# Family-aware extra_body und Reasoning-Strip aus ai_agent wiederverwenden,
# damit Qwen/Gemma-Unterschiede an EINER Stelle leben.
from modules.ai_agent import (
    MODEL_FAMILY,
    _thinking_extra_body,
    _strip_reasoning_markers,
)

_log = logging.getLogger(__name__)

STREAM_DEBUG = os.environ.get("STREAM_DEBUG", "") == "1"
DEBUG_DIR = os.path.join(os.getcwd(), "data", "debug")

BAIL_PHRASES = [
    "i am stuck", "i'm stuck", "no combination left",
    "already tried this", "cannot find a way", "i have no",
    "there is no way", "impossible to match",
    "repeating the same", "going in circles",
    "i cannot determine", "no valid approach",
    "exhausted all", "tried everything",
]

# Phrasen die in Loops typischerweise oft auftauchen.
# Gemma 4 zeigt im Loop "wait", "let's try X", "actually" repetitiv;
# Qwen zeigt eher "I'll output", "final check". Beides hier abgedeckt.
LOOP_PHRASES = [
    "i'll output", "i will output", "final check",
    "one last check", "one more check", "i'll stick",
    "this is correct", "proceed", "ready",
    "output matches", "i will provide",
    # Gemma 4 Loop-Symptome (siehe func_80016B30 Log)
    "let's try", "let me try", "wait,", "actually,",
    "one more idea", "another possibility",
]

STREAM_BATCH_SIZE = int(os.environ.get("STREAM_BATCH_SIZE", "50"))


def stream_thinking_response(client, model, messages, temperature,
                              max_tokens=8192, func_name=""):
    t_start = time.time()

    # Sampling + Thinking-Konfiguration aus dem family-aware Helper.
    # Setzt top_p, top_k UND chat_template_kwargs passend zur Modell-Familie:
    #   Qwen3:  top_k=20, chat_template_kwargs={enable_thinking, preserve_thinking}
    #   Gemma4: top_k=64, chat_template_kwargs={enable_thinking}
    extra_body = _thinking_extra_body(MODEL_FAMILY, enable_thinking=True)

    stream = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
        stream=True,
        extra_body=extra_body,
    )

    raw_chunks = []
    reasoning_chunks = []
    bailed = False
    total_chunks = 0
    batch_counter = 0
    last_log_time = t_start
    content_started = False
    bail_window_size = 80

    debug_file = None
    if STREAM_DEBUG:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        debug_path = os.path.join(DEBUG_DIR, f"{func_name}.txt")
        # Append-Modus: jeder Thinking-Call wird angehaengt
        debug_file = open(debug_path, "a", encoding="utf-8")
        debug_file.write(f"\n{'='*60}\n")
        debug_file.write(f"[{time.strftime('%H:%M:%S')}] Thinking Call | temp={temperature}\n")
        debug_file.write(f"{'='*60}\n\n")
        debug_file.write("[REASONING]\n")
        debug_file.flush()

    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if not delta:
                continue

            total_chunks += 1
            batch_counter += 1

            rc = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if rc:
                reasoning_chunks.append(rc)
                if debug_file:
                    debug_file.write(rc)
                    if total_chunks % 20 == 0:
                        debug_file.flush()

            if delta.content:
                raw_chunks.append(delta.content)
                if debug_file and not content_started:
                    content_started = True
                    debug_file.write(f"\n\n[CODE OUTPUT] ({time.time() - t_start:.1f}s)\n")
                    debug_file.flush()
                if debug_file:
                    debug_file.write(delta.content)
                    if total_chunks % 20 == 0:
                        debug_file.flush()

            if batch_counter >= STREAM_BATCH_SIZE:
                batch_counter = 0
                now = time.time()

                if now - last_log_time >= 15:
                    elapsed = now - t_start
                    _log.debug(f"[{func_name}] Streaming: {elapsed:.0f}s, "
                               f"reasoning={len(reasoning_chunks)}, content={len(raw_chunks)}")
                    last_log_time = now

                # === NOTBREMSEN ===
                # Schwellen wurden gesenkt nach Beobachtung des func_80016B30 Loops,
                # wo das alte 200-Chunk-Limit zu spaet griff.
                if len(reasoning_chunks) > 40:
                    # Groesseres Fenster (160 statt 80 Chunks) — Loop-Phrasen
                    # bei Gemma sind kurz und kommen oft, brauchen breiteren Blick
                    recent = "".join(reasoning_chunks[-160:]).lower()

                    # 1. Semantische Notbremse: KI sagt explizit dass sie stuck ist
                    if any(phrase in recent for phrase in BAIL_PHRASES):
                        duration = time.time() - t_start
                        _log.warning(f"[{func_name}] Notbremse (semantisch) nach {duration:.1f}s")
                        bailed = True
                        if debug_file:
                            debug_file.write(f"\n\n!!! NOTBREMSE (semantisch) nach {duration:.1f}s !!!\n")
                        break

                    # 2. Repetition-Loop: ab 120 Chunks pruefen (war 200)
                    if len(reasoning_chunks) > 120:
                        last_60 = "".join(reasoning_chunks[-60:])
                        # Wenn die letzten 60 Chunks in den 120 davor schon vorkamen = Loop
                        if last_60 in "".join(reasoning_chunks[-180:-60]):
                            duration = time.time() - t_start
                            _log.warning(f"[{func_name}] Notbremse (Repetition-Loop) nach {duration:.1f}s "
                                         f"({len(reasoning_chunks)} chunks)")
                            bailed = True
                            if debug_file:
                                debug_file.write(f"\n\n!!! NOTBREMSE (Repetition-Loop) nach {duration:.1f}s !!!\n")
                            break

                    # 3. Phrase-Repetition: nutzt jetzt globale LOOP_PHRASES Liste
                    #    Schwelle auf 4 gesenkt (war 5) — bei Gemma kommen die
                    #    Loop-Phrasen schneller und dichter
                    phrase_count = sum(recent.count(p) for p in LOOP_PHRASES)
                    if phrase_count >= 4:
                        duration = time.time() - t_start
                        _log.warning(f"[{func_name}] Notbremse (Phrase-Repetition x{phrase_count}) "
                                     f"nach {duration:.1f}s")
                        bailed = True
                        if debug_file:
                            debug_file.write(f"\n\n!!! NOTBREMSE (Phrase-Repetition x{phrase_count}) "
                                             f"nach {duration:.1f}s !!!\n")
                        break
    finally:
        if debug_file:
            duration = time.time() - t_start
            debug_file.write(f"\n[RESULT] {duration:.1f}s | reasoning={len(reasoning_chunks)} | "
                             f"content={len(raw_chunks)} | bailed={bailed}\n")
            debug_file.close()

    duration = time.time() - t_start
    raw_output = "".join(raw_chunks)
    reasoning = "".join(reasoning_chunks) if reasoning_chunks else None

    # Fallback wenn der vLLM-Reasoning-Parser kein reasoning_content lieferte:
    # Tags aus dem Content extrahieren. Erkennt sowohl Qwen <think>...</think>
    # als auch Gemma 4 <|channel>thought\n...<channel|> (inkl. leerer Bloecke).
    if raw_output:
        stripped, extracted = _strip_reasoning_markers(raw_output)
        if extracted and not reasoning:
            reasoning = extracted
        # raw_output IMMER strippen — die Tags duerfen nicht in die History
        # wandern (Gemma-Modelcard: "No Thinking Content in History").
        raw_output = stripped

    return raw_output, reasoning, bailed, duration