# modules/live_view.py
# Rich Live-Terminal-Dashboard — stabile Version.
# Nutzt Alternate Screen Buffer (kein Flackern).

import threading

from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.syntax import Syntax
from rich.text import Text
from rich.console import Console


class LivePanel:
    """Thread-safe Rich Live-View mit Alternate Screen Buffer."""

    def __init__(self):
        self._lock = threading.Lock()
        # screen=True: Alternate Screen Buffer (wie vim/top)
        self._console = Console(force_terminal=True)
        self._live = None

        # State (wird via update() geschrieben, via _render() gelesen)
        self._func_name = ""
        self._phase = ""
        self._c_code = ""
        self._context = ""
        self._score = 0.0
        self._struct_score = 0.0
        self._mismatches = 0
        self._round = 0
        self._temperature = 0.0
        self._attempt = 0
        self._max_attempts = 0

    def start(self):
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,   # Entkoppelt: max 4 Redraws/s
            screen=True,            # Alternate Screen Buffer
        )
        self._live.start()

    def stop(self):
        if self._live:
            self._live.stop()

    def update(self, **kwargs):
        """Thread-safe State-Update. Schreibt nur in Variablen, kein Redraw."""
        with self._lock:
            self._func_name = kwargs.get("func_name", self._func_name)
            self._phase = kwargs.get("phase", self._phase)
            self._score = kwargs.get("score", self._score)
            self._struct_score = kwargs.get("struct_score", self._struct_score)
            self._mismatches = kwargs.get("mismatches", self._mismatches)
            self._round = kwargs.get("round", self._round)
            self._temperature = kwargs.get("temperature", self._temperature)
            self._attempt = kwargs.get("attempt", self._attempt)
            self._max_attempts = kwargs.get("max_attempts", self._max_attempts)

            if "c_code" in kwargs and kwargs["c_code"]:
                self._c_code = kwargs["c_code"]
            if "error" in kwargs:
                self._context = kwargs["error"]
            elif "diff" in kwargs:
                self._context = kwargs["diff"]

        # Live.update setzt nur das Renderable — der Refresh-Thread zeichnet
        if self._live:
            self._live.update(self._render())

    def _render(self):
        with self._lock:
            return self._build_layout()

    def _build_layout(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=4),
        )

        # === HEADER ===
        phase_style = {
            "syntax": ("red", "SYNTAX FIX"),
            "permuter": ("yellow", "PERMUTER"),
            "diff": ("cyan", "DIFF MINIMIZE"),
            "ai_call": ("magenta", "AI CALL"),
        }
        color, phase_label = phase_style.get(self._phase, ("white", self._phase.upper()))

        header = Text()
        header.append(f" ⚙ {self._func_name} ", style="bold white on dark_blue")
        header.append(f"  ", style="dim")
        header.append(f" {phase_label} ", style=f"bold white on {color}")

        if self._phase == "syntax" and self._max_attempts > 0:
            header.append(f"  Versuch {self._attempt}/{self._max_attempts}", style="dim")
        elif self._round > 0:
            header.append(f"  Runde {self._round}", style="dim")

        layout["header"].update(Panel(header, border_style=color))

        # === BODY: Code links, Kontext rechts ===
        layout["body"].split_row(
            Layout(name="code", ratio=1),
            Layout(name="context", ratio=1),
        )

        # C-Code Panel (mit Syntax-Highlighting)
        c_text = self._c_code[:4000] if self._c_code else "// (waiting for code...)"
        try:
            code_widget = Syntax(c_text, "c", theme="monokai", line_numbers=True, word_wrap=True)
        except Exception:
            code_widget = Text(c_text)
        layout["code"].update(Panel(code_widget, title="C-Code", border_style="green"))

        # Kontext Panel (JSON-Diff oder Compiler-Error)
        ctx_text = self._context[:4000] if self._context else "// (waiting for context...)"
        if self._phase == "syntax":
            ctx_title = "Compiler Error"
            ctx_border = "red"
            ctx_widget = Text(ctx_text, style="red")
        else:
            ctx_title = "Assembly Diff"
            ctx_border = "blue"
            # JSON Syntax-Highlighting wenn es wie JSON aussieht
            if ctx_text.strip().startswith(("[", "{")):
                try:
                    ctx_widget = Syntax(ctx_text, "json", theme="monokai", word_wrap=True)
                except Exception:
                    ctx_widget = Text(ctx_text)
            else:
                ctx_widget = Text(ctx_text)
        layout["context"].update(Panel(ctx_widget, title=ctx_title, border_style=ctx_border))

        # === FOOTER: Score Grid ===
        grid = Table.grid(expand=True, padding=(0, 2))
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="center", ratio=1)

        if self._score >= 80:
            sc = "bold green"
        elif self._score >= 40:
            sc = "bold yellow"
        else:
            sc = "bold red"

        grid.add_row(
            Text(f"Match: {self._score:.1f}%", style=sc),
            Text(f"Struct: {self._struct_score:.1f}%", style="cyan"),
            Text(f"Mismatches: {self._mismatches}", style="dim white"),
            Text(f"Temp: {self._temperature:.2f}", style="dim white"),
        )

        layout["footer"].update(Panel(grid, border_style="dim"))

        return layout