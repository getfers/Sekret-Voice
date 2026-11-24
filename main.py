"""CustomTkinter desktop interface for Sekret Voice."""

from dataclasses import replace
from math import log10
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Thread
from time import monotonic
import tkinter as tk

import customtkinter as ctk
from PIL import Image, ImageOps

from audio_devices import default_devices, devices
from audio_engine import AudioEngine, audio_worker
from dsp import NOTES, SCALES
from presets import DEFAULT_PRESET, PRESETS


BASE = Path(__file__).resolve().parent
WHITE = "#FFFFFF"
BLACK = "#000000"
MUTED = "#515151"
LINE = "#E4E4E4"
SURFACE = "#F7F7F7"
HOVER = "#EDEDED"
FONT = "Segoe UI"
EMPTY_DEVICE = "Нет доступных устройств"


def label(parent, value, size=13, bold=False, **kwargs):
    return ctk.CTkLabel(
        parent, text=value, font=ctk.CTkFont(FONT, size, weight="bold" if bold else "normal"),
        text_color=kwargs.pop("text_color", BLACK), anchor=kwargs.pop("anchor", "w"),
        height=kwargs.pop("height", 1), **kwargs,
    )


def button(parent, value, command, primary=False, **kwargs):
    return ctk.CTkButton(
        parent, text=value, command=command, corner_radius=8, border_width=0 if primary else 1,
        border_color=LINE, fg_color=BLACK if primary else WHITE,
        text_color=WHITE if primary else BLACK, hover_color="#292929" if primary else HOVER,
        font=ctk.CTkFont(FONT, 13, weight="bold"), **kwargs,
    )


def menu(parent, values, command=None, width=110, **kwargs):
    return ctk.CTkOptionMenu(
        parent, values=list(values), command=command, width=width, height=32, corner_radius=7,
        fg_color=SURFACE, button_color=SURFACE, button_hover_color=HOVER,
        text_color=BLACK, dropdown_fg_color=WHITE, dropdown_hover_color=HOVER,
        dropdown_text_color=BLACK, font=ctk.CTkFont(FONT, 12),
        dropdown_font=ctk.CTkFont(FONT, 12), **kwargs,
    )


def brand_image():
    with Image.open(BASE / "logo.png") as source:
        # The original file also contains a separate square mark used in sekret.ico.
        wordmark = ImageOps.autocontrast(source.convert("L").crop((23, 180, 607, 330)), cutoff=1)
    return wordmark.point(lambda value: 255 if value > 246 else value).convert("RGB")


class PresetCard(ctk.CTkFrame):
    def __init__(self, parent, name, description, activate):
        super().__init__(parent, fg_color=WHITE, border_color=LINE, border_width=1, corner_radius=10)
        tk.Frame.configure(self, takefocus=1, cursor="hand2")
        self.selected = False
        title = label(self, name, 14, True)
        title.pack(anchor="w", padx=13, pady=(11, 3))
        self.subtitle = label(self, description, 11, text_color=MUTED, wraplength=153, justify="left", width=1)
        self.subtitle.pack(fill="x", padx=13, pady=(0, 10))
        for part in (self, title, self.subtitle):
            part.bind("<Button-1>", lambda event: activate(name))
        tk.Misc.bind(self, "<Return>", lambda event: activate(name))
        tk.Misc.bind(self, "<space>", lambda event: activate(name))
        tk.Misc.bind(self, "<FocusIn>", lambda event: self.configure(border_color=BLACK))
        tk.Misc.bind(self, "<FocusOut>", lambda event: self.update_selection(self.selected))

    def update_selection(self, selected):
        self.selected = selected
        self.configure(border_color=BLACK if selected else LINE, fg_color=SURFACE if selected else WHITE)


class SekretVoice(ctk.CTk):
    def __init__(self, splash=True):
        super().__init__(fg_color=WHITE)
        self.withdraw()
        self.title("Sekret Voice")
        self.minsize(790, 590)
        width = min(1190, max(790, self.winfo_screenwidth() - 90))
        height = min(800, max(590, self.winfo_screenheight() - 96))
        left = (self.winfo_screenwidth() - width) // 2
        top = max(0, (self.winfo_screenheight() - height) // 2 - 12)
        self.geometry(f"{width}x{height}+{left}+{top}")
        if self.tk.call("tk", "windowingsystem") == "win32":
            self.iconbitmap(str(BASE / "sekret.ico"))
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.engine = AudioEngine()
        self.inputs = {}
        self.outputs = {}
        self.default_output = None
        self.monitor_preference = None
        self.monitor_device = None
        self.monitor_choices = {}
        self.mode = None
        self.busy = False
        self.closing = False
        self.audio_tasks = SimpleQueue()
        self.audio_requests = SimpleQueue()
        self.selected = None
        self.columns = 0
        self.card_wrap = 153
        self.visible_card_wrap = 153
        self.layout_job = None
        self.level = 0.0
        self.last_meter = None
        self.last_audio_info = None
        self._logo = brand_image()

        self._build_header()
        self._build_body()
        self._build_footer()
        self._select(DEFAULT_PRESET)
        self.refresh_devices()
        self._filter_presets()
        self.bind("<Control-f>", self._focus_search)
        self.bind("<Escape>", self._clear_search)
        self.audio_thread = Thread(
            target=audio_worker, args=(self.engine, self.audio_requests, self.audio_tasks),
            daemon=True, name="audio-setup",
        )
        self.audio_thread.start()
        self.after(42, self._poll_audio)
        if splash:
            self._show_splash()
        else:
            self.deiconify()

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=WHITE, corner_radius=0)
        header.pack(fill="x", padx=28, pady=(20, 17))
        wordmark = ctk.CTkImage(self._logo, size=(141, 36))
        ctk.CTkLabel(header, text="", image=wordmark, width=141, height=36).pack(side="left")
        label(header, "VOICE", 14, True).pack(side="left", padx=(13, 0))
        self.run_button = button(header, "Включить голос", self._toggle_stream, primary=True, width=156, height=40)
        self.run_button.pack(side="right")
        self.state_label = label(header, "Голос выключен", 12, text_color=MUTED)
        self.state_label.pack(side="right", padx=(0, 16))
        ctk.CTkFrame(self, fg_color=LINE, height=1, corner_radius=0).pack(fill="x")

    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color=WHITE, corner_radius=0)
        body.pack(fill="both", expand=True, padx=28, pady=(20, 0))
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)
        sidebar = ctk.CTkScrollableFrame(
            body, width=288, fg_color=WHITE, border_width=1, border_color=LINE, corner_radius=12,
            scrollbar_button_color="#E5E5E5", scrollbar_button_hover_color="#8F8F8F",
        )
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 20))
        sidebar.grid_columnconfigure(0, weight=1)
        self._build_audio(sidebar)
        self._build_tune(sidebar)

        library = ctk.CTkFrame(body, fg_color=WHITE, corner_radius=0)
        library.grid(row=0, column=1, sticky="nsew")
        active = ctk.CTkFrame(library, fg_color=SURFACE, corner_radius=12)
        active.pack(fill="x", pady=(0, 17))
        active_text = ctk.CTkFrame(active, fg_color="transparent")
        active_text.pack(side="left", fill="x", expand=True, padx=16, pady=12)
        label(active_text, "Активный голос", 11, True, text_color=MUTED).pack(anchor="w", pady=(0, 2))
        self.active_name = label(active_text, "", 25, True)
        self.active_name.pack(anchor="w")
        self.active_detail = label(
            active_text, "", 12, text_color=MUTED, width=1, height=31,
            anchor="nw", wraplength=350, justify="left",
        )
        self.active_detail.pack(fill="x", pady=(3, 0))

        section = ctk.CTkFrame(library, fg_color=WHITE, corner_radius=0)
        section.pack(fill="x", pady=(0, 9))
        label(section, "Пресеты голоса", 23, True).pack(side="left")
        self.count = label(section, "", 12, text_color=MUTED)
        self.count.pack(side="left", padx=(12, 0), pady=(4, 0))
        label(library, "Поиск по названию", 11, text_color=MUTED).pack(anchor="w", pady=(0, 4))
        self.query = tk.StringVar()
        self.search = ctk.CTkEntry(
            library, textvariable=self.query, height=35, corner_radius=8,
            fg_color=WHITE, border_width=1, border_color=LINE, text_color=BLACK,
            font=ctk.CTkFont(FONT, 13),
        )
        self.search.pack(fill="x", pady=(0, 12))
        self.search.bind("<Return>", self._select_first)
        self.cards_area = ctk.CTkScrollableFrame(
            library, fg_color=WHITE, corner_radius=0, border_width=0,
            scrollbar_button_color="#CDCDCD", scrollbar_button_hover_color="#8F8F8F",
        )
        self.cards_area.pack(fill="both", expand=True)
        self.cards = {
            name: PresetCard(self.cards_area, name, preset.description, self._select)
            for name, preset in PRESETS.items()
        }
        self.no_results = label(self.cards_area, "По этому запросу ничего не найдено.", 14, text_color=MUTED)
        self.query.trace_add("write", self._search_changed)
        tk.Misc.bind(library, "<Configure>", self._schedule_layout, add="+")

    def _build_audio(self, parent):
        panel = ctk.CTkFrame(parent, fg_color="transparent")
        panel.grid(row=0, column=0, sticky="ew", padx=11, pady=(7, 0))
        header = ctk.CTkFrame(panel, fg_color="transparent")
        header.pack(fill="x", pady=(0, 15))
        label(header, "Устройства", 19, True).pack(side="left")
        self.refresh_button = button(
            header, "Обновить", lambda: self.refresh_devices(refresh=True), width=79, height=27,
        )
        self.refresh_button.pack(side="right")
        label(panel, "1. Ваш микрофон", 13, True).pack(anchor="w", pady=(0, 5))
        self.input_menu = menu(panel, [EMPTY_DEVICE], width=235, dynamic_resizing=False)
        self.input_menu.pack(fill="x", pady=(0, 13))
        label(panel, "2. Куда отправлять голос", 13, True).pack(anchor="w", pady=(0, 5))
        self.output_menu = menu(panel, [EMPTY_DEVICE], self._output_changed, width=235, dynamic_resizing=False)
        self.output_menu.pack(fill="x", pady=(0, 4))
        self.output_hint = label(panel, "", 11, text_color=MUTED)
        self.output_hint.pack(anchor="w", pady=(0, 13))

        label(panel, "Наушники для прослушивания", 12, True).pack(anchor="w", pady=(0, 5))
        self.monitor_menu = menu(panel, [EMPTY_DEVICE], self._set_monitor, width=235, dynamic_resizing=False)
        self.monitor_menu.pack(fill="x", pady=(0, 6))
        self.monitor_button = button(panel, "Прослушать себя", self._toggle_monitor, height=34)
        self.monitor_button.pack(fill="x", pady=(0, 4))

        ctk.CTkFrame(panel, fg_color=LINE, height=1, corner_radius=0).pack(fill="x", pady=(10, 10))
        gain_line = ctk.CTkFrame(panel, fg_color="transparent")
        gain_line.pack(fill="x")
        label(gain_line, "Громкость", 16, True).pack(side="left")
        self.gain_value = label(gain_line, "0 дБ", 12, text_color=MUTED)
        self.gain_value.pack(side="right")
        gain = self._slider(panel, -18, 24, 42, self._set_gain, height=20)
        gain.set(0)
        gain.pack(fill="x", pady=(5, 7))
        meter_line = ctk.CTkFrame(panel, fg_color="transparent")
        meter_line.pack(fill="x", pady=(0, 4))
        label(meter_line, "Уровень микрофона", 11, text_color=MUTED).pack(side="left")
        self.meter_value = label(meter_line, "—", 11, text_color=MUTED)
        self.meter_value.pack(side="right")
        self.meter = ctk.CTkProgressBar(panel, height=6, corner_radius=3, fg_color=LINE, progress_color=BLACK)
        self.meter.set(0)
        self.meter.pack(fill="x")

    def _build_tune(self, parent):
        tune = ctk.CTkFrame(parent, fg_color=SURFACE, corner_radius=9)
        tune.grid(row=1, column=0, sticky="ew", padx=9, pady=(11, 8))
        heading = ctk.CTkFrame(tune, fg_color="transparent")
        heading.pack(fill="x", padx=10, pady=(8, 4))
        label(heading, "Auto-Tune", 16, True).pack(side="left")
        self.tune_switch = ctk.CTkSwitch(
            heading, text="", width=37, fg_color="#B6B6B6", progress_color=BLACK,
            button_color=WHITE, button_hover_color=HOVER, command=self._set_tune,
        )
        self.tune_switch.pack(side="right")

        scale_line = ctk.CTkFrame(tune, fg_color="transparent")
        scale_line.pack(fill="x", padx=10)
        label(scale_line, "Тон", 11, text_color=MUTED).grid(row=0, column=0, sticky="w")
        label(scale_line, "Гамма", 11, text_color=MUTED).grid(row=0, column=1, sticky="w", padx=(8, 0))
        scale_line.grid_columnconfigure(1, weight=1)
        self.key_menu = menu(scale_line, NOTES, self._set_note, width=52)
        self.key_menu.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self.scale_menu = menu(scale_line, SCALES, self._set_scale, width=135, dynamic_resizing=False)
        self.scale_menu.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(2, 0))
        for control in (self.key_menu, self.scale_menu):
            control.configure(fg_color=WHITE, button_color=WHITE)

        amount_line = ctk.CTkFrame(tune, fg_color="transparent")
        amount_line.pack(fill="x", padx=10, pady=(5, 0))
        label(amount_line, "Сила коррекции", 11, text_color=MUTED).pack(side="left")
        self.amount_value = label(amount_line, "70%", 11, text_color=MUTED)
        self.amount_value.pack(side="right")
        self.amount_slider = self._slider(tune, 0, 100, 100, self._set_amount)
        self.amount_slider.set(70)
        self.amount_slider.pack(fill="x", padx=10, pady=(4, 8))
        self._set_tune()

    @staticmethod
    def _slider(parent, minimum, maximum, steps, command, height=14):
        return ctk.CTkSlider(
            parent, from_=minimum, to=maximum, number_of_steps=steps, command=command,
            fg_color="#DADADA", progress_color=BLACK, button_color=BLACK,
            button_hover_color="#2A2A2A", height=height,
        )

    def _build_footer(self):
        footer = ctk.CTkFrame(self, fg_color=WHITE, corner_radius=0)
        footer.pack(fill="x", padx=29, pady=(8, 9))
        self.audio_info = label(footer, "", 11, text_color=MUTED)
        self.audio_info.pack(side="right")
        self.message = label(footer, "", 11, text_color=MUTED, wraplength=560, justify="left")
        self.message.pack(side="left", fill="x", expand=True)

    def _show_splash(self):
        splash = ctk.CTkToplevel(self, fg_color=WHITE)
        splash.withdraw()
        splash.overrideredirect(True)
        width, height = 500, 254
        x = (self.winfo_screenwidth() - width) // 2
        y = (self.winfo_screenheight() - height) // 2
        splash.geometry(f"{width}x{height}+{x}+{y}")
        shell = ctk.CTkFrame(splash, fg_color=WHITE, border_color=LINE, border_width=1, corner_radius=0)
        shell.pack(fill="both", expand=True)
        wordmark = ctk.CTkImage(self._logo, size=(288, 74))
        ctk.CTkLabel(shell, text="", image=wordmark).pack(pady=(66, 15))
        label(shell, "Вас приветствует Sekret Voice", 19, True).pack()
        splash.attributes("-alpha", 0.)
        splash.deiconify()
        splash.lift()

        def open_window():
            self.attributes("-alpha", 0.)
            self.deiconify()
            self._fade(self, 0., 1., .30, self._finish_window_fade)
            self._fade(splash, 1., 0., .26, splash.destroy)

        self._fade(splash, 0., 1., .40, lambda: self.after(720, open_window))

    @staticmethod
    def _fade(window, start, end, duration, complete=None):
        started = monotonic()

        def tick():
            progress = min(1., (monotonic() - started) / duration)
            smooth = progress * progress * (3 - 2 * progress)
            window.attributes("-alpha", start + (end - start) * smooth)
            if progress < 1.:
                window.after(16, tick)
            elif complete:
                complete()

        tick()

    def _finish_window_fade(self):
        if self.tk.call("tk", "windowingsystem") != "win32":
            return
        # Tk leaves the layered style set after opacity returns to 100%.
        # Restore ordinary painting so child controls redraw during a resize.
        import ctypes
        from ctypes.wintypes import HWND, LONG

        user32 = ctypes.WinDLL("user32")
        user32.GetParent.argtypes = (HWND,)
        user32.GetParent.restype = HWND
        user32.GetWindowLongW.argtypes = (HWND, ctypes.c_int)
        user32.GetWindowLongW.restype = LONG
        user32.SetWindowLongW.argtypes = (HWND, ctypes.c_int, LONG)
        user32.SetWindowLongW.restype = LONG
        hwnd = user32.GetParent(self.winfo_id())
        extended_style = -20
        layered = 0x80000
        style = user32.GetWindowLongW(hwnd, extended_style)
        if style & layered:
            user32.SetWindowLongW(hwnd, extended_style, style & ~layered)

    def _set_monitor(self, value):
        self.monitor_preference = value

    def _set_gain(self, value):
        db = round(value)
        self.gain_value.configure(text=f"{db:+d} дБ" if db else "0 дБ")
        self.engine.controls = replace(self.engine.controls, gain_db=db)

    def _set_tune(self):
        enabled = bool(self.tune_switch.get())
        self.engine.controls = replace(self.engine.controls, tune=enabled)
        for widget in (self.key_menu, self.scale_menu, self.amount_slider):
            widget.configure(state="normal" if enabled else "disabled")

    def _set_note(self, value):
        self.engine.controls = replace(self.engine.controls, root=NOTES.index(value))

    def _set_scale(self, value):
        self.engine.controls = replace(self.engine.controls, scale=value)

    def _set_amount(self, value):
        amount = round(value)
        self.amount_value.configure(text=f"{amount}%")
        self.engine.controls = replace(self.engine.controls, amount=amount / 100)

    def _select(self, name):
        if name == self.selected:
            return
        preset = PRESETS[name]
        if not self.busy:
            self.engine.set_preset(preset)
        if self.selected is not None:
            self.cards[self.selected].update_selection(False)
        self.selected = name
        self.cards[name].update_selection(True)
        self.active_name.configure(text=name)
        self.active_detail.configure(text=preset.description)

    def _schedule_layout(self, event):
        width = event.width / event.widget._get_widget_scaling()
        target_columns = 3 if width >= 650 else 2
        column_width = width / target_columns
        wrap = 197 if column_width >= 240 else 173 if column_width >= 213 else 153
        if target_columns == self.columns and wrap == self.card_wrap:
            return
        self.columns = target_columns
        self.card_wrap = wrap
        if self.layout_job:
            self.after_cancel(self.layout_job)
        self.layout_job = self.after_idle(self._layout_presets)

    def _layout_presets(self):
        self.layout_job = None
        if self.visible_card_wrap != self.card_wrap:
            for card in self.cards.values():
                card.subtitle.configure(wraplength=self.card_wrap)
            self.visible_card_wrap = self.card_wrap
        self._filter_presets()

    def _focus_search(self, event):
        self.search.focus_set()

    def _clear_search(self, event):
        focused = self.focus_get()
        if focused is self.search or (focused and focused.master is self.search):
            self.query.set("")

    def _search_changed(self, *_):
        self._filter_presets()
        self.cards_area.master.yview_moveto(0)

    def _select_first(self, event):
        if self.visible_presets:
            self._select(self.visible_presets[0])

    def _filter_presets(self):
        query = self.query.get().casefold().strip().replace("ё", "е")
        if query == getattr(self, "_last_query", None) and self.columns == getattr(self, "_last_columns", None):
            return
        self._last_query, self._last_columns = query, self.columns
        count = self.columns or 3
        for i in range(3):
            self.cards_area.grid_columnconfigure(i, weight=int(i < count), uniform="cards" if i < count else "")
        previous_positions = {name: index for index, name in enumerate(getattr(self, "visible_presets", ()))}
        previous_columns = getattr(self, "_visible_columns", count)
        self._visible_columns = count
        self.visible_presets = []
        self.no_results.grid_forget()
        for name, card in self.cards.items():
            if query in name.casefold().replace("ё", "е"):
                row, column = divmod(len(self.visible_presets), count)
                old_index = previous_positions.get(name)
                if old_index is None or divmod(old_index, previous_columns) != (row, column):
                    card.grid(row=row, column=column, sticky="nsew", padx=(0, 8), pady=(0, 8))
                self.visible_presets.append(name)
            elif name in previous_positions:
                card.grid_remove()
        shown = len(self.visible_presets)
        self.count.configure(text=f"{shown} / {len(PRESETS)}" if query else str(len(PRESETS)))
        if not shown:
            self.no_results.grid(row=0, column=0, columnspan=count, sticky="w", padx=1, pady=17)

    @staticmethod
    def _device_choices(widget, choices, preferred=None):
        current = widget.get()
        names = list(choices) or [EMPTY_DEVICE]
        widget.configure(values=names, state="normal" if choices else "disabled")
        widget.set(current if current in choices else preferred or names[0])

    def refresh_devices(self, *, refresh=False):
        if self.busy or self.mode:
            return
        try:
            inputs, outputs = devices(refresh=refresh)
            microphone, output, system_output = default_devices(inputs, outputs)
        except Exception as error:
            self.message.configure(text=f"Не удалось прочитать аудиоустройства: {error}")
            return
        self.inputs = {item.label: item for item in inputs}
        self.outputs = {item.label: item for item in outputs}
        self.default_output = system_output
        self._device_choices(self.input_menu, self.inputs, microphone.label if microphone else None)
        self._device_choices(self.output_menu, self.outputs, output.label if output else None)
        self._sync_audio_controls()
        self.message.configure(text=(
            "Выберите голос и нажмите «Включить голос»."
            if self.inputs and self.outputs else "Подключите микрофон и устройство для воспроизведения."
        ))

    def _output_changed(self, _):
        self._sync_audio_controls()

    def _sync_audio_controls(self):
        output = self.outputs.get(self.output_menu.get())
        live = self.mode == "live"
        preview = self.mode == "preview"
        options = {
            key: item for key, item in self.outputs.items()
            if not item.virtual and (not live or item != output)
        }
        self.monitor_choices = options
        preferred = self.monitor_device.label if self.monitor_device else self.monitor_preference
        if preferred not in options:
            preferred = next((key for key, item in options.items() if item.headphones), None)
        if preferred not in options and self.default_output and self.default_output.label in options:
            preferred = self.default_output.label
        self._device_choices(self.monitor_menu, options)
        if preferred in options:
            self.monitor_menu.set(preferred)

        self.output_hint.configure(text=(
            "Виртуальный кабель" if output and output.virtual else
            "Голос будет слышен на этом устройстве" if output else ""
        ))
        listening = preview or self.monitor_device is not None
        self.monitor_menu.configure(state="normal" if options and not listening and not self.busy else "disabled")
        self.monitor_button.configure(
            text="Остановить прослушивание" if listening else "Прослушать себя",
            state="normal" if not self.busy and (listening or bool(options and self.inputs)) else "disabled",
        )
        self.run_button.configure(
            text="Выключить голос" if live else "Включить голос",
            state="normal" if not self.busy and self.inputs and self.outputs else "disabled",
        )
        self.state_label.configure(
            text="Подключение…" if self.busy else "Голос включён" if live else "Только наушники" if preview else "Голос выключен",
            text_color=BLACK if live or preview else MUTED,
        )
        can_select = not self.busy and not self.mode
        self.input_menu.configure(state="normal" if can_select and self.inputs else "disabled")
        self.output_menu.configure(state="normal" if can_select and self.outputs else "disabled")
        self.refresh_button.configure(state="normal" if can_select else "disabled")

    def _audio_job(self, action, complete):
        self.busy = True
        self._sync_audio_controls()
        self.audio_requests.put((action, complete))

    def _toggle_stream(self):
        if self.busy:
            return
        if self.mode == "live":
            self._stop_stream()
        else:
            self._start_flow("live")

    def _start_flow(self, mode, *, stable=False):
        microphone = self.inputs.get(self.input_menu.get())
        if mode == "preview":
            output = self.monitor_choices.get(self.monitor_menu.get())
        else:
            output = self.outputs.get(self.output_menu.get())
        if microphone is None or output is None:
            self.message.configure(text="Выберите микрофон и устройство вывода.")
            return
        monitor = self.engine.output if self.mode == "preview" and mode == "live" else self.monitor_device

        def start():
            try:
                self.engine.start(microphone, output, stable=stable)
            except Exception:
                self.engine.stop()
                raise
            if monitor and monitor != output:
                try:
                    self.engine.start_monitor(monitor)
                except (RuntimeError, ValueError) as error:
                    return None, str(error)
                return monitor, ""
            return None, ""

        def started(result, error):
            if error:
                self._stream_stopped(str(error))
                return
            self.monitor_device, warning = result
            self.mode = mode
            status = "Прослушивание только на выбранном устройстве." if mode == "preview" else f"Голос идёт на «{output.label}»."
            if stable:
                status = "Звук переведён в устойчивый режим."
            self.message.configure(text=warning or status)

        if stable:
            self.message.configure(text="Подстраиваем звук для стабильной работы…")
        self._audio_job(start, started)

    def _stop_stream(self, reason="Обработка голоса остановлена."):
        self._audio_job(self.engine.stop, lambda _, error: self._stream_stopped(str(error) if error else reason))

    def _stream_stopped(self, reason):
        self.mode = None
        self.monitor_device = None
        self.message.configure(text=reason)

    def _stop_monitor(self, reason):
        def stopped(_, error):
            self.monitor_device = None
            self.message.configure(text=str(error) if error else reason)

        self._audio_job(self.engine.stop_monitor, stopped)

    def _toggle_monitor(self):
        if self.busy:
            return
        if self.mode == "preview":
            self._stop_stream("Прослушивание остановлено.")
        elif self.monitor_device:
            self._stop_monitor("Прослушивание остановлено. Голос по-прежнему идёт на выбранный выход.")
        elif self.mode == "live":
            selected = self.monitor_choices.get(self.monitor_menu.get())
            if selected is None:
                return

            def started(_, error):
                if error:
                    self.message.configure(text=f"Не удалось включить прослушивание. {error}")
                else:
                    self.monitor_device = selected
                    self.message.configure(text=f"Прослушивание на «{selected.label}».")

            self._audio_job(lambda: self.engine.start_monitor(selected), started)
        else:
            self._start_flow("preview")

    def _poll_audio(self):
        if self.closing:
            if self.audio_thread.is_alive():
                self.after(42, self._poll_audio)
            else:
                self.destroy()
            return
        if self.busy:
            try:
                complete, result, error = self.audio_tasks.get_nowait()
            except Empty:
                pass
            else:
                self.busy = False
                if self.engine.preset != PRESETS[self.selected]:
                    self.engine.set_preset(PRESETS[self.selected])
                complete(result, error)
                self._sync_audio_controls()

        if self.mode and not self.busy:
            if not self.engine.running:
                self._stop_stream(self.engine.error or "Звуковое устройство отключилось. Включите голос повторно.")
            elif self.engine.needs_more_buffer:
                self._start_flow(self.mode, stable=True)
            elif self.monitor_device and not self.engine.monitoring:
                self._stop_monitor("Устройство прослушивания отключилось.")
        running = bool(self.mode) and not self.busy
        self.level = max(self.engine.level, .79 * self.level) if running else 0.
        db = 20 * log10(max(self.level, 1e-6))
        meter = round(db)
        if meter != self.last_meter:
            self.meter.set(max(0., min(1., (meter + 60.) / 60.)))
            self.meter_value.configure(text=f"{meter} дБ" if meter > -60 else "—")
            self.last_meter = meter
        audio_info = f"Задержка устройств  ·  {self.engine.latency_ms} мс" if running else ""
        if audio_info != self.last_audio_info:
            self.audio_info.configure(text=audio_info)
            self.last_audio_info = audio_info
        self.after(42, self._poll_audio)

    def close(self):
        if not self.closing:
            self.closing = True
            self.withdraw()
            self.audio_requests.put(None)


if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    SekretVoice().mainloop()
