"""CustomTkinter desktop interface for Sekret Voice."""

from dataclasses import replace
from math import log10
from pathlib import Path
from time import monotonic
import tkinter as tk

import customtkinter as ctk
from PIL import Image, ImageOps

from audio_engine import AudioEngine, default_devices, devices
from dsp import NOTES, SCALES
from presets import DRY, PRESETS


BASE = Path(__file__).resolve().parent
WHITE = "#FFFFFF"
BLACK = "#000000"
MUTED = "#515151"
LINE = "#E4E4E4"
SURFACE = "#F7F7F7"
HOVER = "#EDEDED"
FONT = "Segoe UI"
EMPTY_DEVICE = "Нет доступных устройств"
BLOCKS = {"128 сэмплов": 128, "256 сэмплов": 256, "512 сэмплов": 512, "1024 сэмпла": 1024}


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
        subtitle = label(self, description, 11, text_color=MUTED, wraplength=175, justify="left", width=1)
        subtitle.pack(fill="x", padx=13, pady=(0, 10))
        for part in (self, title, subtitle):
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
        self.minsize(880, 670)
        width = min(1190, max(880, self.winfo_screenwidth() - 110))
        height = min(800, max(670, self.winfo_screenheight() - 120))
        left = (self.winfo_screenwidth() - width) // 2
        top = max(0, (self.winfo_screenheight() - height) // 2 - 12)
        self.geometry(f"{width}x{height}+{left}+{top}")
        if self.tk.call("tk", "windowingsystem") == "win32":
            self.iconbitmap(str(BASE / "sekret.ico"))
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.engine = AudioEngine()
        self.block_size = 256
        self.inputs = {}
        self.outputs = {}
        self.default_output = None
        self.monitor_preference = None
        self.selected = None
        self.columns = 0
        self.layout_job = None
        self.level = 0.0
        self._logo = brand_image()

        self._build_header()
        self._build_body()
        self._build_footer()
        self.refresh_devices()
        self._filter_presets()
        self.bind("<Control-f>", self._focus_search)
        self.bind("<Escape>", self._clear_search)
        self.after(38, self._poll_audio)
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
        self.run_button = button(header, "Запустить", self._toggle_stream, primary=True, width=126, height=36)
        self.run_button.pack(side="right")
        self.state_label = label(header, "●  Остановлен", 12, text_color=MUTED)
        self.state_label.pack(side="right", padx=(0, 16))
        ctk.CTkFrame(self, fg_color=LINE, height=1, corner_radius=0).pack(fill="x")

    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color=WHITE, corner_radius=0)
        body.pack(fill="both", expand=True, padx=28, pady=(20, 0))
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)
        sidebar = ctk.CTkFrame(body, fg_color=WHITE, border_width=1, border_color=LINE, corner_radius=12)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 24))
        sidebar.grid_columnconfigure(0, minsize=266, weight=1)
        self._build_audio(sidebar)
        self._build_tune(sidebar)

        library = ctk.CTkFrame(body, fg_color=WHITE, corner_radius=0)
        library.grid(row=0, column=1, sticky="nsew")
        active = ctk.CTkFrame(library, fg_color=SURFACE, corner_radius=12)
        active.pack(fill="x", pady=(0, 21))
        reset = button(active, "Сбросить", lambda: self._select(None), width=89, height=30)
        reset.pack(side="right", anchor="n", padx=16, pady=14)
        active_text = ctk.CTkFrame(active, fg_color="transparent")
        active_text.pack(side="left", fill="x", expand=True, padx=16, pady=12)
        label(active_text, "Активный голос", 11, True, text_color=MUTED).pack(anchor="w", pady=(0, 2))
        self.active_name = label(active_text, "Оригинальный голос", 25, True)
        self.active_name.pack(anchor="w")
        self.active_detail = label(
            active_text, DRY.description, 12, text_color=MUTED, width=1, height=34,
            anchor="nw", wraplength=360, justify="left",
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
        panel.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 0))
        header = ctk.CTkFrame(panel, fg_color="transparent")
        header.pack(fill="x", pady=(0, 8))
        label(header, "Аудио", 19, True).pack(side="left")
        self.refresh_button = button(header, "Обновить", self.refresh_devices, width=78, height=26)
        self.refresh_button.pack(side="right")
        label(panel, "Input  /  Микрофон", 12, True).pack(anchor="w", pady=(0, 3))
        self.input_menu = menu(panel, [EMPTY_DEVICE], width=235, dynamic_resizing=False)
        self.input_menu.pack(fill="x", pady=(0, 8))
        label(panel, "Output  /  Устройство вывода", 12, True).pack(anchor="w", pady=(0, 3))
        self.output_menu = menu(panel, [EMPTY_DEVICE], self._output_changed, width=235, dynamic_resizing=False)
        self.output_menu.pack(fill="x", pady=(0, 8))
        monitor_line = ctk.CTkFrame(panel, fg_color="transparent")
        monitor_line.pack(fill="x", pady=(0, 3))
        label(monitor_line, "Наушники", 12, True).pack(side="left")
        self.monitor_button = button(monitor_line, "Monitor вкл.", self._toggle_monitor, width=96, height=24)
        self.monitor_button.pack(side="right")
        self.monitor_menu = menu(panel, [EMPTY_DEVICE], self._set_monitor, width=235, dynamic_resizing=False)
        self.monitor_menu.pack(fill="x", pady=(0, 8))
        buffer_line = ctk.CTkFrame(panel, fg_color="transparent")
        buffer_line.pack(fill="x", pady=(0, 7))
        label(buffer_line, "Аудиобуфер", 12, True).pack(side="left")
        self.buffer_menu = menu(buffer_line, BLOCKS, self._set_buffer, width=114)
        self.buffer_menu.set("256 сэмплов")
        self.buffer_menu.pack(side="right")

        ctk.CTkFrame(panel, fg_color=LINE, height=1, corner_radius=0).pack(fill="x", pady=(3, 9))
        gain_line = ctk.CTkFrame(panel, fg_color="transparent")
        gain_line.pack(fill="x")
        label(gain_line, "Gain  /  Усиление", 12, True).pack(side="left")
        self.gain_value = label(gain_line, "0 дБ", 12, text_color=MUTED)
        self.gain_value.pack(side="right")
        gain = self._slider(panel, -18, 24, 42, self._set_gain)
        gain.set(0)
        gain.pack(fill="x", pady=(5, 7))
        meter_line = ctk.CTkFrame(panel, fg_color="transparent")
        meter_line.pack(fill="x", pady=(0, 4))
        label(meter_line, "Уровень входа", 11, text_color=MUTED).pack(side="left")
        self.meter_value = label(meter_line, "−∞ dBFS", 11, text_color=MUTED)
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
    def _slider(parent, minimum, maximum, steps, command):
        return ctk.CTkSlider(
            parent, from_=minimum, to=maximum, number_of_steps=steps, command=command,
            fg_color="#DADADA", progress_color=BLACK, button_color=BLACK,
            button_hover_color="#2A2A2A", height=14,
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

    def _set_buffer(self, value):
        self.block_size = BLOCKS[value]

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
        preset = PRESETS[name] if name else DRY
        self.engine.set_preset(preset)
        if self.selected is not None:
            self.cards[self.selected].update_selection(False)
        self.selected = name
        if name is not None:
            self.cards[name].update_selection(True)
        self.active_name.configure(text=name or "Оригинальный голос")
        self.active_detail.configure(text=(
            preset.description + "  ·  " + " / ".join(preset.effects) if name else preset.description
        ))

    def _schedule_layout(self, event):
        target_columns = 3 if event.width >= 650 else 2
        if target_columns == self.columns:
            return
        self.columns = target_columns
        if self.layout_job:
            self.after_cancel(self.layout_job)
        self.layout_job = self.after_idle(self._filter_presets)

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
        self.layout_job = None
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

    def refresh_devices(self):
        try:
            available = devices()
            microphone, output = default_devices(available)
        except Exception as error:
            self.message.configure(text=f"Не удалось прочитать аудиоустройства: {error}")
            return
        self.inputs = {item.label: item for item in available if item.inputs}
        self.outputs = {item.label: item for item in available if item.outputs}
        self.default_output = output
        self._device_choices(self.input_menu, self.inputs, microphone.label if microphone else None)
        self._device_choices(self.output_menu, self.outputs, output.label if output else None)
        self._output_changed(self.output_menu.get())
        ready = bool(self.inputs and self.outputs)
        self.run_button.configure(state="normal" if ready else "disabled")
        self.message.configure(text=(
            "Выберите микрофон и выход. Для прослушивания используйте наушники."
            if ready else "Подключите микрофон и устройство вывода."
        ))

    def _output_changed(self, value):
        output = self.outputs.get(value)
        options = {
            key: item for key, item in self.outputs.items()
            if item != output and (
                output is None or item.name.casefold() != output.name.casefold()
            )
        }
        preferred = (
            self.default_output.label
            if self.default_output and self.default_output.label in options and not self.default_output.virtual
            else next((key for key, item in options.items() if not item.virtual and output and item.api == output.api), None)
        )
        if preferred is None:
            preferred = next((key for key, item in options.items() if not item.virtual), None)
        self._device_choices(self.monitor_menu, options, preferred)
        if self.monitor_preference in options:
            self.monitor_menu.set(self.monitor_preference)
        elif preferred:
            self.monitor_menu.set(preferred)
        self.monitor_button.configure(state="normal" if options and self.engine.running else "disabled")

    def _toggle_stream(self):
        if self.engine.running:
            self._stop_stream()
            return
        microphone = self.inputs.get(self.input_menu.get())
        output = self.outputs.get(self.output_menu.get())
        if microphone is None or output is None:
            self.message.configure(text="Выберите микрофон и устройство вывода.")
            return
        try:
            self.engine.start(microphone, output, self.block_size)
        except Exception as error:
            self.message.configure(text=str(error))
            return
        self.run_button.configure(text="Остановить")
        self.state_label.configure(text="●  В эфире", text_color=BLACK)
        for widget in (self.input_menu, self.output_menu, self.buffer_menu, self.refresh_button):
            widget.configure(state="disabled")
        self._output_changed(self.output_menu.get())
        self.message.configure(text=f"Выход  ·  {output.name}")

    def _stop_stream(self, reason="Поток остановлен."):
        self.engine.stop()
        self.run_button.configure(text="Запустить")
        self.state_label.configure(text="●  Остановлен", text_color=MUTED)
        for widget in (self.input_menu, self.output_menu, self.buffer_menu, self.refresh_button):
            widget.configure(state="normal")
        self.monitor_button.configure(text="Monitor вкл.", state="disabled")
        self._output_changed(self.output_menu.get())
        self.message.configure(text=reason)

    def _toggle_monitor(self):
        if self.engine.monitoring:
            self.engine.stop_monitor()
            self.monitor_button.configure(text="Monitor вкл.")
            self.monitor_menu.configure(state="normal")
            self.message.configure(text="Локальное прослушивание выключено.")
            return
        selected = self.outputs.get(self.monitor_menu.get())
        if selected is None:
            return
        try:
            self.engine.start_monitor(selected, self.block_size)
        except Exception as error:
            self.message.configure(text=f"Monitor: {error}")
            return
        self.monitor_button.configure(text="Monitor выкл.")
        self.monitor_menu.configure(state="disabled")
        self.message.configure(text=f"Прослушивание  ·  {selected.name}")

    def _poll_audio(self):
        if self.engine.rate and not self.engine.running:
            self._stop_stream(self.engine.error or "Звуковое устройство отключилось. Запустите поток повторно.")
        if self.engine.running and self.monitor_menu.cget("state") == "disabled" and not self.engine.monitoring:
            if self.monitor_button.cget("text") == "Monitor выкл.":
                self.engine.stop_monitor()
                self._output_changed(self.output_menu.get())
                self.monitor_button.configure(text="Monitor вкл.")
                self.message.configure(text="Устройство Monitor отключилось.")
        current = self.engine.level
        self.level = max(current, .79 * self.level) if self.engine.running else 0.
        db = 20 * log10(max(self.level, 1e-6))
        self.meter.set(max(0., min(1., (db + 60.) / 60.)))
        self.meter_value.configure(text=f"{db:.0f} dBFS" if db > -60 else "−∞ dBFS")
        if self.engine.running:
            audio_info = f"{self.engine.rate / 1000:g} кГц  ·  {self.engine.latency_ms} мс"
            if self.engine.xruns:
                audio_info += f"  ·  Сбои буфера: {self.engine.xruns}"
        else:
            audio_info = ""
        self.audio_info.configure(text=audio_info)
        self.after(38, self._poll_audio)

    def close(self):
        self.engine.stop()
        self.destroy()


if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    SekretVoice().mainloop()
