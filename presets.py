"""Voice recipes. Pitch and formants are specified in semitones."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Preset:
    description: str
    pitch: float = 0.0
    formant: float = 0.0
    highpass: int = 0
    lowpass: int = 0
    drive: float = 0.0
    ring_hz: float = 0.0
    ring_depth: float = 0.0
    chorus: float = 0.0
    chorus_rate: float = 0.7
    echo: float = 0.0
    echo_ms: int = 180
    reverb: float = 0.0

    @property
    def effects(self) -> tuple[str, ...]:
        return tuple(
            name for enabled, name in (
                (self.pitch, "Тон"),
                (self.formant, "Форманты"),
                (self.ring_depth, "Модуляция"),
                (self.highpass or self.lowpass, "Фильтр"),
                (self.drive, "Перегруз"),
                (self.chorus, "Хорус"),
                (self.echo, "Эхо"),
                (self.reverb, "Реверб"),
            ) if enabled
        )


DEFAULT_PRESET = "Робот"

PRESETS: dict[str, Preset] = {
    "Робот": Preset("Металлическая вибрация", ring_hz=78, ring_depth=.83, highpass=140, lowpass=4800, drive=.26),
    "Демон": Preset("Низкий голос с мрачным хвостом", pitch=-9, formant=-5, drive=.33, reverb=.33),
    "Бурундук": Preset("Высокая мультяшная интонация", pitch=8, formant=5, highpass=180),
    "Анонимус": Preset("Приглушённый голос с зерном", pitch=-4, formant=-3, highpass=180, lowpass=3300, drive=.32),
    "Радио": Preset("Полоса старого радиоприёмника", highpass=340, lowpass=2850, drive=.45),
    "Эхо": Preset("Отчётливый повтор фраз", echo=.63, echo_ms=300),
    "Киборг": Preset("Тяжёлый голос с механическим оттенком", pitch=-3, ring_hz=125, ring_depth=.55, drive=.32, chorus=.18),
    "Глубокий бас": Preset("Плотный низ и мягкие верхи", pitch=-6, formant=-3, lowpass=4500, drive=.13),
    "Гелий": Preset("Лёгкий, очень высокий голос", pitch=11, formant=7, highpass=220),
    "Телефон": Preset("Узкая полоса телефонной трубки", highpass=450, lowpass=2450, drive=.18),
    "Рация": Preset("Резкая и плотная связь", highpass=610, lowpass=2700, drive=.66, ring_hz=46, ring_depth=.14),
    "Мегафон": Preset("Жёсткая середина громкоговорителя", formant=2, highpass=560, lowpass=3900, drive=.77, echo=.11, echo_ms=45),
    "Под водой": Preset("Глухой, покачивающийся тембр", pitch=-2, lowpass=800, chorus=.48, chorus_rate=.36),
    "Пещера": Preset("Дальние отражения в камне", formant=-2, lowpass=3500, echo=.45, echo_ms=320, reverb=.65),
    "Собор": Preset("Широкая реверберация и мягкое эхо", highpass=95, reverb=.76, echo=.23, echo_ms=155),
    "Ванная": Preset("Яркие близкие отражения", highpass=170, echo=.23, echo_ms=66, reverb=.42),
    "Тоннель": Preset("Тёмный голос с гулким отражением", formant=-3, lowpass=2200, reverb=.43, echo=.34, echo_ms=225),
    "Арена": Preset("Голос с трибун и дальний повтор", highpass=220, drive=.24, reverb=.48, echo=.36, echo_ms=245),
    "Призрак": Preset("Воздушное дрожание и длинный хвост", pitch=3, lowpass=5600, chorus=.5, chorus_rate=.32, reverb=.64),
    "Зомби": Preset("Низкое неровное рычание", pitch=-7, formant=-4, ring_hz=29, ring_depth=.39, drive=.32),
    "Гигант": Preset("Глубокий тембр с объёмным эхом", pitch=-10, formant=-7, lowpass=3900, echo=.18, echo_ms=190),
    "Гоблин": Preset("Высокий носовой тембр с хрипом", pitch=5, formant=6, highpass=380, drive=.45),
    "Эльф": Preset("Светлый звонкий голос с хорусом", pitch=4, formant=3, chorus=.26, chorus_rate=.56, reverb=.24),
    "Сказочник": Preset("Округлый мягкий низ", pitch=-2, formant=-2, highpass=80, lowpass=6350, reverb=.10),
    "Инопланетянин": Preset("Высокая пульсация и волнообразный тон", pitch=4, ring_hz=196, ring_depth=.48, chorus=.36, chorus_rate=1.65),
    "Андроид": Preset("Чёткий синтетический металлический тембр", highpass=250, ring_hz=155, ring_depth=.66, chorus=.13, formant=1),
    "Дроид": Preset("Тонкий электронный писк", pitch=7, highpass=490, lowpass=5300, ring_hz=260, ring_depth=.72),
    "Нейро": Preset("Холодный хор с лёгкой модуляцией", formant=2, ring_hz=52, ring_depth=.26, chorus=.42, chorus_rate=1.2),
    "Скафандр": Preset("Приглушённая связь из шлема", pitch=-1, highpass=280, lowpass=3100, drive=.28, echo=.17, echo_ms=95),
    "Космос": Preset("Плавный хорус и далёкий повтор", pitch=2, chorus=.46, chorus_rate=.27, reverb=.52, echo=.28, echo_ms=390),
    "Астрал": Preset("Высокий голос с мерцающими отражениями", pitch=6, formant=2, chorus=.45, reverb=.63, echo=.21, echo_ms=247),
    "Шёпот ветра": Preset("Мягкие верхи и медленная пульсация", highpass=680, lowpass=6600, ring_hz=19, ring_depth=.19, reverb=.43),
    "Хор": Preset("Несколько близких голосов в унисон", chorus=.72, chorus_rate=.57, reverb=.26),
    "Двойник": Preset("Близкая отложенная копия голоса", chorus=.29, chorus_rate=.39, echo=.36, echo_ms=55),
    "Тремоло": Preset("Ритмичная пульсация громкости", ring_hz=7, ring_depth=.38),
    "Вибрато": Preset("Быстрые колебания тембра", chorus=.68, chorus_rate=5.4, formant=1),
    "Стекло": Preset("Звонкая середина и лёгкий хвост", pitch=3, formant=5, ring_hz=315, ring_depth=.26, reverb=.23),
    "Фарфор": Preset("Тонкий высокий оттенок без перегруза", pitch=2, formant=6, highpass=280, lowpass=6700),
    "Бархат": Preset("Сглаженные верхи и тёплая середина", formant=-3, lowpass=5100, drive=.08),
    "Кассета": Preset("Мягкое насыщение и плавание звука", highpass=95, lowpass=4600, drive=.23, chorus=.25, chorus_rate=.43),
    "Граммофон": Preset("Тёплый узкий тембр старой записи", formant=2, highpass=410, lowpass=2200, drive=.5, chorus=.14, chorus_rate=.23),
    "Монохром": Preset("Сдержанная середина с мягким краем", formant=-1, highpass=260, lowpass=3400, drive=.10),
    "Перегруз": Preset("Плотное насыщение с жёсткой атакой", highpass=120, lowpass=6800, drive=.94),
    "Искры": Preset("Яркий хрип с частой модуляцией", formant=4, highpass=530, drive=.52, ring_hz=212, ring_depth=.38),
    "Лёд": Preset("Холодный высокий голос с отражением", pitch=5, formant=2, highpass=390, reverb=.41, echo=.18, echo_ms=114),
    "Пламя": Preset("Тёплый хрип с тёмным хвостом", pitch=-3, formant=-4, drive=.67, reverb=.24),
    "Гроза": Preset("Гулкий низ с быстрой пульсацией", pitch=-8, lowpass=2000, ring_hz=35, ring_depth=.34, echo=.43, echo_ms=340),
    "Тень": Preset("Приглушённое удвоение низкого голоса", pitch=-5, formant=-2, lowpass=3100, chorus=.38, chorus_rate=.34),
    "Зеркало": Preset("Яркое короткое отражение", formant=3, chorus=.17, echo=.49, echo_ms=96),
    "Неон": Preset("Светлый электронный хорус", pitch=3, formant=2, highpass=180, ring_hz=92, ring_depth=.22, chorus=.41, chorus_rate=1.8),
}
