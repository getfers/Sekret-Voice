"""The 50 voice recipes, grouped in the order shown in the app."""

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class Room:
    size: float
    decay: float
    early_ms: float
    damping_hz: int


@dataclass(frozen=True, slots=True)
class Preset:
    description: str
    category: str = ""
    pitch: float = 0.
    formant: float | None = None
    carrier: str = ""
    carrier_hz: float = 0.
    tune: float = 0.
    highpass: int = 0
    lowpass: int = 0
    eq: tuple[tuple[float, float, float], ...] = ()
    drive: float = 0.
    distortion: str = "tube"
    ring_hz: float = 0.
    ring_depth: float = 0.
    tremolo_hz: float = 0.
    tremolo_depth: float = 0.
    flutter_ms: float = 0.
    flutter_hz: float = 1.
    bits: int = 0
    crush_rate: int = 0
    chop_hz: float = 0.
    chorus: float = 0.
    chorus_rate: float = .7
    flange: float = 0.
    flange_rate: float = .6
    echo: float = 0.
    echo_ms: int = 180
    room: Room | None = None

    @property
    def effects(self) -> tuple[str, ...]:
        return tuple(name for enabled, name in (
            (self.pitch, "Тон"), (self.formant is not None, "Форманты"),
            (self.carrier, "Вокодер"), (self.tune, "Auto-Tune"),
            (self.highpass or self.lowpass or self.eq, "Фильтр"), (self.drive, "Перегруз"),
            (self.ring_depth, "Кольцевая модуляция"), (self.tremolo_depth, "Тремоло"),
            (self.flutter_ms, "Вибрато"), (self.bits, "Биткрашер"), (self.chop_hz, "Прерыватель"),
            (self.chorus, "Хорус"), (self.flange, "Фленджер"), (self.echo, "Эхо"),
            (self.room, "Реверберация"),
        ) if enabled)


DEFAULT_PRESET = "Киборг"

_RECIPES = {
    "Персонажи": {
        "Мужской бас": Preset("Грудной низ и тёплые гласные", pitch=-4, formant=-5.5,
                              highpass=55, eq=((150, 7, .8), (420, -4, 1.2)), drive=.16),
        "Женский голос": Preset("Светлый тембр с ясными согласными", pitch=3, formant=6,
                                highpass=120, eq=((3200, 4, .9),)),
        "Ребенок": Preset("Высокий тон и лёгкие короткие гласные", pitch=6.5, formant=9,
                          highpass=190, eq=((1700, 4, .9),), flutter_ms=.23, flutter_hz=4.8),
        "Старик": Preset("Дрожащая, истончённая интонация", pitch=-1.5, formant=-2.5,
                         highpass=235, lowpass=4300, flutter_ms=1.7, flutter_hz=5.7,
                         tremolo_hz=7.1, tremolo_depth=.38),
        "Великан": Preset("Тяжёлый голос с близким двойным ударом", pitch=-10, formant=-9,
                          eq=((205, 8, .8),), drive=.32, echo=.29, echo_ms=29),
        "Гном": Preset("Короткий, гнусавый и упругий тембр", pitch=2.2, formant=5.5,
                       eq=((900, 11, 2.6), (2500, -7, 1.4)), drive=.43, flutter_ms=.65, flutter_hz=3.4),
        "Бурундук": Preset("Мультяшный голос на октаву выше", pitch=12,
                           highpass=245, eq=((3300, 5, 1.1),)),
        "Анонимус": Preset("Замаскированный низ в узкой цифровой полосе", pitch=-5.5, formant=-1.5,
                           highpass=280, lowpass=2380, bits=7, crush_rate=8500),
        "Диктор ТВ": Preset("Ровный, плотный голос из студии", pitch=-.8, formant=-1,
                            highpass=65, lowpass=10800, eq=((125, 4.5, .8), (2900, 5, 1.1)), drive=.25),
        "Шепот маньяка": Preset("Дыхание и шорох согласных", formant=-3.5, carrier="noise",
                               highpass=680, lowpass=7500, tremolo_hz=2.1, tremolo_depth=.37),
    },
    "Монстры": {
        "Демон из бездны": Preset("Октава вниз, вибрация и тёмный гул", pitch=-12, formant=-10,
                                  highpass=68, lowpass=2650, ring_hz=43, ring_depth=.86,
                                  drive=.7, distortion="fold", room=Room(1.2, 1.65, 12, 1750)),
        "Зомби": Preset("Прерывистый хрип с неровной высотой", pitch=-3, lowpass=3100,
                        flutter_ms=3.7, flutter_hz=3.3, tremolo_hz=2.7, tremolo_depth=.8,
                        drive=.52, distortion="clip", bits=7, crush_rate=9900),
        "Вампир": Preset("Холодный бархат с коротким шлейфом", pitch=1.5, formant=-1.5,
                         highpass=90, eq=((540, -7, 1.2), (1750, 5, 1.7)),
                         room=Room(.58, 1.05, 7, 5500)),
        "Орк": Preset("Низкий рык с грубой атакой", pitch=-7, eq=((720, 8, 1.2),),
                      drive=.95, distortion="clip", highpass=145, lowpass=4100, echo=.22, echo_ms=51),
        "Призрак": Preset("Парящие голоса с длинным светлым хвостом", pitch=7, formant=4,
                          highpass=350, chorus=.9, chorus_rate=.31, room=Room(1.6, 3.8, 10, 6400)),
        "Дракон": Preset("Гулкое рычание с шипящими гармониками", pitch=-11,
                          eq=((270, 9, .9), (3000, 8, 1.7)), ring_hz=27, ring_depth=.66,
                          drive=.84, distortion="fold", flange=.53, flange_rate=2.2),
        "Голлум": Preset("Носовой шёпот с нервной дрожью", pitch=1, formant=8.5,
                         eq=((1180, 9, 2.8), (2900, -6, 1.8)), highpass=360,
                         flutter_ms=1.4, flutter_hz=9, tremolo_hz=6.3, tremolo_depth=.35, drive=.29),
        "Ведьма": Preset("Резкий высокий голос со скрипом", pitch=9.5, formant=2.5,
                         highpass=410, lowpass=5650, flutter_ms=.8, flutter_hz=8.5,
                         drive=.61, distortion="clip"),
        "Оборотень": Preset("Хриплый вой с пульсирующим тембром", pitch=-6, formant=-4,
                            flutter_ms=4.2, flutter_hz=4.4, flange=-.74, flange_rate=1.7,
                            drive=.6, distortion="clip"),
        "Темный лорд": Preset("Низкий синтетический аккорд в каменном зале", formant=-5,
                              carrier="chord", carrier_hz=49, highpass=105, lowpass=3550,
                              room=Room(1.35, 2.8, 17, 2940)),
    },
    "Роботы": {
        "Киборг": Preset("Металлические боковые тона и плотный перегруз", pitch=-3,
                         highpass=185, lowpass=4650, ring_hz=118, ring_depth=.9, drive=.38),
        "Андроид": Preset("Ровная речь на квадратном носителе", carrier="pulse", carrier_hz=137,
                          highpass=235, lowpass=6400, eq=((2200, 4, 1.2),)),
        "Меха-титан": Preset("Три низких двигателя и короткий гидравлический отзвук", carrier="chord",
                             carrier_hz=62, formant=-3, highpass=95, lowpass=6230, drive=.49,
                             distortion="clip", echo=.29, echo_ms=78),
        "Пришелец": Preset("Высокие металлические переливы", pitch=5, ring_hz=317, ring_depth=.79,
                           chorus=.78, chorus_rate=1.45, lowpass=5750),
        "ИИ-ассистент": Preset("Чистый синтезаторный голос на фиксированной ноте", carrier="saw",
                               carrier_hz=211, formant=2, highpass=130, lowpass=8200),
        "Сломанный дроид": Preset("Короткие обрывы плохо работающего синтезатора", carrier="pulse",
                                 carrier_hz=103, lowpass=3180, bits=5, crush_rate=3880, chop_hz=7.3),
        "Инопланетный жук": Preset("Рой высоких вибраций и щелчков", ring_hz=895, ring_depth=1,
                                   tremolo_hz=28, tremolo_depth=.84, highpass=640,
                                   eq=((4200, 7, 2.3),), drive=.68, distortion="fold"),
        "Глич-вирус": Preset("Жёсткие цифровые обрывы и ломаный фленджер", bits=3, crush_rate=2600,
                             chop_hz=13.5, flange=-.64, flange_rate=3.6, drive=.57, distortion="fold"),
        "Космонавт": Preset("Приглушённая связь из шлема с коротким отскоком", pitch=-1.8,
                            highpass=820, lowpass=2610, eq=((1520, -7, 3.3),),
                            drive=.29, bits=10, crush_rate=10200, echo=.17, echo_ms=53),
        "Трансформер": Preset("Низкий голос с медленным движением металлических резонансов", pitch=-8,
                              ring_hz=71, ring_depth=.72, flange=1., flange_rate=.47,
                              drive=.64, eq=((190, 5, .9), (2850, 5, 1.3))),
    },
    "Связь": {
        "Полицейская рация": Preset("Резкая импульсная речь с цифровой зернистостью", pitch=-1.8,
                                   highpass=760, lowpass=3360, eq=((2170, 8, 1.7),),
                                   drive=.81, distortion="clip", bits=5, crush_rate=6250,
                                   tremolo_hz=18, tremolo_depth=.32),
        "Старый телефон": Preset("Узкий и чистый звук телефонной трубки", highpass=420,
                                 lowpass=1820, eq=((920, 9, 1.9),), drive=.13),
        "Мегафон на митинге": Preset("Громкая середина и уличный отскок", highpass=240, lowpass=4580,
                                      eq=((1390, 11, 1.6),), drive=.98, distortion="clip",
                                      room=Room(.95, .45, 21, 3410)),
        "Граммофон 1920-х": Preset("Узкий механический рупор и неровное вращение", highpass=305,
                                   lowpass=2210, eq=((560, 10, 1.8),), drive=.48,
                                   distortion="fold", flutter_ms=2.4, flutter_hz=1.4),
        "Кассетный магнитофон": Preset("Тёплая лента с плавным покачиванием высоты", highpass=90,
                                       lowpass=5480, eq=((950, -4, .9),), drive=.51, flutter_ms=4.8,
                                       flutter_hz=.67, tremolo_hz=3.2, tremolo_depth=.21),
        "Радиоведущий": Preset("Близкая, яркая речь с плотной компрессией тембра", highpass=160,
                               lowpass=9500, eq=((2800, 8, 1.2), (5100, 4, .8)), drive=.69),
        "Диспетчер аэропорта": Preset("Высокая полоса системы оповещения с длинным отскоком", pitch=-.5,
                                      highpass=980, lowpass=3620, eq=((2070, 9, 2.2),),
                                      drive=.39, echo=.24, echo_ms=137),
        "8-битная игра": Preset("Высокий голос с грубыми цифровыми ступенями", pitch=7.5,
                                highpass=245, lowpass=4000, bits=4, crush_rate=5520),
        "Голос из телевизора": Preset("Корпус старого телевизора с полой серединой", highpass=245,
                                      lowpass=5400, eq=((470, 6, 2.), (1250, -15, 2.6), (2850, 7, 1.5)),
                                      flange=.22, flange_rate=.23, drive=.4),
        "Подводная лодка": Preset("Глухой интерком с мерным повтором", pitch=-2.6,
                                  highpass=170, lowpass=925, eq=((430, 9, 2.3),),
                                  flutter_ms=2.1, flutter_hz=.8, echo=.36, echo_ms=172),
    },
    "Пространство": {
        "Эхо в пещере": Preset("Длинные отдельные повторы в тёмной пещере", highpass=110, lowpass=3750,
                              echo=.64, echo_ms=287, room=Room(1.38, 2.6, 13, 2500)),
        "Огромный собор": Preset("Светлый реверберационный хвост до пяти секунд", highpass=105,
                                 room=Room(1.85, 5.4, 9, 6100)),
        "Стадион": Preset("Голос с трибун и разнесённые громкие повторы", highpass=235, lowpass=5700,
                          echo=.44, echo_ms=121, room=Room(2.35, 3.0, 27, 5100)),
        "Под водой": Preset("Плывущий бас с булькающими резонансами", lowpass=540,
                           eq=((205, 9, 1.6),), flutter_ms=6.8, flutter_hz=1.17,
                           flange=-.44, flange_rate=.38),
        "В железной бочке": Preset("Гулкая тесная полость с металлическими пиками", highpass=150, lowpass=5100,
                                   eq=((385, 11, 4.), (960, 9, 5.), (2290, 7, 4.)),
                                   room=Room(.28, .67, 2.6, 5200)),
        "Космический вакуум": Preset("Тонкий синтетический шлейф с провалами громкости",
                                     carrier="chord", carrier_hz=307, highpass=670, lowpass=10000,
                                     tremolo_hz=1.3, tremolo_depth=.94, room=Room(1.1, 1.4, 6, 7800)),
        "Хор двойников": Preset("Три плавающих голоса в унисон", chorus=1., chorus_rate=.48,
                               highpass=110, eq=((3300, 3, .8),)),
        "Психоделический фленджер": Preset("Глубокий свип с движущимися провалами", flange=-1.,
                                           flange_rate=.24, highpass=110, lowpass=8700, drive=.22),
        "Трэп-автотюн": Preset("Резкие ступени мелодической коррекции", tune=1., highpass=130,
                               lowpass=11200, eq=((3700, 5, 1.4),), chorus=.21, chorus_rate=2.3),
        "Рок-мегафон": Preset("Широкий жёсткий перегруз с пульсом и комнатным хвостом", highpass=115,
                              lowpass=7770, eq=((240, 7, 1.1), (3410, 8, 1.25)),
                              drive=1., distortion="fold", tremolo_hz=6.2, tremolo_depth=.43,
                              room=Room(.4, .53, 4, 6100)),
    },
}

CATEGORIES = tuple(_RECIPES)
PRESETS: dict[str, Preset] = {
    name: replace(preset, category=category)
    for category, recipes in _RECIPES.items()
    for name, preset in recipes.items()
}
