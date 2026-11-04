# Изучение PCG Book (Shaker, Togelius, Nelson, 2016)

Источник: официальные авторские PDF с https://www.pcgbook.com/  
Локально: `/workspace/PCG_books/chapter01.pdf` … `chapter12.pdf`  
(Папки `PCG_books` изначально в репозитории не было — книги скачаны с официального сайта.)

## Что уже знали / делали в проекте
- Constructive пайплайн: fBm/ridges → сглаживание → thermal + stream-power → Priority-Flood/D8 → озёра/реки → обрывы/лес.
- Mixed-initiative зачаток: GUI с параметрами, слоями и кистями.
- Детерминизм по seed.

## Что нового по главам (релевантное нам)

### Ch.1 — таксономия
- Constructive vs **generate-and-test**; stochastic vs deterministic; online vs offline; **mixed authorship**.
- Требование *playability* контента (у нас пока только «геоморфологическая правдоподобность»).

### Ch.2 — search-based
- Представление + **fitness**; multiobjective (NSGA-II).
- Идея: оптимизировать не «красивый шум», а измеримые свойства карты.

### Ch.4 — ландшафты (ключевая)
- Diamond-square / midpoint, Perlin/gradient noise.
- **Doran & Parberry multi-agent**: coastline → smooth → beach → mountain/hill → **river agents** (путь от берега к хребту, копание клина вниз с расширением).
- Контролируемость числом агентов и lifetime/tokens.
- **Genetic Terrain Programming**: GP-дерево → heightmap; fitness вроде accessibility.
- Musgrave et al.: eroded fractal terrains.

### Ch.5 — растительность
- **L-systems / bracketed L-systems** для формы деревьев/кустов, не только точек на карте.
- Эволюция правил L-system.

### Ch.11 — mixed-initiative
- CAD-помощник vs interactive evolution.
- Примеры: Tanagra, **SketchaWorld**, Sentient Sketchbook (sketch → автодостройка + оценка).

### Ch.12 — оценка генераторов
- **Expressive range**: метрики, далёкие от входных параметров; визуализация пространства выходов.
- Сравнение диапазона при разных конфигах; опросы игроков.

## Идеи улучшения проекта (приоритет)

| # | Идея | Откуда | Зачем |
|---|------|--------|-------|
| 1 | River/coastline/mountain **agents** (Doran) поверх или вместо чистого SPE | Ch.4 | Контроль числа рек, устья, ширины; меньше «случайных» русел |
| 2 | **Generate-and-test**: браковать карты без связного стока / с озёрами без выхода / с деревьями на воде | Ch.1–2,12 | Гарантии качества |
| 3 | Fitness: доля проходимого леса, длина главной реки, число замкнутых котловин, связность суши | Ch.2,4 | Search-based / отбор кандидатов |
| 4 | **Expressive range** dashboard (уклон vs доля воды, число озёр vs длина рек) по сидам | Ch.12 | Отладка и научная оценка |
| 5 | L-system спрайты деревьев по биому/влажности | Ch.5 | Визуал вместо условных знаков |
| 6 | Усилить GUI как SketchaWorld: кисть задаёт constraint, генератор достраивает | Ch.11 | Настоящий mixed-initiative |
| 7 | Diamond-square / simplex как альтернативный бэкенд DEM | Ch.4 | Сравнение и разнообразие |
| 8 | Multiobjective: realism hydro vs gameplay (проходимость для DnD) | Ch.2 | DnD-карты ≠ только геология |
