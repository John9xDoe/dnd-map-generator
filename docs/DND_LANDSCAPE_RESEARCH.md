# Исследование: ландшафт в D&D / OSR / hexcrawl и смежные темы

## Источники

| Источник | Тип | Ключевые идеи |
|----------|-----|----------------|
| Doran & Parberry, *Controlled Procedural Terrain Generation Using Software Agents* (IEEE TCIAIG 2010) | статья | Агенты coastline/smooth/beach/mountain/river; контроль числом агентов и lifetime |
| Hexmancer (Martin Ralya) | OSR tool | Таблицы соседства террейнов; реки/тропы continuity; features на byway/waterway |
| Ground Rules (PBE Games) | hex generator | Коды соседства A1–A7; roads/rivers соединять вручную; «реки текут вниз» |
| Hexer / Hexes on Automatic / Hexscribbler | продукты | Биомы (elev+moisture+temp); settlements scoring; smart rivers A*; POI + lore |
| PCG Book Ch.4–5,11–12 | учебник | Agents, L-systems, mixed-initiative, expressive range |

Локальные копии: `docs/dnd_refs/doran_parberry_terrain.pdf`, `docs/dnd_refs/hexmancer.pdf`.

## Что уже внедрено в код (этот проход)

1. Mountain + river **agents** (`agents.py`)
2. **Generate-and-test** + DnD fitness (passable / largest CC / camps) (`quality.py`)
3. **Expressive range** CLI/GUI (`expressive_range.py`, `--expressive-range`)
4. **L-system** деревья (`lsystem.py`)
5. Кисти как **hard constraints** SketchaWorld-style (`map_session.py`)

## Дополнительные идеи (ещё не сделаны)

### A. Hexcrawl / кампания
1. **Hex overlay** (6-mile hex): агрегировать DEM → terrain code в духе Hexmancer (Plains/Woods/Hills/…).
2. **Таблица соседства**: при генерации соседнего гекса bias по текущему террейну (continuity).
3. **POI scoring**: лагеря, руины, логово — denser у waterways/byways, реже в deep wilderness.
4. **Roads as A\*** между settlements с cost = slope + water + cliff.
5. **Travel time** слой: дни пути пешком между точками (GURPS/DFRPG style).
6. **Gazetteer JSON**: имя гекса, террейн, features, encounter table key.

### B. Климат / биомы
7. Поля **temperature + moisture** (orographic rain shadow за хребтами) → biome map (лес/болото/луг/скалы).
8. Разные L-system / плотность леса по биому.

### C. Игровая пригодность (DnD)
9. **Sight lines / choke points** метрики для боёвок.
10. **Encounter budget** по удалённости от цивилизации (Hexmancer region: wilderness/borderlands/settled).
11. Слой **underdark / verticality** (пещеры под обрывами) — как в Hexer Pro.

### D. UX мастера
12. Экспорт в **Foundry/Roll20** JSON или SVG с легендой.
13. «Сгенерировать регион 20×20 hex» + редактор constraints на гексах.
14. AI lore (опционально) поверх таблицы — как Hexscribbler, но не обязательно.

## Рекомендуемый порядок следующих итераций
1. Biome (temp/moisture) + hex aggregation  
2. Settlement/POI scoring + roads  
3. Export + gazetteer  
