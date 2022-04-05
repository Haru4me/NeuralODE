# Имплементация NeuralODE для задачи прогнозирования ЗПВ верхних слоев почвы

Проект представляет из себя исправление `torchdiffreq` для задачи вида
```text
dy/dt = f(t, y, v)    y(t_0, v_0) = y_0
```
## Тест
Пока можно только проверить работу самой библиотеки на написанном скрипте. Для этого
1. Выполнить установку __poetry__ через
```bash
pip install poetry
```
2. В корневой папке проекта
```bash
poetry install
```
3. Запустить скрипт
```bash
poetry run python src/testODE.py -n test
```

Если хочется настроить другие параметры, выполнить ```poetry run python src/testODE.py -h```

Пример работы скрипта:
<p align="center">
<img align="middle" src="./assets/dopri5/dopri5.gif" alt="ODE Demo" width="500" height="250" />
</p>

## Запуск с GUI
Возможен только запуск через __poetry__. Для этого (если не запускали тест)
1. Выполнить установку __poetry__ через
```bash
pip install poetry
```
2. В корневой папке проекта
```bash
poetry install
```
И далее
3. Запустить скрипт
```bash
poetry run streamlit run src/gui_experiment.py
```

Далее в браузере откроется следующего вида окно
<p align="center">
<img align="middle" src="./assets/GUI.gif" alt="GUI Demo" width="1000" height="400" />
</p>

Далее необходимо выставить настройки и нажать __Start running__.

## Стандартный запуск из терминала
Возможен только запуск через __poetry__. Для этого (если не запускали тест)
1. Выполнить установку __poetry__ через
```bash
pip install poetry
```
2. В корневой папке проекта
```bash
poetry install
```
3. Запустить скрипт
```bash
poetry run python src/experiment.py -n exp_name
```

Чтобы узнать о параметрах скрипта, выполнить
```bash
poetry run python src/experiment.py --help
```


Примечания:
- Все побочные файлы (графики, таблицы с метриками, модель) сохраняются в папку `assets/exp_name` (именную папку эксперимента)
- Все логи пишутся в `logs/exp_name.log` файл


## Запуск прописанных экспериментов

В файлк ```exp_config.json``` лежат настройки к параментрам экперимента вида как в примере
```json
"exp": {
        "lr": 0.01,
        "batch_size": 500,
        "e": 250,
        "m": "euler",
        "lf": "MAE",
        "l": "16 32 16",
        "emb": "16 3",
        "af": "Tanh"
    }
```
Для последовательного запуска всех экспериментов следует запустить скрипт ```runer.py```

```bash
poetry run python runer.py
```
