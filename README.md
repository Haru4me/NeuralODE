# Имплементация NeuralODE для задачи прогнозирования ЗПВ верхних слоев почвы

Проект представляет из себя исправление `torchdiffreq` для задачи вида
```
dy/dt = f(t, y, v)    y(t_0, v_0) = y_0
```
Пока можно только проверить работу самой библиотеки на написанном скрипте. Для этого
1. Выполнить установку __poetry__ через `pip install poetry`
2. В корневой папке проекта `poetry init`
3. `poetry run python src/testODE.py -n test`

Если хочется настроить другие параметры, выполнить `poetry run python src/testODE.py -h`

Пример работы скрипта:
<p align="center">
<img align="middle" src="./assets/dopri5/dopri5.gif" alt="ODE Demo" width="500" height="250" />
</p>
