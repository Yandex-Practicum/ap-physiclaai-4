# Practice 4 — Визуомоторный Behavior Cloning (стартовый набор)

Самодостаточный **стартовый** репозиторий четвёртой практики курса Physical AI.
Здесь рабочий визуомоторный BC-пайплайн: робот учится поднимать куб **по изображению
с камеры на гриппере**. По ходу уроков вы расширите observation space — добавите к
картинке **проприоцепцию** (положения и скорости суставов), проведёте контролируемое
сравнение с baseline и ablation-эксперимент.

> Практика **не требует прохождения Практики 3** — весь рабочий пайплайн
> (среда, опорная политика, обучение, оценка) включён в этот репозиторий.

> Это **baseline-состояние** проекта (вход модели — только изображение). Изменения
> в коде (`env.py`, `collect_data.py`, `model.py`, `train_bc.py`, `inference.py`)
> вы вносите сами, следуя урокам.

Робот — **Franka Emika Panda** ([MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
Apache 2.0). Задача — pick-and-place: поднять куб и доставить в целевую зону.
Наблюдение BC-модели: RGB-кадр **84×84** с камеры на гриппере.

## Требования

- Docker (Desktop на macOS/Windows или Engine на Linux).
- Для обучения BC нужна GPU: локальная NVIDIA или бесплатная в Google Colab.
- ~3 ГБ под Docker-образ + место под `dataset/` и `logs/`.
- Интернет при **первом** запуске обучения: CNN-энкодер (`resnet18`) подгружает
  предобученные веса через `timm`. Дальше веса кэшируются.

## Быстрый старт

```bash
git clone <URL этого репозитория>
cd ap-physiclaai-4
./scripts/run_container.sh          # соберёт образ и откроет shell в /workspace
```

Внутри контейнера: **noVNC** http://localhost:6080/vnc.html, **TensorBoard** http://localhost:6006.
Остановить: `./scripts/kill_container.sh`.

> **Windows.** Скрипты `*.sh` запускаются из **Git Bash** или **WSL2** (не из PowerShell/CMD).
> Запускайте `./scripts/run_container.sh` именно там. Репозиторий настроен (`.gitattributes`)
> так, чтобы скрипты сохраняли LF-переводы строк — иначе Docker выдаёт ошибку вида
> `bash\r: No such file or directory`. Если всё же столкнулись с ней — выполните
> `sed -i 's/\r$//' scripts/*.sh` в Git Bash/WSL2.

## Пайплайн и команды (baseline)

```
checkpoints/rl_expert.pt                  # опорная политика (privileged state)
        │
        ▼
collect_data.py  ──►  dataset/<name>/     # .npz: obs (T,84,84,3), actions (T,8), dones, success
        │
        ▼
train_bc.py      ──►  logs/<exp>/         # TensorBoard + checkpoints/{best,last}.pt
        │
        ▼
inference.py     ──►  Success Rate
```

```bash
# Сбор демонстраций опорной политикой
python3 collect_data.py --checkpoint checkpoints/rl_expert.pt --num_episodes 1000 \
    --save_dir dataset/train --only_success --seed 42
python3 collect_data.py --checkpoint checkpoints/rl_expert.pt --num_episodes 200 \
    --save_dir dataset/eval --only_success --seed 100

# Обучение baseline (вход — изображение)
python3 train_bc.py --train_dir dataset/train --eval_dir dataset/eval \
    --exp_name bc_baseline --epochs 100 --batch_size 64 --lr 1e-4
tensorboard --logdir logs/ --bind_all --port 6006

# Rollout-оценка
python3 inference.py --checkpoint logs/bc_baseline/checkpoints/best.pt --model bc --episodes 50 --seed 999
```

Без локальной GPU — обучение в Colab по [`train_bc.ipynb`](train_bc.ipynb) (GPU runtime).

## Структура проекта

| Путь | Назначение |
|---|---|
| `env.py` | Среда PandaPickCube. `_get_obs()` — кадр камеры; `get_privileged_state()` — для эксперта |
| `model.py` | `BCPolicy` — CNN-энкодер изображения (ResNet-18) → MLP-декодер; `RLPolicy` (эксперт) |
| `collect_data.py` | Сбор датасета опорной политикой: пишет `obs` и `actions` |
| `train_bc.py` | Обучение визуомоторной BC-модели, TensorBoard, чекпоинты |
| `inference.py` | Rollout-оценка (Success Rate) |
| `assets/` | MuJoCo-модель Panda + сцена (стол, куб, целевая зона) |
| `checkpoints/rl_expert.pt` | Опорная политика по privileged state (источник демонстраций) |
| `scripts/` | `run_container.sh`, `kill_container.sh`, дистилляция эксперта, артефакты |

## Опорная политика

`rl_expert.pt` работает по **privileged state** (точные координаты объектов) и служит
источником качественных демонстраций. Получена дистилляцией скриптового IK-контроллера
(`expert_scripted.py`) в `RLPolicy` (`scripts/distill_expert.py`); студентам этот шаг
проходить не нужно. Опорная политика не меняется по ходу практики.

## Лицензия моделей

Меши и kinematics Franka Panda — из MuJoCo Menagerie (Apache License 2.0).
