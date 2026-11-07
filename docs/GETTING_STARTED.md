# Getting started

Everything runs from one Python 3.12 venv, `~/venvs/smolvla`, which `scripts/setup_env.sh` builds: torch, LeRobot 0.6.1, and TensorRT 10.16 where the GPU supports it. Three ways in, cheapest first:

| You have | Run | You get |
|---|---|---|
| Any Linux machine, no GPU, no Hub access | the [offline tests](#offline-smoke-test-no-gpu-no-hub) | the split graphs and the C++ runtime checked against LeRobot on a tiny random model |
| An NVIDIA GPU | the [profile and TensorRT run](#on-a-gpu-machine-lab-server-colab-kaggle) | per-stage latency of every precision variant on the real checkpoint, with ONNX and TensorRT parity |
| A Jetson Xavier NX | the [Jetson guide](JETSON.md) | the measurements on that page: export on a PC, engines and the C++ bench on the board |

The split-graph export and the ONNX Runtime reference are in [How it works](HOW_IT_WORKS.md#split-graph-export), the C++ runtime's build in [C++ runtime](CPP_RUNTIME.md).

## On a GPU machine (lab server, Colab, Kaggle)

```bash
git clone https://github.com/jakhon37/smolvla-edge && cd smolvla-edge
bash scripts/setup_env.sh                 # Python 3.12 venv in ~/venvs/smolvla: torch cu128, LeRobot 0.6.1, TensorRT 10.16
bash scripts/run_p0.sh                    # env check, profile, ONNX export, TensorRT fp16 + fp32
```

Everything lands in `results/<gpu>_<utc>/`:

| File | Contents |
|---|---|
| `env_check.txt` | Machine, driver, bf16 support, headless MuJoCo rendering |
| `profile.json` | Full per-component and per-variant results |
| `breakdown.png`, `breakdown.svg`, `breakdown.md` | Chart plus the same numbers as a table |
| `onnx/export_report.json` | ONNX export and ONNX Runtime parity |
| `onnx/trt_report_*.json` | TensorRT build, parity and latency vs PyTorch |

On a shared server:
- Pick an idle GPU first, e.g. `CUDA_VISIBLE_DEVICES=1 bash scripts/run_p0.sh`.
- Another job on the same GPU shows up in p99.
- `setup_env.sh` picks cu126 torch wheels for GPUs below compute capability 7.5 (P100, V100), because the cu128 wheels do not cover them. TensorRT 10 targets Turing and newer, so on such GPUs only the PyTorch profile applies.

On Kaggle: turn on internet and a GPU, then run [`notebooks/kaggle_p0.ipynb`](../notebooks/kaggle_p0.ipynb). It puts the venv in `/tmp`, so it doesn't count against the 20 GB output limit.

## Individual steps

```bash
PY=~/venvs/smolvla/bin/python
$PY profile_smolvla.py --policy lerobot/smolvla_libero --iters 1000   # default: fp32,lerobot,lerobot_amp,bf16vlm,fp16vlm,fp16all,amp
$PY export_vision.py   --policy lerobot/smolvla_libero --out-dir onnx
$PY trt_vision.py      --onnx onnx/vision_fp16.onnx --torch-baseline      # strongly typed FP16 (TensorRT 10 and 11)
$PY trt_vision.py      --onnx onnx/vision_fp32.onnx --mode fp16-flag       # weakly typed FP16, TensorRT 10 only
$PY plot_results.py    results/*/profile.json --out results/compare        # several GPUs on one chart
```

## Offline smoke test (no GPU, no Hub)

```bash
bash tests/smoke_test.sh ~/venvs/smolvla/bin/python          # tiny random model, a few minutes
bash tests/smoke_test.sh ~/venvs/smolvla/bin/python --full   # SmolVLM2-500M layer sizes, about 5-6 GB RAM
~/venvs/smolvla/bin/python -m pytest tests                   # split-graph parity tests, tiny model, about 3 minutes
bash tests/cpp_parity.sh ~/venvs/smolvla/bin/python          # C++ runtime vs PyTorch, tiny model, a few minutes
```

GitHub Actions runs the first, third and fourth on a CPU runner for every push ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)).
