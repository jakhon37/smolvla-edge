# C++ runtime

[`cpp/`](../cpp) runs the split graphs from C++17, with no Python at inference time. Per action chunk it runs `vision` for each camera, `prefix` once, then the 10-step Euler loop over `expert_step`, accumulating in float32 with LeRobot's update rule. Every buffer is allocated and bound when the runtime is built, and `expert_step` reads the prefix K/V in place.

| Part | What it is |
|---|---|
| `smolvla_rt` library | `Graph`, an interface over one graph with named, statically shaped inputs and outputs. `OrtGraph` implements it with the ONNX Runtime C++ API, on the CPU or the CUDA execution provider. `TrtGraph` implements it with the TensorRT runtime API of versions 8.5 and 10 (`deserializeCudaEngine`, `setTensorAddress`, `enqueueV3`) and is built only when CMake finds TensorRT. `SmolVLARuntime` checks the three graphs against the contract and runs the chunk. Also readers for `split_manifest.json` and for bundles |
| `smolvla_rt_bench` | Checks the chunk against a bundle's PyTorch actions (max abs, cosine) and, stage by stage, against LeRobot's intermediate tensors. Times every component (p50/p90/p99 after warm-up). Optionally runs the control-loop harness. Writes a JSON report |
| Control-loop harness | A 50 Hz control thread takes one action per tick from a chunk buffer, which an inference thread refills when half a chunk is left. It reports wake-up latency, period error, missed deadlines and starved ticks per 10k ticks |
| `smolvla_rt` Python module | Optional pybind11 build with `ort_flow.SplitRuntime`'s `predict_action_chunk(inputs, noise)` |

**Build**

```bash
bash cpp/scripts/fetch_onnxruntime.sh                 # ONNX Runtime 1.31.0 C/C++ SDK into cpp/third_party/onnxruntime
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release  # fetches GoogleTest and nlohmann/json if not installed
cmake --build cpp/build -j
```

- `fetch_onnxruntime.sh` downloads the official release archive from GitHub. If GitHub release downloads are blocked, it falls back to the NuGet package, which has the same CPU library. `--flavor cuda12` or `--flavor cuda13` fetches a GPU build.
- `-DONNXRUNTIME_ROOT=<dir>` uses another SDK. `-DTENSORRT_ROOT=<dir>` adds `TrtGraph` (TensorRT 10 and the CUDA toolkit).
- `-DSMOLVLA_RT_BUILD_PYTHON=ON -DPython_EXECUTABLE=$PY` builds the Python module for the venv's Python, with the pybind11 that `setup_env.sh` installs there. Without `Python_EXECUTABLE`, CMake builds for the first Python it finds. The module lands in `cpp/build/python`: `PYTHONPATH=cpp/build/python $PY -c "import smolvla_rt; rt = smolvla_rt.Runtime('onnx_split')"`.

**Run**

```bash
B=cpp/build/smolvla_rt_bench
$B --onnx-dir onnx_split --bundle onnx_split/bundle.bin --iters 100 --out onnx_split/bench.json   # parity + latency, fp32, CPU
$B --onnx-dir onnx_split --bundle onnx_split/bundle.bin --precision fp16 --provider cuda
$B --onnx-dir onnx_split --bundle onnx_split/bundle.bin --precision fp16 --backend trt  # <graph>_fp16.engine files
$B --onnx-dir onnx_split --bundle onnx_split/bundle.bin --threads 1 --control-loop-ticks 10000 --control-cpu 2 --inference-cpu 3
bash tests/cpp_parity.sh ~/venvs/smolvla/bin/python   # build, export the tiny test policy, check C++ against PyTorch
```

- The bench exits with status 1 if the fp32 chunk differs from the bundle's PyTorch actions by more than `--max-abs`, 1e-4 by default. The tests hold the CPU build to 1e-5; the default leaves room for GPU kernels, which add up in a different order than PyTorch on the CPU. fp16 has no default gate; `--min-cosine` sets one.
- TensorRT engines come from the ONNX graphs, for example `trtexec --onnx=onnx_split/prefix_fp16.onnx --stronglyTyped --saveEngine=onnx_split/prefix_fp16.engine`, once per graph. Build fp32 engines with `--noTF32` as well: TensorRT computes fp32 layers in TF32 unless told not to. TensorRT 8.5 has no `--stronglyTyped`: see [Jetson Xavier NX](JETSON.md).
- `--control-cpu` and `--inference-cpu` pin the control thread and the thread that calls `predict_action_chunk`, not ONNX Runtime's worker threads. With the default `--threads 0`, ONNX Runtime starts its own worker threads and pins them to cores itself, ignoring `taskset` and these options, so a worker can share the control thread's CPU; the bench warns about it. With `--threads 1`, as in the measurements below, ONNX Runtime has no workers and runs the graphs on the inference thread.

In C++:

```cpp
#include "smolvla_rt/loader.h"

auto rt = smolvla_rt::load_runtime(smolvla_rt::Manifest::read("onnx_split"), {});  // ONNX Runtime, fp32, CPU
smolvla_rt::Observation obs{{pixels_cam0, pixels_cam1}, lang_tokens, lang_mask, state};  // preprocessed, as in docs/SPLIT_GRAPH.md
rt->predict_action_chunk(obs, noise, actions);  // noise [1, 50, 32] -> actions [1, 50, 7], normalised
```

**Measured** on a 4-vCPU Xeon cloud VM (CPU only), ONNX Runtime 1.31.0, with the tiny random test policy that `tests/cpp_parity.sh` exports: `make_fake_smolvlm.py --tiny --kv-heads 2`, so 4 VLM layers with 2 key/value heads. The parity numbers were taken with the bundle's PyTorch reference computed on 1 and on 4 threads, and ONNX Runtime on 1 and on 4 threads. A range means the PyTorch thread count moved the last digits; the ONNX Runtime thread count changed nothing.

| Check | Result |
|---|---|
| fp32 action chunk vs PyTorch `predict_action_chunk` | max abs 4.8e-7; 9.5e-7 on the GitHub Actions ubuntu-24.04 runner (gate in the tests: 1e-5) |
| fp32 and fp16 chunks vs `ort_flow.py`, same inputs | bit-identical |
| Pipeline stages vs LeRobot: image tokens, prefix K/V, `v_t` over 10 steps | max abs 3.5e-6, 1.1e-6 to 1.2e-6, 1.2e-6 to 1.4e-6; pad mask equal |
| LeRobot's own `x_t` trajectory through the C++ Euler update | bit-identical at all 10 steps, the last one giving the action chunk |
| fp16 action chunk vs PyTorch fp32 | max abs 1.9e-4 to 2.0e-4, cosine 0.9999999974 to 0.9999999978 |

With one key/value head, K/V read in the wrong memory layout would be the same numbers. With two, a C++ runtime that read the prefix K/V as `[1, P, Hkv, Dh]` instead of `[1, Hkv, P, Dh]` (injected on purpose) moved the fp32 chunk by 0.24 and failed the tests and the bench. On a one-head model the same mistake passes every numeric check, so the parity tests also require more than one head.

Latency on one core (`--threads 1`, pinned to CPU 3, five alternating rounds of 30 chunks against `ort_flow.py`; median of the rounds' p50s, and their range):

| | C++ | `ort_flow.py` |
|---|---|---|
| Action chunk | 288 ms (274-302) | 295 ms (271-305) |
| `vision`, per camera | 94.0 ms (90.5-101.6) | 96.8 ms (90.1-99.6) |
| `prefix` | 14.8 ms (14.3-16.0) | 15.5 ms (15.1-16.3) |
| `expert_step`, per step | 7.32 ms (6.89-7.81) | 7.44 ms (7.11-7.77) |

- ONNX Runtime's graph execution dominates, so on this model C++ and Python are within noise of each other: the rounds spread more than they differ.
- The C++ runtime's own work per chunk is copying the inputs into its buffers, 2.4 ms for two 512x512 cameras, plus the Euler updates, 0.03 ms. Removing the camera-pixel copy, for example by letting the caller write into the runtime's input buffers, is a known follow-up.
- The tiny model only shows runtime overhead. For the full-size graphs with real weights, see [Real weights (CPU)](EXPERIMENTS.md#real-weights-on-the-cpu): there C++ and Python took about the same time, roughly 10 s per chunk on one core.

Control-loop harness: 10k ticks at 50 Hz on the same model (`--threads 1`), with the next chunk requested when 25 actions are left:

| | Control thread on CPU 2, inference on CPU 3 | Both on CPU 3 |
|---|---|---|
| Period error p50 / p99 / max | 0.04 / 3.4 / 78 ms | 0.01 / 2.3 / 34 ms |
| Wake-up latency p50 / p99 / max | 0.20 / 1.9 / 78 ms | 0.14 / 1.3 / 34 ms |
| Missed deadlines (woke a full period late), per 10k ticks | 9 | 4 |
| Ticks without an action, per 10k | 0 | 0 |
| Inference per chunk, p50 | 287 ms | 291 ms |

- This is soft real-time: a stock kernel (Linux 6.18, `PREEMPT_DYNAMIC`, not `PREEMPT_RT`) in a 4-vCPU VM shared with other jobs, the process at nice 15, no `SCHED_FIFO`.
- One run per layout, back to back, on a VM where CPU 2 also served other jobs. Here, and in an earlier pair of runs on the one-key/value-head test model, the split layout had the larger tails; these runs cannot say whether that comes from the layout or from the load. A dedicated machine with `--rt-priority` and isolated cores would do better, but nothing here bounds the worst case.
- Inference took about 290 ms per chunk, and each request left 500 ms of actions, so no tick went without one.

**Not tested**
- `TrtGraph` on TensorRT 10. It has only run on TensorRT 8.5.2, in [Jetson Xavier NX](JETSON.md). Against 10 it compiles, with `-Wall -Wextra -Wpedantic -Wshadow -Werror`, with the TensorRT 10.16 headers from NVIDIA's OSS repository, but was never linked to it or given an engine. This configuration repeats that check; it needs neither ONNX Runtime nor GoogleTest:
  `cmake -S cpp -B cpp/build-trt -DSMOLVLA_RT_TRT_COMPILE_CHECK=ON -DSMOLVLA_RT_WERROR=ON -DSMOLVLA_RT_WITH_ORT=OFF -DSMOLVLA_RT_BUILD_TESTS=OFF -DTENSORRT_ROOT=<TensorRT/include parent> -DSMOLVLA_RT_CUDA_INCLUDE_DIR=<dir with cuda_runtime_api.h>`, then `cmake --build cpp/build-trt`.
- Several GPUs. The current CUDA device belongs to each thread, and the runtime can be driven from threads other than the one that built it: the control loop's inference thread, or Python threads. `CudaDevice` therefore selects its device on the calling thread before every allocation, copy and free. `cpp/tests/test_cuda_device.cpp` checks that on the CPU against a stand-in for the CUDA runtime, one operation per new thread. `TrtGraph` selects its device in `run()` and in its destructor; that part is compile-checked only.
- The real CUDA calls behind `CudaDevice` (`cpp/src/cuda_api.cpp`) ran on the Jetson, on CUDA 11.4. They were linked against CUDA 12.9's `libcudart` too, but there only the "no CUDA driver" error path ran.
- ONNX Runtime's CUDA execution provider: the SDK used here is CPU-only. With it, `OrtGraph` keeps buffers in host memory, so ONNX Runtime copies the prefix K/V to the GPU on every expert step; `TrtGraph` keeps them on the device.
- The guard against a fused multiply-add in the Euler update (`Contract.EulerUpdateRoundsTheProductSeparately`, for the build's `-ffp-contract=off`) can only fail where the compiler may emit FMA instructions: aarch64, as on Jetson, or x86-64 built with `-march=native` on a CPU that has them. The default x86-64 target has no FMA, so on this VM and on the CI runner the test would pass without the flag too. The Jetson is aarch64, but the C++ unit tests were not built there.
- GPU latency that says anything about SmolVLA, other than on the [Xavier NX](JETSON.md). The numbers in this section come from random weights. Real weights in fp16 were probed tensor by tensor on the CPU only, in [Real weights (CPU)](EXPERIMENTS.md#real-weights-on-the-cpu). There the fp16 `vision` graph's attention scores stayed within ±48. The only overflow, inside the vision tower's GELU, did no harm on ONNX Runtime's CPU provider, nor with every node evaluated in IEEE fp16 (numpy). TensorRT's fused kernels on the Xavier NX cannot be probed that way; their fp16 outputs stayed finite.
- The aarch64 branch of `fetch_onnxruntime.sh`, and ONNX Runtime on aarch64. Everything on this page ran on x86-64 with GCC 13; the only aarch64 run is on the [Jetson](JETSON.md).
