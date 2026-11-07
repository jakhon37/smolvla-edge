# Precision layout and method

Which dtype each part of SmolVLA runs in is decided by LeRobot's code, not by the checkpoint file alone, and it decides what "fp32", "bf16" and "fp16" mean in every result here. This page records that layout, the variants measured, and how timing and parity were taken.

## What the code says before any measurement (LeRobot 0.6.1, transformers 5.5)

These come from the source, not from measurement.
- `tests/check_lerobot_dtypes.py` confirms the dtype layout against LeRobot itself: same per-parameter dtypes, and bit-identical actions on a test model.
- The dtype rules depend on `load_vlm_weights`. LeRobot's LIBERO, Meta-World and RoboCasa training commands pass `true`, so fine-tuned checkpoints usually save it.

1. **With `load_vlm_weights=true`, the VLM runs in bf16.**
   - `SmolVLMWithExpertModel` first loads the base VLM with `torch_dtype="bfloat16"` ([smolvlm_with_expert.py#L90-L96](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/smolvla/smolvlm_with_expert.py#L90-L96)).
   - The checkpoint is then copied into those bf16 parameters.
2. **The action expert follows the VLM into bf16, with one exception.**
   - It is built from a copy of the VLM's text config. Under transformers 5, `from_config` uses that config's dtype, so the expert is bf16 too.
   - The exception: the cross-attention layers' `k_proj`/`v_proj` are replaced with plain `nn.Linear` after construction, so they stay fp32.
3. **The small projections are always fp32.** `state_proj`, `action_in_proj`, `action_out_proj` and the time MLP are fp32, and `denoise_step` casts to fp32 before `action_out_proj`.
   - A plain `policy.half()` therefore crashes.
   - The variants below cast only the VLM and the expert.
4. **`lerobot-eval` sets some global flags.** It always sets `cudnn.benchmark=True` and TF32 matmul. With `--policy.use_amp=true` it adds fp16 autocast, which on such checkpoints runs over the bf16 weights.
5. **On Turing (T4), bf16 has no hardware support.** On compute capability 7.5, `torch.cuda.is_bf16_supported(including_emulation=False)` is `False`. How much the bf16 VLM and expert cost there is the first thing this repo measures.

## Variants

| Name | VLM | Expert | Note |
|---|---|---|---|
| `fp32` | fp32 | fp32 | Reference for parity. Strict fp32: TF32 off |
| `lerobot` | bf16 | bf16, cross-attn k/v fp32 | What `lerobot-eval` runs for a `load_vlm_weights=true` checkpoint, with its backend flags |
| `lerobot_amp` | same | same | Plus fp16 autocast (`--policy.use_amp=true`). CUDA only: on CPU, autocast leaves the bf16 LayerNorm weights uncast |
| `bf16vlm` | bf16 | fp32 | Not a layout LeRobot builds |
| `fp16vlm` | fp16 | fp32 | |
| `fp16all` | fp16 | fp16 | |
| `bf16all` | bf16 | bf16 | Opt-in. Everything bf16, including cross-attn k/v |
| `amp` | fp32 + autocast fp16 | fp32 + autocast fp16 | `use_amp` on a `load_vlm_weights=false` checkpoint, without LeRobot's backend flags |

## Method notes

**Timing**
- Every region boundary calls `torch.cuda.synchronize()`, so GPU work is charged to the region that launched it.
- That costs a little, so each variant also runs uninstrumented. The headline end-to-end number comes from that run; the instrumentation overhead is reported.
- `trt_vision.py` times with CUDA events on one stream, inputs already on the GPU. Its optional PyTorch baseline uses the same method, so the two are comparable.

**Inputs**
- Synthetic, shaped like the checkpoint's input features: images in [0, 1], 256×256, two cameras for LIBERO.
- Image content does not change latency.
- Language token count does change it, so `--task` is a real LIBERO instruction.

**Parity**
- Each variant's raw action chunk (normalised space, before unnormalising) is compared with the fp32 chunk under identical noise: max-abs, mean-abs, relative L2, cosine.
- The fp32 reference holds the checkpoint's stored values exactly. The policy is built with `load_vlm_weights=false` and cast to fp32 *before* the weights are copied in, so a bf16 dtype in the base VLM's config cannot round them.
- The dtypes the constructor produced are recorded as `built_dtypes`.
- Every variant starts from that reference and casts parameters directly. Nothing is rounded through a dtype and back.

**Backend flags**
- Our variants run with TF32 and `cudnn.benchmark` off.
- The `lerobot*` variants turn both on, as `lerobot_eval.py` does.
- Each result records the flags it ran with.
- The ONNX parity reference is computed with TF32 off.

**Not claimed here**
- Closed-loop task success. Numerical parity is not closed-loop parity; paired simulator rollouts are the next project.

**Record clocks and versions**
- GPU clocks vary on cloud machines. `sysinfo` records clocks and throttle reasons at idle and after each variant's timed loop, plus driver, CUDA, TensorRT and package versions.
- It reads them from the device torch actually uses, so `CUDA_VISIBLE_DEVICES` is respected.
