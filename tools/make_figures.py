"""Draw the figures of README.md and docs/ from results/*.json.

    python tools/make_figures.py                       # docs/figures/<name>.svg and <name>-dark.svg
    python tools/make_figures.py --png /tmp/preview    # also PNGs, to look at them

Every number in a chart is read from results/. The three diagrams (overview, one action chunk, architecture) are
drawn here with the names and shapes of docs/SPLIT_GRAPH.md. Each figure is written for a light and a dark page; the
Markdown picks one with <picture>. Colours are one categorical palette's light and dark steps, and they follow the
meaning: fp32 is always blue, fp16 orange, bf16 aqua.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CAMERAS, STEPS = 2, 10  # lerobot/smolvla_libero: two cameras, 10 flow-matching steps (docs/SPLIT_GRAPH.md)

THEMES = {
    "light": {"surface": "#fcfcfb", "plane": "#f3f2ef", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "critical": "#d03b3b", "tint": 0.13,
              "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
              "violet": "#6250d6"},
    "dark": {"surface": "#1a1a19", "plane": "#242422", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "critical": "#d03b3b", "tint": 0.22,
             "blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70", "yellow": "#c98500", "magenta": "#d55181",
             "violet": "#9085e9"},
}
PRECISION = {"fp32": "blue", "fp16": "orange", "bf16": "aqua"}


def results(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text())


def blend(fg: str, bg: str, a: float) -> str:
    """fg over bg at opacity a, as an opaque colour: SVG viewers then agree on it."""
    f, b = (tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in (fg, bg))
    return "#" + "".join(f"{round(a * x + (1 - a) * y):02x}" for x, y in zip(f, b))


def on(fill: str, t: dict) -> str:
    """Ink or white for text set inside a filled shape, by the fill's luminance."""
    r, g, b = (int(fill[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0b0b0b" if lum > 0.55 else "#ffffff"


def style() -> None:
    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 10, "svg.fonttype": "path",
                         "svg.hashsalt": "smolvla-edge", "axes.linewidth": 1.0, "axes.unicode_minus": True})


# ---------------------------------------------------------------------------------------------- chart helpers


def chart(t: dict, width: float, plot_height: float, title: str, subtitle: str, left: float):
    """A figure laid out in inches: title, subtitle, a legend row, then the plot, left edge at `left` inches."""
    top, bottom = 1.15, 0.45
    height = top + plot_height + bottom
    fig = plt.figure(figsize=(width, height), facecolor=t["surface"])
    ax = fig.add_axes((left / width, bottom / height, (width - left - 0.35) / width, plot_height / height),
                      facecolor=t["surface"])
    fig.text(0.2 / width, 1 - 0.22 / height, title, color=t["ink"], fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.2 / width, 1 - 0.5 / height, subtitle, color=t["ink2"], fontsize=9.5, va="top")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, labelsize=9.5)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    fig.legend_y = 1 - 0.88 / height
    return fig, ax


def legend(fig, t: dict, items, x_in: float, marker: str = "s") -> None:
    """Swatch and label pairs in one row under the subtitle; labels in ink, identity in the swatch."""
    width = fig.get_figwidth()
    x = x_in / width
    for colour, label in items:
        fig.add_artist(plt.Line2D([x], [fig.legend_y], marker=marker, markersize=8, color=colour, linewidth=0,
                                  transform=fig.transFigure))
        txt = fig.text(x + 0.13 / width, fig.legend_y, label, color=t["ink2"], fontsize=9.5, va="center")
        fig.canvas.draw()
        x = fig.transFigure.inverted().transform(txt.get_window_extent().corners()[-1])[0] + 0.3 / width


def fmt(v: float) -> str:
    """0.22, 0.060, 6.0e-3: two significant digits, scientific below 0.01."""
    return f"{v:.2g}" if v >= 0.1 else (f"{v:.3f}" if v >= 0.01 else f"{v:.1e}".replace("e-0", "e-"))


def log_axis(ax, t: dict, lo: int, hi: int, room: float = 1.0) -> None:
    """Decades lo..hi labelled; `room` times 10**hi of space on the right for value labels."""
    ax.set_xscale("log")
    ax.set_xlim(10.0 ** lo, 10.0 ** hi * room)
    ax.xaxis.set_major_locator(FixedLocator([10.0 ** e for e in range(lo, hi + 1)]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"10$^{{{e}}}$" if e not in (0, 1) else ("1" if e == 0 else "10") for e in range(lo, hi + 1)])


# ---------------------------------------------------------------------------------------------- charts


def jetson_latency(t: dict):
    """Where one chunk's time goes on the Jetson, fp32 against fp16; p50 of each stage, frame 0."""
    runs = [("fp32", results("jetson_xavier_nx_trt85_real_fp32.json")["latency_ms"]["frame_0"]),
            ("fp16, fp32 layers pinned", results("jetson_xavier_nx_trt85_real_fp16.json")["latency_ms"]["frame_0"])]
    parts = [("Vision, 2 cameras", "blue", lambda r: CAMERAS * r["vision"]["p50"]),
             ("VLM prefix", "orange", lambda r: r["prefix"]["p50"]),
             ("Action expert, 10 steps", "aqua", lambda r: STEPS * r["expert_step"]["p50"]),
             ("Inputs and Euler updates", "yellow", lambda r: r["inputs"]["p50"] + r["euler"]["p50"])]
    fig, ax = chart(t, 9.6, 1.6, "One action chunk on a Jetson Xavier NX: 1.0 s in fp32, 0.33 s in fp16",
                    "p50 of 50 chunks after 5 warm-up, real lerobot/smolvla_libero weights, LIBERO frame 0, "
                    "TensorRT 8.5.2", left=2.0)
    fp32_chunk = runs[0][1]["chunk"]["p50"]
    for row, (name, r) in enumerate(runs):
        y = 1 - row
        left = 0.0
        for label, hue, value in parts:
            w = value(r)
            ax.barh(y, w, left=left, height=0.46, color=t[hue], edgecolor=t["surface"], linewidth=1.5)
            if w > 70:
                ax.text(left + w / 2, y, f"{w:.0f}", color=on(t[hue], t), ha="center", va="center", fontsize=9)
            left += w
        chunk = r["chunk"]["p50"]
        note = f"{chunk:.0f} ms" + ("" if row == 0 else f"   {fp32_chunk / chunk:.1f}× faster")
        ax.text(left + 14, y, note, color=t["ink"], va="center", fontsize=10, fontweight="bold")
    ax.set_yticks([1, 0], [name for name, _ in runs], color=t["ink"], fontsize=10)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlim(0, 1260)
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_xticklabels(["0", "250", "500", "750", "1000 ms"])
    legend(fig, t, [(t[hue], label) for label, hue, _ in parts], 0.25)
    return fig


def precision_accuracy(t: dict):
    """Chunk error per frame against LeRobot's fp32, for every precision and runtime measured."""
    cpu, j32, j16 = (results(n) for n in ("real_weights_cpu.json", "jetson_xavier_nx_trt85_real_fp32.json",
                                          "jetson_xavier_nx_trt85_real_fp16.json"))
    pyt = cpu["p0_pytorch_cpu"]["variants"]
    rows = [
        ("fp32", "ONNX Runtime, CPU", cpu["split_graphs"]["fp32"]["actions_max_abs_per_frame"]),
        ("fp32", "TensorRT 8.5, Jetson", j32["parity"]["frames_16"]["chunk_max_abs_per_frame"]),
        ("fp16", "ONNX Runtime, CPU", cpu["split_graphs"]["fp16"]["actions_max_abs_per_frame"]),
        ("fp16", "PyTorch fp16all, CPU (4 frames)", list(pyt["fp16all"]["max_abs_per_frame"].values())),
        ("fp16", "TensorRT 8.5, fp32 layers pinned, Jetson", j16["parity"]["frames_16"]["chunk_max_abs_per_frame"]),
        ("bf16", "PyTorch, as LeRobot runs it, CPU", list(pyt["lerobot"]["max_abs_per_frame"].values())),
        ("fp16", "TensorRT 8.5, plain --fp16, Jetson", j16["without_the_flags"]["chunk_max_abs_per_frame"]),
    ]
    fig, ax = chart(t, 9.6, 3.3, "How far each precision moves the action chunk from LeRobot's fp32",
                    "Max abs difference per LIBERO frame, in action std units; each dot is a frame, the bar the "
                    "median. Real lerobot/smolvla_libero weights.", left=3.2)
    for i, (precision, label, values) in enumerate(rows):
        y = len(rows) - 1 - i
        offsets = [(-0.13, 0.0, 0.13)[k % 3] for k in range(len(values))]
        ax.scatter(values, [y + o for o in offsets], s=34, color=t[PRECISION[precision]],
                   edgecolors=t["surface"], linewidths=1.2, zorder=3)
        med = statistics.median(values)
        ax.plot([med, med], [y - 0.3, y + 0.3], color=t["ink"], linewidth=2, solid_capstyle="round", zorder=4)
        worst = max(values)
        ax.text(worst * 1.6, y, fmt(worst), color=t["ink2"], va="center", fontsize=9)
    labels = [f"{p}  ·  {label}" for p, label, _ in rows]
    ax.set_yticks(range(len(rows) - 1, -1, -1), labels, color=t["ink"], fontsize=9.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    log_axis(ax, t, -6, 1)
    ax.axhspan(-0.5, 0.5, color=blend(t["critical"], t["surface"], t["tint"] * 0.8), zorder=0, linewidth=0)
    ax.text(10 ** -5.85, 0, "✕ wrong: actions unrelated to LeRobot's (cosine down to −0.13)", color=t["ink"],
            fontsize=9, va="center")
    legend(fig, t, [(t[PRECISION[p]], p) for p in ("fp32", "fp16", "bf16")], 0.25, marker="o")
    return fig


def fp16_pinning(t: dict):
    """Each fp16 engine alone on LeRobot's inputs, without and with the pinned fp32 layers."""
    j16 = results("jetson_xavier_nx_trt85_real_fp16.json")
    plain = j16["without_the_flags"]["each_graph_alone_on_lerobot_inputs_frames_16"]
    pinned = j16["parity"]["each_graph_alone_on_lerobot_inputs_frames_16"]
    rows = [("vision  →  image tokens", "image_tokens"), ("prefix  →  K/V cache", "prefix_kv"),
            ("expert_step  →  velocity v_t", "v_t")]
    fig, ax = chart(t, 9.6, 1.5, "Plain --fp16 breaks vision and prefix; pinning the fp32 layers fixes them",
                    "Relative L2 against LeRobot's fp32, each engine alone on LeRobot's inputs, 16 LIBERO frames, "
                    "TensorRT 8.5.2", left=2.6)
    gray = t["muted"]
    for i, (label, key) in enumerate(rows):
        y = len(rows) - 1 - i
        a, b = plain[key]["rel_l2"], pinned[key]["rel_l2"]
        ax.plot([b, a], [y, y], color=t["axis"], linewidth=2, zorder=2)
        ax.scatter([a], [y], s=60, color=gray, edgecolors=t["surface"], linewidths=1.2, zorder=3)
        ax.scatter([b], [y], s=60, color=t["orange"], edgecolors=t["surface"], linewidths=1.2, zorder=3)
        ax.text(a * 1.35, y, fmt(a), color=t["ink2"], va="center", fontsize=9)
        ax.text(b / 1.35, y, fmt(b), color=t["ink2"], va="center", ha="right", fontsize=9)
    ax.set_yticks(range(len(rows) - 1, -1, -1), [r[0] for r in rows], color=t["ink"], fontsize=10)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    log_axis(ax, t, -4, 0, room=3)
    legend(fig, t, [(gray, "plain --fp16"), (t["orange"], "with <graph>_fp16.trt85_flags")], 0.25, marker="o")
    return fig


# ---------------------------------------------------------------------------------------------- diagram helpers


def canvas(t: dict, w: float, h: float):
    """A blank figure measured in tenths of an inch: (0, 0) bottom left, (w, h) top right."""
    fig = plt.figure(figsize=(w / 10, h / 10), facecolor=t["surface"])
    ax = fig.add_axes((0, 0, 1, 1), facecolor=t["surface"])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def node(ax, t: dict, x, y, w, h, title, body=(), hue="blue", title_size=10.5, body_size=8.6, fill=None,
         align="center", dash=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", linewidth=1.3,
                                facecolor=fill or blend(t[hue], t["surface"], t["tint"]), edgecolor=t[hue],
                                linestyle=(0, (4, 3)) if dash else "-", zorder=2))
    lines = [(title, title_size, "bold", t["ink"])] + [(b, body_size, "normal", t["ink2"]) for b in body]
    step = [s * 0.205 for _, s, _, _ in lines]
    total = sum(step)
    cy = y + h / 2 + total / 2
    tx = x + w / 2 if align == "center" else x + 1.2
    for (text, size, weight, colour), dy in zip(lines, step):
        cy -= dy
        ax.text(tx, cy + dy * 0.5, text, fontsize=size, fontweight=weight, color=colour, ha=align, va="center",
                zorder=3)


def arrow(ax, t: dict, a, b, colour=None, rad=0.0, lw=1.4, dash=False, label=None, label_dy=0.9):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>,head_length=5,head_width=3", mutation_scale=1,
                                 connectionstyle=f"arc3,rad={rad}", color=colour or t["ink2"], linewidth=lw,
                                 linestyle=(0, (3, 2.5)) if dash else "-", shrinkA=0, shrinkB=0, zorder=1))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + label_dy, label, fontsize=8, color=t["ink2"],
                ha="center", va="center", zorder=3)


def tile(ax, t: dict, x, y, w, h, value, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", linewidth=0,
                                facecolor=t["plane"], zorder=2))
    ax.text(x + 1.4, y + h * 0.62, value, fontsize=15, fontweight="bold", color=t["ink"], va="center", zorder=3)
    ax.text(x + 1.4, y + h * 0.27, label, fontsize=8.6, color=t["ink2"], va="center", zorder=3)


# ---------------------------------------------------------------------------------------------- diagrams


def overview(t: dict):
    """The main idea: one PyTorch policy, split into three graphs, run by a C++ loop, checked at every stage."""
    j32, j16 = results("jetson_xavier_nx_trt85_real_fp32.json"), results("jetson_xavier_nx_trt85_real_fp16.json")
    fig, ax = canvas(t, 120, 45)
    ax.text(3, 42.3, "smolvla-edge: SmolVLA, split for the edge and checked against LeRobot at every stage",
            fontsize=13, fontweight="bold", color=t["ink"], va="center")
    y, h = 24.5, 13
    node(ax, t, 3, y, 19, h, "LeRobot SmolVLA", ["PyTorch policy, 450M params", "lerobot/smolvla_libero",
                                                   "2 cameras, task, state"], hue="violet")
    node(ax, t, 27, y + 2.2, 21, h - 2.2, "export_split.py", ["three static ONNX graphs,", "fp32 and fp16"],
         hue="blue")
    for i, (g, gx) in enumerate((("vision", 30.6), ("prefix", 36.4), ("expert_step", 43.6))):
        ax.text(gx, y + 3.2, g, fontsize=7.6, color=t["ink"], ha="center", va="center", zorder=3,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": t["surface"], "edgecolor": t["blue"],
                      "linewidth": 0.8})
    node(ax, t, 53, y, 20, h, "Engines", ["TensorRT 8.5 (Jetson)", "TensorRT 10, ONNX Runtime",
                                           "fp16: fp32 layers pinned"], hue="orange")
    node(ax, t, 78, y, 21, h, "C++17 runtime", ["vision × 2 cameras", "prefix: K/V cache, once",
                                                "expert_step × 10, Euler"], hue="aqua")
    node(ax, t, 104, y, 13, h, "Action chunk", ["50 × 7 actions", "into a 50 Hz", "control loop"], hue="yellow")
    for a, b in ((22, 27), (48, 53), (73, 78), (99, 104)):
        arrow(ax, t, (a + 0.3, y + h / 2), (b - 0.3, y + h / 2))
    # The parity gate: LeRobot's fp32 is the reference; the export and the runtime are checked against it.
    gy, gh = 11.5, 8
    node(ax, t, 3, gy, 114, gh, "Parity gate: LeRobot's own fp32 on 16 real LIBERO frames, frozen into bundles",
         ["the export and the runtime are compared with it at every split point: image tokens, prefix K/V, "
          "each v_t, the chunk"], hue="magenta", dash=True)
    arrow(ax, t, (12.5, y), (12.5, gy + gh), colour=t["magenta"], dash=True, lw=1.1)
    for x in (37.5, 63, 88.5):
        arrow(ax, t, (x, gy + gh), (x, y), colour=t["magenta"], dash=True, lw=1.1)
    lat32 = j32["latency_ms"]["frame_0"]["chunk"]["p50"]
    lat16 = j16["latency_ms"]["frame_0"]["chunk"]["p50"]
    tiles = [(f"{lat32 / 1000:.1f} s  →  {lat16 / 1000:.2f} s",
              f"per action chunk on a Jetson Xavier NX, fp32 → fp16 ({lat32 / lat16:.1f}×)"),
             (fmt(j32["parity"]["frames_16"]["chunk"]["max_abs"]), "fp32 chunk error vs LeRobot, worst of 16 frames"),
             (fmt(j16["parity"]["frames_16"]["chunk"]["max_abs"]),
              "fp16 chunk error, action std (LeRobot's own bf16: 0.049)")]
    for i, (value, label) in enumerate(tiles):
        tile(ax, t, 3 + i * 38.5, 1.5, 37, 7.5, value, label)
    return fig


def data_flow(t: dict):
    """One action chunk through the three graphs, with every tensor's shape."""
    fig, ax = canvas(t, 120, 55)
    ax.text(3, 52.3, "One action chunk, graph by graph (shapes of lerobot/smolvla_libero)", fontsize=13,
            fontweight="bold", color=t["ink"], va="center")
    v = "violet"
    node(ax, t, 3, 39, 19, 9.5, "Camera 1, camera 2", ["256×256 frames,", "upsampled to 512×512"], hue=v)
    node(ax, t, 3, 27, 19, 9.5, "Task text", ["48 token ids + mask,", "padded"], hue=v)
    node(ax, t, 3, 15, 19, 9.5, "Robot state", ["8 values, normalised,", "zero-padded to 32"], hue=v)
    node(ax, t, 3, 2, 19, 9.5, "Noise x₁", ["N(0, I), 50 × 32", "same seed as LeRobot"], hue=v)
    node(ax, t, 29, 39, 21, 9.5, "vision, once per camera", ["SigLIP tower + connector",
                                                             "→ 64 tokens × 960"], hue="blue")
    node(ax, t, 57, 15, 23, 33.5, "prefix, once", ["16 SmolVLM layers over", "image, language and state:",
                                                   "177 positions", "= 2 × 64 + 48 + 1", "", "fills the K/V cache"],
         hue="blue")
    node(ax, t, 88, 39, 29, 9.5, "K/V cache", ["16 layers × (K, V), each 5 × 177 × 64,", "+ pad mask over 177"],
         hue="orange")
    node(ax, t, 57, 2, 60, 9.5, "expert_step × 10, the Euler loop in C++",
         ["t = 1.0, 0.9, …, 0.1:   v = expert_step(x, t, K/V),   x ← x − 0.1 · v",
          "x₀ is 50 × 32; its first 7 dims are the 50 × 7 action chunk"], hue="aqua", align="left")
    arrow(ax, t, (22.3, 43.75), (28.7, 43.75))
    arrow(ax, t, (50.3, 43.75), (56.7, 43.75))
    arrow(ax, t, (22.3, 31.75), (56.7, 31.75))
    arrow(ax, t, (22.3, 19.75), (56.7, 19.75))
    arrow(ax, t, (22.3, 6.75), (56.7, 6.75))
    arrow(ax, t, (80.3, 43.75), (87.7, 43.75))
    arrow(ax, t, (102.5, 38.7), (102.5, 11.8), colour=t["orange"])
    ax.text(101.3, 25, "read in place,\nevery step", fontsize=8, color=t["ink2"], ha="right", va="center")
    return fig


def architecture(t: dict):
    """What runs where: the Python side that exports and checks, the contract, the C++ side, the targets."""
    fig, ax = canvas(t, 120, 64)
    ax.text(3, 61.5, "System structure", fontsize=13, fontweight="bold", color=t["ink"], va="center")
    w = 33
    cols = [(3, "Python: export and reference", "violet"), (43.5, "The contract, docs/SPLIT_GRAPH.md", "magenta"),
            (84, "C++17 runtime, cpp/", "aqua")]
    for x, head, hue in cols:
        ax.text(x + w / 2, 57.5, head, fontsize=10.5, fontweight="bold", color=t["ink"], ha="center",
                va="center")
        ax.plot([x, x + w], [55.8, 55.8], color=t[hue], linewidth=2, solid_capstyle="round")
    py = [("export_split.py, smolvla_latency/", ["wrappers around LeRobot's own code;",
                                                  "opset 17, LayerNorm, trt85 flags"]),
          ("tools/", ["LIBERO frames → LeRobot fp32 bundles;", "fp16 range probe, reports"]),
          ("ort_flow.py", ["reference runtime:", "ONNX Runtime + numpy Euler"]),
          ("profile_smolvla.py", ["per-stage PyTorch timing,", "precision variants"])]
    contract = [("split_manifest.json", ["graph files, tensor names,", "shapes, dtypes"]),
                ("<graph>_<p>.onnx", ["vision, prefix, expert_step;", "fp32 and fp16, weights in .data"]),
                ("<graph>_fp16.trt85_flags", ["layer pins for", "TensorRT 8.5 builds"]),
                ("bundle.bin", ["inputs, noise, LeRobot's tensors", "at every split point"])]
    cpp = [("SmolVLARuntime", ["checks the graphs against the contract;", "buffers bound once, Euler loop"]),
           ("Graph: OrtGraph, TrtGraph", ["ONNX Runtime (CPU, CUDA);", "TensorRT 8.5 and 10"]),
           ("smolvla_rt_bench", ["parity per stage, latency p50/p99,", "50 Hz control-loop harness"]),
           ("Python module, pybind11", ["the same predict_action_chunk", "as ort_flow.py"])]
    for (x, _, hue), items in zip(cols, (py, contract, cpp)):
        for i, (title, body) in enumerate(items):
            node(ax, t, x, 45.2 - i * 10.6, w, 9, title, body, hue=hue, body_size=8.2)
    arrow(ax, t, (36.4, 33.9), (43.1, 33.9), lw=1.8, label="writes", label_dy=1.6)
    arrow(ax, t, (76.9, 33.9), (83.6, 33.9), lw=1.8, label="reads", label_dy=1.6)
    node(ax, t, 3, 7.1, 114, 5.2, "Checked by: pytest against LeRobot (tiny model, 2 K/V heads)  ·  "
         "tests/cpp_parity.sh: C++ vs PyTorch  ·  smoke test  ·  GitHub Actions on every push", [], hue="blue",
         title_size=9.3)
    node(ax, t, 3, 1.2, 114, 5.2, "Runs on: x86 CPU (ONNX Runtime)  ·  NVIDIA GPU (TensorRT 10)  ·  "
         "Jetson Xavier NX (TensorRT 8.5.2, JetPack 5)", [], hue="orange", title_size=9.3)
    return fig


FIGURES = {"overview": overview, "data_flow": data_flow, "architecture": architecture,
           "jetson_latency": jetson_latency, "precision_accuracy": precision_accuracy, "fp16_pinning": fp16_pinning}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    ap.add_argument("--png", type=Path, default=None, help="also write PNGs here")
    ap.add_argument("--only", default=None, help="comma list of figure names")
    args = ap.parse_args()
    style()
    args.out.mkdir(parents=True, exist_ok=True)
    names = args.only.split(",") if args.only else list(FIGURES)
    for name in names:
        for mode, t in THEMES.items():
            fig = FIGURES[name](t)
            stem = name if mode == "light" else f"{name}-dark"
            fig.savefig(args.out / f"{stem}.svg", facecolor=t["surface"], metadata={"Date": None})
            if args.png:
                args.png.mkdir(parents=True, exist_ok=True)
                fig.savefig(args.png / f"{stem}.png", facecolor=t["surface"], dpi=110)
            plt.close(fig)
        print(f"wrote {args.out / name}.svg and {name}-dark.svg")


if __name__ == "__main__":
    main()
