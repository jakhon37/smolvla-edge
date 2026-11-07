"""The ONNX rewrites against ONNX Runtime running the original graphs, on small graphs. Needs no torch."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from smolvla_latency.onnx_rewrite import decompose_layer_norm, lower_to_opset17, trt85_fp16_flags

SHAPE = [1, 4, 8]


def layer_norm_model(opset: int, elem: int, bias: bool = True, **attrs) -> onnx.ModelProto:
    dtype = np.float32 if elem == TensorProto.FLOAT else np.float16
    rng = np.random.default_rng(0)
    inits = [numpy_helper.from_array(rng.uniform(0.5, 1.5, SHAPE[-1]).astype(dtype), "scale")]
    inputs = ["x", "scale"]
    if bias:
        inits.append(numpy_helper.from_array(rng.uniform(-1, 1, SHAPE[-1]).astype(dtype), "bias"))
        inputs.append("bias")
    node = helper.make_node("LayerNormalization", inputs, ["y"], name="ln", **attrs)
    graph = helper.make_graph([node], "ln", [helper.make_tensor_value_info("x", elem, SHAPE)],
                              [helper.make_tensor_value_info("y", elem, SHAPE)], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10)


def run(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("opset", [17, 18])
@pytest.mark.parametrize("elem", [TensorProto.FLOAT, TensorProto.FLOAT16])
def test_matches_the_operator(opset, elem, bias):
    dtype = np.float32 if elem == TensorProto.FLOAT else np.float16
    # Up to 663.5, as SigLIP's residual stream with real weights: squares beyond fp16's largest value, 65504.
    x = np.random.default_rng(1).uniform(-663.5, 663.5, SHAPE).astype(dtype)
    original = layer_norm_model(opset, elem, bias, epsilon=1e-6)
    rewritten = layer_norm_model(opset, elem, bias, epsilon=1e-6)
    assert decompose_layer_norm(rewritten) == 1
    assert "LayerNormalization" not in {n.op_type for n in rewritten.graph.node}
    onnx.checker.check_model(rewritten, full_check=True)
    a, b = run(original, x), run(rewritten, x)
    assert b.dtype == dtype and np.isfinite(b).all()
    assert np.abs(a.astype(np.float64) - b.astype(np.float64)).max() < (1e-5 if dtype == np.float32 else 2e-3)


@pytest.mark.parametrize("opset", [17, 18])
def test_fp16_statistics_stay_fp32(opset):
    model = layer_norm_model(opset, TensorProto.FLOAT16)
    decompose_layer_norm(model)
    types = {v.name: v.type.tensor_type.elem_type for v in onnx.shape_inference.infer_shapes(model).graph.value_info}
    for tensor in ("mean", "d", "d2", "var", "var_eps", "std", "norm"):
        assert types[f"y.ln_{tensor}"] == TensorProto.FLOAT, tensor
    assert types["y.ln_norm_low"] == TensorProto.FLOAT16


@pytest.mark.parametrize("attrs, used_mean, message", [
    ({"axis": 1}, False, "axis 1"),
    ({"stash_type": TensorProto.DOUBLE}, False, "stash_type"),
    ({}, True, "Mean or InvStdDev"),
])
def test_refuses_what_it_does_not_reproduce(attrs, used_mean, message):
    model = layer_norm_model(17, TensorProto.FLOAT, **attrs)
    if used_mean:
        model.graph.node[0].output.append("mean")
    with pytest.raises(NotImplementedError, match=message):
        decompose_layer_norm(model)


def test_leaves_other_graphs_alone():
    model = layer_norm_model(17, TensorProto.FLOAT)
    model.graph.node[0].op_type = "Identity"
    assert decompose_layer_norm(model) == 0
    assert [n.op_type for n in model.graph.node] == ["Identity"]


def opset18_model(extra_node=None) -> onnx.ModelProto:
    """x -> ReduceMean(last axis) + Split(two parts, equal), ReduceSum (axes an input since opset 13), ReduceMin."""
    inits = [numpy_helper.from_array(np.array([-1], np.int64), "last"),
             numpy_helper.from_array(np.array([1], np.int64), "second")]
    nodes = [
        helper.make_node("ReduceMean", ["x", "last"], ["mean"], name="mean", keepdims=1, noop_with_empty_axes=0),
        helper.make_node("Sub", ["x", "mean"], ["d"], name="sub"),
        helper.make_node("Split", ["d"], ["a", "b"], name="split", axis=2, num_outputs=2),
        helper.make_node("ReduceMin", ["a", "second"], ["low"], name="low", keepdims=1),
        helper.make_node("ReduceSum", ["b", "last"], ["sum"], name="sum", keepdims=0),
    ]
    if extra_node is not None:
        nodes.append(extra_node)
    outputs = [helper.make_tensor_value_info("low", TensorProto.FLOAT, [1, 1, 4]),
               helper.make_tensor_value_info("sum", TensorProto.FLOAT, [1, 4])]
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, SHAPE)], outputs, inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)


def test_lowering_to_opset17_keeps_the_values():
    x = np.random.default_rng(2).normal(size=SHAPE).astype(np.float32)
    original, lowered = opset18_model(), opset18_model()
    lower_to_opset17(lowered)
    onnx.checker.check_model(lowered, full_check=True)
    assert [o.version for o in lowered.opset_import] == [17]
    nodes = {n.name: n for n in lowered.graph.node}
    assert list(nodes["mean"].input) == ["x"] and list(nodes["low"].input) == ["a"]
    assert {a.name: helper.get_attribute_value(a) for a in nodes["mean"].attribute} == {"keepdims": 1, "axes": [-1]}
    assert {a.name for a in nodes["split"].attribute} == {"axis"}
    assert len(nodes["sum"].input) == 2  # opset 13's ReduceSum already takes its axes as an input
    assert {t.name for t in lowered.graph.initializer} == {"last"}  # "second" had only ReduceMin to serve

    run_all = lambda m: ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})
    for a, b in zip(run_all(original), run_all(lowered)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("node, message", [
    (helper.make_node("Pad", ["x", "last"], ["p"], name="pad"), "changed at opset 18"),
    (helper.make_node("ReduceMax", ["x", "not_constant"], ["m"], name="max"), "not an initializer"),
    (helper.make_node("Split", ["x", "last"], ["s"], name="sized", axis=2), "explicit sizes"),
])
def test_lowering_refuses_what_it_cannot_lower(node, message):
    with pytest.raises(NotImplementedError, match=message):
        lower_to_opset17(opset18_model(node))


def mixed_model(fp32_sigmoid: bool = True, name: str = "sq") -> onnx.ModelProto:
    """fp16 x -> Cast up -> Mul -> Split in two -> Concat -> Cast down -> Sigmoid, the fp32 part in between as in
    the fp16 graphs; with fp32_sigmoid also a Sigmoid on the fp32 square, as the expert's time MLP has."""
    nodes = [
        helper.make_node("Cast", ["x"], ["up"], name="up", to=TensorProto.FLOAT),
        helper.make_node("Mul", ["up", "up"], ["sq"], name=name),
        helper.make_node("Split", ["sq"], ["h1", "h2"], name="halves", axis=2),
        helper.make_node("Concat", ["h1", "h2"], ["join"], name="join", axis=2),
        helper.make_node("Cast", ["join"], ["down"], name="down", to=TensorProto.FLOAT16),
        helper.make_node("Sigmoid", ["down"], ["y"], name="sig16"),
    ]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT16, SHAPE)]
    if fp32_sigmoid:
        nodes.append(helper.make_node("Sigmoid", ["sq"], ["z"], name="sig32"))
        outputs.append(helper.make_tensor_value_info("z", TensorProto.FLOAT, SHAPE))
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT16, SHAPE)], outputs)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)


def specs(flags: list[str]) -> dict[str, str]:
    assert flags[:2] == ["--fp16", "--precisionConstraints=obey"]
    assert not any(" " in f for f in flags)  # the exporter writes them on one line, for $(cat ...)
    return dict(f.split("=", 1) for f in flags[2:])


def test_trt85_flags_pin_the_fp32_nodes():
    flags = specs(trt85_fp16_flags(mixed_model(fp32_sigmoid=False)))
    assert flags["--layerPrecisions"] == "up:fp32,sq:fp32,halves:fp32,join:fp32"
    assert flags["--layerOutputTypes"] == "up:fp32,sq:fp32,join:fp32"  # TensorRT gives the Split one output


def test_trt85_flags_put_every_sigmoid_in_fp32_when_one_is():
    flags = specs(trt85_fp16_flags(mixed_model(fp32_sigmoid=True)))
    assert flags["--layerPrecisions"] == "up:fp32,sq:fp32,halves:fp32,join:fp32,sig32:fp32,sig16:fp32"
    assert flags["--layerOutputTypes"] == "up:fp32,sq:fp32,join:fp32,sig32:fp32,sig16:fp32"


def test_trt85_flags_refuse_what_they_cannot_name_or_type():
    with pytest.raises(ValueError, match="cannot name"):
        trt85_fp16_flags(mixed_model(name="sq:0"))
    model = mixed_model()
    model.graph.node.append(helper.make_node("Unknown", ["x"], ["w"], name="custom", domain="example"))
    model.opset_import.append(helper.make_opsetid("example", 1))
    with pytest.raises(ValueError, match="no dtype"):
        trt85_fp16_flags(model)
