"""Rewrites of exported ONNX graphs for runtimes that lack an operator. Needs only numpy and onnx."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def lower_to_opset17(model: onnx.ModelProto) -> None:
    """Rewrite an opset 18 graph in place as opset 17; refuses any operator it cannot lower.

    ONNX's version converter fails on the prefix and expert_step graphs (it has no Split adapter, for one), and
    torch.onnx then logs that and returns opset 18 although asked for 17. That matters on TensorRT 8.5: its parser
    builds an opset 18 ReduceMean without any error but ignores the axes input and averages over every axis, so
    the engine runs and gives wrong numbers.
    """
    graph = model.graph
    inits = {t.name: t for t in graph.initializer}
    axes_inputs = set()
    for node in graph.node:
        if onnx.defs.get_schema(node.op_type, 18, "").since_version != 18:
            continue  # the same operator at opset 17
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        where = f"{node.op_type} {node.name!r}"
        if node.op_type == "Split":
            if len(node.input) != 1:
                raise NotImplementedError(f"{where}: lowering a Split with explicit sizes")
            keep = [a for a in node.attribute if a.name != "num_outputs"]  # at opset 17 no sizes mean equal parts
        elif node.op_type.startswith("Reduce"):
            if attrs.get("noop_with_empty_axes"):
                raise NotImplementedError(f"{where}: noop_with_empty_axes")
            keep = [a for a in node.attribute if a.name != "noop_with_empty_axes"]
            if len(node.input) > 1 and node.input[1]:
                if node.input[1] not in inits:
                    raise NotImplementedError(f"{where}: its axes are not an initializer")
                axes = numpy_helper.to_array(inits[node.input[1]])
                keep.append(helper.make_attribute("axes", [int(a) for a in axes]))
                axes_inputs.add(node.input[1])
                del node.input[1]
        else:
            raise NotImplementedError(f"{where}: changed at opset 18, and this pass does not lower it")
        del node.attribute[:]
        node.attribute.extend(keep)
    used = {name for node in graph.node for name in node.input}
    kept = [t for t in graph.initializer if t.name not in axes_inputs or t.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            opset.version = 17


def decompose_layer_norm(model: onnx.ModelProto) -> int:
    """Replace every LayerNormalization in place with the basic ops that define it; returns how many.

    TensorRT 8.5, the newest JetPack offers for Xavier, has no importer for the operator. The statistics stay in
    fp32, as the operator's stash_type 1 says: SigLIP's residual stream reaches 663.5 with real weights, and its
    square is beyond fp16. An fp16 runtime has to honour those Casts (docs/SPLIT_GRAPH.md).
    """
    graph = model.graph
    opset = next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    types = {v.name: v.type.tensor_type.elem_type for v in (*graph.input, *graph.value_info, *graph.output)}
    types.update({t.name: t.data_type for t in graph.initializer})

    nodes, count = [], 0
    for node in graph.node:
        if node.op_type != "LayerNormalization":
            nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        where = f"LayerNormalization {node.name!r}"
        if attrs.get("axis", -1) != -1:
            raise NotImplementedError(f"{where}: axis {attrs['axis']}, only the last axis is supported")
        if attrs.get("stash_type", TensorProto.FLOAT) != TensorProto.FLOAT:
            raise NotImplementedError(f"{where}: stash_type {attrs['stash_type']}, only fp32 is supported")
        if any(node.output[1:]):
            raise NotImplementedError(f"{where}: its Mean or InvStdDev output is used")
        if node.input[0] not in types:
            raise ValueError(f"{where}: the graph records no dtype for its input {node.input[0]!r}")

        x, scale = node.input[0], node.input[1]
        bias = node.input[2] if len(node.input) > 2 else ""
        y = node.output[0]
        name = f"{y}.ln_"

        def add(op, inputs, output, **attributes):
            nodes.append(helper.make_node(op, inputs, [output], name=output, **attributes))
            return output

        def mean(tensor, output):
            if opset >= 18:  # ReduceMean's axes moved from an attribute to an input
                return add("ReduceMean", [tensor, name + "axes"], output, keepdims=1)
            return add("ReduceMean", [tensor], output, axes=[-1], keepdims=1)

        if opset >= 18:
            graph.initializer.append(numpy_helper.from_array(np.array([-1], np.int64), name + "axes"))
        graph.initializer.append(numpy_helper.from_array(np.array(attrs.get("epsilon", 1e-5), np.float32),
                                                         name + "epsilon"))
        lowered = types[x] != TensorProto.FLOAT
        stash = add("Cast", [x], name + "stash", to=TensorProto.FLOAT) if lowered else x
        d = add("Sub", [stash, mean(stash, name + "mean")], name + "d")
        var = mean(add("Mul", [d, d], name + "d2"), name + "var")
        std = add("Sqrt", [add("Add", [var, name + "epsilon"], name + "var_eps")], name + "std")
        normed = add("Div", [d, std], name + "norm")
        if lowered:
            normed = add("Cast", [normed], name + "norm_low", to=types[x])
        if bias:
            add("Add", [add("Mul", [normed, scale], name + "scaled"), bias], y)
        else:
            add("Mul", [normed, scale], y)
        count += 1
    del graph.node[:]
    graph.node.extend(nodes)
    return count
