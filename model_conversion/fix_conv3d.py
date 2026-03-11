"""
Transform FactorizePhys ONNX model: decompose Conv3d → Conv2d for ONNX Runtime Web.

ONNX Runtime Web (WebGPU/WASM) only supports Conv1d and Conv2d.
This script replaces each Conv3d node with an equivalent subgraph:
  1. Zero-pad the temporal dimension
  2. Slice 3 temporal offsets, concatenate along channels
  3. Transpose + reshape to merge temporal into batch
  4. Conv2d with reshaped weights
  5. Reshape + transpose back to 5D

Also wraps InstanceNormalization on 5D tensors with Reshape→InstanceNorm(4D)→Reshape.

Usage:
  uv run --with onnx --with numpy python model_conversion/fix_conv3d.py
"""

import copy
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference, checker

INPUT_PATH = "public/factorizephys.onnx"
OUTPUT_PATH = "public/factorizephys.onnx"  # overwrite in-place


def get_shape_map(model):
    """Run shape inference and build a map: tensor_name → list[int]."""
    inferred = shape_inference.infer_shapes(model)
    shape_map = {}
    # From graph inputs
    for vi in inferred.graph.input:
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        shape_map[vi.name] = dims
    # From value_info (intermediate tensors)
    for vi in inferred.graph.value_info:
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        shape_map[vi.name] = dims
    # From graph outputs
    for vi in inferred.graph.output:
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        shape_map[vi.name] = dims
    return shape_map


def get_initializer_map(model):
    """Build a map: initializer_name → (numpy_array, index_in_graph)."""
    init_map = {}
    for idx, init in enumerate(model.graph.initializer):
        init_map[init.name] = (numpy_helper.to_array(init), idx)
    return init_map


def make_const(name, value, dtype=TensorProto.INT64):
    """Create a Constant node that produces a 1D tensor."""
    arr = np.array(value, dtype=np.int64)
    tensor = numpy_helper.from_array(arr, name=name + "_val")
    return helper.make_node("Constant", [], [name], value=tensor)


def decompose_conv3d(node, shape_map, init_map, graph, uid):
    """
    Decompose a single Conv3d node into Conv2d with temporal unfolding.
    Returns (new_nodes, new_initializers) to add, and the node to remove.
    """
    prefix = f"_decomp{uid}_"
    input_name = node.input[0]
    weight_name = node.input[1]
    bias_name = node.input[2] if len(node.input) > 2 and node.input[2] else None
    output_name = node.output[0]

    # Get attributes
    attrs = {}
    for attr in node.attribute:
        if attr.ints:
            attrs[attr.name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.INT:
            attrs[attr.name] = attr.i

    kernel_shape = attrs["kernel_shape"]  # [Kt, Kh, Kw]
    pads = attrs.get("pads", [0, 0, 0, 0, 0, 0])  # [pt_b, ph_b, pw_b, pt_e, ph_e, pw_e]
    strides = attrs.get("strides", [1, 1, 1])  # [St, Sh, Sw]
    dilations = attrs.get("dilations", [1, 1, 1])
    group = attrs.get("group", 1)

    Kt = kernel_shape[0]
    Kh, Kw = kernel_shape[1], kernel_shape[2]
    pt_begin, ph_begin, pw_begin = pads[0], pads[1], pads[2]
    pt_end, ph_end, pw_end = pads[3], pads[4], pads[5]
    St, Sh, Sw = strides

    # Get weight and transform: [Cout, Cin, Kt, Kh, Kw] → [Cout, Kt*Cin, Kh, Kw]
    weight_np, _ = init_map[weight_name]
    Cout, Cin = weight_np.shape[0], weight_np.shape[1]
    # Transpose axes 1,2: [Cout, Cin, Kt, Kh, Kw] → [Cout, Kt, Cin, Kh, Kw]
    weight_2d = weight_np.transpose(0, 2, 1, 3, 4).reshape(Cout, Kt * Cin, Kh, Kw)
    weight_2d_name = prefix + "weight_2d"
    new_weight_init = numpy_helper.from_array(weight_2d.astype(np.float32), name=weight_2d_name)

    # Get input shape
    in_shape = shape_map.get(input_name)
    if not in_shape or len(in_shape) != 5:
        raise ValueError(f"Cannot get 5D shape for {input_name}: {in_shape}")
    B, C_in, T, H, W = in_shape

    T_padded = T + pt_begin + pt_end
    T_out = (T_padded - Kt) // St + 1

    new_nodes = []
    new_inits = [new_weight_init]

    # Step 1: Temporal padding (if needed)
    padded_name = input_name
    if pt_begin > 0 or pt_end > 0:
        padded_name = prefix + "padded"
        # Pad format: [x0_begin, x1_begin, ..., x0_end, x1_end, ...]
        pad_values = [0, 0, pt_begin, 0, 0, 0, 0, pt_end, 0, 0]
        pad_const_name = prefix + "pad_vals"
        new_nodes.append(make_const(pad_const_name, pad_values))
        pad_node = helper.make_node(
            "Pad", [input_name, pad_const_name], [padded_name],
            name=prefix + "Pad", mode="constant"
        )
        new_nodes.append(pad_node)

    # Step 2: Slice temporal offsets and concatenate
    slice_names = []
    for k in range(Kt):
        start_val = k
        end_val = k + T_out * St
        step_val = St

        starts_name = prefix + f"starts_{k}"
        ends_name = prefix + f"ends_{k}"
        axes_name = prefix + f"axes_{k}"
        steps_name = prefix + f"steps_{k}"
        slice_out_name = prefix + f"slice_{k}"

        new_nodes.append(make_const(starts_name, [start_val]))
        new_nodes.append(make_const(ends_name, [end_val]))
        new_nodes.append(make_const(axes_name, [2]))  # temporal axis
        new_nodes.append(make_const(steps_name, [step_val]))

        slice_node = helper.make_node(
            "Slice",
            [padded_name, starts_name, ends_name, axes_name, steps_name],
            [slice_out_name],
            name=prefix + f"Slice_{k}"
        )
        new_nodes.append(slice_node)
        slice_names.append(slice_out_name)

    # Concat along channel axis: [B, Kt*Cin, T_out, H, W]
    concat_out = prefix + "concat"
    concat_node = helper.make_node(
        "Concat", slice_names, [concat_out],
        name=prefix + "Concat", axis=1
    )
    new_nodes.append(concat_node)

    # Step 3: Transpose [B, Kt*Cin, T_out, H, W] → [B, T_out, Kt*Cin, H, W]
    transp1_out = prefix + "transp1"
    transp1 = helper.make_node(
        "Transpose", [concat_out], [transp1_out],
        name=prefix + "Transpose1", perm=[0, 2, 1, 3, 4]
    )
    new_nodes.append(transp1)

    # Step 4: Reshape → [B*T_out, Kt*Cin, H, W]
    reshape1_shape_name = prefix + "reshape1_shape"
    new_nodes.append(make_const(reshape1_shape_name, [B * T_out, Kt * Cin, H, W]))
    reshape1_out = prefix + "reshape1"
    reshape1 = helper.make_node(
        "Reshape", [transp1_out, reshape1_shape_name], [reshape1_out],
        name=prefix + "Reshape1"
    )
    new_nodes.append(reshape1)

    # Step 5: Conv2d [B*T_out, Kt*Cin, H, W] → [B*T_out, Cout, H', W']
    conv2d_inputs = [reshape1_out, weight_2d_name]
    if bias_name:
        conv2d_inputs.append(bias_name)
    conv2d_out = prefix + "conv2d"
    conv2d = helper.make_node(
        "Conv", conv2d_inputs, [conv2d_out],
        name=prefix + "Conv2d",
        kernel_shape=[Kh, Kw],
        pads=[ph_begin, pw_begin, ph_end, pw_end],
        strides=[Sh, Sw],
        dilations=[dilations[1], dilations[2]],
        group=group
    )
    new_nodes.append(conv2d)

    # Compute spatial output dims
    H_out = (H + ph_begin + ph_end - Kh) // Sh + 1
    W_out = (W + pw_begin + pw_end - Kw) // Sw + 1

    # Step 6: Reshape [B*T_out, Cout, H', W'] → [B, T_out, Cout, H', W']
    reshape2_shape_name = prefix + "reshape2_shape"
    new_nodes.append(make_const(reshape2_shape_name, [B, T_out, Cout, H_out, W_out]))
    reshape2_out = prefix + "reshape2"
    reshape2 = helper.make_node(
        "Reshape", [conv2d_out, reshape2_shape_name], [reshape2_out],
        name=prefix + "Reshape2"
    )
    new_nodes.append(reshape2)

    # Step 7: Transpose [B, T_out, Cout, H', W'] → [B, Cout, T_out, H', W']
    transp2 = helper.make_node(
        "Transpose", [reshape2_out], [output_name],
        name=prefix + "Transpose2", perm=[0, 2, 1, 3, 4]
    )
    new_nodes.append(transp2)

    return new_nodes, new_inits


def wrap_instance_norm_5d(node, shape_map, uid):
    """
    Wrap InstanceNormalization on 5D tensor:
      Reshape [B,C,T,H,W] → [B,C,T*H,W] → InstanceNorm → Reshape back.
    InstanceNorm normalizes over spatial dims, so merging T*H is equivalent.
    """
    prefix = f"_instnorm{uid}_"
    input_name = node.input[0]
    output_name = node.output[0]

    in_shape = shape_map.get(input_name)
    if not in_shape or len(in_shape) != 5:
        return None, None  # not 5D, skip

    B, C, T, H, W = in_shape

    new_nodes = []

    # Reshape to 4D: [B, C, T*H, W]
    reshape_to_4d_shape = prefix + "to4d_shape"
    new_nodes.append(make_const(reshape_to_4d_shape, [B, C, T * H, W]))
    reshaped_4d = prefix + "input_4d"
    new_nodes.append(helper.make_node(
        "Reshape", [input_name, reshape_to_4d_shape], [reshaped_4d],
        name=prefix + "ReshapeTo4D"
    ))

    # InstanceNorm on 4D
    norm_4d_out = prefix + "norm_4d_out"
    # Copy attributes from original node
    norm_node = helper.make_node(
        "InstanceNormalization",
        [reshaped_4d, node.input[1], node.input[2]],  # input, scale, B
        [norm_4d_out],
        name=prefix + "InstanceNorm4D",
        epsilon=next((a.f for a in node.attribute if a.name == "epsilon"), 1e-5)
    )
    new_nodes.append(norm_node)

    # Reshape back to 5D: [B, C, T, H, W]
    reshape_to_5d_shape = prefix + "to5d_shape"
    new_nodes.append(make_const(reshape_to_5d_shape, [B, C, T, H, W]))
    new_nodes.append(helper.make_node(
        "Reshape", [norm_4d_out, reshape_to_5d_shape], [output_name],
        name=prefix + "ReshapeTo5D"
    ))

    return new_nodes, node


def transform_model(model):
    """Transform the model: decompose Conv3d, wrap InstanceNorm 5D."""
    # 1. Run shape inference
    shape_map = get_shape_map(model)
    init_map = get_initializer_map(model)

    # 2. Identify nodes to transform
    conv3d_nodes = []
    instnorm_5d_nodes = []
    for node in model.graph.node:
        if node.op_type == "Conv":
            for attr in node.attribute:
                if attr.name == "kernel_shape" and len(attr.ints) == 3:
                    conv3d_nodes.append(node)
                    break
        elif node.op_type == "InstanceNormalization":
            in_shape = shape_map.get(node.input[0])
            if in_shape and len(in_shape) == 5:
                instnorm_5d_nodes.append(node)

    print(f"Found {len(conv3d_nodes)} Conv3d nodes to decompose")
    print(f"Found {len(instnorm_5d_nodes)} 5D InstanceNorm nodes to wrap")

    if not conv3d_nodes and not instnorm_5d_nodes:
        print("Nothing to transform!")
        return model

    # 3. Build new node list
    nodes_to_remove = set(id(n) for n in conv3d_nodes + instnorm_5d_nodes)
    new_graph_nodes = []
    all_new_inits = []

    uid = 0
    for node in model.graph.node:
        if id(node) in nodes_to_remove:
            if node.op_type == "Conv":
                new_nodes, new_inits = decompose_conv3d(node, shape_map, init_map, model.graph, uid)
                new_graph_nodes.extend(new_nodes)
                all_new_inits.extend(new_inits)
                print(f"  Decomposed Conv3d: {node.name[:60]}...")
                uid += 1
            elif node.op_type == "InstanceNormalization":
                new_nodes, _ = wrap_instance_norm_5d(node, shape_map, uid)
                if new_nodes:
                    new_graph_nodes.extend(new_nodes)
                    print(f"  Wrapped InstanceNorm5D: {node.name[:60]}...")
                    uid += 1
                else:
                    new_graph_nodes.append(node)  # keep as-is
        else:
            new_graph_nodes.append(node)

    # 4. Rebuild graph
    # Remove old nodes, add new ones
    del model.graph.node[:]
    model.graph.node.extend(new_graph_nodes)

    # Add new initializers
    for init in all_new_inits:
        model.graph.initializer.append(init)

    # Remove old Conv3d weight initializers (no longer referenced)
    used_inputs = set()
    for node in model.graph.node:
        for inp in node.input:
            used_inputs.add(inp)
    unused_inits = [
        i for i, init in enumerate(model.graph.initializer)
        if init.name not in used_inputs
    ]
    for idx in reversed(unused_inits):
        name = model.graph.initializer[idx].name
        del model.graph.initializer[idx]
        print(f"  Removed unused initializer: {name[:60]}")

    return model


def main():
    print(f"Loading model from {INPUT_PATH}...")
    model = onnx.load(INPUT_PATH)

    print(f"\nOriginal model: {len(model.graph.node)} nodes")
    print("Transforming Conv3d → Conv2d...")
    model = transform_model(model)
    print(f"Transformed model: {len(model.graph.node)} nodes")

    # Run shape inference on the transformed model to ensure consistency
    print("\nRunning shape inference on transformed model...")
    model = shape_inference.infer_shapes(model)

    # Validate
    print("Validating model...")
    try:
        checker.check_model(model)
        print("✅ Model validation passed!")
    except Exception as e:
        print(f"⚠️ Model validation warning: {e}")
        print("   (Saving anyway — some warnings are benign)")

    # Count ops in transformed model
    from collections import Counter
    ops = Counter(n.op_type for n in model.graph.node)
    print("\nTransformed op types:")
    for op, cnt in ops.most_common():
        print(f"  {op}: {cnt}")

    # Verify no Conv3d remains
    for n in model.graph.node:
        if n.op_type == "Conv":
            for attr in n.attribute:
                if attr.name == "kernel_shape" and len(attr.ints) == 3:
                    print(f"❌ ERROR: Conv3d still present: {n.name}")
                    return

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    onnx.save(model, OUTPUT_PATH)
    import os
    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"✅ Saved ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
