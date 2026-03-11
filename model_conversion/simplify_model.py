"""Analyze the simplified ONNX model to understand the size bloat."""
import onnx
from onnx import numpy_helper
import numpy as np

model = onnx.load("/data/efficientphys_sim.onnx")

print("=== Simplified model analysis ===")
print("Nodes: {}".format(len(model.graph.node)))

# Analyze initializers (where onnxsim moved the constants)
inits_by_size = []
for init in model.graph.initializer:
    arr = numpy_helper.to_array(init)
    size_mb = arr.nbytes / (1024 * 1024)
    inits_by_size.append((init.name, list(arr.shape), str(arr.dtype), size_mb))

inits_by_size.sort(key=lambda x: -x[3])
print("\nTop 20 largest initializers:")
for name, shape, dtype, size_mb in inits_by_size[:20]:
    print("  {:.2f} MB  {}  shape={}  dtype={}".format(size_mb, name[:60], shape, dtype))

total = sum(x[3] for x in inits_by_size)
print("\nTotal initializer size: {:.1f} MB".format(total))
print("Initializer count: {}".format(len(inits_by_size)))

# Analyze node types
from collections import Counter
op_counts = Counter(n.op_type for n in model.graph.node)
print("\nNode types:")
for op, count in op_counts.most_common():
    print("  {} x {}".format(op, count))
