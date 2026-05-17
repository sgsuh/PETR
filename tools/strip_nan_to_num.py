"""
Strip torch.nan_to_num's ONNX expansion (IsNaN / IsInf nodes) so the graph
imports cleanly into TensorRT 8.2.4, which lacks native IsInf / IsNaN ops.

The transform replaces each IsNaN / IsInf node's output with a constant
all-False tensor of the same shape. The downstream Where (used by
nan_to_num to swap NaN/+Inf/-Inf for finite fallbacks) then always picks
the original input branch, leaving inference output unchanged whenever
the upstream tensor is in fact finite — which it is for trained PETR.
"""

import argparse

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def strip(src, dst):
    graph = gs.import_onnx(onnx.load(src))
    targets = [n for n in graph.nodes if n.op in ("IsInf", "IsNaN")]

    if not targets:
        print(f"no IsInf / IsNaN nodes in {src}")
        onnx.save(gs.export_onnx(graph), dst)
        return

    print(f"stripping {len(targets)} IsInf / IsNaN nodes")

    for node in targets:
        out_var = node.outputs[0]
        shape = out_var.shape

        if shape is None or any(s is None or isinstance(s, str) for s in shape):
            shape = node.inputs[0].shape

        const = gs.Constant(
            name=f"{out_var.name}_false_const",
            values=np.zeros(shape, dtype=np.bool_),
        )

        for consumer in list(out_var.outputs):
            for i, inp in enumerate(consumer.inputs):
                if inp is out_var:
                    consumer.inputs[i] = const

        node.outputs.clear()

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), dst)
    print(f"saved: {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    strip(args.src, args.dst)


if __name__ == "__main__":
    main()
