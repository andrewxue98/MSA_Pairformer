"""CPU-only reproductions for the screening review; not model validation.

Run from any directory with Python and PyTorch installed. No checkpoints,
network, CUDA work, or changes to model code are required. Extracted methods
are executed verbatim from the checkout to avoid optional model dependencies.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
from functools import partial
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]


def extract_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name))
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


def main():
    torch.set_num_threads(1)
    path = ROOT / "MSA_Pairformer/chunk_layer.py"
    spec = importlib.util.spec_from_file_location("audit_chunk_layer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    chunk_layer = module.chunk_layer
    findings = []

    def expect_error(name, call, error_type, message):
        try:
            call()
        except error_type as exc:
            if message not in str(exc):
                raise AssertionError(f"Unexpected error for {name}: {exc}") from exc
            findings.append({"probe": name, "confirmed": True, "exception": type(exc).__name__, "message": str(exc)})
        else:
            raise AssertionError(f"Expected upstream failure not observed: {name}")

    x = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    control = chunk_layer(lambda x: x + 1, {"x": x}, chunk_size=2, no_batch_dims=1)
    assert torch.equal(control, x + 1)
    findings.append({"probe": "chunk helper single-input control", "confirmed": True})
    expect_error("chunk helper low_mem references undefined _chunk_slice",
                 lambda: chunk_layer(lambda x: x, {"x": x}, 2, 1, low_mem=True), NameError, "_chunk_slice")
    expect_error("chunk helper multiplies dimensions across input keys",
                 lambda: chunk_layer(lambda x, y: x + y, {"x": x, "y": x}, 2, 1), RuntimeError, "size")
    x2 = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    expect_error("chunk helper ignores later batch dimensions",
                 lambda: chunk_layer(lambda x: x, {"x": x2}, 1, 2), RuntimeError, "shape")

    namespace = {"torch": torch, "partial": partial, "chunk_layer": chunk_layer}
    opm_chunk = extract_method(ROOT / "MSA_Pairformer/outer_product.py",
                              "PresoftmaxDifferentialOuterProductMean", "_chunk", namespace)
    def requires_denom(a, b, pair_denom):
        raise AssertionError("Should fail before the operation is entered")
    fake_opm = SimpleNamespace(_opm=requires_denom)
    a = torch.zeros(1, 3, 2, 2)
    expect_error("differential OPM chunk omits pair_denom",
                 lambda: opm_chunk(fake_opm, a, a, 2), TypeError, "pair_denom")

    msa = torch.zeros(1, 8, 3, 4)
    query_view = msa[:, :1]
    query_copy = query_view.clone()
    assert query_view.untyped_storage().nbytes() == msa.untyped_storage().nbytes()
    assert query_copy.untyped_storage().nbytes() == query_copy.numel() * query_copy.element_size()
    assert torch.equal(query_view, query_copy)
    findings.append({"probe": "query slice retains full MSA storage", "confirmed": True,
                     "logical_bytes": query_view.numel() * query_view.element_size(),
                     "view_storage_bytes": query_view.untyped_storage().nbytes(),
                     "copy_storage_bytes": query_copy.untyped_storage().nbytes()})

    notebook = json.loads((ROOT / "MSA_Pairformer_with_MMseqs2.ipynb").read_text())
    helper = None
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "prepare_sequences":
                helper = ast.Module(body=[node], type_ignores=[])
    assert helper is not None
    namespace = {"re": re}
    exec(compile(ast.fix_missing_locations(helper), "notebook:prepare_sequences", "exec"), namespace)
    with contextlib.redirect_stdout(io.StringIO()):
        default_chains, default_breaks = namespace["prepare_sequences"]("ACD:ACD:ACD:ACD")
        kept_chains, kept_breaks = namespace["prepare_sequences"]("ACD:ACD:ACD:ACD", remove_duplicates=False)
    assert len(default_chains) == 1 and default_breaks == []
    assert len(kept_chains) == 4 and kept_breaks == [3, 6, 9]
    findings.append({"probe": "notebook default collapses identical chains", "confirmed": True,
                     "default_chains": len(default_chains), "explicit_preserve_chains": len(kept_chains)})

    source_paths = ["MSA_Pairformer/chunk_layer.py", "MSA_Pairformer/outer_product.py",
                    "MSA_Pairformer/model.py", "MSA_Pairformer_with_MMseqs2.ipynb"]
    result = {"reviewed_upstream_commit": "875363570df1ae484cf725aba382790444223005",
              "torch_version": torch.__version__, "device": "cpu", "checkpoint_loaded": False,
              "scope": "Helper behavior and tensor-storage probes only; no full-model equivalence or GPU benchmarks",
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths},
              "findings": findings}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
