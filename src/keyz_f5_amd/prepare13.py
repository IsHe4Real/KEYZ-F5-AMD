"""Reproduce the tested full-trajectory 13-step graph; never modify the original."""
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
from .assets import HASHES, digest

ORIGINAL = 'c63aeb96953ccae551df6717d892f73a8408e59f1a6fe8edc2aed8063bd195b8'


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists(): raise ValueError('Refusing to overwrite an existing model')
    if digest(source) != ORIGINAL: raise ValueError('Pinned original model hash mismatch')
    model = onnx.load(source)
    tensors = {tensor.name:tensor for tensor in model.graph.initializer}
    delta_name, time_name = '/Constant_1_output_0', 'onnx::Gather_372'
    deltas = numpy_helper.to_array(tensors[delta_name]).copy()
    times = numpy_helper.to_array(tensors[time_name]).copy()
    if deltas.shape != (31,) or times.shape != (1,32,1024) or not np.isfinite(deltas).all() or not np.all(deltas>0) or not np.isclose(deltas.sum(),1,atol=1e-6):
        raise ValueError('Unexpected full-trajectory schedule')
    boundaries = np.rint(np.linspace(0,31,14)).astype(np.int64)
    coarse = np.array([deltas[a:b].sum(dtype=np.float32) for a,b in zip(boundaries[:-1],boundaries[1:])],dtype=deltas.dtype)
    tensors[delta_name].CopyFrom(numpy_helper.from_array(coarse,name=delta_name))
    tensors[time_name].CopyFrom(numpy_helper.from_array(times[:,boundaries,:].copy(),name=time_name))
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation prevents replacing user files, including the source.
    with destination.open('xb') as output: output.write(model.SerializeToString())
    if digest(destination) != HASHES['F5_Transformer.onnx']:
        raise ValueError('Export differs from tested graph; do not use this output')
    return {'steps':13,'sha256':HASHES['F5_Transformer.onnx']}
