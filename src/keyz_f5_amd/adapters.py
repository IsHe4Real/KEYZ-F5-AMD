"""Verify a real DirectML MatMul on the discrete adapter, without TTS claims."""
import ctypes as c
from ctypes import wintypes as w
import json
from pathlib import Path
import time
import uuid

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent

class Desc(c.Structure):
    _fields_ = [('Description', w.WCHAR * 128), ('VendorId', w.UINT),
                ('DeviceId', w.UINT), ('SubSysId', w.UINT), ('Revision', w.UINT),
                ('DedicatedVideoMemory', c.c_size_t), ('DedicatedSystemMemory', c.c_size_t),
                ('SharedSystemMemory', c.c_size_t), ('LuidLow', w.DWORD),
                ('LuidHigh', w.LONG), ('Flags', w.UINT)]

def method(obj, index, result, *args):
    table = c.cast(obj, c.POINTER(c.POINTER(c.c_void_p))).contents
    return c.WINFUNCTYPE(result, c.c_void_p, *args)(table[index])

def adapters():
    factory = c.c_void_p()
    iid = (c.c_ubyte * 16).from_buffer_copy(uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
    create = c.WinDLL('dxgi').CreateDXGIFactory1
    create.argtypes = [c.c_void_p, c.POINTER(c.c_void_p)]
    create.restype = w.LONG
    if create(c.byref(iid), c.byref(factory)) != 0:
        raise RuntimeError('Cannot enumerate DXGI adapters')
    rows = []
    try:
        index = 0
        while True:
            adapter = c.c_void_p()
            status = method(factory, 12, w.LONG, w.UINT, c.POINTER(c.c_void_p))(factory, index, c.byref(adapter))
            if status != 0:
                if status & 0xffffffff != 0x887A0002:
                    raise RuntimeError('DXGI enumeration failed')
                break
            try:
                desc = Desc()
                if method(adapter, 10, w.LONG, c.POINTER(Desc))(adapter, c.byref(desc)) != 0:
                    raise RuntimeError('Cannot read adapter identity')
                rows.append({'device_id': index, 'name': desc.Description, 'vendor': desc.VendorId,
                             'pci_device': desc.DeviceId, 'dedicated_bytes': desc.DedicatedVideoMemory})
            finally:
                method(adapter, 2, w.ULONG)(adapter)
            index += 1
    finally:
        method(factory, 2, w.ULONG)(factory)
    return rows

def main():
    rows = adapters()
    target = next(row for row in rows if row['vendor'] == 0x1002 and row['pci_device'] == 0x73bf)
    if 'DmlExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('DirectML provider missing')
    graph = helper.make_graph([helper.make_node('MatMul', ['a', 'b'], ['out'])], 'dml-smoke',
                             [helper.make_tensor_value_info(name, TensorProto.FLOAT, [256,256]) for name in ['a','b']],
                             [helper.make_tensor_value_info('out', TensorProto.FLOAT, [256,256])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
    onnx.checker.check_model(model)
    opts = ort.SessionOptions()
    opts.enable_mem_pattern = False
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    opts.enable_profiling = True
    opts.profile_file_prefix = str(ROOT / 'dml-smoke-profile')
    session = ort.InferenceSession(model.SerializeToString(), sess_options=opts,
                                   providers=[('DmlExecutionProvider', {'device_id': str(target['device_id'])})])
    session.disable_fallback()
    a = np.full((256,256), 0.25, np.float32)
    b = np.full((256,256), 0.5, np.float32)
    seconds=[]
    for _ in range(3):
        started=time.perf_counter()
        result=session.run(None, {'a':a, 'b':b})[0]
        seconds.append(time.perf_counter()-started)
        np.testing.assert_allclose(result, 32.0, rtol=1e-5)
    profile=Path(session.end_profiling())
    providers=sorted({event.get('args',{}).get('provider') for event in json.loads(profile.read_text()) if event.get('args',{}).get('provider')})
    if 'DmlExecutionProvider' not in providers or 'CPUExecutionProvider' in providers:
        raise RuntimeError('Profile did not verify exclusive DirectML compute')
    report={'result':'PASS', 'scope':'MatMul only; F5 speech not yet tested', 'adapters':rows,
            'selected_adapter':target, 'ort_version':ort.__version__, 'available_providers':ort.get_available_providers(),
            'profile_providers':providers,'run_seconds':seconds,'profile_file':profile.name}
    (ROOT/'directml-smoke-result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__ == '__main__':
    main()
