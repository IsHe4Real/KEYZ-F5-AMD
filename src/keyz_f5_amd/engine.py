"""Single-owner persistent 13-step engine. Models and voice references are external."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time
import threading

import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly
import soundfile as sf

from .adapters import adapters
from .assets import verify_assets


def validate_request(request):
    if not isinstance(request, dict):
        raise ValueError('Request must be an object')
    identifier = request.get('reply_id')
    text = request.get('text')
    if not isinstance(identifier, str) or not re.fullmatch(r'[a-z0-9-]{1,40}', identifier):
        raise ValueError('Invalid reply identifier')
    if not isinstance(text, str) or not text.strip() or len(text) > 180 or not text.isascii():
        raise ValueError('Use one to 180 ASCII characters')
    return identifier, text


class Engine:
    def __init__(self, label, models, reference, transcript, output_dir, device_id=None,
                 optimization='basic', profiling=False, fixed_frames=None):
        if optimization not in {'basic', 'all'}:
            raise ValueError('Unknown optimization level')
        if fixed_frames is not None and (type(fixed_frames) is not int or not 1 <= fixed_frames <= 1600):
            raise ValueError('Invalid bounded exact frame count')
        self.optimization = optimization
        self.profiling = profiling
        self.fixed_frames = fixed_frames
        if not re.fullmatch(r'[a-z0-9-]{1,40}', label):
            raise ValueError('Invalid engine label')
        self.closed = False
        self.results = {}
        self.label = label
        self.owner = threading.Lock()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        models = Path(models).resolve()
        started = time.perf_counter()
        verify_assets(models)
        choices = [a for a in adapters() if a['vendor'] == 0x1002]
        candidates = [a for a in choices if a['device_id'] == device_id] if device_id is not None else [a for a in choices if a['pci_device'] == 0x73bf]
        if not candidates:
            raise ValueError('No selected AMD adapter. Use adapters, then explicitly select an experimental device.')
        self.target = candidates[0]
        self.sessions = {}
        paths = {'Preprocess': models / 'F5_Preprocess.onnx',
                 'Transformer': models / 'F5_Transformer.onnx', 'Decode': models / 'F5_Decode.onnx'}
        load_start = time.perf_counter()
        for name, path in paths.items():
            options = ort.SessionOptions()
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.intra_op_num_threads = 6
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            if name == 'Transformer' and optimization == 'all':
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if name == 'Transformer' and fixed_frames is not None:
                options.add_free_dimension_override_by_name('max_duration', fixed_frames)
            options.enable_profiling = profiling
            options.profile_file_prefix = str(self.output_dir / f'{label}-{name}')
            providers = [('DmlExecutionProvider', {'device_id': str(self.target['device_id'])}), 'CPUExecutionProvider'] if name == 'Transformer' else ['CPUExecutionProvider']
            session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            session.disable_fallback()
            self.sessions[name] = session
        self.model_load_seconds = time.perf_counter() - load_start
        if self.sessions['Transformer'].get_providers()[0] != 'DmlExecutionProvider':
            raise RuntimeError('DirectML not selected')
        ref_start = time.perf_counter()
        audio, rate = sf.read(Path(reference), dtype='float32')
        self.ref_text = re.sub(r'\s+', ' ', Path(transcript).read_text(encoding='utf-8')).strip() + ' '
        if not 1 <= len(self.ref_text) <= 1000 or not self.ref_text.isascii():
            raise ValueError('Use an exact matching ASCII reference transcript up to 1000 characters')
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if rate != 24000:
            divisor = math.gcd(rate, 24000)
            audio = resample_poly(audio, 24000 // divisor, rate // divisor)
        peak = float(np.max(np.abs(audio)))
        if not peak or not np.isfinite(audio).all() or len(audio) / 24000 > 15:
            raise ValueError('Invalid or oversized reference')
        self.ref_frames = len(audio) // 256 + 1
        self.audio = (audio * (32767 / peak)).astype(np.int16).reshape(1, 1, -1)
        if self.sessions['Preprocess'].get_inputs()[0].type == 'tensor(float)':
            self.audio = self.audio.astype(np.float32)
        self.vocab = {line.rstrip('\n\r'): i for i, line in enumerate((models / 'vocab.txt').open(encoding='utf-8'))}
        self.reference_prepare_seconds = time.perf_counter() - ref_start
        self.initialization_seconds = time.perf_counter() - started
        self.session_load_count = 3
        self.reference_prepare_count = 1

    def synthesize(self, request):
        if not self.owner.acquire(blocking=False):
            raise RuntimeError('Engine is busy; use one sequential owner')
        try:
            return self._synthesize(request)
        finally:
            self.owner.release()

    def _synthesize(self, request):
        if self.closed:
            raise RuntimeError('Engine closed')
        identifier, text = validate_request(request)
        if identifier in self.results:
            previous = self.results[identifier]
            if previous['text'] != text:
                raise ValueError('Reply identifier already used with different text')
            return dict(previous, duplicate=True)
        if len(self.results) >= 256:
            raise ValueError('Session limit reached; close and create a fresh engine')
        path = self.output_dir / f'{self.label}-{identifier}.wav'
        if path.exists():
            raise ValueError('Refusing existing output overwrite')
        started = time.perf_counter()
        combined = self.ref_text + text
        if any(char not in self.vocab for char in combined):
            raise ValueError('Unknown vocabulary character')
        duration = self.ref_frames + int(self.ref_frames / len(self.ref_text.encode()) * len(text.encode()))
        if duration > 1600:
            raise ValueError('Estimated context exceeds unchanged 1600-frame limit')
        if self.fixed_frames is not None and duration != self.fixed_frames:
            raise ValueError('Text does not match this exact-size session; no padding or truncation allowed')
        text_ids = np.array([[self.vocab[char] for char in combined]], dtype=np.int32)
        pre = self.sessions['Preprocess']
        ort.set_seed(9527)
        values = pre.run(None, dict(zip([i.name for i in pre.get_inputs()], [self.audio, text_ids, np.array([duration], dtype=np.int64)])))
        phases = {'text_and_preprocess_seconds': time.perf_counter() - started}
        phase_start = time.perf_counter()
        if len(values) != 8:
            raise RuntimeError('Unexpected preprocessing contract')
        noise, *conditioning, ref_len = values
        noise = np.random.default_rng(9527).standard_normal(noise.shape).astype(noise.dtype)
        step = np.array([0], dtype=np.int32)
        transformer = self.sessions['Transformer']
        names = [i.name for i in transformer.get_inputs()]
        tensors = [ort.OrtValue.ortvalue_from_numpy(v, 'dml', self.target['device_id']) for v in [noise, *conditioning, step]]
        binding = transformer.io_binding()
        for name, value in zip(names, tensors):
            binding.bind_ortvalue_input(name, value)
        current = [tensors[0], tensors[-1]]
        future = [ort.OrtValue.ortvalue_from_numpy(np.zeros_like(v), 'dml', self.target['device_id']) for v in [noise, step]]
        phases['noise_and_binding_seconds'] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        for _ in range(13):
            binding.bind_ortvalue_input(names[0], current[0])
            binding.bind_ortvalue_input(names[-1], current[1])
            for output, value in zip(transformer.get_outputs(), future):
                binding.bind_ortvalue_output(output.name, value)
            transformer.run_with_iobinding(binding)
            current, future = future, current
        noise = current[0].numpy()
        # Includes GPU completion wait: not a misleading host-launch-only timer.
        phases['transformer_and_download_seconds'] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        decode = self.sessions['Decode']
        generated = decode.run(None, dict(zip([i.name for i in decode.get_inputs()], [noise, ref_len])))[0].reshape(-1)
        if not generated.size or not np.isfinite(generated).all() or not np.any(generated):
            raise RuntimeError('Invalid generated audio')
        phases['decode_and_validate_seconds'] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        sf.write(path, generated, 24000, subtype='PCM_16')
        phases['write_seconds'] = time.perf_counter() - phase_start
        result = {'reply_id': identifier, 'text': text, 'output': path.name,
                  'file_ready_seconds': time.perf_counter() - started,
                  'audio_seconds': len(generated) / 24000,
                  'conditioning_sha256': hashlib.sha256(b''.join(v.tobytes() for v in conditioning)).hexdigest(),
                  'duplicate': False, 'session_load_count': self.session_load_count,
                  'reference_prepare_count': self.reference_prepare_count}
        result['phases'] = phases
        result['optimization'] = self.optimization
        result['fixed_frames'] = self.fixed_frames
        result['profiling'] = self.profiling
        self.results[identifier] = result
        return result

    def close(self):
        if self.closed:
            return {}
        self.closed = True
        profiles = {}
        for name, session in self.sessions.items():
            if not self.profiling:
                profiles[name] = {'file': None, 'operators': {}, 'notice': 'Profiling disabled; no runtime placement proof for this run'}
                continue
            path = Path(session.end_profiling())
            counts = Counter(e.get('args', {}).get('provider') for e in json.loads(path.read_text()) if e.get('args', {}).get('provider'))
            profiles[name] = {'file': path.name, 'operators': dict(counts)}
        self.sessions.clear()
        return profiles
