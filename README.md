# KEYZ-F5-AMD

Experimental, installable Windows DirectML runtime for F5-TTS ONNX.
Verified locally on an AMD Radeon RX 6900 XT. Other RX 6000 cards are candidates
for testing, **not certified supported hardware**. No Linux/ROCm or all-GPU claim:
the transformer uses DirectML; preprocessing and decoding use the CPU.

This is an early developer release, not a one-click voice application. It does
not include KEYZ/Aurora, browser playback, microphone capture or a network server.

## What changed

- Persistent sessions and reference preparation across sequential requests.
- Separate alternating GPU input/output buffers; inputs are never overwritten
  by outputs during a step.
- Fixed, experimentally coarsened 13-step full trajectory, deterministic seed,
  FP32 inference, and exact model/vocabulary hash checks.
- Bounded JSON-lines worker with request IDs and duplicate protection within
  one process. No automatic services, driver installation or telemetry.

The project's accepted voice is not distributed. Supply a reference recording
you have permission to use, with the exact corresponding transcript.

## Install (Windows, Python 3.12 x64)

Use a separate environment, not an existing Qwen or ComfyUI environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\keyz-f5.exe --help
.\.venv\Scripts\keyz-f5.exe adapters
```

Installation downloads the pinned Python dependencies from the configured
package index. It does not download models or replace graphics drivers.
Do not install `onnxruntime` or `onnxruntime-gpu` over `onnxruntime-directml`
in this environment. Pin availability and installation on a completely clean
second PC still require acceptance testing.

## Obtain compatible model assets

This runtime deliberately rejects arbitrary or newer exports. Use the reviewed
[community export](https://huggingface.co/H5N1AIDS/F5-TTS-ONNX/tree/d3af73ed2db92509a45e3edd4bf43d281306b013),
revision `d3af73ed2db92509a45e3edd4bf43d281306b013`, `CPU_F32.zip`.
Archive SHA256: `09c3b2cf37fc51db6d42cafc5d257b457f389209decb8341dedba446d647f0d5`.
Check it before extracting; never execute scripts from the archive.
Review the archive paths before extraction. Place the original FP32 transformer,
preprocess and decode ONNX files outside the repository.

Obtain `F5TTS_v1_Base/vocab.txt` from
[SWivid's pinned model revision](https://huggingface.co/SWivid/F5-TTS/tree/84e5a410d9cead4de2f847e7c9369a6440bdfaca).
Its checksum is checked by `import-assets`.

```powershell
keyz-f5 prepare13 --source original\F5_Transformer.onnx --destination variant13\F5_Transformer.onnx
keyz-f5 import-assets --preprocess original\F5_Preprocess.onnx --transformer variant13\F5_Transformer.onnx --decode original\F5_Decode.onnx --vocab original\vocab.txt --destination models
keyz-f5 verify-assets --models models
```

Run these commands from the activated environment, or use their full `.venv`
executable paths. Bundle creation refuses an existing destination. It copies
assets without changing the originals. Models remain local and ignored by Git.
Expected per-file hashes are in `src/keyz_f5_amd/assets.py`.

## Speak

Use a short, clean reference (approximately 5–10 seconds; hard maximum 15 seconds)
and an exact matching ASCII transcript in a UTF-8 text file. Start with a short
ASCII phrase. Unknown vocabulary and estimated context above 1600 frames fail
explicitly; text is never silently truncated. Maximum input per request is 180
characters; some shorter inputs may exceed the context bound for your reference.

```powershell
keyz-f5 speak --models models --reference private\reference.wav --transcript private\transcript.txt --output-dir outputs --text "Ready when you are."
```

This writes a WAV and timing metadata; it does not play audio. `speak` is a
single cold process. For persistent warm generation use the worker or Python API:

```powershell
keyz-f5 worker --models models --reference private\reference.wav --transcript private\transcript.txt --output-dir outputs
```

Then send one JSON object per line, sequentially:

```json
{"reply_id":"example-01","text":"Ready when you are."}
```

The worker reports readiness, then a WAV filename and timings per request. Same
ID and same text returns the prior result without regenerating. Reusing an ID
for different text is rejected. IDs must be 1–40 lowercase letters, digits or
hyphens. At most 256 input lines per worker / 256 cached results per engine.
EOF exits. Generated WAVs are retained in the explicit output folder; manage
them yourself. This package has no automatic file deletion.

```python
from keyz_f5_amd.engine import Engine
engine = Engine("demo", "models", "private/reference.wav",
                "private/transcript.txt", "outputs")
try:
    print(engine.synthesize({"reply_id": "demo-01", "text": "Ready when you are."}))
finally:
    engine.close()
```

One sequential owner only; simultaneous synthesis is rejected. Cancellation
does not stop a GPU step mid-flight. An embedding application must discard stale
results and stop its player separately. Do not call `close()` during synthesis.

If no default compatible AMD adapter is found, use `adapters`, then explicitly
select `--device N` (or `device_id=N`). Other AMD adapters are experimental;
the PCI family identifier alone does not establish quality or performance.

## Results and limits

The original integration check produced two WAVs byte-identical to its accepted
13-step grouped sample. Cold file-ready time was 19.260 seconds, then 3.759 seconds
warm for the next 94-character segment on the same worker. These are two samples,
not a guarantee, not phone first-audio latency, and not a controlled Qwen speedup
comparison. The upstream model and its converter deserve credit for the model.

The standalone package also passed local wheel installation, asset recreation,
seven unit tests and two byte-identical audio parity checks. See `VALIDATION.md`
for actual package results and untested areas.
No other GPU, OS, fresh PC, long-reference quality, or production reliability
certification is implied. This is completed-segment output, not token streaming.

## Tests and removal

```powershell
python -m unittest discover -s tests -v
```

Stop the worker with EOF or Ctrl+C. Removing the isolated environment removes
the installed package; it does not remove your external models or recordings.
No KEYZ defaults or other engines are modified by installing this package.

## License and privacy

Code: Apache-2.0; see `LICENSE` and `NOTICE`.
[F5's pretrained weights have separate terms](https://github.com/SWivid/F5-TTS#license),
including the documented non-commercial restriction. Public code availability
does not grant commercial rights to weights or anyone's voice. Model downloads
go to their hosting provider; synthesis itself is local. No private voice,
Vault, settings, credentials or user artwork belong in this repository.
