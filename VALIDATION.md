# Validation — 2026-10-15

Status: experimental developer alpha `0.1.0a1`, Windows Python 3.12.14 x64.
GPU: AMD Radeon RX 6900 XT. No other card has been tested by this project.

## Standalone package checks actually performed

- Built a wheel using the existing local build toolchain.
- Installed that wheel into an isolated target directory, with dependencies
  already available in the separate research environment. Existing Qwen and
  production app environments were not modified.
- Ran seven package unit tests: valid/invalid requests, asset import and original
  preservation, incompatible source rejection, missing/tampered assets, metadata
  import, and no-overwrite bundle behavior. All passed.
- Recreated the 13-step graph from the pinned original. Its full-file SHA256
  matched the tested variant in `assets.py`.
- Imported and verified all four model/vocabulary assets.
- Loaded the installed package, not a workspace source-path shortcut.
- Synthesized two 94-character segments with the same loaded engine. Both full
  WAV files matched the accepted reference runtime byte-for-byte.
- Repeating each request ID/text returned its prior output without regeneration.
- Closed the test engine afterward. No live services were restarted.

| Package measurement | Time |
| --- | ---: |
| Engine initialization, including asset checks | 2.202 s |
| First segment, already initialized | 4.290 s |
| Next segment, same engine | 3.691 s |

Each generated segment contained 7.019 seconds of audio. Two trials are not a
latency distribution. Python process startup, network delivery, autoplay and
phone playback are not included in these package generation times. Asset/model
files may already have been cached by Windows; this is not a fresh-PC cold test.

## Prior browser-adapter check (separate from package)

The original app adapter was tested through a temporary loopback HTTP server.
It reused a worker and returned two byte-identical accepted WAVs through its
existing protected audio endpoint. File-ready times: 19.260 s first request
(including worker/import/model startup) and 3.759 s next request. Explicit
worker unload passed. This does not prove playback on a user's phone.

The source application's 88 Python tests and 12 browser-logic scripts passed
using existing runtimes. Those are application regressions, not extra standalone
package tests. They do not certify every device's microphone or permissions.

## Still pending

- Completely fresh environment installation with dependency downloads.
- A second PC and other RX 6000-family GPUs.
- Browser/phone integration acceptance for this standalone package.
- Longer endurance, varied reference voices, multilingual support and detailed
  resource/failure distributions for the standalone distribution.
- Mid-inference cancellation, automated asset downloading and a one-click GUI.

The app's earlier GPU-placement profiles verified DirectML transformer execution
with CPU preprocessing/decoding. Profiling is disabled in these package parity
runs, so these are not independent new per-operator placement proofs.
