# @yourdevice/engine

**Local-first, in-browser speech-to-text.** Whisper running entirely in the
browser — WebGPU where it works, a single-threaded WASM (SIMD) floor everywhere
else — with timestamps, speaker labels, and SRT/VTT/TXT/JSON export. **The audio
never leaves the device: nothing is uploaded.**

This is the engine behind the free, unlimited transcription at
**[yourdevice.app/transcribe](https://yourdevice.app/transcribe)** — try it there
(drop in a recording; turn your Wi‑Fi off after it loads and it still works).

MIT. Framework-free (no React, no DOM framework) so it drops into anything.

---

## Why it exists

Cloud transcription meters you because their GPU costs money per minute. Run the
model on the user's own device and the marginal cost is ~zero — so it can be
genuinely free and unlimited, and private by construction rather than by policy.

## What's interesting about it

- **A runtime ladder, not feature sniffing.** `detectBackends` reports what
  _might_ work; the worker proves it by try-initializing each backend
  (`planBackendOrder`) until one actually builds the graph. A backend can exist
  and still fail — detection is never trusted on its own.
- **Output validation, not just init checks.** Some GPUs/drivers initialize fine
  and then emit garbage (a quantized decoder that silently produces `!!!!`). The
  engine validates real output at init and demotes to the WASM floor when a
  backend is present but wrong — the failure mode device allowlists miss.
- **No cross-origin isolation required.** It runs single-threaded WASM rather
  than demanding COOP/COEP, so it stays embeddable (no SharedArrayBuffer
  prerequisite).
- **Typed worker protocol.** `HostMessage`/`WorkerMessage` unions with runtime
  guards (`isHostMessage`/`isWorkerMessage`) on both sides of `postMessage` —
  never trust a raw message.
- **Sensible defaults per device.** `whisper-base` (multilingual) everywhere;
  the heavier `whisper-small` only where an accelerated backend and enough
  memory are actually present.

Also included: on-device **speaker diarization** (segmentation + embeddings +
agglomerative clustering) and **exporters** for SRT, VTT, TXT and JSON.

## Usage

The model runs in a Web Worker; `EngineClient` is the typed handle to it.

```ts
import { EngineClient, toSrt } from '@yourdevice/engine';

// The worker is a dedicated entry point (it registers postMessage handlers).
const worker = new Worker(new URL('@yourdevice/engine/worker', import.meta.url), {
  type: 'module',
});
const engine = new EngineClient(worker);

await engine.init({ onDownloadProgress: (p) => console.log('model', p) });

const { segments } = await engine.transcribe(float32PcmMono16k, {
  language: 'en',
  onSegment: (s) => console.log(s.start, s.text),
});

console.log(toSrt(segments));
```

Feed it 16 kHz mono PCM — `resampleTo16kMono` will convert a decoded
`AudioBuffer` for you. Model weights download on first run and are cached.

## Status

Powering production transcription today; the public API is stabilizing toward a
1.0 and an npm publish under `@yourdevice/engine`. Until then this repository is
the source mirror — pin a commit if you build on it.

## License

MIT.
