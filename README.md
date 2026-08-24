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

## Install

```sh
npm install @yourdevice/engine
```

`@huggingface/transformers` comes along as a dependency; it is what actually
runs the ONNX graph. Nothing else is required, and there is no server component.

## Quick start

The model runs in a Web Worker. `EngineClient` is the typed handle to it, and it
takes a **worker factory**, not a worker — when a backend initializes and then
fails, the client throws that worker away and builds a fresh one to retry the
next backend down the ladder (a worker whose WASM init has failed cannot be
reused).

```ts
import {
  EngineClient,
  buildCapabilityReport,
  pickDefaultModel,
  resampleTo16kMono,
  toSrt,
} from '@yourdevice/engine';

export async function transcribeFile(file: File): Promise<string> {
  const engine = new EngineClient(
    () => new Worker(new URL('@yourdevice/engine/worker', import.meta.url), { type: 'module' }),
  );

  // Weights download on first run and come from the browser cache after that.
  engine.onDownloadProgress = ({ loadedBytes, totalBytes }) => {
    console.log(`model ${Math.round((loadedBytes / totalBytes) * 100)}%`);
  };

  const report = buildCapabilityReport({ navigator, WebAssembly });
  const capabilities = await engine.init(pickDefaultModel(report));
  console.log('running on', capabilities.active); // 'webnn' | 'webgpu' | 'wasm'

  // Whisper wants 16 kHz mono. Decode with the Web Audio API, then resample.
  const decoded = await new AudioContext().decodeAudioData(await file.arrayBuffer());
  const pcm = resampleTo16kMono({
    channelData: Array.from({ length: decoded.numberOfChannels }, (_, i) =>
      decoded.getChannelData(i),
    ),
    sampleRate: decoded.sampleRate,
  });

  const segments = await engine.transcribe(
    pcm,
    { task: 'transcribe', language: 'en' }, // omit `language` to auto-detect
    { onPartial: (partial, done, total) => console.log(`${done}/${total}s`, partial.length) },
  );

  engine.dispose();
  return toSrt(segments);
}
```

`transcribe` resolves to `TranscriptSegment[]` — `{ start, end, text, speaker? }`,
seconds from the start of the input. `toSrt`, `toVtt`, `toTxt` and `toJson` turn
that array into a file.

Every call reports its own `requestId` through an `onRequestStart` callback;
pass it to `engine.abort(requestId)` to cancel — including mid-download, which
is the only way to actually stop model weights from arriving.

## How it picks a backend

Feature detection tells you what a browser _claims_; it does not tell you what
will work. This engine treats detection as a filter on what to **try**, and only
a successful initialization selects the active backend.

1. **`detectBackends`** sniffs `navigator.ml`, `navigator.gpu` and
   `WebAssembly`. Cheap, synchronous, and never trusted on its own — WebNN in
   particular can be present and still fail to build the Whisper graph.
2. **`planBackendOrder`** merges any caller preference into the ladder
   `webnn → webgpu → wasm`.
3. The worker **try-initializes** each candidate in turn. A build failure demotes
   to the next one.
4. **Output validation.** Some GPUs and drivers initialize fine and then emit
   garbage — a quantized decoder that silently produces `"!!!!"` forever. After an
   accelerated backend builds, the worker runs a two-second bundled speech clip
   through it and checks the transcript (`isDegenerateGeneration`). A backend that
   is present, initialized, and _wrong_ falls back to the WASM floor. This is the
   failure mode device allowlists miss, and it was found in production, not in CI.

WASM is trusted and never probed: it is the floor, so there is nothing below it
to demote to.

`capabilities.active` on the resolved `init` tells you where you actually landed.

## Models

| id                | size (approx) | languages    |
| ----------------- | ------------- | ------------ |
| `whisper-tiny`    | 40 MB         | multilingual |
| `whisper-base`    | 80 MB         | multilingual |
| `distil-small.en` | 120 MB        | English only |
| `whisper-small`   | 250 MB        | multilingual |

Weights are the ONNX exports published by `onnx-community` on the Hugging Face
Hub, fetched on first use and cached by the browser.

`pickDefaultModel(report)` returns `whisper-base` for phones and for machines
without an accelerated backend, and `whisper-small` only where WebGPU/WebNN is
present _and_ the device reports at least 8 GB of memory. Precision differs per
backend: accelerated backends use a float encoder with a q4 decoder, WASM uses
q8. int8 on the WebGPU execution provider is not used — that is the combination
that produced the garbage output above.

## Speaker diarization

Pass `diarize: true` to `transcribe` and each segment comes back labelled
`"Speaker 1"`…`"Speaker N"`. It runs pyannote segmentation plus WeSpeaker
embeddings, clustered agglomeratively, on the same device the ASR settled on.

It is strictly opt-in and off the fast path: when you never ask for it, the two
extra models are never downloaded and the transcription path is byte-identical.
It pulls down two more models and makes a second pass over the whole file, so
expect it to take noticeably longer.

## Serving your own model mirror

On first init the worker sends one `HEAD` request to
`/models/<hf-repo>/resolve/main/config.json` on its own origin. If that responds,
it fetches all weights from `/models/` instead of the Hugging Face Hub; if it
404s or the request fails, the Hub default stays in place and nothing else
changes.

Worth doing if you ship this at any scale: the Hub's edge returns 503 to browser
requests from some free-hosted origins, and third-party model CDNs show up on
ad-blocker filter lists. Same-origin traffic dodges both. Mirror the Hub's own
path layout (`{model}/resolve/{revision}/`) and it is a straight copy.

## Limits

Read these before building on it.

- **Browser only.** It targets a Web Worker: `WebWorker` lib, no DOM, no Node
  build. There is no server-side or CLI mode.
- **No cross-origin isolation, so WASM is single-threaded.** The engine
  deliberately does not require COOP/COEP headers, which keeps it embeddable but
  rules out `SharedArrayBuffer` and multi-threaded WASM. The WASM floor works,
  but it is markedly slower than an accelerated backend and a long recording on
  it takes real patience. Read `capabilities.active` and tell the user which one
  they got.
- **Pinned to a transformers.js/ORT pair.** Validated against
  `@huggingface/transformers` 4.2.x with `onnxruntime-web` 1.27.0. Whisper's
  quantized kernels have regressed between onnxruntime-web builds before — a
  q8 WASM decode that returns fluent nonsense is a real observed failure, and it
  passes every type check and unit test. If you float those versions, transcribe
  real speech and read the output before you ship.
- **Accelerated backends cannot be validated in CI.** GPU-less runners exercise
  the WASM path only. Changes to dtypes, pipeline options or the ladder need a
  real GPU and a real recording.
- **You bring the audio decoding.** The engine takes 16 kHz mono
  `Float32Array` PCM. `resampleTo16kMono` handles downmix and rate conversion
  (linear interpolation); getting from a container to raw samples is
  `AudioContext.decodeAudioData`'s job, in your code, because the engine holds no
  DOM types.
- **Long files are windowed.** Audio is chunked and de-duplicated at the seams.
  Segment boundaries near a window edge can move by a word.
- **It is Whisper.** Same accuracy characteristics, same failure modes:
  hallucinated text over silence and music, weaker performance on heavy accents
  and low-resource languages, no punctuation guarantees.
- **Pre-1.0.** The API is in use in production but not frozen. Pin an exact
  version.

## Third-party content

- **Model weights** are not bundled. They are downloaded at runtime from
  `onnx-community` on the Hugging Face Hub: Whisper (MIT, OpenAI),
  `distil-small.en` (MIT), and for diarization `pyannote-segmentation-3.0` and
  `wespeaker-voxceleb-resnet34-LM`. Check each repo's own licence before
  redistributing weights or mirroring them yourself.
- **The bundled validation clip** in `validationClip.ts` is two seconds of
  16 kHz mono PCM from John F. Kennedy's 1961 inaugural address — a work of the
  United States federal government, in the public domain — the same sample
  whisper.cpp and transformers.js use in their examples. It exists only so the
  worker can check that an accelerated backend produces real words, and it is the
  only audio the package ships.
- **Everything else in this package** is MIT, © Wangle Media. See `LICENSE`.

## API

`EngineClient`, `EngineError` · `detectBackends`, `buildCapabilityReport`,
`planBackendOrder`, `BACKEND_LADDER` · `MODELS`, `isModelId`, `pickDefaultModel` ·
`resampleTo16kMono` · `diarize`, `assignSpeakers`, `speakerCount`,
`DEFAULT_DIARIZE_CONFIG` · `agglomerative`, `cosineDistance` ·
`toSrt`, `toVtt`, `toTxt`, `toJson` · `isHostMessage`, `isWorkerMessage` ·
`distinctNgramRatio`, `isDegenerateGeneration`.

Types ship with the package. Every export has a doc comment explaining not just
what it does but why it is shaped that way — the source is in the tarball and is
worth reading.

## License

MIT. See [LICENSE](./LICENSE).
