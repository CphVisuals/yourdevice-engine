/**
 * Pure, worker-independent core of one backend attempt inside
 * worker.ts's `handleInit`: build a pipeline for a candidate backend via an
 * injected factory, then — for accelerated backends only — run it once
 * against the bundled validation clip (validationClip.ts) and treat a
 * degenerate transcript exactly like a graph-build failure.
 *
 * Why this exists as its own module: a WebGPU/WebNN backend can pass its
 * runtime handshake *and* build the ONNX graph successfully, and still emit
 * garbage output ("!!!!" repeated, confirmed on real hardware in
 * production) on some devices — a failure mode a `pipeline()` call alone
 * cannot see. `attemptBackend` is the single place that decides "did this
 * backend actually work", so `handleInit` can report it through the exact
 * same `model-load-failed`-with-`backend` path already used for build
 * failures (see worker.ts and client.ts `EngineClient.init`'s cross-worker
 * retry) — a bad-GPU device transparently falls back to WASM.
 *
 * WASM is trusted and never probed (CLAUDE.md rule 9's incident, and this
 * one, are both accelerated-backend-only; the WASM floor has no known
 * garbage-output failure mode, and probing it would only add latency).
 *
 * Extracted out of worker.ts so it is unit-testable with an injected fake
 * pipeline/ASR runner (CLAUDE.md rule 3: pure logic separated from browser
 * globals) — worker.ts itself is a side-effecting entry point
 * (`addEventListener`/`postMessage` run on import) that cannot be imported
 * directly into a test file.
 */
import type {
  AutomaticSpeechRecognitionOutput,
  AutomaticSpeechRecognitionPipeline,
} from '@huggingface/transformers';
import type { BackendId } from './backends.js';
import { isDegenerateOutput } from './degenerateOutput.js';
import { getValidationClip } from './validationClip.js';

/** Builds (or rejects while building) a pipeline for one backend candidate. */
export type PipelineBuilder = (backend: BackendId) => Promise<AutomaticSpeechRecognitionPipeline>;

export type BackendAttemptResult =
  { ok: true; asr: AutomaticSpeechRecognitionPipeline } | { ok: false; error: unknown };

function firstResult(
  output: AutomaticSpeechRecognitionOutput | AutomaticSpeechRecognitionOutput[],
): AutomaticSpeechRecognitionOutput | undefined {
  return Array.isArray(output) ? output[0] : output;
}

/**
 * Attempts one backend candidate: builds its pipeline via `buildPipeline`,
 * then — unless `backend` is `'wasm'` — runs it once against `clip` and
 * rejects the attempt if the transcript is degenerate. `clip` defaults to
 * the bundled validation clip but is injectable so callers (and tests)
 * never have to touch the real bundled audio.
 */
export async function attemptBackend(
  backend: BackendId,
  buildPipeline: PipelineBuilder,
  clip: Float32Array = getValidationClip(),
): Promise<BackendAttemptResult> {
  let asr: AutomaticSpeechRecognitionPipeline;
  try {
    asr = await buildPipeline(backend);
  } catch (error) {
    return { ok: false, error };
  }

  if (backend === 'wasm') {
    return { ok: true, asr };
  }

  try {
    const output = await asr(clip, { return_timestamps: false });
    const text = firstResult(output)?.text ?? '';
    if (isDegenerateOutput(text)) {
      return {
        ok: false,
        error: new Error(
          `${backend}: pipeline built but produced degenerate output on the validation clip (${JSON.stringify(text)})`,
        ),
      };
    }
    return { ok: true, asr };
  } catch (error) {
    return { ok: false, error };
  }
}
