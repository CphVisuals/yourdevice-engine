export type ModelId = 'whisper-base' | 'distil-small.en' | 'whisper-small' | 'whisper-tiny';

/**
 * Per-file dtype map for accelerated (WebGPU/WebNN) inference. int8 kernels
 * are unreliable and slow on the WebGPU execution provider — q8 there
 * produces garbage tokens (verified on real hardware 2026-07-17). The
 * canonical accelerated Whisper config is a float encoder + q4 decoder; the
 * encoder precision is per-model because fp32 encoders of the small-class
 * models exceed practical download/upload sizes, so they use fp16.
 */
export interface AcceleratedDtype {
  encoder_model: 'fp32' | 'fp16';
  decoder_model_merged: 'q4';
}

export interface ModelSpec {
  id: ModelId;
  /** Hugging Face repo the ONNX weights are pulled from (R2 mirror takes over at launch). */
  hfRepo: string;
  /** Approximate download size of the q8 weights, used for progress UI and defaults. */
  approxSizeMb: number;
  multilingual: boolean;
  label: string;
  /** Hidden from the model picker UI (e.g. a tiny model kept around for E2E/manual testing only). */
  hidden: boolean;
  /** Dtypes used on WebGPU/WebNN backends (WASM always uses q8). */
  acceleratedDtype: AcceleratedDtype;
}

export const MODELS: Record<ModelId, ModelSpec> = {
  'whisper-base': {
    id: 'whisper-base',
    hfRepo: 'onnx-community/whisper-base',
    approxSizeMb: 80,
    multilingual: true,
    label: 'Standard (all languages)',
    hidden: false,
    acceleratedDtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
  },
  'distil-small.en': {
    id: 'distil-small.en',
    hfRepo: 'onnx-community/distil-small.en',
    approxSizeMb: 120,
    multilingual: false,
    label: 'Fast (English only)',
    hidden: false,
    acceleratedDtype: { encoder_model: 'fp16', decoder_model_merged: 'q4' },
  },
  'whisper-small': {
    id: 'whisper-small',
    hfRepo: 'onnx-community/whisper-small',
    approxSizeMb: 250,
    multilingual: true,
    label: 'Quality (all languages)',
    hidden: false,
    acceleratedDtype: { encoder_model: 'fp16', decoder_model_merged: 'q4' },
  },
  'whisper-tiny': {
    id: 'whisper-tiny',
    hfRepo: 'onnx-community/whisper-tiny',
    approxSizeMb: 40,
    multilingual: true,
    label: 'Lite (all languages)',
    // Hidden from the picker by default (too low-quality to offer desktops),
    // but it's the iOS default (pickDefaultModel) and the picker always shows
    // the active model — so iPhone users see and can keep "Lite".
    hidden: true,
    acceleratedDtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
  },
};

export function isModelId(value: unknown): value is ModelId {
  // Object.hasOwn, not `in`: `in` walks the prototype chain, so inherited
  // keys like 'toString' would validate as model ids.
  return typeof value === 'string' && Object.hasOwn(MODELS, value);
}
