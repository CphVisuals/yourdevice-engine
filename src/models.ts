export type ModelId = 'whisper-base' | 'distil-small.en' | 'whisper-small';

export interface ModelSpec {
  id: ModelId;
  /** Hugging Face repo the ONNX weights are pulled from (R2 mirror takes over at launch). */
  hfRepo: string;
  /** Approximate download size of the q8 weights, used for progress UI and defaults. */
  approxSizeMb: number;
  multilingual: boolean;
  label: string;
}

export const MODELS: Record<ModelId, ModelSpec> = {
  'whisper-base': {
    id: 'whisper-base',
    hfRepo: 'onnx-community/whisper-base',
    approxSizeMb: 80,
    multilingual: true,
    label: 'Standard (all languages)',
  },
  'distil-small.en': {
    id: 'distil-small.en',
    hfRepo: 'onnx-community/distil-small.en',
    approxSizeMb: 120,
    multilingual: false,
    label: 'Fast (English only)',
  },
  'whisper-small': {
    id: 'whisper-small',
    hfRepo: 'onnx-community/whisper-small',
    approxSizeMb: 250,
    multilingual: true,
    label: 'Quality (all languages)',
  },
};

export function isModelId(value: unknown): value is ModelId {
  // Object.hasOwn, not `in`: `in` walks the prototype chain, so inherited
  // keys like 'toString' would validate as model ids.
  return typeof value === 'string' && Object.hasOwn(MODELS, value);
}
