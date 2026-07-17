/**
 * Pure audio-resampling helper. No DOM types: the site does the actual
 * `AudioContext.decodeAudioData` (DOM belongs in the site, not the
 * framework-free engine — CLAUDE.md rule 2) and hands us plain per-channel
 * `Float32Array`s plus the source sample rate.
 */

const TARGET_SAMPLE_RATE = 16_000;

export interface DecodedAudioLike {
  /** One `Float32Array` per channel, each the same length. */
  channelData: readonly Float32Array[];
  sampleRate: number;
}

/**
 * Downmix to mono (simple channel average) and linearly resample to 16 kHz —
 * the sample rate Whisper expects.
 */
export function resampleTo16kMono(audio: DecodedAudioLike): Float32Array {
  const mono = toMono(audio.channelData);
  if (audio.sampleRate === TARGET_SAMPLE_RATE) {
    return mono;
  }
  return linearResample(mono, audio.sampleRate, TARGET_SAMPLE_RATE);
}

function toMono(channelData: readonly Float32Array[]): Float32Array {
  if (channelData.length === 0) {
    return new Float32Array(0);
  }
  const first = channelData[0];
  if (channelData.length === 1 || first === undefined) {
    return first ?? new Float32Array(0);
  }
  const mono = new Float32Array(first.length);
  for (let i = 0; i < mono.length; i++) {
    let sum = 0;
    for (const channel of channelData) {
      sum += channel[i] ?? 0;
    }
    mono[i] = sum / channelData.length;
  }
  return mono;
}

function linearResample(input: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (input.length === 0 || fromRate === toRate || fromRate <= 0) {
    return input;
  }
  const ratio = fromRate / toRate;
  const outputLength = Math.max(1, Math.round(input.length / ratio));
  const output = new Float32Array(outputLength);
  const lastIndex = input.length - 1;
  for (let i = 0; i < outputLength; i++) {
    const sourcePosition = i * ratio;
    const sourceIndex = Math.min(lastIndex, Math.floor(sourcePosition));
    const fraction = sourcePosition - sourceIndex;
    const sample = input[sourceIndex] ?? 0;
    const nextSample = input[Math.min(lastIndex, sourceIndex + 1)] ?? sample;
    output[i] = sample + (nextSample - sample) * fraction;
  }
  return output;
}
