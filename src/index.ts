export {
  BACKEND_LADDER,
  buildCapabilityReport,
  detectBackends,
  pickDefaultModel,
  planBackendOrder,
  type BackendId,
  type CapabilityReport,
  type DetectionScope,
} from './backends.js';
export { isModelId, MODELS, type ModelId, type ModelSpec } from './models.js';
export {
  isHostMessage,
  isWorkerMessage,
  type EngineErrorCode,
  type HostMessage,
  type TranscribeOptions,
  type TranscriptSegment,
  type WorkerMessage,
} from './protocol.js';
export { resampleTo16kMono, type DecodedAudioLike } from './audio.js';
export { toJson, toSrt, toTxt, toVtt } from './export.js';
export {
  assignSpeakers,
  DEFAULT_DIARIZE_CONFIG,
  diarize,
  speakerCount,
  type DiarizeConfig,
  type SpeakerTurn,
} from './diarize.js';
export { agglomerative, cosineDistance, type Vector } from './clustering.js';
export {
  EngineClient,
  EngineError,
  type DownloadProgressCallback,
  type EngineWorkerLike,
  type InitOptions,
  type TranscribeCallbacks,
} from './client.js';

/**
 * The worker itself is not exported from here (it's an entry point, not a
 * library module — it calls `addEventListener`/`postMessage` as a side
 * effect on import). Hosts construct it directly from its dedicated module
 * export condition:
 *
 *   new Worker(new URL('@yourdevice/engine/worker', import.meta.url), { type: 'module' })
 *
 * See package.json `exports['./worker']` (workspace: `src/worker.ts`;
 * published: `dist/worker.js` via `publishConfig.exports`).
 */
