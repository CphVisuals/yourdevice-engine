import type { ModelId } from './models.js';

export type BackendId = 'webnn' | 'webgpu' | 'wasm';

/** Preferred initialization order: NPU/GPU via WebNN, then WebGPU, then the WASM floor. */
export const BACKEND_LADDER: readonly BackendId[] = ['webnn', 'webgpu', 'wasm'];

/**
 * Feature detection says a backend *may* work; only a successful runtime
 * initialization proves it does (WebNN in particular can exist yet fail to
 * build the Whisper graph). `detected` is the sniff, `active` is the backend
 * that actually initialized.
 */
export interface CapabilityReport {
  detected: Record<BackendId, boolean>;
  active: BackendId | null;
  deviceMemoryGb: number | null;
}

/** The subset of globals the detector reads, injectable for tests. */
export interface DetectionScope {
  navigator?: {
    ml?: unknown;
    gpu?: unknown;
    deviceMemory?: number;
  };
  WebAssembly?: unknown;
}

export function detectBackends(scope: DetectionScope): Record<BackendId, boolean> {
  return {
    webnn: scope.navigator?.ml != null,
    webgpu: scope.navigator?.gpu != null,
    wasm: scope.WebAssembly != null,
  };
}

export function buildCapabilityReport(scope: DetectionScope): CapabilityReport {
  const memory = scope.navigator?.deviceMemory;
  return {
    detected: detectBackends(scope),
    active: null,
    deviceMemoryGb: typeof memory === 'number' && memory > 0 ? memory : null,
  };
}

/**
 * Backends to try, in order. An explicit preference wins for the backends it
 * names; anything detected but not named is appended in ladder order so a
 * failed preferred backend still falls through to the floor.
 */
export function planBackendOrder(
  detected: Record<BackendId, boolean>,
  preference?: readonly BackendId[],
): BackendId[] {
  // Deduplicate: a repeated preference entry must not cause a second init
  // attempt of the same failed backend before falling through the ladder.
  const preferred = [...new Set(preference ?? [])].filter((id) => detected[id]);
  const rest = BACKEND_LADDER.filter((id) => detected[id] && !preferred.includes(id));
  return [...preferred, ...rest];
}

/**
 * Default model choice. `whisper-base` is the smallest multilingual model we
 * ship, so it is the default everywhere; only clearly capable devices (a
 * detected GPU/NPU path and >= 8 GB reported memory) default up to
 * `whisper-small`. The UI always offers an explicit override.
 */
export function pickDefaultModel(report: CapabilityReport): ModelId {
  const accelerated = report.detected.webgpu || report.detected.webnn;
  if (accelerated && report.deviceMemoryGb !== null && report.deviceMemoryGb >= 8) {
    return 'whisper-small';
  }
  return 'whisper-base';
}
