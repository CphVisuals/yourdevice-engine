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
  /** Phone/tablet — used to pick a lighter default model (see pickDefaultModel). */
  isMobile: boolean;
}

/** The subset of globals the detector reads, injectable for tests. */
export interface DetectionScope {
  navigator?: {
    ml?: unknown;
    gpu?: unknown;
    deviceMemory?: number;
    userAgent?: string;
    /** Chromium's client hints; `mobile` is the cleanest phone signal. */
    userAgentData?: { mobile?: boolean };
  };
  WebAssembly?: unknown;
}

/**
 * Whether this is a phone/tablet. Prefers Chromium's `userAgentData.mobile`
 * (covers Android Chrome — where WebGPU is flaky, so mobile lands on the WASM
 * floor and needs a lighter model); falls back to a user-agent sniff for
 * Safari/Firefox which don't expose client hints.
 */
export function detectMobile(scope: DetectionScope): boolean {
  const nav = scope.navigator;
  if (nav?.userAgentData?.mobile === true) return true;
  const ua = typeof nav?.userAgent === 'string' ? nav.userAgent : '';
  return /iPhone|iPad|iPod|Android|Mobile/i.test(ua);
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
    isMobile: detectMobile(scope),
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
 * Default model choice. `whisper-base` (the smaller, faster multilingual model)
 * is the default everywhere except clearly capable *desktops*, which default up
 * to `whisper-small` for quality. The UI always offers an explicit override.
 *
 * Phones are forced to `whisper-base` regardless of the memory signal: WebGPU
 * is flaky on mobile so phones usually run the WASM floor, where whisper-small
 * is slower than real-time (benchmarked ~0.85× on a flagship OnePlus 13),
 * whereas whisper-base runs comfortably faster. `navigator.deviceMemory` caps
 * at 8 in every browser, so a 12 GB phone otherwise looks like an 8 GB desktop
 * and would wrongly get the heavy model.
 */
export function pickDefaultModel(report: CapabilityReport): ModelId {
  if (report.isMobile) return 'whisper-base';
  const accelerated = report.detected.webgpu || report.detected.webnn;
  if (accelerated && report.deviceMemoryGb !== null && report.deviceMemoryGb >= 8) {
    return 'whisper-small';
  }
  return 'whisper-base';
}
