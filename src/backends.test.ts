import { describe, expect, it } from 'vitest';
import {
  buildCapabilityReport,
  detectBackends,
  detectIOS,
  detectMobile,
  pickDefaultModel,
  planBackendOrder,
} from './backends.js';

const ALL_DETECTED = { webnn: true, webgpu: true, wasm: true } as const;

describe('detectBackends', () => {
  it('reports everything absent on an empty scope', () => {
    expect(detectBackends({})).toEqual({ webnn: false, webgpu: false, wasm: false });
  });

  it('detects each backend from its global', () => {
    expect(detectBackends({ navigator: { ml: {}, gpu: {} }, WebAssembly: {} })).toEqual(
      ALL_DETECTED,
    );
  });

  it('treats WebGPU without WebNN correctly (the common Chrome case)', () => {
    expect(detectBackends({ navigator: { gpu: {} }, WebAssembly: {} })).toEqual({
      webnn: false,
      webgpu: true,
      wasm: true,
    });
  });
});

describe('planBackendOrder', () => {
  it('follows the ladder when there is no preference', () => {
    expect(planBackendOrder(ALL_DETECTED)).toEqual(['webnn', 'webgpu', 'wasm']);
  });

  it('skips undetected backends', () => {
    expect(planBackendOrder({ webnn: false, webgpu: true, wasm: true })).toEqual([
      'webgpu',
      'wasm',
    ]);
  });

  it('honors an explicit preference but keeps the floor as fallback', () => {
    expect(planBackendOrder(ALL_DETECTED, ['wasm'])).toEqual(['wasm', 'webnn', 'webgpu']);
    expect(planBackendOrder(ALL_DETECTED, ['webgpu'])).toEqual(['webgpu', 'webnn', 'wasm']);
  });

  it('ignores preferred backends that were not detected', () => {
    expect(planBackendOrder({ webnn: false, webgpu: false, wasm: true }, ['webnn'])).toEqual([
      'wasm',
    ]);
  });

  it('returns empty when nothing is detected', () => {
    expect(planBackendOrder({ webnn: false, webgpu: false, wasm: false })).toEqual([]);
  });
});

describe('buildCapabilityReport', () => {
  it('captures device memory when reported', () => {
    const report = buildCapabilityReport({
      navigator: { gpu: {}, deviceMemory: 8 },
      WebAssembly: {},
    });
    expect(report.deviceMemoryGb).toBe(8);
    expect(report.active).toBeNull();
  });

  it('leaves device memory null when unavailable (Safari/Firefox)', () => {
    expect(buildCapabilityReport({ WebAssembly: {} }).deviceMemoryGb).toBeNull();
  });
});

describe('pickDefaultModel', () => {
  it('defaults to whisper-base', () => {
    const report = buildCapabilityReport({ WebAssembly: {} });
    expect(pickDefaultModel(report)).toBe('whisper-base');
  });

  it('upgrades to whisper-small only with acceleration and >= 8 GB memory', () => {
    const capable = buildCapabilityReport({
      navigator: { gpu: {}, deviceMemory: 8 },
      WebAssembly: {},
    });
    expect(pickDefaultModel(capable)).toBe('whisper-small');

    const lowMemory = buildCapabilityReport({
      navigator: { gpu: {}, deviceMemory: 4 },
      WebAssembly: {},
    });
    expect(pickDefaultModel(lowMemory)).toBe('whisper-base');

    const noGpu = buildCapabilityReport({
      navigator: { deviceMemory: 16 },
      WebAssembly: {},
    });
    expect(pickDefaultModel(noGpu)).toBe('whisper-base');
  });

  it('stays conservative when device memory is unknown', () => {
    const unknownMemory = buildCapabilityReport({ navigator: { gpu: {} }, WebAssembly: {} });
    expect(pickDefaultModel(unknownMemory)).toBe('whisper-base');
  });

  it('forces whisper-base on mobile even when it looks capable', () => {
    // A 12 GB phone reports deviceMemory 8 (browsers cap it) + gpu present —
    // it would otherwise wrongly get the heavy model. isMobile overrides.
    const phone = buildCapabilityReport({
      navigator: { gpu: {}, deviceMemory: 8, userAgentData: { mobile: true } },
      WebAssembly: {},
    });
    expect(phone.isMobile).toBe(true);
    expect(pickDefaultModel(phone)).toBe('whisper-base');
  });

  it('forces whisper-tiny on iOS (WebKit memory cap crashes on larger models)', () => {
    const iphone = buildCapabilityReport({
      navigator: { userAgent: 'iPhone; CPU iPhone OS 17_0 like Mac OS X' },
      WebAssembly: {},
    });
    expect(iphone.isIOS).toBe(true);
    // isIOS wins over the generic mobile → whisper-base rule.
    expect(iphone.isMobile).toBe(true);
    expect(pickDefaultModel(iphone)).toBe('whisper-tiny');
  });
});

describe('detectIOS', () => {
  it('matches iPhone/iPad/iPod user agents', () => {
    expect(detectIOS({ navigator: { userAgent: 'iPhone; CPU iPhone OS 17_0' } })).toBe(true);
    expect(detectIOS({ navigator: { userAgent: 'iPad; CPU OS 16_0' } })).toBe(true);
    expect(detectIOS({ navigator: { userAgent: 'iPod touch' } })).toBe(true);
  });

  it('detects iPadOS 13+ masquerading as desktop Safari via touch points', () => {
    // iPadOS reports a "Macintosh" UA; only maxTouchPoints gives it away.
    expect(
      detectIOS({ navigator: { userAgent: 'Macintosh; Intel Mac OS X', maxTouchPoints: 5 } }),
    ).toBe(true);
    // A real Mac (trackpad, not touch) must NOT be treated as iOS.
    expect(
      detectIOS({ navigator: { userAgent: 'Macintosh; Intel Mac OS X', maxTouchPoints: 0 } }),
    ).toBe(false);
  });

  it('is false for Android and desktop', () => {
    expect(detectIOS({ navigator: { userAgent: 'Android 14; Mobile' } })).toBe(false);
    expect(detectIOS({ navigator: { userAgent: 'X11; Linux x86_64' } })).toBe(false);
    expect(detectIOS({})).toBe(false);
  });
});

describe('detectMobile', () => {
  it('trusts Chromium userAgentData.mobile', () => {
    expect(detectMobile({ navigator: { userAgentData: { mobile: true } } })).toBe(true);
    expect(detectMobile({ navigator: { userAgentData: { mobile: false } } })).toBe(false);
  });

  it('falls back to a user-agent sniff for browsers without client hints', () => {
    expect(detectMobile({ navigator: { userAgent: 'iPhone; CPU iPhone OS' } })).toBe(true);
    expect(detectMobile({ navigator: { userAgent: 'Android 14; Mobile' } })).toBe(true);
    expect(detectMobile({ navigator: { userAgent: 'X11; Linux x86_64' } })).toBe(false);
    expect(detectMobile({ navigator: {} })).toBe(false);
    expect(detectMobile({})).toBe(false);
  });
});
