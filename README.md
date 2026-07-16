# engine

Local-first, in-browser speech-to-text engine. Whisper running entirely in the
browser — on WebGPU where available, WebNN as a progressive enhancement, and a
single-threaded WASM (SIMD) floor. Audio never leaves the device.

> Status: pre-release skeleton. The typed worker protocol and backend ladder are
> in place; inference lands next. This package is framework-free and will be
> published under its final name at launch.

## Design

- **Runtime ladder, not feature sniffing.** `detectBackends` reports what _may_
  work; the worker proves it by try-initializing each backend in
  `planBackendOrder` until one succeeds. WebNN can exist yet fail to build the
  Whisper graph — detection alone is never trusted.
- **Typed worker protocol.** `HostMessage` / `WorkerMessage` unions with runtime
  guards (`isHostMessage`, `isWorkerMessage`) on both sides of `postMessage`.
- **Conservative defaults.** `whisper-base` (multilingual, ~80 MB q8) everywhere;
  `whisper-small` only on accelerated devices with ≥ 8 GB reported memory.

## License

MIT
