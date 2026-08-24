/**
 * Refuses a publish run by npm rather than pnpm.
 *
 * This package resolves to TypeScript sources inside the workspace
 * (`exports['.'] === './src/index.ts'`) and swaps them for the compiled
 * `dist/` via `publishConfig.exports`. Rewriting the manifest from
 * `publishConfig` is a *pnpm* feature: `npm publish` treats `publishConfig`
 * as config keys only and copies `exports` through verbatim, so it would
 * upload a package pointing at `./src/index.ts` — a path `files` does not
 * even ship as an entry point — and every consumer would fail to resolve it.
 * A public version cannot be re-uploaded, so this has to fail before the
 * tarball is built, not after.
 *
 * Runs as `prepublishOnly`, which both package managers honour.
 */
const agent = process.env.npm_config_user_agent ?? '';

if (!agent.startsWith('pnpm')) {
  console.error(
    [
      '',
      'Refusing to publish @yourdevice/engine with npm.',
      '',
      `  detected package manager: ${agent || '(none reported)'}`,
      '',
      'npm ignores publishConfig.exports, so it would publish a package whose',
      '"exports" still point at ./src/index.ts and break on install.',
      '',
      'Publish with pnpm instead:',
      '',
      '  pnpm --filter @yourdevice/engine publish --access public',
      '',
    ].join('\n'),
  );
  process.exit(1);
}
