# Release Checklist

## Release Policy

MPS is currently in the `0.x` maturity stage (`0.5.5`).

Recommended interpretation:

- `v0.6.x`
  Stable iteration for adopters already aligned with the current API surface.
- `v1.0`
  Requires stricter quality gates and API stability guarantees.

## Required Gates for `v0.6`

### Quality Gates

- `cargo test --all-targets` must pass.
- `cargo check --examples` must pass.
- `cargo check` for declared target platforms should pass in CI:
  - `x86_64-unknown-linux-gnu`
  - `aarch64-apple-darwin`
  - iOS targets in active support matrix

### Performance Gates

- `mps_microbench` median metrics should be compared against previous release baseline.
- Include `parallel(raw)`, `gflops`, and elapsed CV in release notes.
- No major regression in median throughput on primary host classes.

### Documentation Gates

- README install snippet and version references must match the release tag.
- `docs/` index and core docs must be up to date with user-facing behavior.

## Additional Gates for `v1.0`

- Clippy policy decided and enforced (for example `-D warnings`) across main targets.
- CI workflow published and required for release branches.
- Public API stability statement documented.
- Compatibility notes for legacy `thread_pool` vs `TaskDispatcher` migration finalized.
- Release notes include explicit supported target matrix and known limitations.

## Current Known Gaps (as of this checklist draft)

- Strict clippy gate (`cargo clippy --all-targets -- -D warnings`) currently fails due to existing warnings in multiple modules.
- No repo-local CI workflow files are currently present.
- Crates.io distribution is not yet enabled; installation uses git tags.

## Suggested Go/No-Go Heuristic

Go for `v0.6` if all are true:

1. Test/check gates pass for supported targets.
2. No critical benchmark regression versus previous tag.
3. Docs and changelog are synchronized with code behavior.

Hold `v1.0` until all are true:

1. CI and lint policy are fully enforced.
2. API/compatibility boundaries are documented and stable.
3. Release process is repeatable without manual exceptions.
