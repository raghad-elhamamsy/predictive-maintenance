# Reproducibility Audit

## Scope

The audit covers the generic channel loader, synthetic workbook generator, eight analysis notebooks, scoring utilities, and local demonstration interface.

## Verification

Run `python tests/fresh_clone_smoke.py` from the repository root. It builds a temporary copy, creates synthetic input data, executes notebooks 01–08, and verifies expected generated artifacts. The test does not use local data files.

## Limitations

- All values and event labels are generated for code-path testing.
- No result estimates performance on an organization, asset, or industry.
- The dashboard is a local demonstration and is not an operational alerting system.
- Public material should remain limited to synthetic or suitably aggregated examples.
