# Project audit — 2026-10-07

Reviewed the local working project, including the NumPy keypoint CNN and geometry,
legacy segmentation and bootstrap pipeline, Flask APIs, annotation template,
archive handling, desktop launcher, Windows installer, dependency pins, tests and
Git publication. The Git baseline contained only four files; the existing working
pipeline and tests must be committed with the fixes for a reproducible checkout.

## Confirmed defects and fixes

| Problem | Impact | Fix and evidence |
| --- | --- | --- |
| Archive tests read an ignored local fixture | Three tests failed in a clean source copy | Generate a tiny image in each test; optional external RAR backend can be explicitly skipped |
| Explicit manifest group IDs bypassed perceptual auditing | Re-encoded near duplicates could cross train/test boundaries with different group IDs | Recompute hashes from actual images and audit supplied partitions; a regression re-encodes identical pixels with distinct file hashes |
| Archive pixel limits were checked after OpenCV decoding | A small compressed image could allocate an excessive pixel buffer before rejection | Inspect the encoded dimensions first for ZIP/RAR in CLI and web paths; regression asserts OpenCV is never called for an oversized header |
| JSON state/history could be overwritten partially | Failed writes could destroy a previously valid document | Serialize, flush and sync a unique sibling file, then atomically replace; regression simulates a failed replacement and verifies old data and temp cleanup |
| A broken keypoint project blocked the dashboard | Healthy projects became unavailable when another project's JSON or annotation was unreadable | Isolate errors per project and show affected names in the UI without altering their files |
| Pseudo-label metadata keys used unnormalized paths | Windows short-path aliases missed identity metadata and misclassified held-out subjects as missing metadata | Normalize relative/absolute metadata paths against the source root and reject conflicting aliases; the original workflow regression exposed this on GitHub Windows CI |
| Compressed checkpoint bounds ignored expanded NPZ size | Compressed arrays could exceed the memory policy | Check ZIP entry sizes before `np.load`; reject malformed archives as actionable errors and verify keypoint checkpoint dimensions/schema |
| Launcher accepted any HTTP 200 on port 8501 and did not serialize startup | Another installation/server could be displayed; simultaneous clicks could start competing backends | Require `/api/health` to identify Palmistry and the matching directory; hold a named startup mutex until the app is ready |
| Installer allowed nested source/target directories | Copying into a child of the source could recursively copy the destination; ancestor targets could overwrite source files | Reject equal, child and parent targets before creating files; check venv creation exit status and Python 3.11+ |
| Legacy workers could open console windows on Windows | Extra terminal windows interrupted the desktop flow | Start analysis and legacy pipeline subprocesses with `CREATE_NO_WINDOW` |
| Closing the native window bypassed the page's unsaved-edit guard | Unsubmitted keypoint edits could be discarded without the page handling closure | Route native closure through Qt's `RequestClose` and only close on `windowCloseRequested`; desktop smoke verifies `beforeunload` runs |
| No repository CI or standalone desktop smoke check | A clean checkout could silently regress | Add Windows/Linux verification CI and a Windows Qt template/download smoke test; exclude build output and logs from Git |

The desktop shortcut opens the native PySide6 window, with downloads and review
popups contained in the application. The existing backend intentionally stays
alive when the window closes so background work can finish.

## Validation

Commands run on Windows, CPython 3.11.9:

```powershell
python scripts/verify.py --output artifacts/review-20261007
python scripts/verify_desktop.py
```

- Baseline: 99 tests, 3 errors caused by missing local fixtures, 5 legacy skips.
- After fixes: 108 tests, no failures/errors, 5 legacy PyTorch tests skipped
  because that framework is disabled in this NumPy project.
- Compile/import, loopback HTTP, dependency consistency, fixture preprocessing,
  training, classical/model prediction and evaluation, offline UI subprocess,
  and malformed-input handling passed. The bad-image CLI correctly returned 2.
- The desktop smoke test rendered the actual annotation template, executed its
  JavaScript, opened an app popup, saved a JSON download and invoked the page's
  `beforeunload` guard on native closure using a temporary
  local server. It uses no camera or personal dataset.
- PowerShell scripts parsed successfully. Installer paths equal to, below and
  above the source were all rejected before installation.

Raw generated verification data stays under ignored `artifacts/`. The published
CI workflow recreates the same checks from a clean checkout on Windows/Linux;
its GitHub run is separate from local evidence.

## Remaining limitations requiring data or hardware evidence

- The main local queue has 400 pending images and no approved annotations as
  observed during this audit. Its separate one-image guide is an example, not
  training evidence. Real labels and truthful subject/session metadata are
  required before training and evaluating subject-independent performance.
- The 64×64 NumPy CNN remains a baseline. Passing geometry/gradient/fixture tests
  does not establish accuracy, confidence calibration or robustness on real palms.
  Increase resolution or model capacity only after collecting independent
  annotated validation/test data and measuring the current errors.
- Camera/driver behavior and diverse real RAR archives were not newly tested on
  hardware. Archive regressions exercise the bounded import paths with fixtures
  and mocked backends; actual backend availability was detected on this machine.
- The legacy Flask/segmentation file is large and overlaps parts of the newer
  pipeline. Splitting its camera, archive and job services into modules is future
  maintenance work; this audit prioritizes confirmed behavior defects and keeps
  existing interfaces stable.
- Multi-file metadata import is protected against concurrent UI jobs, but is not
  a database transaction across all annotation files. Atomic individual JSON
  writes reduce corruption risk; regular backups remain useful for recovery.

Original images, annotations, checkpoints, temporary uploads and local runtime
logs are excluded from the source commit.
