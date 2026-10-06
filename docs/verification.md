# Verification and limitations

This is a reproducible computer-vision prototype. Evidence was measured on
2026-09-12 using Windows, CPython 3.11.9 and the NumPy-only model.
The source revision is `e729a6d8d339a6ace70c5d4ef868962d215a7d98` plus the
uncommitted changes in this task. Machine-readable evidence also records
the application source SHA-256 hashes and dirty-tree status.

## Reproduction

```powershell
python -m venv .venv-clean
.\.venv-clean\Scripts\python.exe -m pip install -r requirements.txt --timeout 20 --retries 1
.\.venv-clean\Scripts\python.exe scripts/verify.py --output artifacts/verification
python -m unittest discover -s tests -v
git diff --check
```

The verification environment uses no global site packages and installs only the pinned core dependencies.
The final verification uses `.venv-clean`, and all 53 tests pass; five legacy framework-specific tests remain skipped because this project intentionally excludes that framework.

`scripts/verify.py` orchestrates the existing commands; it does not duplicate
training or inference. It records exact argument arrays, exit codes, durations,
stdout/stderr, dependency state, source hashes and metrics. See
[verification.json](verification.json) for the saved measured results. Fresh
logs are written under `artifacts/verification/`. A bad-image command deliberately
returns exit 2 after writing structured JSON. Other verification commands
return exit 0. Runtime values are observations, not performance guarantees.

## Required regression coverage

| Requirement | Verification |
|---|---|
| Syntax/imports, help, doctor | Compile/import subprocesses, Windows UTF-8 help, actual dependency output |
| Preprocessing shape and mask IDs | 128×128 crop/mask checks; deterministic repeated preprocessing |
| Anti-aliasing | A diagonal class-4 line contains only 0 and 4 |
| Two-point palm | Raises without mutating mask; collinear hull also rejected |
| Duplicate filenames | Stable path hashes differ; existing output collision raises without overwriting |
| Ontology | Converter, pipeline and UI agree; unversioned/fate maps rejected; explicit legacy conversion required |
| Split determinism | Input order independent, seed controlled, nonempty train/val/test |
| Leakage | Subject, source, SHA-256 and near-duplicate groups cannot cross splits |
| Independent validation | Missing validation rejected; validation augmentation forbidden; repeated sample identical |
| Model shape | NumPy checkpoint schema and prediction mask shape |
| Accumulation remainder | Actual training loop: 3 batches, accumulation 2, exactly 2 optimizer steps |
| Bad image | JSON error/retake contract; no stale masks or overlays |
| Missing checkpoint | Explicit checkpoint error, including when fallback flag is also supplied |
| Failed inference | Nonzero exit and stale success markers rejected; old snapshot invalidated |
| Overlay/manual lock | Crop overlay, frame ID, mask, reading and metrics copied from one snapshot |
| Queue and reset | Only newest frame retained; in-flight result discarded after reset |
| ZIP/RAR traversal | Malicious traversal archive is rejected; valid images are extracted only inside the run root |
| Flask defaults | Loopback constant, real local HTTP smoke, off-host requests rejected |
| Authentication/network mode | Network mode absent; ngrok refused; CSRF and Origin checks exercised |
| Resource cleanup | Temporary request directory removed on success, failure and timeout; fake camera threads stop |
| Empty dataset | Explicit actionable failure |
| Malformed mask | Invalid IDs, RGB masks and noninteger masks rejected before training loss |
| Invalid parameters | Unsupported dimensions, nonfinite thresholds, missing positive counts rejected |
| Checkpoint safety | SHA mismatch, incompatible schema and arbitrary pickle objects rejected |
| Resource limits | Large archives upload in bounded chunks; compressed/expanded/member/image limits and disk preflight are checked before decode; request rate limit exercised |

Hardware camera capture, driver blocking behavior, real-data
accuracy, statistical confidence calibration and multi-user deployment were
**not verified**. A fake camera and a real offline fixture inference verify
thread and IPC mechanics without claiming actual camera success.

## Dataset and metrics interpretation

The six generated `geometric-v1` images are mathematical test shapes, not photos
of people. Their labels come from deterministic geometry, independently of
the segmenter. Perceptual clustering merges them into three groups: train has
1 image, validation has 3, and test has 2. These group IDs do not represent
human subjects. The demo evaluates one image from the synthetic test group.

Evaluation transforms the geometric target using the same recorded crop and
padding coordinates as inference, with nearest-neighbour label resizing.
It compares 16,384 pixels at 128×128. Every class reports IoU, Dice, precision,
recall, F1 and support; a 6×6 confusion matrix is included. Undefined
denominators are null, not perfect scores. Mean IoU averages nonempty unions.

The one-epoch CPU model is trained on the single synthetic training image
at 32×32. It is a smoke-test checkpoint, not a useful palm-line model.
Validation uses only the three nonaugmented validation samples; test data
is never used by training, early stopping or checkpoint selection. The demo's
`--min_image_quality 0` override is strictly a fixture-testing convenience.
Keep the default quality threshold for local photos.

Classical pseudo-labels and line-point hulls are not independent anatomical
ground truth. Converter boxes become approximate center-lines and the palm
area is an undilated convex hull of line points; three noncollinear points
do not establish a true hand boundary. Independently annotated real masks
are still needed. Source-folder and pHash grouping can miss identity leakage;
fallback results must not be called subject-independent.

## Deliberate fallback decisions

| Area | Implemented/verified | Deferred or blocked |
|---|---|---|
| Inference | Unique request directory, matching success marker, timeout, cleanup | Persistent in-process model loading; each subprocess still reloads weights |
| Readiness | No repeated prediction without checkpoint/hash/NumPy model | A failed prediction disables the worker until application restart |
| Networking | Loopback, peer/Host checks, CSRF, rate limit, CSP and no-store headers | LAN access, authentication, TLS and production WSGI deployment |
| Archives | Explicit ZIP/RAR extraction stays inside dataset/run roots and records skipped entries | License/author attribution-aware remote acquisition |
| Downloads | Remote acquisition disabled; bounded local RGB decode | License/author/attribution-aware remote acquisition |
| Checkpoints | Portable `.npz`, explicit SHA-256, schema/shape validation | Signed provenance and cross-version migration; old checkpoints fail closed |
| Reproducibility | Fresh-run Python/NumPy seed, source/manifest hashes | Cross-device determinism is not applicable; statistical model reruns are deterministic |
| Scores | Uncalibrated heuristic/ranking semantics, quality abstention | Probability calibration and learned out-of-distribution detector |

The checksum only checks integrity against the hash the user explicitly trusts;
it is not an author signature. Do not hash an arbitrary untrusted checkpoint
and assume that establishes trust. Weights-only loading is not a resource
sandbox. Maximum checkpoint file size is 512 MiB, and model architectures and
input sizes are restricted.

The local demo uses Flask's development server. Do not reverse-proxy it to the
Internet or change its loopback restriction. Any local process under the same
account can reach the demo. Rate limits are process-local, and streams are
intended for a single local user.

Camera frames remain in memory except during an explicitly configured model
request. That request uses the account's OS temporary storage, cleans up on
normal success/failure/timeout, and retains no camera frames in the repository.
Retention is fixed to zero; unsupported retention settings fail. Temporary
files inherit OS account permissions, are not encrypted, and an abrupt process
or machine crash can leave remnants. Consent is required before capturing
someone else's hands. Stop the app when finished; Ctrl+C invokes shutdown.
CLI images, outputs and synthetic fixtures persist only at explicitly requested
paths and are excluded from Git by default under `artifacts/`.

## Editing and baseline evidence

The initial tree had the UTF-8 console modification in
`palmistry_strict_auto_onefile.py` and untracked `PROMPT_SUA_CAI_THIEN.md`.
Both were preserved. The repository initially contained no dataset,
checkpoint, tests or dependency manifest. Python 3.11.9 and core imports worked;
The project intentionally excludes PyTorch, scikit-learn and pandas.

The Windows sandbox failed with `helper_unknown_error: apply deny-read ACLs`.
`apply_patch` could create new files but could not read existing ones. Existing
source edits therefore used narrow Python transformations with exact-match
assertions, followed by `git diff --check` and diff inspection. Temporary edit
scripts are ignored under `artifacts/`; no reset, destructive checkout,
commit, push, automatic bundle deletion or unrelated file overwrite was used.

The first suite run had one test-only Windows codepage error while reading
UTF-8 JSON. It was repaired by specifying UTF-8, and subsequent full suites
passed. No failed model or real-data result is being hidden by the fixture.
