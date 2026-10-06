# Prompt sửa và cải thiện Palmistry

Copy toàn bộ prompt dưới đây vào một agent coding khi muốn tiếp tục sửa repo.

```text
You are working directly inside this repository.

Act as a senior AI engineer, senior software developer, engineering manager,
AI startup product executive, and computer-vision professor at the same time.
Your job is to turn this repository into a reproducible computer-vision
prototype within 48 hours. Do not rewrite the entire system and do not claim
production readiness.

## Non-negotiable operating rules

1. First inspect the existing code, README, git status, tracked files, entry
   points, tests, runtime environment, and current limitations.
2. Preserve all existing user work. Never use `git reset --hard`, destructive
   checkout, broad deletion, or overwrite unrelated changes.
3. Do not fabricate metrics, datasets, labels, checkpoint quality, user
   studies, scientific validity, or completed features.
4. Use only data already present or a clearly documented openly licensed
   dataset. Record source, license, attribution, checksum, and split policy.
5. If the real dataset or trained weights are missing, do not hide that fact.
   Add a tiny deterministic fixture only for pipeline tests and clearly state
   that real training requires the documented dataset and checkpoint.
6. Keep the product claim limited to palm-image quality assessment,
   segmentation, and visualization. Do not present palm-line geometry as a
   scientifically validated personality, health, or future prediction system.
7. Make the smallest credible end-to-end slice work before adding optional
   features.
8. Every important claim must be backed by a command, test output, measured
   value, source file, or explicit limitation.

## Current repository context

The repository currently contains:

- `palmistry_strict_auto_onefile.py`: classical-CV pseudo-mask generation,
  dataset build, training, and prediction CLI.
- `tien_xu_ly.py`: Roboflow/YOLO-style label conversion.
- `giao_dien_ui.py`: Flask camera UI and subprocess-based live inference.
- `README.md`: bilingual experimental project description.

The current repository may not contain a real dataset, trained checkpoint,
test suite, dependency lockfile, or benchmark. Verify the actual state instead
of assuming those assets exist.

Known audit targets that must be checked and fixed when confirmed:

- Anti-aliased `cv2.polylines` can create invalid semantic mask IDs.
- A label with fewer than three points can cause the converter to mark the
  entire image as `palm_area`.
- Image-level random splitting can leak the same subject/hand into train and
  validation/test data.
- Falling back from an empty validation split to the augmented training set
  makes validation metrics invalid.
- The Flask UI can read stale prediction files after a failed subprocess.
- The live display may ignore the newest prediction overlay, and manual lock
  may capture a raw frame instead of the overlay.
- The UI writes camera frames to fixed project files and has shared mutable
  state across camera, inference, and HTTP threads.
- The Flask app binds to `0.0.0.0` without authentication by default.
- NumPy `.npz` checkpoint loading and ZIP extraction must be treated as untrusted artifacts
  boundaries.
- Class ID 5 has inconsistent meanings between the converter and model
  pipeline.

Treat this list as hypotheses. Confirm each item against the current source,
then record whether it was fixed, not applicable, or deferred.

## Phase 0 — Baseline and safety checkpoint

Before editing:

1. Run `git status --short --branch` and record the result.
2. List tracked files with `git ls-files`.
3. Identify Python version, installed packages, available GPU, and available
   test/lint tools.
4. Run syntax/import checks for all Python files.
5. Inspect every CLI entry point and the Flask routes.
6. Check whether `SECURITY.md`, `.gitignore`, `pyproject.toml`, requirements,
   datasets, checkpoints, and tests exist.
7. Create a short baseline report in the final response. Do not create fake
   baseline metrics.

If the working tree contains unrelated edits, preserve them and work around
them. If an intended change overlaps an unrelated edit, stop and explain the
conflict.

## Phase 1 — Define the contracts before refactoring

Write down and enforce these contracts:

### Dataset contract

- Image is readable RGB data.
- Mask has exactly one channel and a documented integer class set.
- Canonical classes must be defined once and reused everywhere:
  `0=background`, `1=palm_area`, `2=life_line`, `3=head_line`,
  `4=heart_line`, `5=minor_or_unknown_line` or another explicitly chosen
  meaning. Do not silently mix `fate_line` and `minor_or_unknown_line`.
- Every row contains source path, output paths, checksum, split, label source,
  and quality/rejection reason where applicable.
- Invalid mask IDs, missing files, degenerate geometry, and ambiguous labels
  are rejected with explicit diagnostics.

### Model contract

- Input shape, color order, normalization, and image size are explicit.
- Image size is validated or padded so the U-Net down/up-sampling path cannot
  fail on non-compatible dimensions.
- Output shape is `[batch, num_classes, height, width]` and is tested.
- Checkpoint schema contains model configuration, class map, image size,
  training config, code version, and validation metadata.

### Inference contract

- Every request has a unique ID or isolated temporary workspace.
- A failed inference never returns artifacts from a previous request.
- The response includes status, error code, model-used indicator, quality
  scores, confidence, output artifact paths, and an abstention reason.
- A user-supplied missing checkpoint must fail clearly; it must not silently
  fall back to pseudo-mask output.

### Privacy and network contract

- Local-only mode binds to `127.0.0.1`.
- Network exposure is opt-in and requires authentication, authorization,
  CSRF protection where applicable, rate limiting, and secure deployment.
- Camera frames are not retained longer than necessary and are not committed
  to git.

## Phase 2 — Fix correctness blockers first

Implement and test the smallest safe fixes in this order:

1. Make semantic mask generation produce only canonical class IDs. Avoid
   anti-aliasing on integer label masks, handle overlapping classes
   deterministically, and add post-generation validation.
2. Never fill the whole image as palm area because a polygon is unavailable.
   Reject the sample or use a documented fallback that cannot create false
   foreground labels.
3. Prevent output filename collisions using a stable relative-path/hash
   component.
4. Make train/validation/test splitting deterministic and group-aware. If a
   required split is absent, fail with an actionable message instead of
   measuring training data as validation data.
5. Make validation non-augmented and ensure the test split is evaluated only
   after model selection.
6. Fix live inference so subprocess failures, timeouts, missing outputs, and
   failed writes cannot reuse stale files.
7. Render the current overlay in live mode and ensure manual lock captures a
   consistent frame, mask, metrics, and reading snapshot.
8. Add state synchronization for shared engine state and a graceful camera
   shutdown path.

Do not add a large framework solely to hide these issues.

## Phase 3 — Build one deterministic end-to-end slice

Complete exactly one path:

`fixture/openly-licensed input -> preprocessing -> mask -> model or explicit
fallback -> segmentation output -> evaluation -> visualization`

Requirements:

1. Use existing entry points when they are equivalent. Add `train.py`,
   `infer.py`, and `evaluate.py` only if the current entry points cannot
   provide a clear, testable interface.
2. Add a tiny deterministic fixture if real data is absent. The fixture must
   test the pipeline, not claim model quality.
3. Support explicit `seed`, `config`, `checkpoint`, `device`, and `image_size`
   options.
4. Add an evaluation command that reports appropriate metrics such as
   per-class IoU, Dice, precision, recall, or F1. Handle absent classes
   explicitly and do not report a misleading aggregate alone.
5. If no real checkpoint exists, run the preprocessing and evaluation path on
   the fixture and state that real model training was not performed.
6. Add at least one sample input/output visualization only when the data is
   legally usable. Otherwise use the deterministic fixture and label it as
   synthetic/test-only.

## Phase 4 — Reproducibility and maintainability

Add the minimum project scaffolding needed to reproduce the result:

- `pyproject.toml` or a pinned requirements/lock file.
- `.gitignore` for checkpoints, camera frames, generated bundles, caches, and
  datasets that should not be committed.
- `.env.example` for non-secret configuration; never hardcode tokens.
- A small config surface for paths, device, image size, seed, thresholds, and
  retention policy.
- Unit tests for preprocessing shape, class-ID validation, split integrity,
  model output shape, bad-input behavior, and fixture inference.
- An integration test for the Flask routes. Verify local-only defaults and
  authentication behavior if network mode exists.
- Clear logs and actionable exceptions. Do not swallow broad exceptions
  without recording the reason.

Keep model loading in memory for the live UI. Use a bounded latest-frame
queue or equivalent backpressure strategy instead of spawning a new Python
process and loading a checkpoint for every frame.

## Phase 5 — Security and privacy hardening

Review and fix, or explicitly defer with rationale:

1. Bind the Flask development UI to loopback by default.
2. Protect stream and state-changing APIs before LAN or ngrok exposure.
3. Use safe checkpoint loading and validate state-dict schema, digest, and
   provenance.
4. Extract ZIP members only after path containment and symlink checks.
5. Limit downloaded image bytes, decoded megapixels, archive size, and number
   of files to prevent resource exhaustion.
6. Store license/author/source metadata for downloaded images and document
   attribution obligations.
7. Escape all server-controlled values inserted into HTML.
8. Avoid storing camera frames persistently unless the user explicitly opts in.
9. Add security regression tests for unauthenticated routes, traversal paths,
   stale artifacts, and malformed inputs.

Do not claim “secure” merely because no critical vulnerability was found.

## Phase 6 — README update

Update `README.md` with these exact sections:

1. Problem and motivation.
2. Scope and responsible-use statement.
3. Architecture diagram or concise architecture description.
4. Exact environment setup commands for the tested Python version.
5. Exact commands for fixture tests, preprocessing, training, inference, and
   evaluation.
6. Demo instructions for the Flask UI, including camera/checkpoint
   prerequisites and local-only behavior.
7. One measured result table. Every value must be generated by a command and
   labeled as fixture, validation, test, or real-data evidence.
8. Known limitations, including missing data/weights, no scientific claim,
   hardware assumptions, and untested paths.
9. A truthful two-line resume bullet. It must not say production-ready or
   imply validated palmistry/personality prediction.

If trained weights cannot be included, explain why and provide the precise
reproduction command, expected input layout, and the fact that the command
requires the documented dataset.

## Acceptance criteria

The work is complete only if all applicable criteria pass:

- Existing user changes are preserved.
- All Python files compile and import.
- The deterministic fixture runs from a clean checkout using documented
  commands.
- Preprocessing outputs have valid shape and canonical class IDs.
- The model output shape test passes, or the repository clearly documents that
  the real model path is blocked by a missing dependency/checkpoint.
- Bad image, missing checkpoint, malformed mask, and failed inference paths
  return explicit errors and never stale artifacts.
- Train/validation/test split integrity is tested.
- Evaluation metrics are reproducible with the same seed/config.
- The Flask UI starts locally without requiring a checkpoint, and its status
  clearly reports missing model assets.
- No public-network exposure is enabled by default.
- README commands have actually been executed or are explicitly marked
  unverified.
- No fabricated result, dataset statistic, benchmark, or user claim appears.

## Final response format

At the end, report exactly:

1. Executive verdict: what now works and what is still blocked.
2. Files changed, grouped by purpose.
3. Commands executed, with important exit codes.
4. Test/build results.
5. Measured metrics, with dataset/fixture provenance and split name.
6. Security/privacy changes and remaining risks.
7. Remaining limitations and unverified paths.
8. A truthful two-line resume bullet.

Use file paths and line numbers for important fixes. Distinguish clearly
between implemented, verified, inferred, and deferred work.

Do not describe the project as production-ready. The final wording should
call it a reproducible computer-vision prototype.

## Baseline task requirements to preserve

The original task is:

"Make this project resume-ready within 48 hours, not to rewrite the entire
system. Implement the smallest credible end-to-end slice, run the relevant
tests/builds, and update the README with problem and motivation, architecture,
exact setup and run commands, demo instructions, measured evidence, known
limitations, and a truthful two-line resume bullet.

Turn Palmistry into a reproducible computer-vision prototype.

Tasks:

- Inspect the current preprocessing, dataset, model, training, and camera code.
- Complete one end-to-end path: dataset -> preprocessing ->
  training/inference -> segmentation output -> evaluation.
- Use only data already present or a clearly documented, openly licensed
  dataset. Do not invent dataset statistics.
- If the real dataset is missing, add a tiny deterministic fixture for
  pipeline tests and explicitly state that real training requires the
  documented dataset.
- Add train.py, infer.py, and evaluate.py only if equivalent entry points do
  not already exist.
- Add reproducibility options: seed, config, checkpoint path, device, and
  image size.
- Report appropriate metrics such as IoU, Dice, precision, recall, or F1.
- Add at least one sample input/output visualization if legally permitted.
- Add tests for preprocessing shape, model output shape, and inference on a
  fixture.
- Add a requirements lock or clearly documented environment setup.

Do not describe the project as production-ready. If trained weights cannot be
included, provide a precise explanation and a command for reproducing them.
Update the README with one result table and one honest resume bullet."
```
