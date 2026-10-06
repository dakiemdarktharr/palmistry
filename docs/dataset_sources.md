# Dataset sources for the palm-line pipeline

Checked: 2026-09-12.

## Recommended source

[Tongji Contactless Palmprint Dataset](https://cslinzhang.github.io/ContactlessPalm/) is the closest match for this project:

- 12,000 contactless palm images, above the 8,000-image target.
- 600 distinct palms from 300 volunteers, captured in two sessions.
- The authors describe high contrast/high signal-to-noise images where minute creases are visible.
- The official page provides both original images and ROI images.

The dataset contains palmprint images, not semantic masks for life/head/heart lines. It must therefore go through the project's classical pseudo-label phase and the 400-image human review queue. The official page does not state a permissive commercial license; treat it as research-only and do not redistribute it until the authors confirm the terms.

Important split rule: the same palm appears in both sessions. Do not randomly split individual images. Group by palm identity before train/validation/test splitting, or arrange the extracted files into one source folder per palm before running the pipeline.

## Secondary source

[11K Hands](https://sites.google.com/view/11khands) contains 11,076 images at 1600x1200 from 190 subjects. It includes palmar and dorsal sides on a uniform white background and is explicitly offered for reasonable academic fair use. It is useful for hand detection/cropping and as extra palm images after filtering to the palmar side. It does not provide palm-line masks and the palmar subset is not documented as 8,000 images, so it should not be treated as the primary 8,000-image palm-line source.

## Not the first choice

- [PolyU-IITD Contactless Palmprint v3](https://www4.comp.polyu.edu.hk/~csajaykr/palmprint3.htm) is relevant, but access requires an institutional request and signed agreement; commercial use and redistribution are prohibited.
- [HaGRID](https://github.com/hukenovs/hagrid) is very large, but it is a gesture dataset with varied backgrounds and a very large download. It is suitable for hand detection/gesture pretraining, not for palm-crease tracing.

## Import and review

After obtaining an allowed ZIP or RAR, run:

    .\.venv\Scripts\python.exe pipeline.py --input_zip C:\data\tongji_original_images.zip --run_dir artifacts\tongji_review_001 --prepare_review --review_count 400

Inspect review/queue.html, update review/review.csv, then run bootstrap_review.py as documented in the main README. Keep the downloaded ZIP/RAR archive, source URL, access terms, checksum and any subject/palm metadata next to the run record; do not commit the images to Git.
