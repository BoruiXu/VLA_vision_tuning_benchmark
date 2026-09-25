# Release checks

Completed locally for this release candidate:

- Exported 61 distinct quantitative runs: 32 full Plus, 15 original LIBERO and 14 fixed-subset runs; 358,800 episode outcomes in total.
- Checked unique keys, counts, success totals, four-suite/seven-dimension coverage, 40 source-task families and exact base/tuned pairing.
- Cross-checked available source recording databases against aggregate outcome records. `sqlite_crosschecked` is recorded per run, rather than implying every source had a database.
- Verified Camera-view and Sensor-noise training-ID hashes and zero overlap.
- Recomputed the ten main targeted-tuning CI tables with 20,000 draws; all stored scalar/interval values agree with the source analysis within absolute tolerance 1e-10.
- Parsed all released Python sources and tested launcher plans, CLI help, bootstrap edge cases and selected reference counts without GPU model imports.
- Checked the minimal OpenPI patch against its pinned upstream source.
- Scanned textual files and decompressed episode data for personal paths, known private account/host identifiers and credential patterns.
- Removed raw logs/session UUIDs, video metadata, PDF author/date metadata and generated Python bytecode from the distributable package. Visually inspected the two video frame sequences and a numerical result figure.

`results/validation.json` records the machine-readable validation summary. Archive entries use neutral ownership/permissions and a fixed archive timestamp; no source filesystem extended attributes are copied.

Newly authored benchmark code and associated documentation use the MIT License with anonymous contributor attribution. Third-party terms are preserved separately.
