# 06 — Submission Contract, Container Harness & Reproducible CLI Workflow

To ensure strict zero-leakage evaluation on sealed **`test-v1`** (480 core pages + 24 negative controls) and **`fresh-YYYY`** (80 pages/year), official leaderboard submissions execute inside an **offline OCI/Docker container sandbox**. Local evaluation on public **`dev-v1`** and optional **`research_nc_eval_only`** BYOD suites uses the exact same `omr-northstar` CLI and deterministic reference scorer.

---

## 1. OCI / Docker Container Submission Contract (Sealed `test-v1`)

### 1.1 Mandatory Entrypoint & CLI Interface
Every submitted container image MUST define an `ENTRYPOINT` accepting exactly two required flags (plus optional deterministic seed):

```bash
docker run --rm \
  --network=none \
  --read-only \
  --tmpfs /tmp:rw,size=16g \
  -v /host/sealed-inputs:/input:ro \
  -v /host/submission-outputs:/output:rw \
  <submission-image:tag> \
  --input-dir /input \
  --output-dir /output
```

- **`--input-dir <PATH>` (read-only mount):** Contains the inference manifest (`items.json`) and raw image/PDF files. No ground truth, domain tags, complexity tiers, visual severity levels, staff counts, or instrumentation metadata are present.
- **`--output-dir <PATH>` (writable mount):** Container writes one canonical notation file per input item (MusicXML 4.0 `.musicxml` or MEI `.mei`) plus an optional `telemetry.json` runtime log.
- **Exit Codes:** `0` on full suite completion; non-zero exit code triggers an automated harness failure audit (partial outputs are not silently promoted).

### 1.2 Input Directory Layout (Single-Page & Multi-Page Sequences)
`/input/items.json` lists every evaluation unit in deterministic order without exposing any oracle labels:

```json
{
  "schema_version": "1.0.0",
  "split": "test-v1",
  "items": [
    {
      "item_id": "ns_t1_0042",
      "mode": "single_page",
      "files": ["pages/ns_t1_0042.png"]
    },
    {
      "item_id": "ns_seq_0017",
      "mode": "multi_page_sequence",
      "files": [
        "sequences/ns_seq_0017/page_01.png",
        "sequences/ns_seq_0017/page_02.png",
        "sequences/ns_seq_0017/page_03.png"
      ]
    }
  ]
}
```

- **Supported Input Formats:** PNG (1-bit, 8-bit grayscale, 24-bit RGB), JPEG, TIFF (single- or multi-page), and PDF (when provided, canonical rasterized PNG/TIFF pages are always supplied alongside so systems are not forced to bundle a specific PDF renderer).
- **Multi-Page Sequence Contract:** For `mode = "multi_page_sequence"`, `files` are strictly ordered (`page_index_in_work = 1..k`). Systems may emit either a single combined multi-page `<item_id>.musicxml` (with explicit `<print new-page="yes"/>` breaks) OR per-page files (`<item_id>_p01.musicxml`, ..., `<item_id>_p0k.musicxml`); the harness canonicalizes both into the same multi-page CER graph.
- **EXIF & DPI Policy:**
  - Standard pages have EXIF orientation pre-applied and stripped (upright pixel arrays).
  - Family `3B` Orientation Traps bake `90°`, `180°`, or `270°` rotations directly into pixel arrays with stripped/neutral EXIF so models cannot read camera orientation tags.
  - Physical DPI headers are omitted or neutralized to prevent domain leakage (`CDR` vs `ALS` vs `RWP`).

### 1.3 Output Naming & Negative-Control Rejection Contract
For each `item_id` in `items.json`, the container writes:
- **Music pages:** `/output/<item_id>.musicxml` (preferred) or `/output/<item_id>.mei`.
- **Negative-Control / Non-Music pages (`n_gt = 0`):** To pass the **Negative-Control Gate (`NC-Gate`)**, the system MUST either:
  1. Emit a valid empty MusicXML/MEI score containing **zero notes** (`n_pred = 0`), OR
  2. Emit an explicit empty rejection sentinel `/output/<item_id>.empty.json` (`{"item_id": "<item_id>", "has_music": false}`).
  - *Note:* Emitting `.empty.json` on a real music page (`n_gt > 0`) counts as an empty output (`Q(P) = 0`, `CF = 1`, Anti-Abstention Rule #3).

---

## 2. Sandbox Isolation, Determinism & Resource Disclosure

### 2.1 Strict Network & Filesystem Isolation
- **Zero Network Access (`--network=none`):** All model weights, tokenizers, and runtime libraries MUST be baked into the OCI image prior to submission. Any attempted outbound connection fails immediately.
- **Zero Exfiltration:** Containers have read-only access to `/input` and write access only to `/output` and ephemeral `/tmp`. Only scored aggregate metrics (`report.json`)—never raw images or model debug dumps containing input pixels—leave the sealed runner.

### 2.2 Zero Oracle Routing Enforcement
- Submissions MUST run a **single unified pipeline** across all 4 domains (`CDR`, `ALS`, `RWP`, `HWM`), 6 complexity tiers (`T0`–`T5`), and 4 visual severity levels (`V1`–`V4`).
- Learned internal routing/classification from raw pixels is explicitly **permitted**.
- Conditioning on filenames, file sizes, EXIF tags, or manual per-domain model selection is strictly **prohibited**; the sealed harness anonymizes all `item_id` filenames into opaque hashes (`item_0001.png` ... `item_0504.png`) before mounting `/input`.

### 2.3 Deterministic Timeouts & Hardware/Memory Disclosure
- **Per-Page Soft Timeout:** `120 s` per single page (or `120 s × k` for a `k`-page sequence). Timed-out pages are recorded as unparseable (`Q(P) = 0`, `CF = 1`).
- **Total Suite Wall-Clock Cap:** `4 hours` for the complete `test-v1` suite (480 core + 24 negative controls).
- **Mandatory Efficiency & Determinism Disclosure:** Every leaderboard submission discloses:
  - Hardware profile used (CPU-only vs. GPU model/VRAM),
  - Peak host RAM and peak GPU VRAM (`peak_memory_gb`),
  - Median and P95 per-page latency (`median_latency_s`, `p95_latency_s`),
  - Active parameter count (`parameter_count`),
  - Fixed inference seed & deterministic decoding mode (re-running a random 10-page subset must reproduce identical `Q(P)` within `±0.05` points).

---

## 3. Evaluation-Only Non-Commercial Research Track (`research_nc_eval_only`)

Alongside the 100% permissive Core Track (`Public-Domain`, `CC0-1.0`, `CC-BY-4.0`), `OMR-NS` supports an optional **Evaluation-Only Non-Commercial Research Track (`research_nc_eval_only`)** using **Bring-Your-Own-Data (BYOD) manifest pointers**:
- Users download non-commercial handwritten/archival research corpora (e.g., `CC-BY-NC` datasets such as CVC-MUSCIMA / MUSCIMA++) directly from their upstream academic hosts under their original licenses.
- `OMR-NS` distributes only lightweight BYOD manifest pointers (file checksums, canonical split IDs, and ground-truth alignment metadata)—never redistributing `CC-BY-NC` images or annotations commercially.
- Commercial and permissive OMR models that never train on `CC-BY-NC` data gain a **guaranteed zero-leakage, high-diversity handwritten evaluation track** without removing permissive handwritten pages from their training pools.
- Track results are reported separately (`track = "research_nc_eval_only"`) and never mixed into the permissive `test-v1` headline `NS`.

---

## 4. Reproducible `omr-northstar` CLI Workflow

The zero-dependency reference package `omr_northstar` provides five deterministic CLI subcommands:

### 4.1 `score` — Compute 4-Layer Scorecard & Canonical `report.json`
```bash
python -m omr_northstar score path/to/results.json \
  --system-name "my-omr-v1" \
  --split dev-v1 \
  --json-out report_a.json \
  --include-per-page
```
- Computes per-page `Q(P)` (`40/25/20/15`), domain strata `Q_s` & `CFR_s`, primary `NS`, companion `NS_tier`, `wFPP` (median/P90/P99), `NTSR`, `NUPR`, paired deltas (`Δ_engraver`, `Δ_photo`, `Δ_handwriting`), 6×4 matrix, 8-bucket error taxonomy, and Worst-10/Best-10 forensic roster.
- Pass `--include-per-page` to include `per_page_scores` in `layer4_forensic_pack` for downstream paired permutation testing via `compare`.
- Exit code `0` on success; `1` on invalid input payload.

### 4.2 `compare` — Pairwise System Comparison with Work-Clustered Permutation Test
```bash
python -m omr_northstar compare report_a.json report_b.json \
  --n-perm 1000 \
  --seed 42 \
  --alpha 0.05 \
  --json-out diff_a_vs_b.json
```
- Takes two canonical `report.json` files (produced with `--include-per-page`) and computes $\Delta NS = NS(B) - NS(A)$, per-domain $\Delta Q_s$, per-tier $\Delta Q_t$, $\Delta \text{wFPP}$, and two-sided work-clustered permutation p-values (`--alpha 0.05` significance gate) so multi-page works are never treated as independent samples.
- Exit code `0` on success; `1` on mismatched page sets or invalid input.

### 4.3 `validate-manifest` — Schema & Licensing Integrity Check
```bash
python -m omr_northstar validate-manifest path/to/dev_manifest.json
```
- Verifies all required fields (`page_id`, `work_id`, `split`, `domain`, `complexity_tier`, `visual_tier`, `repertoire_stratum`, `staff_space_px`, `geometry`, `photometry`, `optics_codec`, `provenance`, `license`), uniqueness of `page_id`, enum validity, and permissive licensing rules (`PDM-1.0` / `Public-Domain` with `--allow-legacy-pd`, `CC0-1.0`, `CC-BY-4.0` for core splits; `CC-BY-NC-*` permitted exclusively under `research_nc_eval_only`).
- Exit code `0` if valid; `1` if any record fails validation.

### 4.4 `validate-splits` — Zero-Leakage Disjointness Audit
```bash
python -m omr_northstar validate-splits path/to/dev_manifest.json path/to/test_manifest.json
```
- Validates both `dev` and `test` manifests and enforces all anti-contamination disjointness invariants across splits:
  1. **Zero `work_id` overlap** across splits.
  2. **Zero `writer_id` overlap** (`HWM`) across splits.
  3. **`>= 50%` unseen `capture_device_id`** (`RWP`) in `test-v1` relative to `dev-v1`.
  4. **Engraver/publisher diversity cap (`<= 15%`)** on printed/clean slices.
- Exit code `0` if all leakage and diversity gates pass; `1` with explicit violation diagnostics otherwise.

### 4.5 `validate-report` — Canonical Report Schema Verification
```bash
python -m omr_northstar validate-report report_a.json
```
- Validates a generated `report.json` against `schemas/report.schema.json`. Exit code `0` if valid; `1` otherwise.
