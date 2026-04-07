# VRU Trajectory Prediction at Smart Intersections

**Diploma Thesis** — Trajectory prediction of Vulnerable Road Users (VRUs) at smart
intersections for road safety and autonomous driving applications.

---

## 1. Project Overview

This project addresses the problem of **pedestrian trajectory prediction** in complex
urban intersection environments. The goal is to model the future motion of a target
pedestrian given:

- Their observed trajectory (position, speed, heading)
- The local scene context (ground surface type, crosswalk proximity)
- Traffic signal states (pedestrian crossing signals f1, f2, f3)
- The presence and motion of neighboring agents (VRUs, vehicles)

Accurate pedestrian trajectory prediction is essential for autonomous driving systems
and smart infrastructure applications, particularly at intersections where multiple
road users interact under the control of traffic signals.

The dataset used is the **IMPTC Dataset** (IEEE IV 2023), recorded at an instrumented
public intersection in Germany using LiDAR and camera sensors at 25 Hz.

---

## 2. Data Sources

### Primary Source: `grid_dataset_final.json`

This file serves as the **single source of truth** for the entire pipeline. It was
constructed by processing the raw IMPTC sequence archives and contains, for each
pedestrian track, a temporally ordered sequence of frames at **10 Hz** (downsampled
from the original 25 Hz).

Each frame contains:

| Field | Description |
|---|---|
| `ts` | Exact UTC timestamp (microseconds) |
| `ax_global`, `ay_global` | Global position of pedestrian A (meters, East/North) |
| `speed_a` | Scalar speed (m/s) |
| `theta_a` | Heading angle in global frame (radians, 0 = East) |
| `grid.vrus` | 5×5 binary occupancy map — VRU presence |
| `grid.vehicles` | 5×5 binary occupancy map — vehicle presence |
| `grid.ground` | 5×5 ground type map (road, sidewalk, crosswalk, etc.) |
| `grid.rel_x/y` | Relative position of agents per cell in **local frame of A** |
| `grid.rel_vx/y` | Relative velocity of agents per cell in **local frame of A** |
| `traffic_lights` | Pedestrian signal states: f1, f2, f3 |

**Dataset statistics:**

| Metric | Value |
|---|---|
| Total pedestrian tracks | 1,920 |
| Total frames @ 10 Hz | 570,902 |
| Scenes | 264 |
| Frames on crosswalk | 81,515 (14.3%) |
| Frames with ≥1 neighbor | 193,494 (33.9%) |

---

## 3. Initial Approach

The initial pipeline attempted to enrich the pedestrian context dataset with neighbor
class labels by:

1. Loading a pre-processed wide-format file (`ALL_VRUs_FINAL.csv`) containing
   trajectory data as per-frame columns (`x_0..x_49`, `y_0..y_49`).
2. Melting this file to long format indexed by `(seq_datetime, frame_idx)`.
3. Reconstructing approximate absolute neighbor positions as:
   ```
   nb_abs_x = pedestrian_x + nb_rel_x
   nb_abs_y = pedestrian_y + nb_rel_y
   ```
4. Querying a KDTree at the corresponding `(scene, frame_idx)` to assign class labels.

The approach produced **0% match rate**, indicating fundamental design errors.

---

## 4. Identified Problems

### 4.1 Incorrect Coordinate Transformation

The most critical error was the assumption that `nb_rel_x`, `nb_rel_y` represent
offsets in the **global** coordinate frame. In reality, these values are stored in the
**local frame of pedestrian A**, defined by:

```
R_θA (global → local):
  [[cos θ,  sin θ],
   [-sin θ, cos θ]]
```

The correct reconstruction of neighbor B's global position requires the **inverse
rotation**:

```
p_B = p_A + R_θA^T · [rel_x, rel_y]

where R_θA^T = [[cos θ, -sin θ],
                [sin θ,  cos θ]]
```

Using the identity `p_B = p_A + [rel_x, rel_y]` (without rotation) introduces a
systematic error proportional to the heading angle, making matching impossible
whenever `θ ≠ 0`.

### 4.2 Temporal Mismatch (frame_idx vs. ts)

The initial pipeline indexed agents by `frame_idx` (sequential integer), while the
pedestrian context dataset used exact UTC timestamps (`ts`). Since different agents
start at different absolute times and the sequences are not synchronized by
frame_idx, this produced misaligned lookups.

**Fix:** Use exact `ts` values as the temporal key throughout the pipeline.

### 4.3 Inconsistent Data Sources

Multiple CSV files were used simultaneously with incompatible identifiers:

| Source | Scene format | Track ID format |
|---|---|---|
| `pedestrian_context.csv` | `0000_20230322_081506` | `0000_.../000` |
| `ALL_VRUs_FINAL.csv` | `20230322_081506` | `0`, `1`, `2`... |

Only 121 out of 264 scenes overlapped, and within those, the track ID formats
differed. This caused systematic lookup failures.

**Fix:** Eliminate all external CSVs and use `grid_dataset_final.json` as the
single source of truth, where all scene, track, and timestamp identifiers are
already consistent.

### 4.4 Grid-based Neighbor Representation

The `rel_x/y` values stored in the grid are the **center coordinates of occupied
cells**, not exact agent positions. Each cell covers 1m × 1m, introducing up to
`√2 × 0.5 ≈ 0.71m` of quantization error per neighbor. This must be accounted for
in the matching threshold.

---

## 5. Methodological Fixes

### 5.1 Inverse Rotation for Coordinate Reconstruction

```python
def local_to_global_offset(rel_x, rel_y, theta):
    c, s = math.cos(theta), math.sin(theta)
    dx = c * rel_x - s * rel_y   # R_θ^T applied
    dy = s * rel_x + c * rel_y
    return dx, dy

def reconstruct_global(px, py, theta, rel_x, rel_y):
    dx, dy = local_to_global_offset(rel_x, rel_y, theta)
    return px + dx, py + dy
```

### 5.2 True Timestamp Alignment

Agent index is keyed by `(scene, ts)` using exact microsecond timestamps from
`grid_dataset_final.json`. This guarantees temporal alignment across all agents.

### 5.3 Unified Data Schema

All pipeline stages use identical identifiers:

| Variable | Format | Example |
|---|---|---|
| `scene` | `{seq_id}_{datetime}` | `0000_20230322_081506` |
| `track_id` | `{scene}/{local_id}` | `0000_20230322_081506/000` |
| `ts` | UTC microseconds (int) | `1679472907040035` |
| `x, y` | Global meters (East, North) | `−4.1459, −4.1853` |
| `class` | Original IMPTC label | `person`, `bicycle` |

### 5.4 KDTree Matching Strategy

For each neighbor cell, the reconstructed global position is queried against a
KDTree of exact agent positions at the same `(scene, ts)`. A match threshold of
**1.5m** is used, accounting for:

- Cell quantization error: ≤ 0.71m
- Floating-point precision: negligible
- Minor position drift between 10 Hz frames: ≤ 0.2m

---

## 6. Final Pipeline (`pipeline.py`)

The pipeline consists of five stages:

### Stage 1 — Load and Build Agent Index

Load `grid_dataset_final.json` and construct a `(scene, ts) → KDTree` index over
exact agent positions. This index is used for all subsequent matching operations.

### Stage 2 — Neighbor Extraction from Grid

For each frame, scan the 5×5 `vrus` layer and extract occupied cells (excluding
the center cell `[2,2]` = self). For each occupied cell, compute:
- Local frame coordinates `(rel_x, rel_y)`
- Reconstructed global coordinates `(abs_x, abs_y)` via inverse rotation
- Euclidean distance to A

### Stage 3 — Class Matching via KDTree

For each extracted neighbor, query the agent index at `(scene, ts)` and assign
the class label of the nearest matched agent within threshold.

### Stage 4 — Build Final CSV

Write one row per (pedestrian, timestamp) containing all fields from the unified
schema, including both local and global neighbor coordinates and matched class labels.

### Stage 5 — Validation

Compute and report:
- Neighbor match rate
- Match error distribution (m)
- Ground type distribution
- Traffic light state distribution
- Closest neighbor distance statistics

---

## 7. Results

### Dataset Size

| Split | Tracks | Frames @ 10Hz | Sliding Window Samples |
|---|---|---|---|
| Train (80%) | 1,536 | ~457K | 332,146 |
| Eval (10%) | 192 | ~57K | 37,794 |
| Test (10%) | 192 | ~57K | 38,871 |

### Neighbor Statistics

| Metric | Value |
|---|---|
| Frames with ≥1 neighbor | 193,494 (33.9%) |
| Frames with neighbor < 2m | 146,200 (25.6%) |
| Frames with neighbor < 5m | 193,494 (33.9%) |
| Mean neighbors per frame | 0.486 |

### Neighbor Class Distribution

| Class | Count |
|---|---|
| sidewalk | 80.0% of pedestrian frames |
| crosswalk | 14.3% |
| road | 2.3% |

### Baseline Model (Constant Velocity)

| Split | ADE (m) | FDE (m) |
|---|---|---|
| Eval | 0.5341 | 1.3869 |
| Test | 0.5495 | 1.4049 |

**By traffic light state (f1):**

| State | ADE (m) | FDE (m) |
|---|---|---|
| green | 0.4068 | 1.0122 |
| red | 0.6290 | 1.6659 |

**By ground type:**

| Ground | ADE (m) | FDE (m) |
|---|---|---|
| road | 0.4080 | 1.0651 |
| crosswalk | 0.5154 | 1.3381 |
| curb | 0.6593 | 1.6514 |

---

## 8. Limitations

### 8.1 Grid-based Quantization Error

Neighbor positions are derived from 1m × 1m grid cells. The stored `rel_x/y` values
represent cell centers, introducing up to 0.71m of positional uncertainty per
neighbor. This makes exact matching impossible and limits the achievable match rate.

### 8.2 Incomplete Class Recovery

Due to grid quantization, not all neighbors can be matched within the 1.5m threshold.
Unmatched neighbors are labeled `unknown`. Furthermore, the grid only records
VRU presence; vehicle neighbors require separate processing of the `vehicles` layer.

### 8.3 Class Imbalance

The IMPTC dataset contains predominantly pedestrians (80% of VRU tracks), which
may bias the learned model toward pedestrian-like motion patterns.

---

## 9. Future Work

- **Exact neighbor positions:** Replace grid-based representation with exact
  per-agent relative positions to eliminate quantization error.
- **Vehicle layer integration:** Explicitly model vehicle neighbors using the
  `grid.vehicles` layer with the same inverse rotation methodology.
- **Refined matching:** Use velocity information (`rel_vx/y`) to disambiguate
  multiple close agents during matching.
- **Deep trajectory model:** Train an LSTM or Transformer-based model using the
  prepared training samples (`train_X.npy`, `train_X_tl.npy`, `train_Y.npy`).
- **Context-aware prediction:** Incorporate traffic light state transitions as
  a conditioning signal for the trajectory prediction model.

---

## 10. Repository Structure

```
imptc_project/
├── pipeline.py                  # Main pipeline (clean rebuild)
├── build_grid_dataset.py        # Build 5×5 local grid per pedestrian
├── build_grid_with_ground.py    # Add ground plane segmentation
├── add_traffic_lights.py        # Add traffic light signals
├── build_training_samples.py    # Sliding window training samples
├── baseline_cv_new.py           # Constant velocity baseline
├── export_pedestrian_context.py # Export context CSV
├── analyze_context.py           # Statistical analysis
├── load_full_dataset.py         # Load IMPTC trajectory dataset
├── download_and_build.py        # Download sequence archives
├── matching.py                  # Temporal matching of VRU pairs
├── relative_motion.py           # Relative geometry computation
└── results/
    ├── final_dataset.csv        # Final unified dataset
    ├── pedestrian_context.csv   # Context per frame
    ├── baseline_cv_eval_results.csv
    └── baseline_cv_test_results.csv
```

---

## 11. Requirements

```bash
pip install numpy scipy pandas requests tqdm
```

---

## 12. References

- Hetzel et al., *The IMPTC Dataset: An Infrastructural Multi-Person Trajectory
  and Context Dataset*, IEEE IV 2023.
  DOI: [10.1109/IV55152.2023.10186776](https://doi.org/10.1109/IV55152.2023.10186776)
- Dataset download: [Zenodo](https://zenodo.org/records/14811016)
- Ground plane segmentation map: [Google Drive](https://drive.google.com/file/d/1uPcvJv-etmImUCoVugBAJUJmFrLcd5kR/view)
