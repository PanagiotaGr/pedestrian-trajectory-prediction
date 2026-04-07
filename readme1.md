
---

# VRU Trajectory Prediction at Smart Intersections

**Diploma Thesis Project**
Trajectory prediction of Vulnerable Road Users (VRUs) in urban intersections.

---

## 1. Overview

This project focuses on **pedestrian trajectory prediction** using real-world data from a smart intersection.

The goal is to predict the future motion of a pedestrian given:

* Past trajectory (position, speed, heading)
* Scene context (ground type, crosswalk)
* Traffic lights (f1, f2, f3)
* Nearby agents (VRUs)

The pipeline builds a **fully structured dataset** suitable for machine learning models.

---

## 2. Dataset

* Source: IMPTC Dataset (IEEE IV 2023)
* Original frequency: 25 Hz
* Processed frequency: **10 Hz**

### Statistics:

* Tracks: 1,920
* Frames: 570,902
* Scenes: 264

---

## 3. Data Representation

### 3.1 Global Frame

* x: East
* y: North

### 3.2 Local Frame (Pedestrian-centric)

Each pedestrian defines a local coordinate system:

* x-axis: heading direction
* y-axis: left direction

---

## 4. Local Grid Representation

For every pedestrian and timestamp:

* Construct a **5×5 grid (1m × 1m per cell)**
* Covers a 5m × 5m area around the pedestrian

Each grid contains:

* VRUs (binary)
* Vehicles (binary)
* Ground type
* Relative positions (rel_x, rel_y)
* Relative velocities (rel_vx, rel_vy)

---

## 5. Key Challenge & Fix

### Problem:

Neighbor positions (`rel_x`, `rel_y`) are in the **local frame**, not global.

### Wrong:

```
p_B = p_A + rel
```

### Correct:

```
p_B = p_A + R_θ^T · rel
```

Where:

```
R_θ^T = [[cosθ, -sinθ],
         [sinθ,  cosθ]]
```

This inverse rotation is critical for correct spatial reconstruction.

---

## 6. Pipeline

### Stage 1 — Data Loading

* Load `grid_dataset_final.json`
* Extract:

  * positions
  * velocities
  * headings
  * grid data
  * traffic lights

---

### Stage 2 — Agent Index

* Build index using:

```
(scene, ts) → KDTree
```

* Stores exact agent positions

---

### Stage 3 — Neighbor Extraction

From 5×5 grid:

* Extract occupied cells
* Compute:

  * local coordinates
  * reconstructed global position
  * distance

---

### Stage 4 — Matching

* Match each neighbor to closest agent
* Using KDTree with threshold:

```
1.5 meters
```

Accounts for:

* grid quantization error (~0.7m)
* noise

---

### Stage 5 — Final Dataset

Each row represents:

* one pedestrian
* one timestamp

Includes:

* motion features
* environment
* traffic lights
* up to 3 nearest neighbors

---

### Stage 6 — Validation

Computed:

* match rate
* error distribution
* distance statistics
* class distribution

---

## 7. Results

### Neighbor Matching

* Match rate: **~46%**
* Mean error: **0.13 m**

### Dataset Coverage

* Frames with neighbors: **33.9%**

### Ground Types

* Sidewalk: 80%
* Crosswalk: 14%
* Road: 2%

---

## 8. Training Samples

Sliding window:

* Input: 38 frames (~3.8 sec)
* Output: 48 frames (~4.8 sec)

Dataset split:

* Train: 332K samples
* Eval: 37K
* Test: 38K

---

## 9. Baseline Model

### Constant Velocity Model

Performance:

| Metric | Value  |
| ------ | ------ |
| ADE    | 0.53 m |
| FDE    | 1.39 m |

---

## 10. Key Contributions

* Built **end-to-end data pipeline**
* Fixed **coordinate transformation bug**
* Ensured **temporal alignment using timestamps**
* Unified dataset under single source of truth
* Integrated:

  * traffic lights
  * ground segmentation
  * social context

---

## 11. Limitations

* Grid-based representation introduces ~0.7m error
* Not all neighbors matched → `unknown`
* Class imbalance (mostly pedestrians)

---

## 12. Future Work

* Deep learning model (LSTM / Transformer)
* Better neighbor matching using velocity
* Include vehicle interactions
* Context-aware prediction (traffic lights)

---

## 13. Project Structure

```
imptc_project/
├── pipeline.py
├── build_grid_dataset.py
├── build_grid_with_ground.py
├── add_traffic_lights.py
├── build_training_samples.py
├── baseline_cv_new.py
├── results/
│   ├── final_dataset.csv
│   ├── final_dataset_validation.csv
```

---

## 14. Requirements

```bash
pip install numpy scipy pandas
```

---

## 15. Conclusion

This project establishes a **complete preprocessing and dataset construction pipeline** for pedestrian trajectory prediction, enabling the transition to machine learning models.

---


