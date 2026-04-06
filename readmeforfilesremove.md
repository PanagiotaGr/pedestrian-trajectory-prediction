# Repository Cleanup Plan

## Final Pipeline Dependencies

The final pipeline consists of these stages:

```
Stage 1: Download data
Stage 2: Build grid dataset
Stage 3: Add ground + traffic lights
Stage 4: Build training samples
Stage 5: Baseline model
Stage 6: Analysis & export
```

---

## ✅ FILES TO KEEP

### Core Pipeline
| File | Stage | Purpose |
|------|-------|---------|
| `pipeline.py` | All | Clean unified pipeline — single source of truth |
| `download_and_build.py` | 1 | Download all 17GB archives from Zenodo |
| `load_full_dataset.py` | 1 | Load trajectory dataset (2.9GB) |
| `build_grid_dataset.py` | 2 | Build 5×5 local grid per pedestrian @ 10Hz |
| `build_grid_with_ground.py` | 2 | Add ground plane segmentation |
| `add_traffic_lights.py` | 2 | Add f1/f2/f3 traffic light signals |
| `build_training_samples.py` | 3 | Sliding window → train/eval/test .npy |
| `baseline_cv_new.py` | 4 | Constant velocity baseline (ADE/FDE) |
| `export_pedestrian_context.py` | 5 | Export context CSV with neighbors |
| `analyze_context.py` | 5 | Statistical analysis of context CSV |

### Scripts folder (keep)
| File | Purpose |
|------|---------|
| `scripts/build_pedestrian_math_dataset_v2.py` | Pairwise interaction dataset (used for analysis) |

---

## ❌ FILES TO REMOVE (Legacy / Outdated)

### Superseded by `pipeline.py`
| File | Reason |
|------|--------|
| `build_final_dataset.py` | Replaced by `pipeline.py` (had coordinate bug) |
| `match_neighbor_classes.py` | Replaced by `pipeline.py` (wrong frame_idx matching) |
| `relative_motion.py` | Logic absorbed into `pipeline.py` |
| `matching.py` | Superseded by agent index in `pipeline.py` |
| `build_dataset.py` | Early version, replaced by grid approach |

### Superseded by newer versions
| File | Reason |
|------|--------|
| `load_imptc_trajectories.py` | Superseded by `load_full_dataset.py` |
| `build_final_csv.py` | Old export, replaced by `export_pedestrian_context.py` |
| `build_final_all_vrus.py` | Merged into pipeline |
| `build_per_vru_csv.py` | Not used in final pipeline |
| `build_social_interactions.py` | Experimental, not in final pipeline |
| `build_interactions.py` | Superseded by `scripts/build_pedestrian_math_dataset_v2.py` |
| `build_interactions_v2.py` | Same as above |
| `enrich_crosswalk_cases.py` | One-off analysis, not core pipeline |
| `add_ground_type.py` | Replaced by `build_grid_with_ground.py` |
| `add_traffic_lights_final.py` | Replaced by `add_traffic_lights.py` |
| `build_relative_geometry_semantic_tls.py` | Experimental |
| `build_training_samples_chunked.py` | Old version with RAM issue |
| `preprocess_imptc.py` | Early preprocessing, not used |
| `preprocess_rotation.py` | Logic moved to `pipeline.py` |
| `save_csv.py` | Utility replaced inline |
| `find_matches.py` | Replaced by agent index in `pipeline.py` |
| `imptc_mapper.py` | Experimental mapper |
| `baseline_cv.py` | Old baseline, replaced by `baseline_cv_new.py` |
| `quick_model_test.py` | Throwaway test script |
| `test_first_chunk.py` | Throwaway test script |

---

## 📁 Proposed Clean Structure

```
imptc_project/
│
├── README.md
├── requirements.txt
│
├── data/                          # Downloaded archives (not in git)
│   ├── imptc_set_01.tar.gz
│   ├── ...
│   ├── imptc_trajectory_dataset.tar.gz
│   └── ground_plane_map.csv
│
├── pipeline/                      # Core pipeline scripts
│   ├── 01_download.py             # (= download_and_build.py)
│   ├── 02_load_dataset.py         # (= load_full_dataset.py)
│   ├── 03_build_grid.py           # (= build_grid_dataset.py)
│   ├── 04_add_ground.py           # (= build_grid_with_ground.py)
│   ├── 05_add_traffic_lights.py   # (= add_traffic_lights.py)
│   ├── 06_build_samples.py        # (= build_training_samples.py)
│   └── pipeline.py                # Unified pipeline (neighbor matching)
│
├── models/                        # (future) neural network models
│   └── baseline_cv.py             # (= baseline_cv_new.py)
│
├── analysis/                      # Analysis & export scripts
│   ├── export_context.py          # (= export_pedestrian_context.py)
│   └── analyze_context.py        # (= analyze_context.py)
│
└── results/                       # Output files (not in git except CSVs)
    ├── final_dataset.csv
    ├── pedestrian_context.csv
    ├── baseline_cv_eval_results.csv
    └── baseline_cv_test_results.csv
```

---

## Safe Cleanup Commands

```bash
cd ~/imptc_project

# Create backup first!
mkdir -p legacy_scripts
cp *.py legacy_scripts/

# Move legacy files to legacy folder
mv build_final_dataset.py legacy_scripts/
mv match_neighbor_classes.py legacy_scripts/
mv relative_motion.py legacy_scripts/
mv matching.py legacy_scripts/
mv build_dataset.py legacy_scripts/
mv load_imptc_trajectories.py legacy_scripts/
mv build_final_csv.py legacy_scripts/ 2>/dev/null
mv build_final_all_vrus.py legacy_scripts/
mv build_per_vru_csv.py legacy_scripts/
mv build_social_interactions.py legacy_scripts/
mv build_interactions.py legacy_scripts/
mv build_interactions_v2.py legacy_scripts/
mv enrich_crosswalk_cases.py legacy_scripts/
mv add_ground_type.py legacy_scripts/ 2>/dev/null
mv add_traffic_lights_final.py legacy_scripts/ 2>/dev/null
mv build_relative_geometry_semantic_tls.py legacy_scripts/
mv build_training_samples_chunked.py legacy_scripts/ 2>/dev/null
mv preprocess_imptc.py legacy_scripts/ 2>/dev/null
mv preprocess_rotation.py legacy_scripts/ 2>/dev/null
mv save_csv.py legacy_scripts/ 2>/dev/null
mv find_matches.py legacy_scripts/ 2>/dev/null
mv imptc_mapper.py legacy_scripts/ 2>/dev/null
mv baseline_cv.py legacy_scripts/
mv quick_model_test.py legacy_scripts/ 2>/dev/null
mv test_first_chunk.py legacy_scripts/ 2>/dev/null

echo "Done! Files remaining:"
ls *.py
```

---

## Files remaining after cleanup

```
imptc_project/
├── pipeline.py
├── download_and_build.py
├── load_full_dataset.py
├── build_grid_dataset.py
├── build_grid_with_ground.py
├── add_traffic_lights.py
├── build_training_samples.py
├── baseline_cv_new.py
├── export_pedestrian_context.py
├── analyze_context.py
└── legacy_scripts/   ← backup, can delete later
```

Total: **10 active scripts** instead of 25+

---

## Git cleanup

```bash
# Update .gitignore to exclude legacy_scripts
echo "legacy_scripts/" >> .gitignore

# Update git
git add -A
git commit -m "Cleanup: move legacy scripts to legacy_scripts/"
git push origin main
```
