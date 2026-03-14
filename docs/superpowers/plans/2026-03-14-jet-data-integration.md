# Jet Data Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable training RAN on real jet substructure data (Pythia26/Herwig Z+jets Delphes) downloaded directly from Zenodo, bypassing the broken `energyflow` package.

**Architecture:** Two-file separation — `download_jet_data.py` handles one-time download and variable extraction, `load_jet_data.py` handles routine loading, z-score standardization, and dataset construction via existing `RAN_Dataset` infrastructure.

**Tech Stack:** NumPy, urllib (stdlib), existing `datasets.py` infrastructure.

---

## Chunk 1: Implementation

### Task 1: Create `download_jet_data.py`

**Files:**
- Create: `download_jet_data.py`

Downloads 17 `.npz` files per generator (Pythia26, Herwig) from Zenodo record 3548091. Loads only the needed array keys (skipping `particles`, `Zs`, `lhas`, `ang2s`). Extracts 6 substructure variables (`m`, `M`, `w`, `tau21`, `zg`, `sdm`) using the `get_var` logic from legacy `jet_data.py`. Saves per-variable `.npz` files (`m.npz`, `M.npz`, etc.) to `.cache/`, each containing keys `z_true`, `x_data`, `z_gen`, `x_sim`. Deletes raw Zenodo files after extraction.

- [ ] **Step 1:** Write `download_jet_data.py` with download, extraction, and cleanup logic
- [ ] **Step 2:** Verify by running `python download_jet_data.py` (or defer to user if Zenodo is slow)

### Task 2: Create `load_jet_data.py`

**Files:**
- Create: `load_jet_data.py`

Checks `.cache/` for per-variable `.npz` files. If missing, invokes `download_jet_data`. Loads each variable, subsamples to `n_samples`, standardizes each variable using MC gen-level (`z_gen`) mean/std applied to all 4 arrays. Stacks into `(n_samples, 6)` feature arrays. Feeds into `RAN_Dataset._build_dataset` / `_split_dataset`.

- [ ] **Step 1:** Write `load_jet_data.py` with cache check, standardization, and dataset construction
- [ ] **Step 2:** Verify imports and dataset shape

### Task 3: Update `main.py`

**Files:**
- Modify: `main.py`

Add `dataset: str = "gaussian"` CLI parameter. When `"jets"`, call `load_jet_dataset()` and override `dim` from its return value.

- [ ] **Step 1:** Add dataset parameter and conditional loading
- [ ] **Step 2:** Update config dump to include dataset type
- [ ] **Step 3:** Verify `python main.py --dataset jets` runs end-to-end

### Task 4: Clean up legacy file

- [ ] **Step 1:** Delete `jet_data.py` (superseded by the new files)
