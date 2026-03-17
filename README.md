# MRI-US Image Fusion for Endometriosis Detection

This project is a small experiment harness for MRI/Ultrasound image fusion. It includes a PALM baseline (optimization-based), a DDFM baseline (diffusion-based), and hybrid runs that use PALM outputs to initialize DDFM.

## What’s in this repo

- `PALM/`: PALM implementation (El Mansouri et al., 2020)
- `DDFM/`: DDFM implementation (Zhao et al., 2023) with small glue-code changes
- `scripts/run.py`: one entry point to run baselines, hybrids, and evaluation
- `data/`: input images used by the runners
- `experiments/`: outputs produced by `scripts/run.py`

## Data

- `data/irm.png`, `data/us.png` (used by `scripts/run.py`)

## Quick start

Create an environment, install dependencies, then run:

```bash
python scripts/run.py --palm
python scripts/run.py --ddfm
python scripts/run.py --hybrids
python scripts/run.py --evaluate
```

You can also run everything in one go:

```bash
python scripts/run.py --all
```