# EE 513 Project 1: Camera Characterization

## Problem

Characterize an iPhone 16 camera using RAW images by measuring its lens geometry,
resolution, noise, dynamic range, and color accuracy.

## Method

The notebook processes RAW captures to calibrate camera intrinsics and lens
distortion from checkerboard images; measure center and corner MTF50 from slanted
edges; estimate conversion gain, read noise, and dark current from flat and dark
frames; calculate dynamic range; and evaluate color error before and after a
color correction matrix.  All resources and calibration tools are from open-source libraries.  

## Results

Values and uncertainties below are taken from [`results.json`](./results.json),
which is the source of truth. Uncertainties use the definitions recorded by the
notebook; they are not necessarily independent measurement uncertainties.

| Measurement | Result | Unit |
|---|---:|---|
| Mean focal length | 3125.3 ± 4.9 | pixels |
| Principal point, x | 1990.7 ± 7.1 | pixels |
| Principal point, y | 1524.7 ± 8.3 | pixels |
| Pooled reprojection RMS | 2.11 ± 0.61 | pixels |
| Center MTF50 | 0.228 ± 0.008 | cycles/original sensor pixel |
| Mean corner MTF50 | 0.191 ± 0.017 | cycles/original sensor pixel |
| Worst corner MTF50 | 0.170 ± 0.007 | cycles/original sensor pixel |
| Conversion gain | 0.262 ± 0.006 | DN/electron |
| Read noise | 6.86 ± 0.16 | electrons |
| Dark current | 2.90 ± 0.54 | electrons/second |
| Dynamic range | 10.95 ± 0.00 | stops |
| Mean color error before correction | 7.63 ± 3.69 | ΔE00 |
| Mean color error after correction | 2.14 ± 0.88 | ΔE00 |

## Reproduce

From the repository root, execute the notebook with:

```bash
uv run --project . jupyter nbconvert --to notebook --execute projects/01_project/project_main.ipynb --output project_main.executed.ipynb
```
