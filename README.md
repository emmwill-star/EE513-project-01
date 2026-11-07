# EE513 Project 1: Camera Characterization

- Camera model: iPhone 16
- RAW capture app: TSRAW Camera

## Lens calibration

Pass image dimensions to `calibrate_camera` in OpenCV `(width, height)` order.
Checkerboard corners are refined on unannotated pixels before visualization.
Calibration and target RAW decoding use `postprocess(user_flip=0)` to keep sensor
coordinates consistent even when capture orientation metadata differs. Images
therefore display in native sensor orientation rather than automatic portrait mode.
`calculate_mean_reprojection_error` returns the mean per-view RMS and the individual
per-view RMS errors, in pixels. The notebook reports a point-weighted overall RMS;
the accompanying spread in `reprojection_rms_px` is the standard deviation across
views, not an uncertainty estimate.

After changing calibration helpers, rerun checkerboard detection, calibration,
result saving, and undistortion in order. Previously saved outputs are stale until
rerun. The target must have the same decoded dimensions, orientation, and camera
geometry as the calibration images; the notebook rejects dimension mismatches.

## Choosing an MTF square

```python
selection = utils.select_square_edge(greenPlane)  # Largest square (default).
selection = utils.select_square_edge(greenPlane, target="top-left")
```

`target` accepts `"largest"`, `"top-left"`, `"top-right"`, `"bottom-left"`, and
`"bottom-right"`. Corners refer to the input array's orientation, relative to the
largest square's center. Each corner selects the largest smaller square in that
quadrant. The selected square's right edge is used, with the same return keys and
ROI cleanup as before. Missing targets raise `ValueError`; if a small corner square
is filtered out, reduce `minAreaFraction` from its default `0.005`.
Inspect the overlay to confirm the intended square before measuring MTF.
Corner detection applies square-root tone compression only to the detection copy
to separate shaded paper from dark targets. MTF is always measured on the original
linear green samples; the default largest-square detector is unchanged.

The Measurement 2 corner cell measures all four targets in each of `edge1.DNG`
through `edge5.DNG` using `calculate_mtf50_from_mtf(..., target=option)`.
It reports a per-image table and each corner's arithmetic mean across all five
images, in cycles/micrometer. The **worst corner average** is the smallest of
these four means, not the mean of each image's worst corner. Missing images,
failed selections, and unresolved MTF50 crossings stop reporting rather than
silently averaging an incomplete set. The helper still defaults to `"largest"`
for the central-square measurement. Rerun the image-loading and Measurement 2
cells after updating the helper.
The cell also reports cycles per original sensor pixel by multiplying each
cycles/micrometer value by `p` (the sensor pitch in micrometers). The converted
per-image values are in `edgeSquareMTF50CyclesPerPixel`, corner means in
`edgeSquareAverageCyclesPerPixel`, and the lowest mean in
`worstCornerAverageCyclesPerPixel`. These are not cycles per subsampled
green-plane pixel, which would instead use `2 * p`.
The result-saving cell uses original-sensor-pixel units for all three MTF50
fields; rerun the center and corner measurement cells before saving.

## Saving measurement results

Use `add_to_results` from `helper/utils.py` (with `helper` on the Python import path):

```python
from utils import add_to_results

add_to_results(
    {"mean": 12.5, "std": 0.8},
    "results.yml",
    measurement="noise",
)
```

This saves the values under the `noise` section. Repeated calls overwrite matching
keys while preserving other keys and sections. Nested values are replaced, not
recursively merged. Missing or empty files and missing sections are initialized
automatically; the parent directory must already exist. Use YAML-safe Python values
(convert NumPy arrays/scalars with `.tolist()`/`.item()`). YAML comments and original
formatting are not preserved.