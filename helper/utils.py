from collections.abc import Iterable, Mapping
import json
from pathlib import Path
import re
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from psu_imaging import srgb_to_linear

import rawpy
import sys
import os
from pathlib import Path
import importlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, ExifTags
import imageio
import tifffile

RAW_EXTENSIONS = {".dng", ".raw", ".nef", ".arw", ".cr2", ".cr3", ".raf", ".orf", ".rw2", ".pef", ".srw"}


def update_json_values(jsonPath: str | Path, updates: Mapping[str, object]) -> None:
    """Replace values of existing top-level keys in a JSON file.

    Pass a mapping of keys to replacement values, for example:
        update_json_values("results.json", {
            "camera": "My camera",
            "focal_length_mm": [4.25, 0.02],
            "read_noise_e": None,
        })

    All keys are checked before writing; unknown keys raise KeyError.
    Unspecified values and the top-level key order are preserved. Values are
    replaced, not appended or merged. An empty mapping leaves the file unchanged.
    The file must already exist and contain a JSON object (otherwise ValueError).
    Output uses UTF-8 and two-space indentation; original whitespace is not retained.
    File errors, JSON decoding errors, and serialization errors propagate.
    Invalid updates never overwrite the file, including NaN/infinite values.
    """
    if not isinstance(updates, Mapping):
        raise TypeError("updates must be a mapping of existing keys to replacement values")

    jsonPath = Path(jsonPath)
    with jsonPath.open(encoding="utf-8") as jsonFile:
        data = json.load(jsonFile)
    if not isinstance(data, dict):
        raise ValueError("JSON file must contain a top-level object")

    for key in updates:
        if key not in data:
            raise KeyError(f"JSON key does not exist: {key!r}")
    if not updates:
        return

    data.update(updates)
    serialized = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    jsonPath.write_text(serialized + "\n", encoding="utf-8")



def lens_id(path):
    """Return a dictionary of lens and exposure metadata.

    Keys: lensModel, focalLengthMm, focalLength35Mm, imageSize (width, height),
    iso, and exposureTimeSeconds.
    Missing required EXIF tags raise KeyError; file reading errors propagate.
    """
    with tifffile.TiffFile(path) as imageFile:
        page = imageFile.pages[0]
        exif = page.tags["ExifTag"].value
        focalLength = exif["FocalLength"]
        iso = exif["ISOSpeedRatings"]
        exposureTime = exif["ExposureTime"]

        return {
            "lensModel": exif.get("LensModel", "unknown"),
            "focalLengthMm": round(focalLength[0] / focalLength[1], 3),
            "focalLength35Mm": exif.get("FocalLengthIn35mmFilm"),
            "imageSize": (page.tags["ImageWidth"].value, page.tags["ImageLength"].value),
            "iso": iso,
            "exposureTimeSeconds": exposureTime[0] / exposureTime[1],
        }

def _load_image(imagePath: Path) -> Image.Image:
	if imagePath.suffix.lower() in RAW_EXTENSIONS:
		import rawpy

		with rawpy.imread(str(imagePath)) as raw:
			return Image.fromarray(raw.postprocess(use_camera_wb=True))
	with Image.open(imagePath) as image:
		return ImageOps.exif_transpose(image).convert("RGB")


def _read_camera_id(imagePath: Path) -> str | None:
	if imagePath.suffix.lower() != ".dng":
		return None

	import tifffile

	with tifffile.TiffFile(imagePath) as dngFile:
		tags = dngFile.pages[0].tags
		cameraId = "Unknown camera"
		for tagCode in (272, 271):
			tag = tags.get(tagCode)
			if tag is not None:
				value = str(tag.value).strip("\x00 ")
				if value:
					cameraId = value
					break

		exifTag = tags.get(34665)
		exif = exifTag.value if exifTag is not None else {}
		cameraDescriptions = [exif.get("LensModel", "")]
		for tagCode in (50708, 42036):
			tag = tags.get(tagCode)
			if tag is not None:
				cameraDescriptions.append(tag.value)
		for description in cameraDescriptions:
			cameraNumber = re.search(
				r"\bcamera\s+(?:#\s*|(?:number|no\.?)\s*)?(\d+)\b(?!\.|\s*(?:mm|cm|m)\b)",
				str(description),
				re.IGNORECASE,
			)
			if cameraNumber is not None:
				return f"{cameraId} | Camera {cameraNumber.group(1)}"
		return cameraId


def get_noise_profile_status(imagePath: str | Path) -> dict:
	"""Return NoiseProfile presence and coefficient values for a TIFF/DNG."""
	import tifffile

	with tifffile.TiffFile(imagePath) as dngFile:
		for page in dngFile.pages:
			noiseProfile = page.tags.get(51041)
			if noiseProfile is not None:
				return {"present": True, "values": tuple(noiseProfile.value)}
	return {"present": False, "values": None}


def has_noise_profile(imagePath: str | Path) -> bool:
	"""Return whether a TIFF/DNG contains the DNG NoiseProfile tag."""
	return get_noise_profile_status(imagePath)["present"]

## Measurement 1
def create_contact_sheet(
	imagePaths: Iterable[str | Path],
	*,
	columns: int = 5,
	tileWidth: int = 480,
	tileHeight: int = 380,
	labelHeight: int = 40,
	fontSize: int = 24,
	outputPath: str | Path | None = None,
) -> Image.Image:
	"""Create a labeled contact sheet from RAW or Pillow-supported image files.

	Paths are used in the supplied order; rows are calculated automatically.
	Tile and label dimensions are in pixels. Images retain their aspect ratio.
	Filenames appear above photos; DNG camera IDs appear in a bottom label
	band of labelHeight pixels. IDs use Model, falling back to Make or "Unknown camera".
	Explicit camera numbers in LensModel or UniqueCameraModel are appended;
	focal lengths and internal hardware model numbers are not camera numbers.
	Long labels use a smaller font to fit within their tile.
	Pass labelHeight=0 to hide labels. Relative paths use the working directory.
	If outputPath is provided, save there (its parent directory must exist).
	Return the RGB Pillow image without printing or displaying it.

	Raises ValueError for empty inputs or invalid layout dimensions, and
	FileNotFoundError for missing input files. Image decoding errors propagate.
	RAW decoding requires rawpy and uses camera white balance.
    
    Usage: 
        from pathlib import Path
        from utils import create_contact_sheet

        imagePaths = sorted(
            Path("images").glob("cb*.DNG"),
            key=lambda imagePath: int(imagePath.stem[2:]),
        )

        sheet = create_contact_sheet(
            imagePaths,
            columns=5,
            outputPath="images/cb_contact_sheet.png",
        )
        display(sheet)  # Notebook only
 
	"""
	layoutValues = (columns, tileWidth, tileHeight, labelHeight, fontSize)
	if any(not isinstance(value, int) or isinstance(value, bool) for value in layoutValues):
		raise ValueError("Layout dimensions and fontSize must be integers")
	if columns < 1 or fontSize < 1 or labelHeight < 0:
		raise ValueError("columns and fontSize must be positive; labelHeight must be nonnegative")
	if tileWidth <= 24 or tileHeight <= labelHeight + 16:
		raise ValueError("Tiles must leave room for images and padding")
	if labelHeight and labelHeight < fontSize + 8:
		raise ValueError("labelHeight must be at least fontSize + 8, or zero to hide labels")

	paths = [Path(imagePath) for imagePath in imagePaths]
	if not paths:
		raise ValueError("At least one image path is required")
	for imagePath in paths:
		if not imagePath.is_file():
			raise FileNotFoundError(f"Image file not found: {imagePath}")

	rows = (len(paths) + columns - 1) // columns
	sheet = Image.new("RGB", (columns * tileWidth, rows * tileHeight), "white")
	draw = ImageDraw.Draw(sheet)
	font = ImageFont.load_default(size=fontSize)

	for index, imagePath in enumerate(paths):
		with _load_image(imagePath) as image:
			cameraId = _read_camera_id(imagePath) if labelHeight else None
			footerHeight = labelHeight if cameraId is not None else 0
			imageAreaHeight = tileHeight - labelHeight - footerHeight
			if imageAreaHeight <= 16:
				raise ValueError("Tiles must leave room for images, camera labels, and padding")
			image.thumbnail(
				(tileWidth - 24, imageAreaHeight - 16),
				Image.Resampling.LANCZOS,
			)
			left = (index % columns) * tileWidth
			top = (index // columns) * tileHeight
			if labelHeight:
				labels = [(imagePath.stem, top + 8)]
				if cameraId is not None:
					labels.append((cameraId, top + tileHeight - footerHeight + 8))
				for label, labelTop in labels:
					labelFont = font
					labelFontSize = fontSize
					while draw.textlength(label, font=labelFont) > tileWidth - 24 and labelFontSize > 1:
						labelFontSize -= 1
						labelFont = ImageFont.load_default(size=labelFontSize)
					draw.text((left + 12, labelTop), label, fill="black", font=labelFont)
			imageLeft = left + (tileWidth - image.width) // 2
			imageTop = top + labelHeight + (imageAreaHeight - image.height) // 2
			sheet.paste(image, (imageLeft, imageTop))
			draw.rectangle(
				(left, top, left + tileWidth - 1, top + tileHeight - 1),
				outline="#b8b8b8",
				width=2,
			)

	if outputPath is not None:
		sheet.save(outputPath)
	return sheet

def checkerboard_found(imagePath: Path, boardSize):
    """Detect corners in unrotated sensor coordinates and return an annotated BGR image."""
    imgPoints = []
    objPoints = []
    
    objp = np.zeros((boardSize[0] * boardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:boardSize[0], 0:boardSize[1]].T.reshape(-1, 2)
    with rawpy.imread(str(imagePath)) as raw:
        rgb = raw.postprocess(user_flip=0)  # Keep all views in the same sensor coordinates.

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ret, corners = cv2.findChessboardCorners(bgr, boardSize, None)

    if ret:
        print(f"Chessboard corners found for {imagePath.name}.")
        objPoints.append(objp.copy())

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria=(
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            ),
        )
        imgPoints.append(corners)
        cv2.drawChessboardCorners(bgr, boardSize, corners, ret)
        
    else:
        print(f"Chessboard corners not found for {imagePath.name}.")
        
    return imgPoints, objPoints, bgr

def measure_checkerboard_straightness(imagePoints, boardSize):
    """Fit orthogonal lines to every checkerboard row and column.

    Return rmsPx and maxPx perpendicular residuals pooled over both directions,
    plus fitted line segments and per-line rms values. These describe grid
    straightness, not reprojection error or independent measurement uncertainty.
    """
    columns, rows = boardSize
    points = np.asarray(imagePoints, dtype=float)
    if columns < 3 or rows < 3:
        raise ValueError("Straightness requires at least three corners in each direction")
    if points.shape not in ((rows * columns, 2), (rows * columns, 1, 2)):
        raise ValueError("imagePoints must match boardSize with two coordinates per corner")
    if not np.isfinite(points).all():
        raise ValueError("imagePoints must be finite")
    grid = points.reshape(rows, columns, 2)
    lines = []
    residuals = []
    for direction, groups in (("row", grid), ("column", grid.transpose(1, 0, 2))):
        for index, group in enumerate(groups):
            center = group.mean(axis=0)
            _, singularValues, axes = np.linalg.svd(group - center, full_matrices=False)
            if singularValues[0] == 0:
                raise ValueError("A checkerboard line must contain distinct points")
            along = (group - center) @ axes[0]
            perpendicular = (group - center) @ axes[1]
            residuals.extend(perpendicular)
            lines.append({
                "direction": direction,
                "index": index,
                "points": group,
                "segment": center + np.outer([along.min(), along.max()], axes[0]),
                "rmsPx": float(np.sqrt(np.mean(perpendicular ** 2))),
            })
    residuals = np.asarray(residuals)
    return {
        "rmsPx": float(np.sqrt(np.mean(residuals ** 2))),
        "maxPx": float(np.max(np.abs(residuals))),
        "lines": lines,
    }


def calibrate_camera(imgPoints, objPoints, imageShape):
    """Calibrate using imageShape in OpenCV (width, height) order."""
    if not imgPoints or not objPoints:
        raise ValueError("imgPoints and objPoints must not be empty")
    if len(imgPoints) != len(objPoints):
        raise ValueError("imgPoints and objPoints must have the same length")
    
    objPoints = [np.array(op, dtype=np.float32) for op in objPoints]
    # imgPoints = [np.array(ip, dtype=np.float32) for ip in imgPoints]
    (
        ret, cameraMatrix, distCoeffs, rvecs, tvecs,
        stdIntrinsics, stdExtrinsics, perViewErrors,
    ) = cv2.calibrateCameraExtended(objPoints,
            imgPoints,
            imageShape,
            None,
            None,
            flags=cv2.CALIB_FIX_K3
    )
    if not ret:
        raise RuntimeError("Camera calibration failed")
    return cameraMatrix, distCoeffs, rvecs, tvecs, stdIntrinsics, stdExtrinsics, perViewErrors
    
def calculate_mean_reprojection_error(imgPoints, objPoints, cameraMatrix, distCoeffs, rvecs, tvecs):
    """Return the mean and list of per-view RMS Euclidean errors in pixels."""
    meanError = 0
    errors = []
    for i in range(len(objPoints)):
        imgPoints2, _ = cv2.projectPoints(
            objPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs
        )
        observed = imgPoints[i].reshape(-1, 2)
        projected = imgPoints2.reshape(-1, 2)
        error = np.sqrt(np.mean(np.sum((observed - projected) ** 2, axis=1)))
        meanError += error
        errors.append(error)

    meanError /= len(objPoints)
    return meanError, errors
    
def print_calibrations(cameraMatrix, distCoeffs, rvecs, tvecs):
    fx = float(cameraMatrix[0, 0])
    fy = float(cameraMatrix[1, 1])
    cx = float(cameraMatrix[0, 2])
    cy = float(cameraMatrix[1, 2])
    k1, k2, p1, p2, k3 = distCoeffs.ravel()[:5]
    
    print(f"Calibrated focal lengths: fx = {fx:.3f} px, fy = {fy:.3f} px")
    print(f"Principal point: cx = {cx:.3f} px, cy = {cy:.3f} px")
    print(f"Distortion coefficients: k1 = {k1:.6f}, k2 = {k2:.6f}, p1 = {p1:.6f}, p2 = {p2:.6f}, k3 = {k3:.6f}")
    
    
## Measurement 2
def select_square_edge(
	image: np.ndarray,
	*,
	target: Literal["largest", "top-left", "top-right", "bottom-left", "bottom-right"] = "largest",
	minAreaFraction: float = 0.005,
	roiWidth: int = 48,
	edgeFraction: float = 0.5,
) -> dict:
	"""Find a solid dark square-like target and its right-edge ROI.

	Select target="largest" (default) or "top-left", "top-right", "bottom-left",
	or "bottom-right". Corner targets are the largest smaller square in that
	quadrant relative to the largest square's center, in input-image coordinates.
	Lower minAreaFraction if the smaller corner targets are filtered out.

	Accept a finite 2D green plane. Threshold only a normalized detection copy;
	do not change or resample the input. Allow moderate perspective distortion.
	Return squareCorners, edgeEndpoints, roi=(x, y, width, height), and edgeAngleDeg
	in input-image coordinates. The ROI spans the middle edgeFraction of a
	near-vertical dark-to-light edge, with roiWidth pixels of cross-edge padding.
	Try narrower padding and retain the longest clean run of rows when nearby
	markings contaminate the ROI. Keep at least one third of the requested rows.
	Raise ValueError if no suitable target or unclipped edge ROI is available.
	This is a geometric detector: inspect its overlay before measuring MTF.
	"""
	imageArray = np.asarray(image)
	if target not in ("largest", "top-left", "top-right", "bottom-left", "bottom-right"):
		raise ValueError("target must be largest, top-left, top-right, bottom-left, or bottom-right")
	if imageArray.ndim != 2 or imageArray.size == 0:
		raise ValueError("image must be a non-empty 2D intensity array")
	if not np.issubdtype(imageArray.dtype, np.number) or not np.isfinite(imageArray).all():
		raise ValueError("image must contain finite numeric intensities")
	if not 0 < minAreaFraction < 0.5 or not 0 < edgeFraction < 1:
		raise ValueError("minAreaFraction must be in (0, 0.5); edgeFraction must be in (0, 1)")
	if not isinstance(roiWidth, int) or isinstance(roiWidth, bool) or roiWidth < 8:
		raise ValueError("roiWidth must be an integer of at least 8 pixels")
	lowerLevel, upperLevel = np.percentile(imageArray, (1, 99))
	if upperLevel <= lowerLevel:
		raise ValueError("image has insufficient intensity contrast")
	detectionLevels = np.clip(
		(imageArray.astype(np.float64) - lowerLevel) / (upperLevel - lowerLevel),
		0, 1,
	)
	if target != "largest":
		detectionLevels = np.sqrt(detectionLevels)  # Separate shaded paper from dark corner targets.
	detectionImage = (255 * detectionLevels).astype(np.uint8)
	detectionImage = cv2.GaussianBlur(detectionImage, (5, 5), 0)
	_, mask = cv2.threshold(detectionImage, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	imageHeight, imageWidth = imageArray.shape
	candidates = []
	for contour in contours:
		area = cv2.contourArea(contour)
		if not minAreaFraction * imageArray.size <= area <= 0.5 * imageArray.size:
			continue
		polygon = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
		if len(polygon) != 4 or not cv2.isContourConvex(polygon):
			continue
		corners = polygon.reshape(4, 2)
		if np.any(corners <= 2) or np.any(corners[:, 0] >= imageWidth - 3) or np.any(corners[:, 1] >= imageHeight - 3):
			continue
		_, (boxWidth, boxHeight), _ = cv2.minAreaRect(polygon)
		if min(boxWidth, boxHeight) == 0 or max(boxWidth, boxHeight) / min(boxWidth, boxHeight) > 1.6:
			continue
		polygonMask = np.zeros_like(mask)
		cv2.fillPoly(polygonMask, [polygon], 255)
		if np.mean(mask[polygonMask != 0] != 0) < 0.9:
			continue
		candidates.append((area, corners))
	if not candidates:
		raise ValueError("No large solid square target found; inspect the capture or minAreaFraction")
	largestArea, squareCorners = max(candidates, key=lambda candidate: candidate[0])
	if target != "largest":
		centerX, centerY = np.mean(squareCorners, axis=0)
		cornerCandidates = []
		for area, corners in candidates:
			candidateX, candidateY = np.mean(corners, axis=0)
			isHorizontalMatch = candidateX < centerX if target.endswith("left") else candidateX > centerX
			isVerticalMatch = candidateY < centerY if target.startswith("top") else candidateY > centerY
			if area < largestArea and isHorizontalMatch and isVerticalMatch:
				cornerCandidates.append((area, corners))
		if not cornerCandidates:
			raise ValueError(
				f"No {target} square target found; inspect the capture or lower minAreaFraction"
			)
		_, squareCorners = max(cornerCandidates, key=lambda candidate: candidate[0])
	edges = []
	for index in range(4):
		endpoints = squareCorners[[index, (index + 1) % 4]].astype(np.float64)
		deltaX, deltaY = endpoints[1] - endpoints[0]
		if abs(deltaY) > abs(deltaX):
			edges.append(endpoints)
	if not edges:
		raise ValueError("The square has no near-vertical edge suitable for this MTF function")
	edgeEndpoints = max(edges, key=lambda endpoints: float(np.mean(endpoints[:, 0])))
	midpoint = np.mean(edgeEndpoints, axis=0)
	innerEndpoints = midpoint + (edgeEndpoints - midpoint) * edgeFraction
	top = int(np.ceil(np.min(innerEndpoints[:, 1])))
	bottom = int(np.floor(np.max(innerEndpoints[:, 1]))) + 1
	if top < 0 or bottom > imageHeight or bottom - top < 8:
		raise ValueError("Selected edge ROI is too small or extends outside the image")
	cleanRois = []
	for paddingWidth in dict.fromkeys((roiWidth, max(8, roiWidth // 2), max(8, roiWidth // 4))):
		left = int(np.floor(np.min(innerEndpoints[:, 0]) - paddingWidth / 2))
		right = int(np.ceil(np.max(innerEndpoints[:, 0]) + paddingWidth / 2))
		if left < 0 or right > imageWidth:
			continue
		roiImage = imageArray[top:bottom, left:right]
		plateauWidth = max(1, roiImage.shape[1] // 5)
		darkLevels = np.median(roiImage[:, :plateauWidth], axis=1)
		lightPlateau = roiImage[:, -plateauWidth:]
		lightLevels = np.median(lightPlateau, axis=1)
		cleanRows = (
			(lightLevels > darkLevels)
			& (np.min(lightPlateau, axis=1) > darkLevels + 0.25 * (lightLevels - darkLevels))
		)
		transitions = np.diff(np.concatenate(([False], cleanRows, [False])).astype(np.int8))
		for start, stop in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
			if stop - start >= max(8, int(np.ceil((bottom - top) / 3))):
				cleanRois.append((int(stop - start), (left, top + int(start), right - left, int(stop - start))))
		if cleanRows.all():
			break
	if not cleanRois:
		raise ValueError("Selected edge has no sufficiently long clean dark-to-light ROI")
	_, roi = max(cleanRois, key=lambda candidate: candidate[0])
	deltaX, deltaY = edgeEndpoints[1] - edgeEndpoints[0]
	return {
		"squareCorners": squareCorners,
		"edgeEndpoints": edgeEndpoints,
		"roi": roi,
		"edgeAngleDeg": float(np.rad2deg(np.arctan(deltaX / deltaY))),
	}

def measure_slanted_edge_mtf(
    image,
    roi,
    sample_spacing=1.0,
    oversampling=4,
):
    """Measure slanted-edge MTF from a 2D image and an edge-containing ROI.

    roi is (x, y, width, height) in input-image sample coordinates. Frequencies
    are returned in inverse units of sample_spacing.
    """
    imageArray = np.asarray(image)
    if imageArray.ndim != 2 or imageArray.size == 0:
        raise ValueError("image must be a non-empty 2D intensity array.")
    if not np.issubdtype(imageArray.dtype, np.number):
        raise TypeError("image must contain numeric intensity values.")
    if not np.isfinite(imageArray).all():
        raise ValueError("image must contain only finite intensity values.")

    if not isinstance(roi, (tuple, list)) or len(roi) != 4:
        raise ValueError("roi must be (x, y, width, height).")
    if any(
        not isinstance(coordinate, (int, np.integer))
        or isinstance(coordinate, (bool, np.bool_))
        for coordinate in roi
    ):
        raise TypeError("ROI coordinates must be integers.")

    roiX, roiY, roiWidth, roiHeight = map(int, roi)
    if roiX < 0 or roiY < 0 or roiWidth < 3 or roiHeight < 3:
        raise ValueError("ROI origin must be non-negative and size at least 3 by 3.")
    if (
        roiX + roiWidth > imageArray.shape[1]
        or roiY + roiHeight > imageArray.shape[0]
    ):
        raise ValueError("ROI must lie entirely inside the image.")

    if (
        isinstance(sample_spacing, (bool, np.bool_))
        or not isinstance(sample_spacing, (int, float, np.integer, np.floating))
    ):
        raise TypeError("sample_spacing must be a numeric scalar.")
    if not np.isfinite(sample_spacing) or sample_spacing <= 0:
        raise ValueError("sample_spacing must be finite and positive.")
    if (
        not isinstance(oversampling, (int, np.integer))
        or isinstance(oversampling, (bool, np.bool_))
        or oversampling < 1
    ):
        raise ValueError("oversampling must be a positive integer.")

    roiImage = imageArray[
        roiY : roiY + roiHeight, roiX : roiX + roiWidth
    ].astype(np.float64)
    if np.ptp(roiImage) == 0:
        raise ValueError("ROI has no intensity contrast; choose a crop containing an edge.")

    # Locate the half-contrast edge in each row to get subpixel edge coordinates.
    plateauWidth = max(1, roiWidth // 5)
    edgeRows = np.arange(roiY, roiY + roiHeight, dtype=np.float64)
    edgeColumns = np.empty(roiHeight, dtype=np.float64)
    for rowIndex, rowValues in enumerate(roiImage):
        darkLevel = float(np.median(rowValues[:plateauWidth]))
        lightLevel = float(np.median(rowValues[-plateauWidth:]))
        if lightLevel <= darkLevel:
            raise ValueError(
                f"ROI row {roiY + rowIndex} does not transition dark-to-light left-to-right."
            )

        threshold = (darkLevel + lightLevel) / 2
        crossingIndices = np.flatnonzero(
            (rowValues[:-1] < threshold)
            & (rowValues[1:] >= threshold)
        )
        if crossingIndices.size == 0:
            raise ValueError(
                f"No dark-to-light threshold crossing found in ROI row {roiY + rowIndex}."
            )

        crossingGradients = (
            rowValues[crossingIndices + 1] - rowValues[crossingIndices]
        )
        crossingIndex = crossingIndices[np.argmax(crossingGradients)]
        firstValue = rowValues[crossingIndex]
        secondValue = rowValues[crossingIndex + 1]
        crossingFraction = (threshold - firstValue) / (secondValue - firstValue)
        edgeColumns[rowIndex] = roiX + crossingIndex + crossingFraction

    # Fit x as a function of y because the edge is nearly vertical.
    edgeSlope, edgeIntercept = np.polyfit(edgeRows, edgeColumns, 1)
    edgeAngleDeg = float(np.rad2deg(np.arctan(edgeSlope)))
    if abs(edgeSlope) >= 1:
        raise ValueError("Edge must be closer to vertical than horizontal.")

    # Signed perpendicular distances align samples from all rows into one edge profile.
    pixelColumns = np.arange(roiX, roiX + roiWidth, dtype=np.float64)
    pixelRows = np.arange(roiY, roiY + roiHeight, dtype=np.float64)
    xCoordinates, yCoordinates = np.meshgrid(pixelColumns, pixelRows)
    pixelDistances = (
        xCoordinates - edgeSlope * yCoordinates - edgeIntercept
    ) / np.sqrt(1 + edgeSlope**2)
    pixelDistances *= sample_spacing

    # Bin at a fraction of one sample spacing to form the oversampled ESF.
    binWidth = sample_spacing / oversampling
    distanceMinimum = float(np.min(pixelDistances))
    distanceMaximum = float(np.max(pixelDistances))
    binCount = int(np.ceil((distanceMaximum - distanceMinimum) / binWidth)) + 1
    binIndices = np.floor(
        (pixelDistances - distanceMinimum) / binWidth
    ).astype(np.int64)
    intensitySums = np.bincount(
        binIndices.ravel(), weights=roiImage.ravel(), minlength=binCount
    )
    sampleCounts = np.bincount(binIndices.ravel(), minlength=binCount)
    binCenters = distanceMinimum + (np.arange(binCount) + 0.5) * binWidth
    populatedBins = sampleCounts > 0
    if np.count_nonzero(populatedBins) < 3:
        raise ValueError("ROI does not provide enough samples to build an ESF.")
    esf = np.interp(
        binCenters,
        binCenters[populatedBins],
        intensitySums[populatedBins] / sampleCounts[populatedBins],
    )

    # Normalize the dark and light tails so the MTF is independent of image contrast.
    plateauBinCount = max(1, binCount // 10)
    darkLevel = float(np.median(esf[:plateauBinCount]))
    lightLevel = float(np.median(esf[-plateauBinCount:]))
    if lightLevel <= darkLevel:
        raise ValueError("Binned ESF must transition from dark to light.")
    esf = (esf - darkLevel) / (lightLevel - darkLevel)

    # Differentiate the ESF; windowing its ends reduces finite-ROI FFT artifacts.
    lsf = np.gradient(esf, binWidth)
    windowedLsf = lsf * np.hanning(lsf.size)

    # Zero-pad the transform for a smoother frequency grid, then normalize at DC.
    fftLength = 1 << int(np.ceil(np.log2(4 * windowedLsf.size)))
    spectrum = np.abs(np.fft.rfft(windowedLsf, n=fftLength))
    if spectrum[0] <= np.finfo(np.float64).eps:
        raise ValueError("LSF has no measurable DC response.")
    allFrequencies = np.fft.rfftfreq(fftLength, d=binWidth)
    allMtf = spectrum / spectrum[0]

    # Restrict reported frequencies to the input image's Nyquist limit.
    nyquistFrequency = 0.5 / sample_spacing
    belowNyquist = allFrequencies <= nyquistFrequency
    frequency = allFrequencies[belowNyquist]
    mtf = allMtf[belowNyquist]

    # Interpolate the first downward 0.5 crossing; NaN means it is not resolved.
    crossingIndices = np.flatnonzero((mtf[:-1] >= 0.5) & (mtf[1:] < 0.5))
    mtf50 = np.nan
    if crossingIndices.size:
        crossingIndex = crossingIndices[0]
        frequencyFraction = (
            (0.5 - mtf[crossingIndex])
            / (mtf[crossingIndex + 1] - mtf[crossingIndex])
        )
        mtf50 = frequency[crossingIndex] + frequencyFraction * (
            frequency[crossingIndex + 1] - frequency[crossingIndex]
        )

    return {
        "roiImage": roiImage,
        "edgeRows": edgeRows,
        "edgeColumns": edgeColumns,
        "edgeAngleDeg": edgeAngleDeg,
        "esfDistance": binCenters,
        "esf": esf,
        "lsf": lsf,
        "frequency": frequency,
        "mtf": mtf,
        "nyquistFrequency": nyquistFrequency,
        "mtf50": mtf50,
    }

def calculate_mtf50_from_mtf(
    greenPlanes,
    p,
    nF,
    *,
    target: Literal["largest", "top-left", "top-right", "bottom-left", "bottom-right"] = "largest",
):
    """Measure every image at the selected square; return selections, MTF50, and Nyquist MTF.

    Pixel pitch p is in micrometers; MTF50 and nF are in cycles/micrometer.
    Selection or measurement failures raise ValueError with image/target context.
    An unresolved MTF50 crossing remains NaN, as in measure_slanted_edge_mtf.
    """
    squareSelections = {}
    measuredModulationAtNyquist = {}
    mtf50 = {}
    for imageName, greenPlane in greenPlanes.items():
        try:
            selection = select_square_edge(greenPlane, target=target)
            syntheticEdgeMeasurement = measure_slanted_edge_mtf(
                greenPlane,
                roi=selection["roi"],
                sample_spacing=2 * p,
                oversampling=4,
            )
        except ValueError as error:
            raise ValueError(f"{imageName} ({target}): {error}") from error
        squareSelections[imageName] = selection
        mtf50[imageName] = syntheticEdgeMeasurement["mtf50"]
        measuredModulationAtNyquist[imageName] = float(
            np.interp(
                nF,
                syntheticEdgeMeasurement["frequency"],
                syntheticEdgeMeasurement["mtf"],
            )
        )
    return squareSelections, mtf50, measuredModulationAtNyquist
            

## Excerise 3
def read_raw_or_array(imageSource):
	if isinstance(imageSource, (str, Path)):
		with rawpy.imread(str(imageSource)) as raw:
			return raw.raw_image_visible[0::2, 1::2].astype(np.float64)
	imageArray = np.asarray(imageSource, dtype=np.float64)
	if imageArray.ndim != 2:
		raise ValueError("Flat-field inputs must be 2D raw mosaics")
	return imageArray

def measure_flat_field_pairs(imagePairs, blackLevel, roiSize=128):
	imagePairs = list(imagePairs)
	if not imagePairs:
		raise ValueError("At least one pair of flat-field images is required")
	if not isinstance(roiSize, int) or isinstance(roiSize, bool) or roiSize < 2:
		raise ValueError("roiSize must be an integer of at least 2 pixels")

	meanSignalDN = []
	noiseVarianceDN2 = []
	for imageSourceA, imageSourceB in imagePairs:
		imageA = read_raw_or_array(imageSourceA)
		imageB = read_raw_or_array(imageSourceB)
		if imageA.shape != imageB.shape:
			raise ValueError("Paired flat fields must have matching dimensions")

		height, width = imageA.shape
		if height < roiSize or width < roiSize:
			raise ValueError(f"Images must be at least {roiSize} x {roiSize} pixels")
		top = (height - roiSize) // 2
		left = (width - roiSize) // 2
		imageA = imageA[top:top + roiSize, left:left + roiSize]
		imageB = imageB[top:top + roiSize, left:left + roiSize]

		meanSignalDN.append((imageA.mean() + imageB.mean()) / 2 - blackLevel)
		noiseVarianceDN2.append(np.var(imageA - imageB, ddof=1) / 2)

	return np.asarray(meanSignalDN), np.asarray(noiseVarianceDN2)

def measure_flat_field_single(images, blackLevel, roiSize=128):
	images = list(images)
	if not images:
		raise ValueError("At least one flat-field image is required")
	if not isinstance(roiSize, int) or isinstance(roiSize, bool) or roiSize < 2:
		raise ValueError("roiSize must be an integer of at least 2 pixels")

	meanSignalDN = []
	noiseVarianceDN2 = []
	for imageSource in images:
		image = read_raw_or_array(imageSource)
		if image.ndim != 2:
			raise ValueError("Flat-field inputs must be 2D raw mosaics")

		height, width = image.shape
		if height < roiSize or width < roiSize:
			raise ValueError(f"Images must be at least {roiSize} x {roiSize} pixels")
		top = (height - roiSize) // 2
		left = (width - roiSize) // 2
		image = image[top:top + roiSize, left:left + roiSize]

		meanSignalDN.append(image.mean() - blackLevel)
		noiseVarianceDN2.append(np.var(image, ddof=1))

	return np.asarray(meanSignalDN), np.asarray(noiseVarianceDN2)

def plot_photon_transfer(
	meanSignalDN,
	noiseVarianceDN2,
	saturationDN,
	saturationFraction=0.7,
	fitRange=None,
	ax=None,
):
	meanSignalDN = np.asarray(meanSignalDN, dtype=float)
	noiseVarianceDN2 = np.asarray(noiseVarianceDN2, dtype=float)
	if meanSignalDN.ndim != 1 or noiseVarianceDN2.ndim != 1:
		raise ValueError("Photon-transfer measurements must be one-dimensional")
	if meanSignalDN.size != noiseVarianceDN2.size or meanSignalDN.size == 0:
		raise ValueError("Mean and variance arrays must have the same nonzero length")
	if not np.isfinite(meanSignalDN).all() or not np.isfinite(noiseVarianceDN2).all():
		raise ValueError("Photon-transfer measurements must be finite")
	if not np.isfinite(saturationDN) or saturationDN <= 0:
		raise ValueError("saturationDN must be a positive finite signal range")
	if not 0 < saturationFraction <= 1:
		raise ValueError("saturationFraction must be in (0, 1]")

	saturationLimitDN = saturationDN * saturationFraction
	belowSaturation = meanSignalDN < saturationLimitDN
	if np.count_nonzero(belowSaturation) < 3:
		raise ValueError("At least three measurements must be below the saturation cutoff")
	fitMask = belowSaturation.copy()
	if fitRange is not None:
		fitRange = np.asarray(fitRange, dtype=float)
		if fitRange.shape != (2,) or not np.isfinite(fitRange).all() or fitRange[0] >= fitRange[1]:
			raise ValueError("fitRange must contain increasing finite DN bounds")
		fitMask &= (meanSignalDN >= fitRange[0]) & (meanSignalDN <= fitRange[1])
	if np.count_nonzero(fitMask) < 3:
		raise ValueError("At least three plotted measurements must be inside fitRange")
	if ax is None:
		_, ax = plt.subplots()

	ax.scatter(meanSignalDN[belowSaturation], noiseVarianceDN2[belowSaturation])
	ax.set_xlabel("Mean signal, E[D] - B (DN)")
	ax.set_ylabel("Temporal variance (DN^2)")
	ax.set_title(f"Photon-transfer curve below {saturationFraction:.0%} saturation")

	slope, intercept = np.polyfit(meanSignalDN[fitMask], noiseVarianceDN2[fitMask], 1)
	# if slope <= 0 or intercept < 0:
	# 	raise ValueError("Fit must have a positive slope and nonnegative intercept")
	fitSignalDN = np.linspace(
		meanSignalDN[fitMask].min(), meanSignalDN[fitMask].max(), 100
	)
	ax.plot(fitSignalDN, slope * fitSignalDN + intercept, label="Linear fit")
	ax.legend()

	readNoiseDN = np.sqrt(intercept/slope**2)
	return {
		"slopeDNPerElectron": float(slope),
		"interceptDN2": float(intercept),
		"conversionGainElectronPerDN": float(1 / slope),
		"readNoiseDN": float(readNoiseDN),
		"readNoiseElectrons": float(readNoiseDN / slope),
		"saturationLimitDN": float(saturationLimitDN),
		"saturationFraction": float(saturationFraction),
		"fitRangeDN": (
			float(meanSignalDN[fitMask].min()),
			float(meanSignalDN[fitMask].max()),
		),
	}

def calculate_signal_to_noise(signalElectrons, readNoiseElectrons):
	signalElectrons = np.asarray(signalElectrons, dtype=float)
	readNoiseElectrons = np.asarray(readNoiseElectrons, dtype=float)
	if not np.isfinite(signalElectrons).all() or np.any(signalElectrons < 0):
		raise ValueError("signalElectrons must contain finite nonnegative values")
	if not np.isfinite(readNoiseElectrons).all() or np.any(readNoiseElectrons < 0):
		raise ValueError("readNoiseElectrons must be finite and nonnegative")

	denominator = np.sqrt(signalElectrons + readNoiseElectrons ** 2)
	snr = np.divide(
		signalElectrons,
		denominator,
		out=np.zeros_like(signalElectrons, dtype=float),
		where=denominator > 0,
	)
	return float(snr) if snr.ndim == 0 else snr


def plot_snr_vs_signal(meanSignalDN, conversionGainElectronPerDN, readNoiseElectrons, ax=None):
	"""Plot predicted shot-plus-read-noise SNR over the supplied signal range.

	Accept black-subtracted mean signal in DN, gain in electrons/DN, and
	read noise in electrons RMS. Draw a smooth model curve in electron units,
	not lines connecting the input samples. Return the Matplotlib axes.
	Raise ValueError for invalid signals, gain, or read noise.
	"""
	meanSignalDN = np.asarray(meanSignalDN, dtype=float)
	if meanSignalDN.ndim != 1 or meanSignalDN.size == 0:
		raise ValueError("meanSignalDN must be a nonempty one-dimensional array")
	if not np.isfinite(meanSignalDN).all() or np.any(meanSignalDN < 0):
		raise ValueError("meanSignalDN must contain finite nonnegative values")
	gainArray = np.asarray(conversionGainElectronPerDN, dtype=float)
	if gainArray.ndim != 0 or not np.isfinite(gainArray) or gainArray <= 0:
		raise ValueError("conversionGainElectronPerDN must be a finite positive scalar")
	signalElectrons = meanSignalDN * float(gainArray)
	if not np.isfinite(signalElectrons).all():
		raise ValueError("Gain-scaled signals must be finite")
	readNoiseArray = np.asarray(readNoiseElectrons, dtype=float)
	if readNoiseArray.ndim != 0 or not np.isfinite(readNoiseArray) or readNoiseArray < 0:
		raise ValueError("readNoiseElectrons must be a finite nonnegative scalar")
	readNoiseElectrons = float(readNoiseArray)

	positiveSignal = signalElectrons > 0
	if not np.any(positiveSignal):
		raise ValueError("At least one positive signal is required for logarithmic axes")
	curveSignalElectrons = np.geomspace(
		signalElectrons[positiveSignal].min(), signalElectrons[positiveSignal].max(), 400,
	)
	curveSnr = calculate_signal_to_noise(curveSignalElectrons, readNoiseElectrons)
	if ax is None:
		_, ax = plt.subplots()

	ax.loglog(curveSignalElectrons, curveSnr, label="Shot + read noise prediction")
	ax.set_xlabel("Signal (electrons)")
	ax.set_ylabel("Signal-to-noise ratio")
	ax.set_title("Signal-to-noise ratio versus signal")
	if readNoiseElectrons > 0:
		kneeSignalElectrons = readNoiseElectrons ** 2
		ax.axvline(
			kneeSignalElectrons,
			color="tab:orange",
			linestyle="--",
			label=f"Knee = {kneeSignalElectrons:.2g} e-",
		)
	ax.legend()
	return ax


def plot_dark_current(exposureTimesSeconds, darkSignalElectrons, ax=None):
	exposureTimesSeconds = np.asarray(exposureTimesSeconds, dtype=float)
	darkSignalElectrons = np.asarray(darkSignalElectrons, dtype=float)
	if exposureTimesSeconds.ndim != 1 or darkSignalElectrons.ndim != 1:
		raise ValueError("Exposure times and dark signals must be one-dimensional")
	if exposureTimesSeconds.size != darkSignalElectrons.size or exposureTimesSeconds.size < 2:
		raise ValueError("At least two matching exposure and dark-signal values are required")
	if not np.isfinite(exposureTimesSeconds).all() or np.any(exposureTimesSeconds < 0):
		raise ValueError("Exposure times must be finite and nonnegative")
	if not np.isfinite(darkSignalElectrons).all():
		raise ValueError("Dark signal values must be finite")
	if np.unique(exposureTimesSeconds).size < 2:
		raise ValueError("At least two distinct exposure times are required")

	order = np.argsort(exposureTimesSeconds)
	exposureTimesSeconds = exposureTimesSeconds[order]
	darkSignalElectrons = darkSignalElectrons[order]
	slope, intercept = np.polyfit(exposureTimesSeconds, darkSignalElectrons, 1)
	if ax is None:
		_, ax = plt.subplots()

	ax.plot(exposureTimesSeconds, darkSignalElectrons, "o", label="Measured dark signal")
	fitTimes = np.linspace(exposureTimesSeconds.min(), exposureTimesSeconds.max(), 100)
	ax.plot(fitTimes, slope * fitTimes + intercept, label="Linear fit")
	ax.set_xlabel("Exposure time (s)")
	ax.set_ylabel("Mean dark signal above black (electrons)")
	ax.set_title("Dark signal versus exposure time")
	ax.legend()
	return {
		"darkCurrentElectronsPerSecond": float(slope),
		"interceptElectrons": float(intercept),
		"ax": ax,
	}
 
def find_checkerboard(checkerboardPath):
	"""Return an unresampled 2D RAW green-plane crop inside the checkerboard.

	Use the first green position in the 2 x 2 Bayer pattern. Detect 9 x 6
	inner corners on a normalized copy and extrapolate to the outer edges.
	Return original sensor DN with the original dtype and black pedestal;
	no demosaicing, rotation, gamma encoding, or interpolation is applied.
	Use an axis-aligned rectangle inside a roughly axis-aligned board, omitting
	narrow edge strips to exclude surrounding paper. Dimensions need not be
	multiples of eight. Raise ValueError for unsupported CFA layouts, failed
	detection, or an invalid crop. Decoding errors propagate.
	"""
	with rawpy.imread(str(checkerboardPath)) as raw:
		if raw.raw_pattern is None or raw.raw_pattern.shape != (2, 2):
			raise ValueError("Checkerboard RAW extraction requires a 2 x 2 Bayer pattern")
		greenPositions = [
			(row, column)
			for row in range(2) for column in range(2)
			if raw.color_desc[int(raw.raw_pattern[row, column])] == ord("G")
		]
		if len(greenPositions) != 2:
			raise ValueError("Checkerboard RAW extraction requires two Bayer green positions")
		greenRow, greenColumn = greenPositions[0]
		greenPlane = raw.raw_image_visible[greenRow::2, greenColumn::2].copy()

	if greenPlane.min() == greenPlane.max():
		raise ValueError(f"RAW green plane has no contrast in {checkerboardPath}")
	checkerboardGray = cv2.normalize(
		greenPlane, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U,
	)
	patternSize = (9, 6)  # 10 x 7 squares have 9 x 6 internal corners.
	checkerboardFound, checkerboardCorners = cv2.findChessboardCorners(
		checkerboardGray,
		patternSize,
		cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
	)
	if not checkerboardFound:
		raise ValueError(f"Could not find a {patternSize} checkerboard in {checkerboardPath}")

	refinementCriteria = (
		cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
		30,
		0.001,
	)
	checkerboardCorners = cv2.cornerSubPix(
		checkerboardGray,
		checkerboardCorners,
		(11, 11),
		(-1, -1),
		refinementCriteria,
	)

	gridColumns, gridRows = patternSize
	boardGrid = np.mgrid[0:gridColumns, 0:gridRows].T.reshape(-1, 2).astype(np.float32)
	boardToImage, _ = cv2.findHomography(boardGrid, checkerboardCorners.reshape(-1, 2))
	if boardToImage is None:
		raise ValueError(f"Could not fit the checkerboard boundary in {checkerboardPath}")
	boardBoundary = np.array(
		[[-1, -1], [gridColumns, -1], [gridColumns, gridRows], [-1, gridRows]],
		dtype=np.float32,
	)
	imageBoundary = cv2.perspectiveTransform(boardBoundary[None], boardToImage)[0]
	imageHeight, imageWidth = greenPlane.shape
	if (
		not np.isfinite(imageBoundary).all()
		or np.any(imageBoundary < 0)
		or np.any(imageBoundary[:, 0] > imageWidth - 1)
		or np.any(imageBoundary[:, 1] > imageHeight - 1)
	):
		raise ValueError(f"Estimated checkerboard extends outside {checkerboardPath}")
	topCorners, bottomCorners = np.split(imageBoundary[np.argsort(imageBoundary[:, 1])], 2)
	topLeft, topRight = topCorners[np.argsort(topCorners[:, 0])]
	bottomLeft, bottomRight = bottomCorners[np.argsort(bottomCorners[:, 0])]
	left = int(np.ceil(max(topLeft[0], bottomLeft[0])))
	right = int(np.floor(min(topRight[0], bottomRight[0])))
	top = int(np.ceil(max(topLeft[1], topRight[1])))
	bottom = int(np.floor(min(bottomLeft[1], bottomRight[1])))
	cropCorners = ((left, top), (right, top), (right, bottom), (left, bottom))
	if right <= left or bottom <= top or any(
		cv2.pointPolygonTest(imageBoundary, corner, False) < 0 for corner in cropCorners
	):
		raise ValueError(f"Could not select an interior RAW crop in {checkerboardPath}")
	return greenPlane[top:bottom, left:right]


## Exercise 4 

def raw_to_linear_rgb(imagePath: str | Path, saturationValue: float):
    """Return quarter-pixel-count linear camera RGB from a Bayer RAW file.

    Subtract each site's metadata black level and divide by
    (saturationValue - black level), using the measured saturation DN.
    Each visible 2 x 2 block contributes R, mean(G1, G2), B without
    interpolation, white balance, color conversion, gamma, or rotation.
    The float64 output has shape (height // 2, width // 2, 3).
    Negative values and values above one are preserved, not clipped.

    Require an even-sized visible Bayer mosaic containing R, G, G, B.
    Raise ValueError for unsupported layouts or invalid normalization levels.
    File reading and RAW decoding errors propagate.
    """
    if not np.isscalar(saturationValue) or not np.isreal(saturationValue):
        raise ValueError("saturationValue must be a finite real scalar")
    saturationValue = float(saturationValue)
    if not np.isfinite(saturationValue):
        raise ValueError("saturationValue must be a finite real scalar")

    with rawpy.imread(str(imagePath)) as rawFile:
        if rawFile.raw_pattern is None or rawFile.raw_pattern.shape != (2, 2):
            raise ValueError("Linear RGB extraction requires a 2 x 2 Bayer pattern")
        mosaic = rawFile.raw_image_visible
        if mosaic.ndim != 2 or min(mosaic.shape) < 2 or any(size % 2 for size in mosaic.shape):
            raise ValueError("The visible RAW mosaic must have positive even dimensions")

        # Visible color indices account for sensor margins shifting the CFA origin.
        pattern = rawFile.raw_colors_visible[:2, :2]
        colorNames = [chr(rawFile.color_desc[int(index)]) for index in pattern.flat]
        if sorted(colorNames) != ["B", "G", "G", "R"]:
            raise ValueError("The visible Bayer block must contain R, G, G, B")
        blackLevels = np.asarray(rawFile.black_level_per_channel, dtype=np.float64)
        siteBlackLevels = blackLevels[pattern]
        if not np.isfinite(siteBlackLevels).all() or np.any(saturationValue <= siteBlackLevels):
            raise ValueError("saturationValue must exceed every finite site black level")

        sitePlanes = []
        for row, column in np.ndindex(2, 2):
            blackLevel = siteBlackLevels[row, column]
            sitePlanes.append(
                (mosaic[row::2, column::2].astype(np.float64) - blackLevel)
                / (saturationValue - blackLevel)
            )

    redPlane = sitePlanes[colorNames.index("R")]
    greenPlanes = [plane for plane, color in zip(sitePlanes, colorNames) if color == "G"]
    bluePlane = sitePlanes[colorNames.index("B")]
    output = {
		'stack': np.stack((redPlane, (greenPlanes[0] + greenPlanes[1]) / 2, bluePlane), axis=-1),
		'saturation': saturationValue,
		'blackLevel': siteBlackLevels,
		'pattern': pattern,
		'colorNames': colorNames,
 		}
    return output

def plot_quarter_resolution():
    pass

def white_level_histogram(imagePath, binMin, binMax, binRange):
    """histogram the raw values and look for a spike at the top of the range indicating the white level"""
    with rawpy.imread(str(imagePath)) as rawFile:
        raw = rawFile.raw_image_visible.copy()
    hist, bin_edges = np.histogram(raw, bins=binMax-binMin, range=binRange)
    return hist, bin_edges

def extract_colorchecker_means(
    linearRgb: np.ndarray,
    *,
    interiorFraction: float = 0.5,
) -> dict:
    """Locate a 24-patch ColorChecker and measure its linear camera RGB.

    Accept an H x W x 3 floating-point RGB image. Detect on an 8-bit
    preview, but measure original pixels without clipping or interpolation.
    interiorFraction controls the retained linear size of each patch.

    Return meanRgb (24 x 3), patchPolygons (24 x 4 x 2), and
    pixelCounts (24,). Patch order follows OpenCV's MCC24 model.
    Raise ValueError for invalid inputs, detection failure, or invalid ROIs.
    Require the pinned OpenCV 4.11 contrib build; its MCC24 detector is
    validated on the assignment charts. OpenCV 5.0 detection was inconsistent.
    """
    linearRgb = np.asarray(linearRgb)
    if (
        linearRgb.ndim != 3
        or linearRgb.shape[2] != 3
        or min(linearRgb.shape[:2]) < 2
        or not np.issubdtype(linearRgb.dtype, np.floating)
    ):
        raise ValueError("linearRgb must be a floating-point H x W x 3 RGB image")
    if not np.isfinite(linearRgb).all():
        raise ValueError("linearRgb must contain only finite values")
    if not np.isfinite(interiorFraction) or not 0 < interiorFraction <= 1:
        raise ValueError("interiorFraction must be in (0, 1]")
    if not hasattr(cv2, "mcc"):
        raise RuntimeError("ColorChecker detection requires OpenCV's mcc contrib module")

    # Adjust only the detection preview; preserve the measurement array.
    previewRgb = np.clip(linearRgb, 0.0, 1.0)
    previewRgb = np.where(
        previewRgb <= 0.0031308,
        12.92 * previewRgb,
        1.055 * previewRgb ** (1 / 2.4) - 0.055,
    )
    previewBgr = cv2.cvtColor(
        np.rint(previewRgb * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )

    detector = cv2.mcc.CCheckerDetector_create()
    if not detector.process(previewBgr, cv2.mcc.MCC24):
        raise ValueError("No 24-patch ColorChecker was detected")
    checker = detector.getBestColorChecker()
    if checker is None:
        raise ValueError("The detector returned no ColorChecker")

    patchPoints = np.asarray(checker.getColorCharts(), dtype=np.float64)
    if patchPoints.size != 24 * 4 * 2 or not np.isfinite(patchPoints).all():
        raise ValueError("The detector did not return 24 valid patch polygons")
    patchPolygons = patchPoints.reshape(24, 4, 2)
    centers = patchPolygons.mean(axis=1, keepdims=True)
    patchPolygons = centers + interiorFraction * (patchPolygons - centers)

    height, width = linearRgb.shape[:2]
    meanRgb = np.empty((24, 3), dtype=np.float64)
    pixelCounts = np.empty(24, dtype=np.int64)

    for patchIndex, polygon in enumerate(patchPolygons):
        if (
            np.any(polygon < 0)
            or np.any(polygon[:, 0] > width - 1)
            or np.any(polygon[:, 1] > height - 1)
        ):
            raise ValueError(f"Patch {patchIndex + 1} extends outside the image")

        left, top = np.floor(polygon.min(axis=0)).astype(int)
        right, bottom = np.ceil(polygon.max(axis=0)).astype(int)
        patchMask = np.zeros((bottom - top + 1, right - left + 1), dtype=np.uint8)
        localPolygon = polygon - [left, top]
        cv2.fillConvexPoly(
            patchMask,
            np.rint(localPolygon * 256).astype(np.int32),
            1,
            shift=8,
        )
        patchPixels = linearRgb[top:bottom + 1, left:right + 1][patchMask != 0]
        if patchPixels.shape[0] == 0:
            raise ValueError(f"Patch {patchIndex + 1} contains no measurement pixels")

        meanRgb[patchIndex] = patchPixels.mean(axis=0, dtype=np.float64)
        pixelCounts[patchIndex] = patchPixels.shape[0]

    return {
        "meanRgb": meanRgb,
        "patchPolygons": patchPolygons,
        "pixelCounts": pixelCounts,
    }


def plot_colorchecker_patches(
    linearRgb: np.ndarray,
    patchPolygons: np.ndarray,
    ax=None,
):
    """Overlay 24 numbered measurement polygons on a linear RGB preview.

    Pass the same H x W x 3 image used by extract_colorchecker_means and its
    patchPolygons array of shape (24, 4, 2), in (x, y) coordinates.
    Clip and apply approximate display gamma only to a new preview array.
    Return the Matplotlib axes without calling show or modifying inputs.
    Raise ValueError for invalid images, polygons, or out-of-image coordinates.
    """
    from matplotlib.patches import Polygon

    linearRgb = np.asarray(linearRgb)
    patchPolygons = np.asarray(patchPolygons, dtype=np.float64)
    if (
        linearRgb.ndim != 3
        or linearRgb.shape[2] != 3
        or min(linearRgb.shape[:2]) < 1
        or not np.issubdtype(linearRgb.dtype, np.floating)
        or not np.isfinite(linearRgb).all()
    ):
        raise ValueError("linearRgb must be a finite floating-point H x W x 3 RGB image")
    if patchPolygons.shape != (24, 4, 2) or not np.isfinite(patchPolygons).all():
        raise ValueError("patchPolygons must be a finite array of shape (24, 4, 2)")
    height, width = linearRgb.shape[:2]
    if (
        np.any(patchPolygons < 0)
        or np.any(patchPolygons[..., 0] > width - 1)
        or np.any(patchPolygons[..., 1] > height - 1)
    ):
        raise ValueError("Patch polygons must lie inside the image")

    previewRgb = np.clip(linearRgb, 0, 1) ** (1 / 2.2)
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(previewRgb, origin="upper")
    for patchNumber, polygon in enumerate(patchPolygons, start=1):
        ax.add_patch(
            Polygon(polygon, closed=True, fill=False, edgecolor="lime", linewidth=1.5)
        )
        centerX, centerY = polygon.mean(axis=0)
        ax.text(
            centerX,
            centerY,
            str(patchNumber),
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1),
        )
    ax.set_title("ColorChecker measurement regions")
    ax.axis("off")
    return ax

def srgb_baseline_color_error(
    meanRgb: np.ndarray,
    referenceXYZ: np.ndarray,
    referenceIlluminant: np.ndarray,
    *,
    whiteIndex: int = 18,
    chromaticAdaptation: str = "Bradford",
) -> dict:
    """Report chart color error before fitting any camera matrix.

    meanRgb (N x 3) is linear camera RGB, e.g. extract_colorchecker_means
    output. Per-channel gains map patch whiteIndex to neutral RGB equal to
    the reference white luminance Y. The balanced RGB is treated as linear
    sRGB (no decoding), converted to XYZ with standard sRGB primaries, and
    adapted from D65 to referenceIlluminant (CIE xy, e.g. the chart's D50).
    Both XYZ sets are converted to Lab using referenceIlluminant.

    Return whiteBalanceGains (3,), balancedRgb, testXYZ, referenceLab,
    testLab (N x 3), deltaE (N,) CIEDE2000, and meanDeltaE, maxDeltaE,
    whiteDeltaE floats. Raise ValueError for invalid shapes, non-finite
    values, a bad whiteIndex, or a non-positive camera white patch.
    """
    import colour

    meanRgb = np.asarray(meanRgb, dtype=np.float64)
    referenceXYZ = np.asarray(referenceXYZ, dtype=np.float64)
    referenceIlluminant = np.asarray(referenceIlluminant, dtype=np.float64)
    if meanRgb.ndim != 2 or meanRgb.shape[1] != 3 or meanRgb.shape != referenceXYZ.shape:
        raise ValueError("meanRgb and referenceXYZ must both be N x 3 arrays")
    if referenceIlluminant.shape != (2,):
        raise ValueError("referenceIlluminant must be CIE xy chromaticity of shape (2,)")
    if not all(np.isfinite(array).all() for array in (meanRgb, referenceXYZ, referenceIlluminant)):
        raise ValueError("Inputs must contain only finite values")
    if isinstance(whiteIndex, bool) or not isinstance(whiteIndex, (int, np.integer)):
        raise ValueError("whiteIndex must be an integer")
    if not 0 <= whiteIndex < meanRgb.shape[0]:
        raise ValueError(f"whiteIndex must be in [0, {meanRgb.shape[0] - 1}]")

    cameraWhite = meanRgb[whiteIndex]
    if np.any(cameraWhite <= 0):
        raise ValueError("The camera white patch must be positive in every channel")
    whiteBalanceGains = referenceXYZ[whiteIndex, 1] / cameraWhite  # White -> neutral at reference Y
    balancedRgb = meanRgb * whiteBalanceGains

    testXYZ = colour.RGB_to_XYZ(
        balancedRgb,
        "sRGB",
        illuminant=referenceIlluminant,  # Adapt sRGB's D65 white to the chart illuminant
        chromatic_adaptation_transform=chromaticAdaptation,
    )
    referenceLab, testLab, deltaE = _chart_delta_e(referenceXYZ, testXYZ, referenceIlluminant)

    return {
        "whiteBalanceGains": whiteBalanceGains,
        "balancedRgb": balancedRgb,
        "testXYZ": testXYZ,
        "referenceLab": referenceLab,
        "testLab": testLab,
        "deltaE": deltaE,
        "meanDeltaE": float(deltaE.mean()),
        "maxDeltaE": float(deltaE.max()),
        "whiteDeltaE": float(deltaE[whiteIndex]),
    }

def _chart_delta_e(referenceXYZ, testXYZ, referenceIlluminant):
    """Return (referenceLab, testLab, CIEDE2000 deltaE) under one illuminant."""
    import colour

    referenceLab = colour.XYZ_to_Lab(referenceXYZ, illuminant=referenceIlluminant)
    testLab = colour.XYZ_to_Lab(testXYZ, illuminant=referenceIlluminant)
    deltaE = np.asarray(colour.delta_E(referenceLab, testLab, method="CIE 2000"))
    return referenceLab, testLab, deltaE

def fit_color_correction_matrix(
    meanRgb: np.ndarray,
    referenceXYZ: np.ndarray,
    referenceIlluminant: np.ndarray,
    *,
    colorGain: np.ndarray | None = None,
    clipLevel: float = 0.98,
    operateOnAllValues: bool = False,
) -> dict:
    """Least-squares fit of a 3 x 3 matrix from white-balanced RGB to XYZ.

    meanRgb (N x 3) is normalized linear camera RGB, where 1 is saturation.
    colorGain (3,) holds per-channel white-balance gains (default ones).
    Patches with any raw channel >= clipLevel are excluded from the fit,
    since clipped values do not follow the linear model.

    Solve min ||balancedRgb @ colorMatrix.T - referenceXYZ|| over the
    unclipped patches, so XYZ = colorMatrix @ (colorGain * rgb). The scale
    is absorbed by the matrix; no exposure normalization is needed.

    Return colorMatrix (3 x 3), cameraToXyz = colorMatrix @ diag(colorGain),
    fitMask (N,), testXYZ, referenceLab, testLab (N x 3), deltaE (N,)
    CIEDE2000 under referenceIlluminant, and meanDeltaE/maxDeltaE over
    fitted patches. Raise ValueError for invalid inputs, fewer than 3
    unclipped patches, or a rank-deficient fit.
    """
    meanRgb = np.asarray(meanRgb, dtype=np.float64)
    referenceXYZ = np.asarray(referenceXYZ, dtype=np.float64)
    colorGain = np.ones(3) if colorGain is None else np.asarray(colorGain, dtype=np.float64)
    if meanRgb.ndim != 2 or meanRgb.shape[1] != 3 or meanRgb.shape != referenceXYZ.shape:
        raise ValueError("meanRgb and referenceXYZ must both be N x 3 arrays")
    if colorGain.shape != (3,) or np.any(colorGain <= 0):
        raise ValueError("colorGain must contain three positive gains")
    if not all(np.isfinite(array).all() for array in (meanRgb, referenceXYZ, colorGain)):
        raise ValueError("Inputs must contain only finite values")
    
    if(operateOnAllValues):
        clipLevel = np.inf
    fitMask = np.all(meanRgb < clipLevel, axis=1)  # Clipping is judged on raw, not balanced, RGB
    if fitMask.sum() < 3:
        raise ValueError("At least 3 unclipped patches are required for the fit")
    balancedRgb = meanRgb * colorGain
    solution, _, rank, _ = np.linalg.lstsq(
        balancedRgb[fitMask], referenceXYZ[fitMask], rcond=None
    )
    if rank != 3:
        raise ValueError("Unclipped patches cannot determine a full-rank matrix")

    testXYZ = balancedRgb @ solution
    referenceLab, testLab, deltaE = _chart_delta_e(referenceXYZ, testXYZ, referenceIlluminant)
    colorMatrix = solution.T  # Column-vector form: xyz = colorMatrix @ rgb
    return {
        "colorMatrix": colorMatrix,
        "cameraToXyz": colorMatrix @ np.diag(colorGain),
        "fitMask": fitMask,
        "testXYZ": testXYZ,
        "referenceLab": referenceLab,
        "testLab": testLab,
        "deltaE": deltaE,
        "meanDeltaE": float(deltaE[fitMask].mean()),
        "maxDeltaE": float(deltaE[fitMask].max()),
    }

def summarize_delta_e(deltaE: np.ndarray, patchNames, mask=None, worstCount: int = 3) -> dict:
    """Return mean, population std, max, and worst patches over masked patches.

    stdDeltaE uses ddof=0 to describe spread across the selected chart patches,
    not uncertainty of the mean or repeatability across captures.
    """
    deltaE = np.asarray(deltaE, dtype=np.float64)
    mask = np.ones(deltaE.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if deltaE.ndim != 1 or mask.shape != deltaE.shape or len(patchNames) != deltaE.size:
        raise ValueError("deltaE, mask, and patchNames must all have length N")
    if not mask.any():
        raise ValueError("mask must select at least one patch")
    if isinstance(worstCount, bool) or not isinstance(worstCount, int) or worstCount < 1:
        raise ValueError("worstCount must be a positive integer")

    candidates = np.flatnonzero(mask)
    worstIndices = candidates[np.argsort(deltaE[candidates])[::-1][:worstCount]]
    return {
        "meanDeltaE": float(deltaE[mask].mean()),
        "stdDeltaE": float(deltaE[mask].std(ddof=0)),
        "maxDeltaE": float(deltaE[mask].max()),
        "worstPatches": [
            (int(index) + 1, patchNames[index], float(deltaE[index])) for index in worstIndices
        ],
    }

def plot_delta_e_before_after(
    beforeDeltaE: np.ndarray,
    afterDeltaE: np.ndarray,
    patchNames,
    *,
    statsMask=None,
    worstCount: int = 3,
    ax=None,
):
    """Grouped bar chart of per-patch CIEDE2000 before and after correction.

    statsMask (N,) selects the patches used for mean/std/max/worst statistics,
    e.g. the fit's fitMask; excluded patches are hatched and labeled in the
    plot so clipped patches remain visible but do not distort the summary.
    Dashed lines mark each condition's mean over the selected patches.

    Return (ax, summary) where summary maps "before"/"after" to
    summarize_delta_e output. Raise ValueError for mismatched inputs.
    """
    beforeDeltaE = np.asarray(beforeDeltaE, dtype=np.float64)
    afterDeltaE = np.asarray(afterDeltaE, dtype=np.float64)
    if beforeDeltaE.shape != afterDeltaE.shape:
        raise ValueError("beforeDeltaE and afterDeltaE must have the same shape")
    if statsMask is None:
        statsMask = np.ones(beforeDeltaE.shape, dtype=bool)
    statsMask = np.asarray(statsMask, dtype=bool)
    summary = {
        "before": summarize_delta_e(beforeDeltaE, patchNames, statsMask, worstCount),
        "after": summarize_delta_e(afterDeltaE, patchNames, statsMask, worstCount),
    }
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 5))

    positions = np.arange(beforeDeltaE.size)
    barWidth = 0.4
    conditions = [
        ("before", beforeDeltaE, -barWidth / 2, "tab:red"),
        ("after", afterDeltaE, barWidth / 2, "tab:blue"),
    ]
    for label, values, offset, color in conditions:
        bars = ax.bar(
            positions + offset,
            values,
            barWidth,
            color=color,
            label=f"{label.capitalize()} (mean {summary[label]['meanDeltaE']:.2f}, "
                  f"std {summary[label]['stdDeltaE']:.2f}, "
                  f"max {summary[label]['maxDeltaE']:.2f})",
        )
        for bar, isUsed in zip(bars, statsMask):
            if not isUsed:  # Hatch patches excluded from statistics
                bar.set_hatch("//")
                bar.set_alpha(0.45)
        ax.axhline(summary[label]["meanDeltaE"], color=color, linestyle="--", linewidth=1)

    tickLabels = [
        f"{index + 1} {name}" + ("" if statsMask[index] else " (clipped)")
        for index, name in enumerate(patchNames)
    ]
    ax.set_xticks(positions, tickLabels, rotation=60, ha="right", fontsize=8)
    ax.set_xlim(-0.6, beforeDeltaE.size - 0.4)
    ax.set_ylabel("CIEDE2000 ΔE")
    ax.set_title("Per-patch color error before and after matrix correction")
    ax.legend()
    ax.figure.tight_layout()
    return ax, summary

def flat_field_correct(linearRgb: np.ndarray, flatRgb: np.ndarray, roiSize: int = 64) -> dict:
    """Divide an image by a per-channel, center-normalized flat field.

    Each channel of flatRgb (H x W x 3, e.g. raw_to_linear_rgb(...)['stack'])
    is divided by the mean of its own central roiSize x roiSize window, so the
    flat equals 1 at the center in R, G, and B independently. linearRgb is
    then divided by that normalized flat, correcting color-dependent falloff
    without a single gray gain. Non-positive flat pixels yield NaN.

    Return corrected (H x W x 3), normalizedFlat (H x W x 3), and
    centerMeans (3,). Raise ValueError for mismatched or invalid inputs.
    """
    linearRgb = np.asarray(linearRgb, dtype=np.float64)
    normalizedFlat, centerMeans = normalize_flat_field(flatRgb, roiSize)
    if linearRgb.shape != normalizedFlat.shape:
        raise ValueError("linearRgb and flatRgb must have matching shapes")

    corrected = np.full_like(linearRgb, np.nan)
    np.divide(linearRgb, normalizedFlat, out=corrected, where=normalizedFlat > 0)
    return {
        "corrected": corrected,
        "normalizedFlat": normalizedFlat,
        "centerMeans": centerMeans,
    }

def normalize_flat_field(flatRgb: np.ndarray, roiSize: int = 64):
    """Scale each channel of an H x W x 3 flat to 1 at its own central ROI.

    Return (normalizedFlat, centerMeans) where centerMeans has shape (3,).
    Raise ValueError for invalid shapes, ROI sizes, or non-positive centers.
    """
    flatRgb = np.asarray(flatRgb, dtype=np.float64)
    if flatRgb.ndim != 3 or flatRgb.shape[2] != 3:
        raise ValueError("flatRgb must be an H x W x 3 RGB image")
    if not isinstance(roiSize, int) or isinstance(roiSize, bool) or roiSize < 1:
        raise ValueError("roiSize must be a positive integer")
    height, width = flatRgb.shape[:2]
    if height < roiSize or width < roiSize:
        raise ValueError(f"Images must be at least {roiSize} x {roiSize} pixels")

    top = (height - roiSize) // 2
    left = (width - roiSize) // 2
    centerMeans = flatRgb[top:top + roiSize, left:left + roiSize].mean(axis=(0, 1))
    if not np.all(np.isfinite(centerMeans)) or np.any(centerMeans <= 0):
        raise ValueError("Flat-field center must be positive in every channel")
    return flatRgb / centerMeans, centerMeans  # Broadcast: each channel by its own center

def plot_flat_field_profiles(
    flatSources: dict,
    saturationValue: float | None = None,
    roiSize: int = 64,
    bandHeight: int = 16,
    ax=None,
):
    """Overlay center-row profiles of per-channel normalized white sheets.

    flatSources maps a round label to a RAW path or an H x W x 3 linear RGB
    array. Paths are converted with raw_to_linear_rgb(path, saturationValue).
    Every flat is normalized with normalize_flat_field, then averaged over
    bandHeight rows centered vertically to suppress noise. Plotted with
    seaborn lineplot: hue encodes round and style encodes channel. The x axis
    is the fraction of image width so differently sized inputs still align.

    Lens falloff is fixed to the frame, so its profiles coincide across
    rounds; lamp falloff moves with the scene and makes them differ.

    Return (ax, profiles), where profiles maps each label to a (W, 3) array.
    Raise ValueError for empty sources, missing saturationValue, or bad bands.
    """
    if not flatSources:
        raise ValueError("At least one flat-field source is required")
    if not isinstance(bandHeight, int) or isinstance(bandHeight, bool) or bandHeight < 1:
        raise ValueError("bandHeight must be a positive integer")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    profiles = {}
    profileFrames = []
    for label, source in flatSources.items():
        if isinstance(source, (str, Path)):
            if saturationValue is None:
                raise ValueError("saturationValue is required when passing RAW paths")
            source = raw_to_linear_rgb(source, saturationValue)["stack"]
        normalizedFlat, _ = normalize_flat_field(source, roiSize)
        height, width = normalizedFlat.shape[:2]
        if bandHeight > height:
            raise ValueError(f"bandHeight must not exceed image height ({height})")
        top = (height - bandHeight) // 2
        profile = normalizedFlat[top:top + bandHeight].mean(axis=0)
        profiles[label] = profile

        xPosition = (np.arange(width) + 0.5) / width
        for channelIndex, channelName in enumerate("RGB"):
            profileFrames.append(pd.DataFrame({
                "position": xPosition,
                "signal": profile[:, channelIndex],
                "round": label,
                "channel": channelName,
            }))

    sns.lineplot(
        data=pd.concat(profileFrames, ignore_index=True),
        x="position",
        y="signal",
        hue="round",
        style="channel",
        style_order=["R", "G", "B"],
        estimator=None,  # One sample per x per line; skip aggregation
        linewidth=1.2,
        ax=ax,
    )
    ax.axhline(1.0, color="gray", linewidth=0.8)
    ax.set_xlabel("Horizontal position (fraction of image width)")
    ax.set_ylabel("Normalized signal (center = 1)")
    ax.set_title("Center-row flat-field profiles by round")
    sns.move_legend(ax, "lower center", ncol=2, fontsize=8)
    return ax, profiles
