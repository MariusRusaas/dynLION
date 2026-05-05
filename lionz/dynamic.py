#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIONZ Dynamic PET Segmentation
-------------------------------

This module extends LION to handle dynamic (4D) PET images. It segments tumors
frame-by-frame, tracks them temporally, and outputs a 4D NIfTI with consistent
per-tumor class labels across all frames.

.. versionadded:: 1.1.0
"""

import os
import time

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, center_of_mass

from lionz import constants
from lionz import file_utilities
from lionz import image_processing
from lionz import models
from lionz import predict
from lionz import system


def extract_frames(img_4d: sitk.Image) -> list[sitk.Image]:
    """Split a 4D SimpleITK image into a list of 3D frames."""
    size = list(img_4d.GetSize())  # [X, Y, Z, T]
    num_frames = size[3]
    frames = []

    for t in range(num_frames):
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize([size[0], size[1], size[2], 0])
        extractor.SetIndex([0, 0, 0, t])
        frame = extractor.Execute(img_4d)
        frames.append(frame)

    return frames


def segment_frames(
    frames: list[sitk.Image],
    model_routine: dict,
    accelerator: str,
    output_manager: system.OutputManager,
    threshold: float | None = None,
) -> list[np.ndarray]:
    """
    Run LION segmentation on each frame, reusing the predictor across frames.
    Returns a list of binary 3D masks in original image space.
    """
    masks = []

    for desired_spacing, model_workflows in model_routine.items():
        for model_workflow in model_workflows:
            model_obj = model_workflow[0]

            # Initialize predictor once
            output_manager.log_update("  Initializing predictor for dynamic segmentation...")
            predictor = predict.initialize_predictor(model_obj, accelerator)

            for i, frame in enumerate(frames):
                output_manager.spinner_update(
                    f"Segmenting frame {i + 1}/{len(frames)}..."
                )
                output_manager.log_update(f"  - Segmenting frame {i + 1}/{len(frames)}")

                # Resample to model spacing
                resampled_array = image_processing.ImageResampler.resample_image_SimpleITK_DASK_array(
                    frame, 'bspline', desired_spacing
                )

                # Predict using shared predictor
                seg_array = predict.predict_with_predictor(predictor, resampled_array, model_obj)

                # Convert to SimpleITK and resample back to original space
                seg_img = sitk.GetImageFromArray(seg_array)
                seg_img.SetSpacing(desired_spacing)
                seg_img.SetOrigin(frame.GetOrigin())
                seg_img.SetDirection(frame.GetDirection())
                resampled_seg = image_processing.ImageResampler.resample_segmentation(frame, seg_img)

                if threshold is not None:
                    resampled_seg = image_processing.threshold_segmentation_sitk(
                        frame, resampled_seg, threshold
                    )

                masks.append(sitk.GetArrayFromImage(resampled_seg))

    return masks


def label_tumors_by_size(binary_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Label connected components in a binary mask, ordered by descending volume.
    Largest tumor gets label 1, second largest gets label 2, etc.

    Returns (labeled_mask, num_tumors).
    """
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=np.int32), 0

    labeled, num_features = label(binary_mask.astype(np.int32))

    # Compute volume of each component
    volumes = []
    for i in range(1, num_features + 1):
        volumes.append((i, np.sum(labeled == i)))

    # Sort by volume descending
    volumes.sort(key=lambda x: x[1], reverse=True)

    # Relabel: largest = 1, second = 2, etc.
    relabeled = np.zeros_like(labeled, dtype=np.int32)
    for new_label, (old_label, _) in enumerate(volumes, start=1):
        relabeled[labeled == old_label] = new_label

    return relabeled, len(volumes)


def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = np.sum((mask_a > 0) & (mask_b > 0))
    union = np.sum((mask_a > 0) | (mask_b > 0))
    if union == 0:
        return 0.0
    return intersection / union


def _compute_centroid_distance_mm(
    mask_a: np.ndarray, mask_b: np.ndarray, spacing: tuple[float, ...]
) -> float:
    """Compute distance in mm between centroids of two binary masks."""
    com_a = center_of_mass(mask_a > 0)
    com_b = center_of_mass(mask_b > 0)
    # spacing is (X, Y, Z) but arrays are (Z, Y, X)
    spacing_zyx = tuple(reversed(spacing))
    dist = 0.0
    for i in range(len(com_a)):
        dist += ((com_a[i] - com_b[i]) * spacing_zyx[i]) ** 2
    return dist ** 0.5


def track_tumors_across_frames(
    labeled_masks: list[np.ndarray],
    spacing: tuple[float, ...],
) -> list[np.ndarray]:
    """
    Track tumors across frames using spatial overlap (IoU) and centroid distance.

    Starts from the last frame (canonical reference) and walks backward.
    After tracking, renumbers labels globally so label 1 = tumor with largest
    max-volume across all frames.
    """
    num_frames = len(labeled_masks)
    if num_frames == 0:
        return []

    tracked = [None] * num_frames
    tracked[-1] = labeled_masks[-1].copy()

    next_new_label = int(np.max(labeled_masks[-1])) + 1

    # Walk backward from second-to-last frame
    for t in range(num_frames - 2, -1, -1):
        current_labels = labeled_masks[t]
        prev_tracked = tracked[t + 1]  # already-tracked adjacent frame

        result = np.zeros_like(current_labels, dtype=np.int32)
        current_unique = set(np.unique(current_labels)) - {0}
        prev_unique = set(np.unique(prev_tracked)) - {0}

        if not current_unique:
            tracked[t] = result
            continue

        # For each label in the previous (already-tracked) frame, find best match in current
        assigned_current = set()
        label_mapping = {}  # current_label -> tracked_label

        for prev_label in prev_unique:
            prev_region = (prev_tracked == prev_label)
            best_iou = 0.0
            best_current = None

            for cur_label in current_unique - assigned_current:
                cur_region = (current_labels == cur_label)
                iou = _compute_iou(prev_region, cur_region)
                if iou > best_iou:
                    best_iou = iou
                    best_current = cur_label

            if best_current is not None and best_iou >= constants.TRACKING_IOU_THRESHOLD:
                label_mapping[best_current] = prev_label
                assigned_current.add(best_current)
            elif best_current is not None:
                # Fallback: check centroid distance
                cur_region = (current_labels == best_current)
                dist = _compute_centroid_distance_mm(prev_region, cur_region, spacing)
                if dist <= constants.TRACKING_MAX_CENTROID_DISTANCE_MM:
                    label_mapping[best_current] = prev_label
                    assigned_current.add(best_current)

        # Assign tracked labels
        for cur_label in current_unique:
            cur_region = (current_labels == cur_label)
            if cur_label in label_mapping:
                result[cur_region] = label_mapping[cur_label]
            else:
                # New tumor not seen in adjacent frame
                result[cur_region] = next_new_label
                next_new_label += 1

        tracked[t] = result

    # Global renumbering: label 1 = tumor with largest max-volume across time
    all_labels = set()
    for mask in tracked:
        all_labels.update(set(np.unique(mask)) - {0})

    if not all_labels:
        return tracked

    max_volumes = {}
    for lbl in all_labels:
        max_vol = 0
        for mask in tracked:
            vol = np.sum(mask == lbl)
            if vol > max_vol:
                max_vol = vol
        max_volumes[lbl] = max_vol

    sorted_labels = sorted(all_labels, key=lambda l: max_volumes[l], reverse=True)
    label_remap = {old: new for new, old in enumerate(sorted_labels, start=1)}

    renumbered = []
    for mask in tracked:
        new_mask = np.zeros_like(mask, dtype=np.int32)
        for old_label, new_label in label_remap.items():
            new_mask[mask == old_label] = new_label
        renumbered.append(new_mask)

    return renumbered


def sanity_check_and_propagate(
    tracked_masks: list[np.ndarray],
    frames: list[sitk.Image],
    spacing: tuple[float, ...],
) -> list[np.ndarray]:
    """
    Sanity check segmentations using the last N frames as ground truth.

    1. Ground truth tumors = tumors present in the last GROUND_TRUTH_FRAMES frames.
    2. Walk backward: if a ground-truth tumor is missing in frame t, copy from frame t+1.
    3. Determine stopping frame: earliest frame where detected tumor volume >= 75% of
       ground-truth volume.
    4. For frames before the stopping frame, copy the mask from the stopping frame.
    """
    num_frames = len(tracked_masks)
    gt_count = min(constants.GROUND_TRUTH_FRAMES, num_frames)
    result = [m.copy() for m in tracked_masks]

    # Identify ground-truth tumor labels (present in all of the last gt_count frames)
    gt_labels = None
    for t in range(num_frames - gt_count, num_frames):
        frame_labels = set(np.unique(result[t])) - {0}
        if gt_labels is None:
            gt_labels = frame_labels
        else:
            gt_labels = gt_labels & frame_labels

    if not gt_labels:
        return result

    # Compute ground-truth total volume (average of last gt_count frames)
    gt_volume = 0
    for t in range(num_frames - gt_count, num_frames):
        for lbl in gt_labels:
            gt_volume += np.sum(result[t] == lbl)
    gt_volume /= gt_count

    # Backward propagation: fill missing ground-truth tumors
    for t in range(num_frames - gt_count - 1, -1, -1):
        for lbl in gt_labels:
            if np.sum(result[t] == lbl) == 0:
                # Copy this tumor region from frame t+1
                region = (result[t + 1] == lbl)
                result[t][region] = lbl

    # Determine stopping frame based on volume ratio
    stopping_frame = 0
    for t in range(num_frames):
        frame_volume = 0
        for lbl in gt_labels:
            frame_volume += np.sum(result[t] == lbl)
        if frame_volume >= constants.STOPPING_VOLUME_FRACTION * gt_volume:
            stopping_frame = t
            break

    # For frames before stopping frame, copy mask from stopping frame
    for t in range(stopping_frame):
        result[t] = result[stopping_frame].copy()

    return result


def assemble_4d_output(
    masks: list[np.ndarray],
    reference_4d: sitk.Image,
) -> sitk.Image:
    """
    Combine per-frame labeled masks into a 4D SimpleITK image.
    Preserves spacing, origin, and direction from the reference 4D image.
    """
    stacked = np.stack(masks, axis=0)  # shape: (T, Z, Y, X)
    img_4d = sitk.GetImageFromArray(stacked.astype(np.int32))

    # Copy metadata from reference
    img_4d.SetSpacing(reference_4d.GetSpacing())
    img_4d.SetOrigin(reference_4d.GetOrigin())
    img_4d.SetDirection(reference_4d.GetDirection())

    return img_4d


def run_dynamic_pipeline(
    subject: str,
    subject_index: int,
    number_of_subjects: int,
    model_routine: dict,
    accelerator: str,
    output_manager: system.OutputManager,
    threshold: float | None = None,
    generate_mip: bool = False,
):
    """
    Orchestrator for the dynamic (4D) PET segmentation pipeline.

    Loads a 4D NIfTI, segments each frame, tracks tumors temporally,
    applies sanity checks, and saves a 4D labeled segmentation.
    """
    subject_name = os.path.basename(subject)

    if output_manager is None:
        output_manager = system.OutputManager(False, False)

    output_manager.log_update(f" SUBJECT (DYNAMIC): {subject_name}")

    # Set up directory structure
    lion_dir, segmentations_dir, stats_dir = file_utilities.lion_folder_structure(subject)
    output_manager.log_update(f" LION directory for subject {subject_name} at: {lion_dir}")

    # Find PT file
    modality_files = []
    for prefix in ('PT_', 'PT-'):
        modality_files = file_utilities.get_files(subject, prefix, ('.nii', '.nii.gz'))
        if modality_files:
            break

    if not modality_files:
        output_manager.warn(f"No PT files found for subject {subject_name}, skipping")
        return

    file_path = modality_files[0]
    file_name = file_utilities.get_nifti_file_stem(file_path)

    # Load 4D image
    start_time = time.time()
    output_manager.spinner_update(
        f"[{subject_index + 1}/{number_of_subjects}] Loading dynamic PET for {subject_name}..."
    )
    img_4d = sitk.ReadImage(file_path)
    spacing = img_4d.GetSpacing()  # (X, Y, Z, T)
    spatial_spacing = spacing[:3]

    output_manager.log_update(f"  4D image size: {img_4d.GetSize()}")

    # Extract frames
    frames = extract_frames(img_4d)
    num_frames = len(frames)
    output_manager.log_update(f"  Extracted {num_frames} frames")
    output_manager.info(f"Dynamic PET: {num_frames} frames detected for {subject_name}")

    # Segment each frame
    output_manager.section("Frame-by-frame Segmentation")
    binary_masks = segment_frames(frames, model_routine, accelerator, output_manager, threshold)

    # Label tumors by size in each frame
    output_manager.spinner_update(
        f"[{subject_index + 1}/{number_of_subjects}] Labeling tumors for {subject_name}..."
    )
    labeled_masks = []
    for i, mask in enumerate(binary_masks):
        labeled, num_tumors = label_tumors_by_size(mask)
        labeled_masks.append(labeled)
        output_manager.log_update(f"  Frame {i + 1}: {num_tumors} tumor(s) found")

    # Track tumors across frames
    output_manager.spinner_update(
        f"[{subject_index + 1}/{number_of_subjects}] Tracking tumors across frames for {subject_name}..."
    )
    tracked_masks = track_tumors_across_frames(labeled_masks, spatial_spacing)

    # Sanity check and backward propagation
    output_manager.spinner_update(
        f"[{subject_index + 1}/{number_of_subjects}] Sanity checking segmentations for {subject_name}..."
    )
    final_masks = sanity_check_and_propagate(tracked_masks, frames, spatial_spacing)

    # Assemble and save 4D output
    output_manager.spinner_update(
        f"[{subject_index + 1}/{number_of_subjects}] Saving dynamic segmentation for {subject_name}..."
    )
    output_4d = assemble_4d_output(final_masks, img_4d)
    output_path = os.path.join(segmentations_dir, f"{file_name}_dynamic_tumor_seg.nii.gz")
    sitk.WriteImage(output_4d, output_path)

    elapsed = (time.time() - start_time) / 60
    output_manager.log_update(f"  Dynamic segmentation saved to: {output_path}")
    output_manager.log_update(
        f"  Total time for {subject_name}: {round(elapsed, 1)} min"
    )

    # Summary
    all_labels = set()
    for m in final_masks:
        all_labels.update(set(np.unique(m)) - {0})
    output_manager.spinner_update(
        f"[{subject_index + 1}/{number_of_subjects}] Done: {subject_name} "
        f"({num_frames} frames, {len(all_labels)} tumor(s)) | {round(elapsed, 1)} min"
    )

    return output_path
