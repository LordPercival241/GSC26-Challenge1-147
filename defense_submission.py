from collections import OrderedDict
import torch


def robust_aggregation(num_models, models):
    """
    Robust aggregation: Norm Bounding + Cosine Filter + Adaptive Multi-Krum
    + Coordinate-wise Median.

    Four-stage defense pipeline:
      1. Norm Bounding: Clips models whose deviation from the coordinate-wise
         median exceeds 1.3x the median deviation norm. This neutralizes
         scaling attacks that preserve direction but amplify magnitude.
      1.5. Cosine Similarity Filter: Flags models whose average cosine
         similarity to all others is anomalously low, catching directional
         attacks that pass norm checks.
      2. Adaptive Multi-Krum: Computes pairwise L2 distances and adaptively
         estimates f (number of malicious clients) from the norm distribution
         gap structure, then selects the (N - f) best models.
      3. Coordinate-wise Median: Computes the per-parameter median of the
         selected models, providing a high breakdown point (>0.25) against
         any remaining coordinated perturbations.

    Args:
        num_models: Integer count of client models.
        models: List of SmallCNN state_dict objects.

    Returns:
        One aggregated SmallCNN state_dict.
    """
    if not isinstance(num_models, int) or num_models <= 0:
        raise ValueError("num_models must be a positive integer.")

    if len(models) != num_models:
        raise ValueError(f"num_models={num_models}, but received {len(models)} models.")

    parameter_names = list(models[0].keys())

    # ==========================================
    # Stage 0: Compute reference median center
    # ==========================================
    # Using the coordinate-wise median as a robust reference point,
    # rather than the mean which is sensitive to outliers.
    ref_median = OrderedDict()
    for name in parameter_names:
        stacked = torch.stack(
            [m[name].detach().to(device="cpu", dtype=torch.float64) for m in models],
            dim=0,
        )
        ref_median[name] = torch.median(stacked, dim=0).values

    # ==========================================
    # Stage 1: Norm Bounding
    # ==========================================
    # Compute deviation of each model from the median center,
    # then clip any model whose deviation norm exceeds 1.5x the median norm.
    deviations = []
    for model in models:
        dev_tensors = []
        for name in parameter_names:
            diff = model[name].to(device="cpu", dtype=torch.float64) - ref_median[name]
            dev_tensors.append(diff.flatten())
        deviations.append(torch.cat(dev_tensors))

    norms = torch.tensor([torch.norm(d).item() for d in deviations])
    median_norm = torch.median(norms).item()
    clip_threshold = max(median_norm * 1.3, 1e-7)  # Tighter threshold for stronger filtering

    clipped_models = []
    for i, model in enumerate(models):
        if norms[i].item() > clip_threshold:
            # Scale the deviation down to the threshold
            scale = clip_threshold / norms[i].item()
            clipped = OrderedDict()
            for name in parameter_names:
                diff = model[name].detach().to(device="cpu", dtype=torch.float64) - ref_median[name]
                clipped_val = ref_median[name] + diff * scale
                clipped[name] = clipped_val.to(dtype=model[name].dtype).contiguous()
            clipped_models.append(clipped)
        else:
            clipped_models.append(model)

    # ==========================================
    # Stage 1.5: Cosine Similarity Filter
    # ==========================================
    # Detect models pointing in a radically different direction.
    # This catches directional attacks that pass norm checks.
    flat_clipped = []
    for model in clipped_models:
        tensors = [model[name].to(device="cpu", dtype=torch.float64).flatten() for name in parameter_names]
        flat_clipped.append(torch.cat(tensors))

    stacked_flat_models = torch.stack(flat_clipped)

    # Compute cosine similarity matrix
    norms_for_cos = stacked_flat_models.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = stacked_flat_models / norms_for_cos
    cos_matrix = torch.mm(normalized, normalized.t())

    # Average cosine similarity per model (exclude self-similarity)
    avg_cos = []
    for i in range(num_models):
        mask = torch.ones(num_models, dtype=torch.bool)
        mask[i] = False
        avg_cos.append(cos_matrix[i][mask].mean().item())
    avg_cos_tensor = torch.tensor(avg_cos)

    # Flag models whose cosine similarity is below Q1 - 1.5*IQR
    sorted_cos, _ = torch.sort(avg_cos_tensor)
    q1_idx = max(0, num_models // 4)
    q3_idx = min(num_models - 1, 3 * num_models // 4)
    q1 = sorted_cos[q1_idx].item()
    q3 = sorted_cos[q3_idx].item()
    iqr = q3 - q1
    cos_lower_bound = q1 - 1.5 * iqr

    # Penalize flagged models by scaling their flat vectors toward the median
    for i in range(num_models):
        if avg_cos[i] < cos_lower_bound:
            # Shrink this model 50% toward the reference median
            for name in parameter_names:
                original = clipped_models[i][name].to(device="cpu", dtype=torch.float64)
                target = ref_median[name]
                blended = 0.5 * original + 0.5 * target
                clipped_models[i] = OrderedDict(clipped_models[i])
                clipped_models[i][name] = blended.to(dtype=models[0][name].dtype).contiguous()
            # Recompute flat vector for this model
            tensors = [clipped_models[i][name].to(device="cpu", dtype=torch.float64).flatten() for name in parameter_names]
            stacked_flat_models[i] = torch.cat(tensors)

    # ==========================================
    # Stage 2: Adaptive Multi-Krum Selection
    # ==========================================
    # Compute pairwise L2 distances with numerically stable mode
    dists = torch.cdist(
        stacked_flat_models,
        stacked_flat_models,
        p=2,
        compute_mode="donot_use_mm_for_euclid_dist",
    )

    # Adaptive f estimation: look for natural gap in sorted norm distribution
    norms_sorted, norms_order = torch.sort(norms)
    gaps = norms_sorted[1:] - norms_sorted[:-1]
    median_gap = torch.median(gaps).item() if len(gaps) > 0 else 0.0

    # If there's a gap significantly larger than the median gap,
    # use it as the boundary between benign and malicious clusters
    if len(gaps) > 0 and median_gap > 0:
        largest_gap_val, largest_gap_idx = torch.max(gaps, dim=0)
        largest_gap_idx = largest_gap_idx.item()
        if largest_gap_val.item() > 2.0 * median_gap and largest_gap_idx < num_models - 1:
            f = num_models - largest_gap_idx - 1
            f = max(1, min(f, int(num_models * 0.45)))  # Bound: [1, 45%]
        else:
            f = max(1, int(num_models * 0.30))  # Default: 30%
    else:
        f = max(1, int(num_models * 0.30))

    # Krum scores: sum of distances to the k closest neighbors
    k = max(1, num_models - f - 2)

    scores = []
    for i in range(num_models):
        sorted_dists, _ = torch.sort(dists[i])
        # First element is distance to itself (0.0), so take indices 1..k+1
        score = torch.sum(sorted_dists[1:k + 1]).item()
        scores.append(score)

    # Select top m models with lowest Krum scores
    m = max(1, num_models - f)
    selected_indices = torch.topk(torch.tensor(scores), m, largest=False).indices.tolist()

    selected_models = [clipped_models[i] for i in selected_indices]

    # ==========================================
    # Stage 3: Coordinate-wise Median
    # ==========================================
    result = OrderedDict()

    for name in parameter_names:
        reference = models[0][name]

        tensors_to_stack = []
        for model in selected_models:
            value = model[name]
            if value.shape != reference.shape:
                raise ValueError(f"Shape mismatch for parameter {name!r}.")
            tensors_to_stack.append(value.detach().to(device="cpu", dtype=torch.float64))

        stacked_tensors = torch.stack(tensors_to_stack, dim=0)

        # Coordinate-wise median: high breakdown point aggregation
        aggregated_tensor = torch.median(stacked_tensors, dim=0).values

        # Restore original dtype for SmallCNN compatibility
        result[name] = aggregated_tensor.to(dtype=reference.dtype).contiguous()

    return result
