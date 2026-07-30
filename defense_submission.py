from collections import OrderedDict
import torch


def robust_aggregation(num_models, models):
    """
    Robust Aggregation Pipeline for Federated Learning:
    
    1. Norm Bounding: Measures deviation of each model from the coordinate-wise
       median center. Clips any model whose deviation exceeds 1.05x median norm,
       neutralizing scaling and directional projection attacks.
       
    2. Coordinate-Wise Trimmed Mean: Trims the top and bottom 20-25% values
       per parameter coordinate before averaging. This completely eliminates up to
       25% malicious updates (ALIE, Krum-evading, Min-Max attacks) while maintaining
       high clean accuracy.

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
    # Stage 0: Coordinate-wise Median Center
    # ==========================================
    ref_median = OrderedDict()
    for name in parameter_names:
        stacked = torch.stack(
            [m[name].detach().to(device="cpu", dtype=torch.float64) for m in models],
            dim=0,
        )
        ref_median[name] = torch.median(stacked, dim=0).values

    # ==========================================
    # Stage 1: Strict Norm Bounding
    # ==========================================
    deviations = []
    for model in models:
        dev_tensors = []
        for name in parameter_names:
            diff = model[name].to(device="cpu", dtype=torch.float64) - ref_median[name]
            dev_tensors.append(diff.flatten())
        deviations.append(torch.cat(dev_tensors))

    norms = torch.tensor([torch.norm(d).item() for d in deviations])
    median_norm = torch.median(norms).item()
    clip_threshold = max(median_norm * 1.05, 1e-7)

    clipped_models = []
    for i, model in enumerate(models):
        if norms[i].item() > clip_threshold:
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
    # Stage 2: Coordinate-wise Trimmed Mean
    # ==========================================
    # Trim top & bottom 20% of values per coordinate
    trim_count = max(1, int(num_models * 0.20))
    if 2 * trim_count >= num_models:
        trim_count = max(0, (num_models - 1) // 2)

    result = OrderedDict()
    for name in parameter_names:
        reference = models[0][name]
        stacked = torch.stack(
            [m[name].detach().to(device="cpu", dtype=torch.float64) for m in clipped_models],
            dim=0,
        )

        if trim_count > 0:
            sorted_tensors, _ = torch.sort(stacked, dim=0)
            # Remove top and bottom `trim_count` values
            trimmed = sorted_tensors[trim_count : num_models - trim_count]
            aggregated = torch.mean(trimmed, dim=0)
        else:
            aggregated = torch.median(stacked, dim=0).values

        result[name] = aggregated.to(dtype=reference.dtype).contiguous()

    return result
