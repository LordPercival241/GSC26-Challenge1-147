from collections import OrderedDict
import torch

def robust_aggregation(num_models, models):
    """
    Robust aggregation function for defense.
    Implements Coordinate-wise Median to filter out malicious outliers.
    Allowed packages: PyTorch, NumPy, standard library.
    """
    if not isinstance(num_models, int) or num_models <= 0:
        raise ValueError("num_models must be a positive integer.")

    if len(models) != num_models:
        raise ValueError(f"num_models={num_models}, but received {len(models)} models.")

    parameter_names = list(models[0].keys())
    result = OrderedDict()

    for name in parameter_names:
        reference = models[0][name]
        
        # Collect all tensors for this parameter across all models
        tensors_to_stack = []
        for model in models:
            value = model[name]
            if value.shape != reference.shape:
                raise ValueError(f"Shape mismatch for parameter {name!r}.")
            tensors_to_stack.append(value.detach().to(device="cpu", dtype=torch.float64))

        # Stack tensors and compute the median along the model dimension (dim=0)
        stacked_tensors = torch.stack(tensors_to_stack, dim=0)
        median_tensor, _ = torch.median(stacked_tensors, dim=0)

        # Store the aggregated (median) parameter
        result[name] = median_tensor.to(dtype=reference.dtype).contiguous()

    return result
