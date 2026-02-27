import torch
import os
import sys

def decode_boxes(layer_output: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    # Input: tl/br coordinates
    # Output: Center position
    xy = layer_output[..., :2] * (anchor[..., 2:] - anchor[..., :2]) + (anchor[..., 2:] + anchor[..., :2]) / 2
    # Output: width/height
    wh = torch.exp(layer_output[..., 2:]) * (anchor[..., 2:] - anchor[..., :2])
    # Convert back to xyxy format
    decoded_box = torch.cat((xy - wh/2, xy+wh/2), -1)
    return decoded_box



def concat_repeat_last(list_1d: list[torch.Tensor]) -> torch.Tensor:
    """
    list_1d: list of 1D tensors (variable length). Assumes each tensor has len>=1.
    Returns: [N, Lmax] tensor where shorter rows repeat their last element.
    """
    assert len(list_1d) > 0
    device = list_1d[0].device
    dtype  = list_1d[0].dtype

    # Lengths (small loop over tensors; unavoidable with Python list input)
    lengths = torch.tensor([t.numel() for t in list_1d], device=device, dtype=torch.long)
    if torch.any(lengths == 0):
        raise ValueError("All tensors must be non-empty (need a 'last' element to repeat).")

    N = lengths.numel()
    Lmax = int(lengths.max().item())

    # Concatenate all values into one 1D buffer
    flat = torch.cat(list_1d, dim=0)  # [sum(lengths)]

    # Row base offsets into flat for each tensor
    # offsets[i] = starting index in `flat` for row i
    offsets = torch.empty(N, device=device, dtype=torch.long)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(lengths[:-1], dim=0)

    # Build per-row indices 0..Lmax-1, clamped to len_i-1
    col = torch.arange(Lmax, device=device, dtype=torch.long)  # [Lmax]
    idx_in_row = torch.minimum(col[None, :], (lengths - 1)[:, None])  # [N, Lmax]

    # Convert row-local indices to flat indices and gather
    flat_idx = offsets[:, None] + idx_in_row  # [N, Lmax]
    out = flat[flat_idx]  # advanced indexing / gather

    return out


def concat_repeat_last_with_classes(
    list_1d: list[torch.Tensor],
    classes: list[int],
) -> torch.Tensor:
    """
    list_1d : list of 1D tensors (variable length)
    classes : list of integers, same length as list_1d

    Behavior:
        - If classes[i] == 0 → tensor is skipped (may be empty).
        - If classes[i] != 0 → tensor must be non-empty.
        - Output contains only tensors with class != 0.
        - Shorter tensors repeat their last element to match max length.

    Returns:
        Tensor of shape [N_valid, Lmax]
    """

    if len(list_1d) != len(classes):
        raise ValueError("list_1d and classes must have the same length.")

    if len(list_1d) == 0:
        raise ValueError("Input list is empty.")

    device = list_1d[0].device
    dtype = list_1d[0].dtype

    # Boolean mask of tensors to keep
    class_tensor = torch.tensor(classes, device=device)
    keep_mask = class_tensor != 0

    if not torch.any(keep_mask):
        # No valid tensors → return empty tensor with shape [0, 0]
        return torch.empty((0, 0), device=device, dtype=dtype)

    # Filter tensors to keep (minimal Python loop, unavoidable)
    filtered = [t for t, keep in zip(list_1d, keep_mask.tolist()) if keep]

    # Compute lengths
    lengths = torch.tensor([t.numel() for t in filtered],
                           device=device, dtype=torch.long)

    if torch.any(lengths == 0):
        raise ValueError("Non-zero class tensors must be non-empty.")

    N = lengths.numel()
    Lmax = int(lengths.max().item())

    # Concatenate all kept tensors
    flat = torch.cat(filtered, dim=0)

    # Compute row offsets
    offsets = torch.empty(N, device=device, dtype=torch.long)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(lengths[:-1], dim=0)

    # Column indices
    col = torch.arange(Lmax, device=device, dtype=torch.long)

    # Clamp to repeat last element
    idx_in_row = torch.minimum(col[None, :], (lengths - 1)[:, None])

    # Map to flat indices
    flat_idx = offsets[:, None] + idx_in_row

    # Gather
    out = flat[flat_idx]

    return out


def prompt_load_if_exists(path: str) -> bool:
    """
    Check whether a file exists and, if so, prompt the user whether to load it.

    Parameters
    ----------
    path : str
        Path to the file.

    Returns
    -------
    bool
        True  -> user chose to load the file
        False -> file does not exist or user declined
    """

    if not os.path.isfile(path):
        return False

    # Interactive terminal check (optional but recommended for robustness)
    if not sys.stdin.isatty():
        # Non-interactive session: default to False
        return False

    prompt = f"File '{path}' found. Load it? [y/N]: "

    while True:
        try:
            answer = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()  # clean newline on Ctrl+D / Ctrl+C
            return False

        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no", ""):
            return False

        print("Please answer 'y' or 'n'.")

