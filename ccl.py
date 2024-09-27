import torch


def label_row_2d(arr):
    transitions = (arr - torch.roll(arr, 1, dims=1)) == 1
    transitions[:, 0] = arr[:, 0] == 1
    labels = torch.cumsum(transitions, dim=1)
    labels = labels * arr
    offset = torch.cumsum(torch.max(labels, dim=1).values, dim=0)[:-1]
    offset = torch.cat((torch.tensor([0], dtype=torch.int64, device=arr.device), offset))
    labels += offset[:, None]
    labels = labels * arr
    return labels


def fill_forward(arr: torch.Tensor) -> torch.Tensor:
    if torch.all(arr == 0):
        return arr
    arr[0] = arr[torch.argmax((arr > 0) * 1)]
    mask = arr == 0
    idx = torch.where(~mask, torch.arange(mask.shape[0], device=arr.device), torch.tensor(0, device=arr.device))
    return arr[torch.cummax(idx, dim=0)[0]]


def vertical(row1, row2, transitions):
    arr = transitions * row1
    last_value = fill_forward(arr)

    result = last_value * (row2 > 0)
    flipped = torch.flip(result, dims=[0])
    arr = flipped

    arr_non_zero = (arr > 0).to(torch.uint8)
    transitions = (arr_non_zero - torch.roll(arr_non_zero, 1)) == 1
    transitions[0] = arr_non_zero[0] == 1
    transitions = transitions.to(torch.uint8)
    transitions = transitions * arr
    ff = fill_forward(transitions)
    ff = torch.flip(ff, dims=[0])
    result = ff * (result > 0)
    result[result == 0] = row2[result == 0]

    # Fix inner
    to_reset = ~torch.isin(row2, row2[row1 > 0])
    result[to_reset] = row2[to_reset]

    return result


def vertical_all(arr_input):
    arr_offset = torch.cat((torch.zeros_like(arr_input[0:1]), arr_input[:-1]), dim=0)
    both_non_zero = (arr_offset * arr_input > 0).to(torch.uint8)
    transitions = (both_non_zero - torch.roll(both_non_zero, 1, dims=1)) == 1
    transitions[:, 0] = both_non_zero[:, 0] == 1
    transitions = transitions.to(torch.uint8)
    for idx in range(1, len(arr_input)):
        arr_input[idx] = vertical(arr_input[idx - 1], arr_input[idx], transitions[idx])
    return arr_input


def label(img_to_label):
    if len(img_to_label.shape) != 2:
        raise ValueError('The input image must be 2D')
    flip_img = img_to_label.shape[0] > img_to_label.shape[1]
    if flip_img:  # Algorithm works on rows
        img_to_label = torch.transpose(img_to_label, 0, 1)

    img_rows = label_row_2d(img_to_label)
    labeled_arr_inv = img_rows.max() + 1 - img_rows
    labeled_arr_inv[img_rows == 0] = 0
    labeled_arr_inv = vertical_all(labeled_arr_inv)

    # Back down-top
    labeled_arr_inv = torch.flipud(labeled_arr_inv)
    pairs = torch.stack((labeled_arr_inv[1:], labeled_arr_inv[:-1]), dim=2).reshape(-1, 2)
    pairs = pairs[pairs[:, 0] != 0]
    pairs = pairs[pairs[:, 1] != 0]
    pairs = torch.unique(pairs, dim=0)
    pairs = pairs[pairs[:, 1].argsort(descending=True)]
    for from_label, to_label in pairs:
        labeled_arr_inv[labeled_arr_inv == from_label] = to_label

    if flip_img:  # Restore original shape
        labeled_arr_inv = torch.transpose(labeled_arr_inv, 0, 1)

    _, labeled_arr_inv = torch.unique(labeled_arr_inv, return_inverse=True)
    return labeled_arr_inv, labeled_arr_inv.max()
