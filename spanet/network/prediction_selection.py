import torch
from torch import Tensor
from typing import List

TArray = Tensor

TFloat32 = torch.float32
TInt64 = torch.int64

NUMBA_DEBUG = False

if NUMBA_DEBUG:
    def script(fn):
        return fn
else:
    script = torch.jit.script


@script
def mask_1(data: Tensor, size: int, index: int, value: float):
    data[index] = value


@script
def mask_2(flat_data: Tensor, size: int, index: int, value: float):
    data = flat_data.view(size, size)
    data[index, :] = value
    data[:, index] = value


@script
def mask_3(flat_data: Tensor, size: int, index: int, value: float):
    data = flat_data.view(size, size, size)
    data[index, :, :] = value
    data[:, index, :] = value
    data[:, :, index] = value


@script
def mask_jet(data: Tensor, num_partons: int, max_jets: int, index: int, value: float):
    if num_partons == 1:
        mask_1(data, max_jets, index, value)
    elif num_partons == 2:
        mask_2(data, max_jets, index, value)
    elif num_partons == 3:
        mask_3(data, max_jets, index, value)


@script
def compute_strides(num_partons: int, max_jets: int) -> Tensor:
    strides = torch.zeros(num_partons, dtype=torch.int64)
    strides[-1] = 1
    for i in range(num_partons - 2, -1, -1):
        strides[i] = strides[i + 1] * max_jets
    return strides


@script
def unravel_index(index: int, strides: Tensor) -> Tensor:
    num_partons = strides.shape[0]
    result = torch.zeros(num_partons, dtype=torch.int64)

    remainder = index
    for i in range(num_partons):
        result[i] = remainder // strides[i]
        remainder %= strides[i]
    return result


@script
def ravel_index(index: Tensor, strides: Tensor) -> int:
    return (index * strides).sum().item()


@script
def maximal_prediction(predictions: List[Tensor]):
    best_jet = -1
    best_prediction = -1
    best_value = -float('inf')

    for i in range(len(predictions)):
        max_jet = torch.argmax(predictions[i]).item()
        max_value = predictions[i][max_jet].item()

        if max_value > best_value:
            best_prediction = i
            best_value = max_value
            best_jet = max_jet

    return best_jet, best_prediction, best_value


@script
def extract_prediction(predictions: List[Tensor], num_partons: Tensor, max_jets: int) -> Tensor:
    float_negative_inf = -float('inf')
    max_partons = num_partons.max().item()
    num_targets = len(predictions)

    strides = []
    for i in range(num_targets):
        strides.append(compute_strides(num_partons[i].item(), max_jets))

    results = torch.full((num_targets, max_partons), -2, dtype=torch.int64)

    for _ in range(num_targets):
        best_jet, best_prediction, best_value = maximal_prediction(predictions)

        if not torch.isfinite(torch.tensor(best_value)):
            return results

        best_jets = unravel_index(best_jet, strides[best_prediction])

        results[best_prediction, :] = -1
        for i in range(num_partons[best_prediction]):
            results[best_prediction, i] = best_jets[i]

        predictions[best_prediction][:] = float_negative_inf
        for i in range(num_targets):
            for jet in best_jets:
                mask_jet(predictions[i], num_partons[i].item(), max_jets, jet.item(), float_negative_inf)

    return results


@script
def _extract_predictions(predictions: List[Tensor], num_partons: Tensor, max_jets: int, batch_size: int) -> Tensor:
    output = torch.zeros(batch_size, len(predictions), num_partons.max().item(), dtype=torch.int64)
    predictions = [p.clone() for p in predictions]

    for batch in range(batch_size):
        current_prediction = [prediction[batch] for prediction in predictions]
        output[batch, :, :] = extract_prediction(current_prediction, num_partons, max_jets)

    return output.permute(1, 0, 2).contiguous()


def extract_predictions(predictions: List[Tensor]):
    flat_predictions = [p.view(p.size(0), -1) for p in predictions]
    num_partons = torch.tensor([len(p.size()) - 1 for p in predictions], dtype=torch.int64)
    max_jets = max(max(p.size()[1:]) for p in predictions)
    batch_size = max(p.size(0) for p in predictions)

    results = _extract_predictions(flat_predictions, num_partons, max_jets, batch_size)
    return [result[:, :partons] for result, partons in zip(results, num_partons)]
