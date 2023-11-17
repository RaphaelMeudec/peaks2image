import pandas as pd
import torch

from torch.utils.data import DataLoader, TensorDataset

from peaks2image.model import (
    peaks2image,
    MAX_NUMBER_OF_PEAKS,
)
from peaks2image.peaks import pad_peaks
from peaks2image.torch_utils import predict


coordinates = pd.DataFrame({
    "X": [1, 2, 3, 4, 5],
    "Y": [4, 5, 6, 7, 8],
    "Z": [7, 8, 9, 10, 11],
    "pmid": [1, 1, 1, 2, 2],
})

pmids, padded_peaks, padded_masks = pad_peaks(
    coordinates,
    max_number_of_peaks=MAX_NUMBER_OF_PEAKS,
    id_column="pmid",
)

peaks_loader = DataLoader(
    TensorDataset(
        torch.from_numpy(padded_peaks).float(),
        torch.from_numpy(padded_masks).float(),
    )
)

difumo = predict(peaks2image, peaks_loader, device="cuda" if torch.cuda.is_available() else "cpu")
print(difumo.shape)
print(difumo)


