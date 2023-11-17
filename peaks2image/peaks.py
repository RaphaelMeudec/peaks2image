import warnings
import numpy as np

def pad_group(peaks, max_number_of_peaks):
    padded_group = np.pad(
        peaks[["X", "Y", "Z"]].values, ((0, max_number_of_peaks - len(peaks)), (0, 0))
    )
    mask = np.concatenate(
        [
            np.ones((len(peaks), 1)),
            np.zeros((max_number_of_peaks - len(peaks), 1)),
        ],
        axis=0,
    )

    return padded_group, mask


def pad_peaks(peaks_df, max_number_of_peaks, id_column="image_path"):
    groups = peaks_df.groupby(id_column)

    images_paths = []
    padded_peaks = []
    masks = []
    for image_path, group in groups:
        if len(group) >= max_number_of_peaks:
            warnings.warn(
                f"Image {image_path} contains {len(group)} peaks which is more "
                f"than {max_number_of_peaks} peaks. Ignoring them."
            )
            group = group.head(max_number_of_peaks)
        images_paths.append(image_path)
        padded_group, mask = pad_group(group, max_number_of_peaks)
        padded_peaks.append(padded_group)
        masks.append(mask)

    return images_paths, np.array(padded_peaks), np.array(masks)
