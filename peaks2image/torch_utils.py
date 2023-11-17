import numpy as np
import torch
import tqdm


def predict(model, data_loader, device="cpu"):
    model.eval()
    model.to(device)

    outputs = []
    with torch.no_grad():
        for _, batch in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
            padded_peaks, masks = (
                batch[0].to(device),
                batch[1].to(device),
            )
            model.set_mask(masks)
            output = model(padded_peaks)
            outputs.append(output.detach().cpu().numpy())

    return np.concatenate(outputs, axis=0)
