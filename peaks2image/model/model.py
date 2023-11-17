import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

from peaks2image.model.set_transformer import SetTransformer, ENC_ARGS, DEC_ARGS


MAX_NUMBER_OF_PEAKS = 150
POSTPROCESS_DIFUMO = lambda x: 1000 * x  # Simple rescaling done in the original model to stabilize training
# get_model(name=config.model.name, **config.model.params)

PEAKS2IMAGE_PARAMS = {
    "in_features": 3,
    "embedding_size": 128,
    "out_features": 1024,
    "num_hidden_layers": 1,
    "dropout": 0.2,
    "num_heads": 16,
    "num_attention_layers": 3
}

class Peaks2Image(nn.Module):
    def __init__(
        self,
        in_features: int,
        embedding_size: int,
        out_features: int = None,
        encoder_kwargs=ENC_ARGS,
        decoder_kwargs=DEC_ARGS,
        normalize: bool = False,
        masked: bool = True,
        squeeze_seeds: bool = True,
        use_fc_out: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.set_transformer = SetTransformer(
            embedding_size,
            embedding_size,
            out_features,
            encoder_kwargs,
            decoder_kwargs,
            normalize,
            masked,
            squeeze_seeds,
        )
        self.fc_in = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.ReLU(),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(out_features, out_features),
        )

    def set_mask(self, mask: torch.Tensor):
        self.mask = mask
        self.set_transformer.set_mask(mask)

    def forward(self, X: torch.Tensor):
        X = self.fc_in(X)
        X = self.set_transformer(X)
        X = self.fc_out(X)
        return POSTPROCESS_DIFUMO(X)
