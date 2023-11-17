from typing import Dict

import torch
import torch.functional as F
import torch.nn as nn

from peaks2image.model.layers import MaskedLinear, Decoder, Encoder


ENC_ARGS = {
    "type": "ISAB",
    "kwargs_per_layer": [{"num_heads": 2, "num_inds": 32}] * 2,
}

DEC_ARGS = {
    "pma_kwargs": {
        "num_heads": 2,
        "num_seeds": 1,
        "rff": True,  # rff False in authors implementation
    },
    "sab_kwargs": {"num_heads": 2},
}


class SetTransformer(nn.Module):
    def __init__(
        self,
        in_features: int,
        embedding_size: int,
        out_features: int = None,
        encoder_kwargs: Dict = ENC_ARGS,
        decoder_kwargs: Dict = DEC_ARGS,
        normalize: bool = False,
        masked: bool = True,
        squeeze_seeds: bool = False,
        **kwargs,
    ):
        """Fully parametrized Set Transformer
        SetTransformer(X) = Decoder(Encoder(Embedder(X)))

        Parameters
        ----------
        in_features: int
            Data point dimension (of elements of input set X)
        embedding_size: int
            Data point dimention after embedding of elements of X
            (`dim` in MAB/SAB/ISAB)
        out_features: int, optional
            Data point dimension of the output
            by default = None.
            if not None, a final Linear layer is applied
            (as in autjors implementation)
        encoder_kwargs: Dict
            Encoder kwargs (type, kwargs_per_layer)
            except dim that is defined in __init__()
        decoder_kwargs: Dict
            Decoder kwargs (pma_kwargs, sab_kwargs)
            except dim that is defined in __init__()
        masked: bool, optional
            if True, uses MaskedLinear layer instead of the vanilla nn.Linear
            in all downstream layers
            by default True
        squeeze_seeds: bool = False
            in the case of a single seed for the PMA, squeeze the 3D output
            --a 1-sized set (B, 1, D)-- to get a 2D output (B, D)
        """
        super(SetTransformer, self).__init__()
        self.masked = masked
        self.embedding_size = embedding_size

        linear = nn.Linear(in_features, embedding_size)
        if self.masked:
            linear = MaskedLinear(in_features, embedding_size)
        self.embedder = nn.Sequential(
            linear, nn.ReLU()  # relu not used in authors embedding
        )

        encoder_kwargs["dim"] = embedding_size
        encoder_kwargs["masked"] = self.masked
        for layer in encoder_kwargs["kwargs_per_layer"]:
            layer["normalize"] = normalize
        self.encoder = Encoder(**encoder_kwargs)

        decoder_kwargs["dim"] = embedding_size
        decoder_kwargs["masked"] = self.masked
        decoder_kwargs["pma_kwargs"]["normalize"] = normalize
        decoder_kwargs["sab_kwargs"]["normalize"] = normalize
        self.decoder = Decoder(**decoder_kwargs)

        if out_features is not None:
            self.fc_out = nn.Linear(embedding_size, out_features)
        else:
            self.fc_out = nn.Identity()

        self.squeeze_seeds = squeeze_seeds
        if squeeze_seeds:
            n_seeds = decoder_kwargs["pma_kwargs"]["num_seeds"]
            assert (
                n_seeds == 1
            ), f"Attempting to squeeze an output set of size {n_seeds} > 1!"

    def set_mask(self, mask: torch.Tensor) -> None:
        """Sets the mask attribute as well as all MaskedLinear layers.

        Parameters
        ----------
        mask : torch.Tensor
            0-1 tensor representing the mask/shape of the set
        """
        self.mask = mask
        # Set mask in all MaskedLinear layers
        for layer in self.modules():
            if isinstance(layer, MaskedLinear):
                layer.set_mask(mask)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """SetTransformer(X) = Decoder(Encoder(Embedder(X)))

        Parameters
        ----------
        X : torch.Tensor
            shape (batch_size, n, in_features)

        Returns
        -------
        torch.Tensor
            shape (batch_size, num_seeds, out_features)
        """
        embedding = self.embedder(X)  # batch_size, n, embedding_size
        code = self.encoder(embedding)  # batch_size, n, embedding_size
        output = self.decoder(code)  # batch_size, num_seeds, embedding_size
        output = self.fc_out(output)  # batch_size, num_seeds, out_features

        if self.squeeze_seeds:
            output = torch.squeeze(output, -2)

        return output
