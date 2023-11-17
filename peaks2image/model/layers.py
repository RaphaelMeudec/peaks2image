import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import List, Dict


class MaskedLinear(nn.Linear):
    """Same as Linear except that the outputs are masked:
    out = mask * out_lin with out_lin = input * W + b.

    Example:
    --------
    We have sets of base length n = 300 and wish to mask them to allow
    variable set lengths n' < n.
    """

    def __init__(self, in_features, out_features, bias=True, mask=None):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        self.mask = mask

    def set_mask(self, mask: torch.Tensor):
        """Set the mask of the layer.

        Parameters
        ----------
        mask: torch.Tensor of size (batch_size, n)
        """
        # A last dimension is added to the mask to allow broadcasting to
        # tensors of shape (batch_size, set_size, input_dim)
        self.mask = mask

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        input: torch.Tensor of size (batch_size, n, in_features)

        Returns:
        --------
        Output: torch.Tensor of size (batch_size, n, out_features)
        """
        return self.mask * F.linear(input, self.weight, self.bias)


class MAB(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        normalize: bool = True,
        masked_X: bool = True,
        masked_Y: bool = True,
        use_query_as_residual: bool = True,
        **kwargs
    ):
        """Multihead Attention Block:
        MAB(X, Y) = LayerNorm(H + rFF(H)) ∈ R n x dim
        where H = LayerNorm(X + Multihead(X, Y, Y))

        Parameters
        ----------
        dim: int
            Data point dimension (of the elements of X and Y)
        num_heads: int
            Number of heads in the multi-head architecture
        normalize: bool, optional
            if True, use LayerNorm layers as part of the architecture
            (as per the original paper),
            by default True
        masked_X: bool, optional
            if True, uses MaskedLinear layer instead of the vanilla nn.Linear
            in multi-head embedding of X (to generate Q) and rFF,
            by default True
        masked_Y: bool, optional
            if True, uses MaskedLinear layer instead of the vanilla nn.Linear
            in multi-head embedding of Y (to generate K and V),
            by default True
        use_query_as_residual: bool, optional
            if True, uses query(X) instead of X for the residual connection of eq. 6.
            Original paper indicates X but their implementation goes towards query(X).
        """

        super(MAB, self).__init__()

        self.dim = dim
        # typical choice for the split dimension of the heads
        self.dim_split = dim // num_heads

        # type of linear layer for embeddings and rFF
        linear_X = MaskedLinear if masked_X else nn.Linear
        linear_Y = MaskedLinear if masked_Y else nn.Linear

        # embeddings for multi-head projections
        self.fc_q = linear_X(dim, dim)
        self.fc_k = linear_Y(dim, dim)
        self.fc_v = linear_Y(dim, dim)

        # row-wise feed-forward layer
        self.rff = nn.Sequential(linear_X(dim, dim), nn.ReLU())

        self.use_query_as_residual = use_query_as_residual
        self.normalize = normalize
        if normalize:
            self.layer_norm_h = nn.LayerNorm(dim)
            self.layer_norm_out = nn.LayerNorm(dim)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute MAB(X,Y)

        Parameters
        ----------
        X: torch.Tensor of size (batch_size, n, dim)
            Embedded to define the Query (Q) vector
        Y: torch.Tensor of size (batch_size, m, dim)
            Embedded to define the Key (K) and Value (V) vectors

        Returns
        -------
        O: torch.Tensor of size (batch_size, n, dim)
            Output MAB(X,Y)
        """
        assert X.shape[-1] == self.dim
        assert Y.shape[-1] == self.dim

        assert X.shape[0] == Y.shape[0]

        batch_size, _, _ = X.shape

        # embedding for multi-head projections (masked or not)
        Q = self.fc_q(X)  # (B, N, D)
        K, V = self.fc_k(Y), self.fc_v(Y)  # (B, M, D)

        # Split into num_head vectors (num_heads * batch_size, n/m, dim_split)
        Q_ = torch.cat(Q.split(self.dim_split, -1), 0)  # (B', N, D')
        K_ = torch.cat(K.split(self.dim_split, -1), 0)  # (B', M, D')
        V_ = torch.cat(V.split(self.dim_split, -1), 0)  # (B', M, D')

        # Attention weights of size (num_heads * batch_size, n, m):
        # measures how similar each pair of Q and K is.
        W = torch.softmax(
            Q_.bmm(K_.transpose(-2, -1)) / math.sqrt(self.dim), -1  # (B', D', M)
        )  # (B', N, M)

        # Multihead output (batch_size, n, dim):
        # weighted sum of V where a value gets more weight if its corresponding
        # key has larger dot product with the query.
        H = torch.cat(
            (
                (W.bmm(V_)).split(  # (B', N, M)  # (B', N, D')
                    batch_size, 0
                )  # [(B, N, D')] * num_heads
            ),
            -1,
        )  # (B, N, D)

        residual_term = Q if self.use_query_as_residual else X
        H = residual_term + H
        if self.normalize:
            H = self.layer_norm_h(H)
        out = H + self.rff(H)
        if self.normalize:
            out = self.layer_norm_out(out)
        return out


class SAB(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        normalize: bool = True,
        masked: bool = True,
        use_query_as_residual: bool = True,
        **kwargs
    ):
        """Self Attention Block:
        SAB(X) = MAB(X, X) ∈ R n x dim

        Parameters
        ----------
        dim: int
            Data point dimension (of the elements of X)
        num_heads: int
            Number of heads in the multi-head architecture
        normalize: bool, optional
            if True, use LayerNorm layers as part of
            the architecture (as per the original paper),
            by default True
        masked: bool, optional
            if True, uses MaskedLinear layer in MAB instead of nn.Linear
            to account for variable set lengths in X,
            by default True
        use_query_as_residual: bool, optional
            if True, uses query(X) instead of X for the residual connection of eq. 6.
            Original paper indicates X but their implementation goes towards query(X).
        """
        super(SAB, self).__init__()
        self.mab = MAB(
            dim,
            num_heads,
            normalize=normalize,
            masked_X=masked,
            masked_Y=masked,
            use_query_as_residual=use_query_as_residual,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute SAB(X)

        Parameters
        ----------
        X: torch.Tensor of size (batch_size, n, dim)
            Used for Query, Key and Value vectors.

        Returns
        -------
        Output SAB(X): torch.Tensor of size (batch_size, n, dim)
        """
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_inds: int,
        normalize: bool = True,
        masked: bool = True,
        **kwargs
    ):
        """Induced Self Attention Block:
        ISAB (X) = MAB(X, H) ∈ R n x dim
        where H = MAB(I, X) ∈ R num_inds x dim

        Parameters
        ----------
        dim: int
            Data point dimension (of the elements of X)
        num_heads: int
            Number of heads in the multi-head architecture
        num_inds: int
            Number of inducing points
        normalize: bool, optional
            if True, use LayerNorm layers as part of
            the architecture (as per the original paper),
            by default True
        masked: bool, optional
            if True, uses MaskedLinear layers instead of nn.Linear
            to account for variable set lengths in X,
            by default True
        """
        super(ISAB, self).__init__()

        self.inducing_points = nn.Parameter(torch.Tensor(1, num_inds, dim))
        nn.init.xavier_uniform_(self.inducing_points)

        # no masks in the multi-head embedding of I, nor the rFF
        # masks needed for the the multi-head embedding of X
        self.mab_h = MAB(
            dim, num_heads, normalize=normalize, masked_X=False, masked_Y=masked
        )
        # masks needed for the multi-head embedding of X and the rFF
        # no masks in the multi-head embedding of H
        self.mab_out = MAB(
            dim, num_heads, normalize=normalize, masked_X=masked, masked_Y=False
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute ISAB(X)

        Parameters
        ----------
        X: torch.Tensor of size (batch_size, n, dim)
            Used for Key and Value vectors in MAB_1 and Query vector in MAB_2

        Returns
        -------
        Output ISAB(X): torch.Tensor of size (batch_size, n, dim)
        """

        H = self.mab_h(
            self.inducing_points.repeat(X.size(0), 1, 1), X  # (B, I, D)  # (B, N, D)
        )  # (B, I, D)
        return self.mab_out(X, H)  # (B, N, D)  # (B, I, D)  # (B, N, D)


class Encoder(nn.Module):
    def __init__(
        self,
        type: str,
        kwargs_per_layer: List[Dict],
        dim: int,
        masked: bool = True,
        **kwargs
    ):
        """Set Transformer encoder
        Stack of SAB or ISAB blocks
           Encoder(X) = SAB(SAB(... X)) ∈ R n x dim
        or Encoder(X) = ISAB(ISAB(... X)) ∈ R n x dim

        Parameters
        ----------
        type : str
            one of ["SAB", "ISAB"]
        kwargs_per_layer: List[Dict]
            kwargs for SAB or ISAB class (num_heads, normalize)
            except dim that has to be the same for all
        dim: int
            Data point dimension (of the elements of X)
        masked: bool, optional
            if True, uses MaskedLinear layer instead of nn.Linear
            to account for variable set lengths in X,
            by default True

        Raises
        ------
        AssertionError
            if type is not in ["SAB", "ISAB"]
        """
        super(Encoder, self).__init__()

        if type not in ["SAB", "ISAB"]:
            raise AssertionError("type should be one of [`SAB`, `ISAB`]")

        for layer_kwarg in kwargs_per_layer:
            layer_kwarg["dim"] = dim
            layer_kwarg["masked"] = masked

        layers = [
            (SAB if type == "SAB" else ISAB)(**layer_kwargs)
            for layer_kwargs in kwargs_per_layer
        ]
        self.seq = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Encoding for Set Transformer (stack of SAB or ISAB layers)

        Parameters
        ----------
        X: torch.Tensor of size (batch_size, n, dim)

        Returns
        -------
        Encoding of size (batch_size, n, dim)
        """
        return self.seq(X)


class PMA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_seeds: int,
        normalize: bool = True,
        rFF: bool = False,
        masked: bool = True,
        **kwargs
    ):
        """Pooling by Multihead Attention block:
        PMA(Z) = MAB(S, rFF(Z)) ∈ R num_seeds x dim
        where S ∈ R num_seeds x dim

        Parameters
        ----------
        dim: int
            Data point dimension (of the elements of Z)
        num_seeds : int
            number of seed vectors
        normalize: bool, optional
            if True, use LayerNorm layers as part of
            the architecture (as per the original paper),
            by default True
        rFF: bool, optional
            if True use rFF to embedd Z (as in paper)
            by default False (as in authors implementation)
        masked: bool, optional
            if True, uses MaskedLinear layer in rFF and MAB instead of
            nn.Linear to account for variable set lengths in Z,
            by default True
        """
        super(PMA, self).__init__()

        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)

        if rFF:
            linear = MaskedLinear(dim, dim) if masked else nn.Linear(dim, dim)
            self.rff = nn.Sequential(linear, nn.ReLU())
        else:
            self.rff = nn.Identity()

        # no masks in the multi-head embedding of S, nore the rFF
        # masks needed for the the multi-head embedding of Z
        self.mab = MAB(
            dim, num_heads, normalize=normalize, masked_X=False, masked_Y=masked
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Compute PMA(Z)

        Parameters
        ----------
        Z: torch.Tensor of size (batch_size, n, dim)
            Output of the Encoder

        Returns
        -------
        Pooled data PMA(Z): torch.Tensor of size (batch_size, num_seeds, dim)
        """
        Z_ = self.rff(Z)  # not in authors implementation

        return self.mab(
            self.S.repeat(Z.size(0), 1, 1), Z_  # (B, S, D)  # (B, N, D)
        )  # (B, S, D)


class Decoder(nn.Module):
    def __init__(
        self,
        pma_kwargs: Dict,  # num_heads, num_seeds, normalize
        sab_kwargs: Dict,  # num_heads, normalize
        dim: int,
        masked: bool = True,
        **kwargs
    ):
        """Set Transformer Decoder
        Decoder(Z) = rFF(SAB(PMA(Z))) ∈ R num_seeds x dim

        Parameters
        ----------
        pma_kwargs : Dict
            PMA kwargs (num_heads, num_seeds, normalize, rff)
            except dim that has to be the same for all
        sab_kwargs : Dict
            SAB kwargs (num_heads, normalize)
            except dim that has to be the same for all
        dim : int
            Data point dimension (of the elements of Z)
        masked: bool, optional
            if True, uses MaskedLinear layer instead of nn.Linear
            to account for variable set lengths in Z,
            by default True
        """
        super(Decoder, self).__init__()

        pma_kwargs["dim"] = dim
        pma_kwargs["masked"] = masked
        self.pma = PMA(**pma_kwargs)

        sab_kwargs["dim"] = dim
        # input of sab is output of pma that already handled the masked case
        sab_kwargs["masked"] = False
        self.sab = SAB(**sab_kwargs)

        # TODO: remove ReLU
        self.rff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

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

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Decoding for Set Transformer

        Parameters
        ----------
        Z: torch.Tensor of size (batch_size, n, dim)
            Output of the Encoder

        Returns
        -------
        Decoded data: torch.Tensor of size (batch_size, num_seeds, dim)
        """
        out_pma = self.pma(Z)  # (batch_size, num_seeds, dim)

        return self.rff(self.sab(out_pma))  # (batch_size, num_seeds, dim)
