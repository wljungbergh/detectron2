import torch
from torch import nn

RB_KERNEL: torch.Tensor = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4
GB_KERNEL: torch.Tensor = torch.tensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4


class LearnableISP(nn.Module):
    """Learnable ISP layer."""

    def __init__(
        self, use_normalize: bool, use_yeojohnson: bool, use_gamma: bool, use_erf: bool, normalize_with_max: bool
    ):
        super().__init__()
        # assert B, 1, H, W
        self.use_normalize = use_normalize
        # modes
        self.use_yeojohnson = use_yeojohnson
        self.use_erf = use_erf
        self.use_gamma = use_gamma
        self.normalize_with_max = normalize_with_max
        # these are mutually exclusive
        assert (
            sum([self.use_yeojohnson, self.use_erf, self.use_gamma]) <= 1
        ), "Only one of use_yeojohnson, use_erf, use_gamma can be True."

        self.rb_kernel = nn.Conv2d(
            1, 1, 3, padding="same", padding_mode="reflect", bias=False
        )
        self.rb_kernel.weight.data = RB_KERNEL.view(1, 1, 3, 3)
        self.g_kernel = nn.Conv2d(
            1, 1, 3, padding="same", padding_mode="reflect", bias=False
        )
        self.g_kernel.weight.data = GB_KERNEL.view(1, 1, 3, 3)

        # remove required gradient
        self.rb_kernel.weight.requires_grad = False
        self.g_kernel.weight.requires_grad = False

        self.lambda_param = nn.Parameter(torch.tensor(0.35), requires_grad=True)
        self.gamma_param = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.mu_param = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.sigma_param = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.luminance_extractor = nn.Conv2d(3, 1, 1, bias=False)
        self.luminance_extractor.weight.data = torch.tensor(
            [[[[0.2126, 0.7152, 0.0722]]]]
        ).view(1, 3, 1, 1)
        self.luminance_extractor.weight.requires_grad = False

    def _decompanding(self, x):
        """Decompanding based on yeo-johnsson."""
        # mask = x == 0
        # x_not_zero = x[~mask]
        x = (torch.pow((x + 1), (self.lambda_param)) - 1) / self.lambda_param
        # x[~mask] = x_not_zero
        # x[mask] = torch.log(x[mask] + 1)

        return x

    def _demosaicing(self, x):
        """Batch demosaic."""
        # start with a zero matrix
        b, _, h, w = x.shape
        channel_masks = torch.zeros(b, 3, h, w, device=x.device)

        channel_masks[:, 0, 0::2, 0::2] = 1
        channel_masks[:, 1, 0::2, 1::2] = 1
        channel_masks[:, 1, 1::2, 0::2] = 1
        channel_masks[:, 2, 1::2, 1::2] = 1

        x = x * channel_masks

        # red channel
        red = x[:, 0, :, :]
        red = self.rb_kernel(red.unsqueeze(1)).squeeze(1)

        # green channel
        green = x[:, 1, :, :]
        green = self.g_kernel(green.unsqueeze(1)).squeeze(1)

        # blue channel
        blue = x[:, 2, :, :]
        blue = self.rb_kernel(blue.unsqueeze(1)).squeeze(1)

        # stack channels
        x = torch.stack([red, green, blue], dim=1)

        return x

    def _reinhardt_tonemapping(self, x):
        """Reinhard tonemapping."""
        # extract luminance
        lum = self.luminance_extractor(x) + 1e-6

        x = x / (self.gamma_param + lum)

        return x

    def _gamma(self, x):
        """Gamma correction."""
        x = torch.pow(x, self.gamma_param)

        return x

    def _learnable_yeo_johnsson(self, x):
        """Learnable yeo-johnsson."""
        x = (torch.pow((x + 1), (self.lambda_param)) - 1) / self.lambda_param
        return x

    def _learnable_gamma(self, x):
        x = self._demosaicing(x)
        x = self._gamma(x)
        return x

    def _learnable_erf(self, x):
        x = torch.erf((x - self.mu_param) / (self.sigma_param * torch.sqrt(2.0)))
        return x

    def _normalize(self, x):
        """Normalize to 0-1 range for 12 bit images."""
        x = x / 4095.0
        return x

    def forward(self, x):
        """Forward pass."""
        b, _, h, w = x.shape
        if self.use_normalize:
            x = self._normalize(x)
        if self.use_yeojohnson:
            x = self._learnable_yeo_johnsson(x)
        elif self.use_gamma:
            x = self._learnable_gamma(x)
        elif self.use_erf:
            x = self._learnable_erf(x)
        if self.normalize_with_max:
            x = x / x.view(b, 3, -1).max(dim=2)[0].view(b, 3, 1, 1)

        return x
