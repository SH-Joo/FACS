"""FACS-Net hybrid encoder and progressive frequency-aware decoder."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, channels, *, elementwise_affine=False):
        super().__init__(channels, elementwise_affine=elementwise_affine)
        if not elementwise_affine:


            self.register_buffer("fixed_weight", torch.ones(channels), persistent=False)
            self.register_buffer("fixed_bias", torch.zeros(channels), persistent=False)

    def forward(self, x):
        weight = self.weight if self.elementwise_affine else self.fixed_weight
        bias = self.bias if self.elementwise_affine else self.fixed_bias
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape,
                            weight, bias, self.eps).permute(0, 3, 1, 2)


class DepthWiseConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, padding=1):
        super().__init__()
        self.DW_conv = nn.Conv2d(in_dim, in_dim, kernel, padding=padding,
                                 groups=in_dim, padding_mode="reflect")
        self.PW_conv = nn.Conv2d(in_dim, out_dim, 1, padding_mode="reflect")

    def forward(self, x):
        return self.PW_conv(self.DW_conv(x))


class DoubleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        hidden_dim = (in_dim + out_dim) // 2
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, kernel, stride, padding, in_dim, out_dim):
        super().__init__()
        self.kernel, self.stride, self.padding = kernel, stride, padding
        self.overlap_patches = nn.Unfold(kernel, padding=padding, stride=stride)
        self.embedding = nn.Conv2d(in_dim * kernel**2, out_dim, 1, padding_mode="reflect")

    def forward(self, x):
        batch, _, height, width = x.shape
        out_h = (height + 2*self.padding - self.kernel) // self.stride + 1
        out_w = (width + 2*self.padding - self.kernel) // self.stride + 1
        patches = self.overlap_patches(x).reshape(batch, -1, out_h, out_w)
        return self.embedding(patches)


class EfficientMSA(nn.Module):
    def __init__(self, dim, n_heads, reduction_ratio, *, norm_affine=False):
        super().__init__()
        self.norm = LayerNorm2d(dim, elementwise_affine=norm_affine)
        self.reshaping_k = nn.Conv2d(dim, dim, reduction_ratio, stride=reduction_ratio, padding_mode="reflect")
        self.reshaping_v = nn.Conv2d(dim, dim, reduction_ratio, stride=reduction_ratio, padding_mode="reflect")
        self.attention = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, x):
        batch, channels, height, width = x.shape
        x = self.norm(x)
        query = x.flatten(2).transpose(1, 2)
        key = self.reshaping_k(x).flatten(2).transpose(1, 2)
        value = self.reshaping_v(x).flatten(2).transpose(1, 2)


        output, _ = self.attention(query, key, value, need_weights=not self.training)
        return output.transpose(1, 2).reshape(batch, channels, height, width)


class MixFFN(nn.Module):
    def __init__(self, dim, expansion_factor, *, norm_affine=False):
        super().__init__()
        self.norm = LayerNorm2d(dim, elementwise_affine=norm_affine)
        latent = dim * expansion_factor
        self.mixffn = nn.Sequential(
            nn.Conv2d(dim, latent, 1, padding_mode="reflect"), DepthWiseConv(latent, latent), nn.GELU(),
            nn.Conv2d(latent, dim, 1, padding_mode="reflect"),
        )

    def forward(self, x):
        return self.mixffn(self.norm(x))


class MiT(nn.Module):
    def __init__(self, channels=3, dims=(64, 256, 512, 1024),
                 n_heads=(1, 2, 8, 8), expansion=(8, 8, 4, 4),
                 reduction_ratio=(8, 4, 2, 1), n_layers=(2, 2, 2, 2),
                 *, norm_affine=False):
        super().__init__()
        if any(len(values) != 4 for values in (dims, n_heads, expansion, reduction_ratio, n_layers)):
            raise ValueError("MiT requires exactly four stages")
        self.stages = nn.ModuleList()
        for in_dim, dim, heads, factor, reduction, depth in zip(
            (channels, *dims[:-1]), dims, n_heads, expansion, reduction_ratio, n_layers,
        ):
            blocks = nn.ModuleList([
                nn.ModuleList([EfficientMSA(dim, heads, reduction, norm_affine=norm_affine),
                               MixFFN(dim, factor, norm_affine=norm_affine)])
                for _ in range(depth)
            ])
            self.stages.append(nn.ModuleList([
                OverlapPatchEmbedding(3, 2, 1, in_dim, dim), blocks,
            ]))

    def forward(self, x):
        features = []
        for embedding, blocks in self.stages:
            x = embedding(x)
            for attention, feedforward in blocks:
                x = x + attention(x)
                x = x + feedforward(x)
            features.append(x)
        return features


class ConvUpsample(nn.Module):
    def __init__(self, scale, in_dim, out_dim):
        super().__init__()
        self.conv = DoubleConv(in_dim, out_dim)
        self.upscale = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)

    def forward(self, x):
        return self.upscale(self.conv(x))


class ResNetEncoder(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        if weights not in (None, "IMAGENET1K_V1", "IMAGENET1K_V2"):
            raise ValueError(f"Unknown ResNet50 weight version: {weights}")
        encoder = resnet50(weights=None if weights is None else ResNet50_Weights[weights])
        self.encoder1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.mp = encoder.maxpool
        self.encoder2, self.encoder3, self.encoder4 = encoder.layer1, encoder.layer2, encoder.layer3

    def forward(self, x):
        first = self.encoder1(x)
        second = self.encoder2(self.mp(first))
        third = self.encoder3(second)
        fourth = self.encoder4(third)
        return first, second, third, fourth


class GlobalPool2d(nn.Module):
    """Same global pooling values; CUDA backward supports deterministic mode."""
    def __init__(self, maximum=False):
        super().__init__()
        self.maximum = maximum

    def forward(self, x):
        if not self.training:

            if self.maximum:
                return F.adaptive_max_pool2d(x, 1)
            return F.adaptive_avg_pool2d(x, 1)
        if self.maximum:
            return x.flatten(2).max(-1, keepdim=True).values.unsqueeze(-1)
        return x.mean((-2, -1), keepdim=True)


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio):
        super().__init__()
        self.avg_pool, self.max_pool = GlobalPool2d(), GlobalPool2d(maximum=True)
        self.fc1 = nn.Conv2d(channels, max(1, channels//ratio), 1, bias=False, padding_mode="reflect")
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(max(1, channels//ratio), channels, 1, bias=False, padding_mode="reflect")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        def branch(pooled):
            return self.fc2(self.relu1(self.fc1(pooled)))
        return self.sigmoid(branch(self.avg_pool(x)) + branch(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2,
                               bias=False, padding_mode="reflect")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pooled = torch.cat((x.mean(1, keepdim=True), x.max(1, keepdim=True).values), dim=1)
        return self.sigmoid(self.conv1(pooled))


class CBAM(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)


class FPCM(nn.Module):
    def __init__(self, in_channels, cutoff_ratio=0.25, *, filter_mode="normalized", cbam=True):
        super().__init__()
        if filter_mode not in ("paper", "normalized") or not 0 < cutoff_ratio < 1:
            raise ValueError("FPCM needs mode paper/normalized and 0 < beta < 1")
        self.filter_mode = filter_mode
        if filter_mode == "paper":
            self.register_buffer("cutoff_ratio", torch.tensor(float(cutoff_ratio)))
        else:
            self.cutoff_ratio = nn.Parameter(torch.tensor(float(cutoff_ratio)))
        def processor():
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels//2, 3, padding=1, padding_mode="reflect"),
                nn.BatchNorm2d(in_channels//2), nn.ReLU(inplace=True),
            )
        self.low_processor, self.high_processor = processor(), processor()
        self.cbam = CBAM(in_channels, 8) if cbam else nn.Identity()
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(in_channels//4), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, 1, 1, padding_mode="reflect"),
        )

    def frequency_filter(self, height, width, *, device=None, dtype=torch.float32):

        yy = torch.arange(height, device=device, dtype=dtype) - height//2
        xx = torch.arange(width, device=device, dtype=dtype) - width//2
        radius = torch.sqrt(yy[:, None].square() + xx[None, :].square())
        beta = self.cutoff_ratio.to(device=device, dtype=dtype)
        if self.filter_mode == "paper":
            exponent = (radius / (2 * height * width * beta)).square()
        else:
            denominator = radius.amax().clamp_min(torch.finfo(dtype).eps)
            exponent = (radius / denominator / beta.clamp_min(1e-3)).square().clamp_max(100)
        return torch.exp(-exponent)[None, None]

    def decompose(self, x):

        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(x.device.type, enabled=False):
            stable = x.to(dtype)
            spectrum = torch.fft.fftshift(torch.fft.fft2(stable), dim=(-2, -1))
            filt = self.frequency_filter(*x.shape[-2:], device=x.device, dtype=dtype)
            low = torch.fft.ifft2(torch.fft.ifftshift(spectrum*filt, dim=(-2, -1))).real
            high = stable - low
        return low.to(x.dtype), high.to(x.dtype)

    def forward(self, x):
        low, high = self.decompose(x)
        fused = torch.cat((self.low_processor(low), self.high_processor(high)), dim=1)
        return self.seg_head(self.cbam(fused))


class FACSNet(nn.Module):
    """RGB BCHW -> one-channel logits; no loss, metric state or device side effects."""
    def __init__(self, *, cnn_weights=None, norm_affine=False, cbam=True, fpcm=True,
                 filter_mode="normalized", cutoff_ratio=0.25):
        super().__init__()
        self.head1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, padding_mode="reflect"), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, padding_mode="reflect"), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.head2 = nn.Sequential(nn.Conv2d(16, 3, 1, padding_mode="reflect"), nn.BatchNorm2d(3), nn.ReLU(inplace=True))
        self.mix_transformer = MiT(norm_affine=norm_affine)
        self.cnn_encoder = ResNetEncoder(cnn_weights)
        self.upsampling_4 = ConvUpsample(2, 2048, 512)
        self.upsampling_3 = ConvUpsample(2, 1536, 256)
        self.upsampling_2 = ConvUpsample(2, 768, 64)
        self.upsampling_1 = ConvUpsample(2, 192, 16)
        for stage, channels, ratio in ((4, 2048, 16), (3, 1536, 16), (2, 768, 16), (1, 192, 8), (0, 32, 4)):
            setattr(self, f"CBAM{stage}", CBAM(channels, ratio) if cbam else nn.Identity())
        self.fpcm_tail = (FPCM(32, cutoff_ratio, filter_mode=filter_mode, cbam=cbam)
                          if fpcm else DoubleConv(32, 8))
        if not fpcm:
            self.segmentation_head = nn.Conv2d(8, 1, 1)
        self.use_fpcm = fpcm

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 64 or any(s % 16 for s in x.shape[-2:]):
            raise ValueError("FACS expects [B,3,H,W] with H,W >= 64 and divisible by 16")
        shallow = self.head1(x)
        shared = self.head2(shallow)
        transformer = self.mix_transformer(shared)
        cnn = self.cnn_encoder(shared)
        x = self.upsampling_4(self.CBAM4(torch.cat((transformer[3], cnn[3]), dim=1)))
        for index in (2, 1, 0):
            fusion = torch.cat((transformer[index], cnn[index], x), dim=1)
            x = getattr(self, f"upsampling_{index+1}")(getattr(self, f"CBAM{index+1}")(fusion))
        x = self.fpcm_tail(self.CBAM0(torch.cat((shallow, x), dim=1)))
        return x if self.use_fpcm else self.segmentation_head(x)
