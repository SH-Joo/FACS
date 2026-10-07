"""Compatibility adapter for the FACS-Net logits model."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from facs.model import (CBAM, ChannelAttention, ConvUpsample, DepthWiseConv,
                        DoubleConv, EfficientMSA, FACSNet, FPCM, LayerNorm2d,
                        MiT, MixFFN, OverlapPatchEmbedding, ResNetEncoder,
                        SpatialAttention)

conv_upsample = ConvUpsample


class Model(FACSNet):
    """Model-name adapter; use explicit options and the logits contract."""
    def __init__(self, args=None, **options):
        if args is not None and hasattr(args, "cnn_weights"):
            options.setdefault("cnn_weights", args.cnn_weights)
        super().__init__(**options)
