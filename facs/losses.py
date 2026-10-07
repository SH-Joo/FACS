"""Independent logits-based objectives, including FACS-Net Eqs. 19–23."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def validate_pair(logits, target):
    if logits.ndim != 4 or logits.shape[1] != 1 or logits.shape != target.shape:
        raise ValueError("Loss expects equal [B,1,H,W] logits and GT")
    if not torch.isfinite(target).all() or not ((target == 0) | (target == 1)).all():
        raise ValueError("GT must be binary 0/1; sigmoid is applied only to predictions")


def soft_skeleton(probability, iterations=25):
    """2D differentiable thinning with the cross erosion from clDice [69]."""
    if probability.ndim != 4 or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("Soft skeleton expects BCHW and nonnegative iterations")
    def erode(image):
        vertical = -F.max_pool2d(-image, (3, 1), 1, (1, 0))
        horizontal = -F.max_pool2d(-image, (1, 3), 1, (0, 1))
        return torch.minimum(vertical, horizontal)
    def opening(image):
        return F.max_pool2d(erode(image), 3, 1, 1)
    residual = F.relu(probability - opening(probability))
    skeleton = residual
    for _ in range(iterations):
        probability = erode(probability)
        residual = F.relu(probability - opening(probability))
        skeleton = skeleton + F.relu(residual * (1 - skeleton))
    return skeleton


class SoftCLDiceLoss(nn.Module):
    def __init__(self, iterations=25, eps=1e-6):
        super().__init__()
        self.iterations, self.eps = iterations, eps

    def forward(self, logits, target):
        probability = logits.sigmoid()
        sp = soft_skeleton(probability, self.iterations)
        sg = soft_skeleton(target, self.iterations)
        dims = (1, 2, 3)
        precision = (sp*target).sum(dims) / (sp.sum(dims)+self.eps)
        recall = (sg*probability).sum(dims) / (sg.sum(dims)+self.eps)
        score = 2*precision*recall / (precision+recall).clamp_min(self.eps)
        return 1-score.mean()


class SoftCTSLoss(nn.Module):
    """Literal Gaussian-affinity score; identical soft fields need not score one.

    Gaussian is sum-normalized. Scores are computed per image before averaging.
    Empty affinity pairs have score zero. No sqrt or perfect-score correction.
    """
    def __init__(self, iterations=25, sigma=2.0, kernel_size=13, eps=1e-6):
        super().__init__()
        if (not isinstance(iterations, int) or iterations < 0 or not math.isfinite(sigma)
                or sigma <= 0 or not isinstance(kernel_size, int) or kernel_size < 1
                or kernel_size % 2 != 1 or not math.isfinite(eps) or eps <= 0):
            raise ValueError("Invalid soft-CTS parameters")
        self.iterations, self.sigma, self.kernel_size, self.eps = iterations, sigma, kernel_size, eps
        axis = torch.arange(kernel_size, dtype=torch.float64) - kernel_size//2
        gaussian = torch.exp(-(axis[:, None].square()+axis[None, :].square())/(2*sigma**2))
        self.register_buffer("gaussian", (gaussian/gaussian.sum())[None, None])

    def affinity(self, mask):
        skeleton = soft_skeleton(mask, self.iterations)
        return F.conv2d(skeleton, self.gaussian.to(skeleton), padding=self.kernel_size//2)

    def score_from_probabilities(self, probability, target):
        ap, ag = self.affinity(probability), self.affinity(target)
        dims = (1, 2, 3)
        overlap = (ap*ag).sum(dims)
        precision = overlap / (ap.sum(dims)+self.eps)
        recall = overlap / (ag.sum(dims)+self.eps)
        return 2*precision*recall / (precision+recall).clamp_min(self.eps)

    def forward(self, logits, target):
        return 1-self.score_from_probabilities(logits.sigmoid(), target).mean()


class EdgeNetwork(nn.Module):
    """SEMEDA Section 3.2: two-class mask -> 16 -> 32 -> two-class edges."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(2, 16, 3, padding=1), nn.Conv2d(16, 32, 3, padding=1),
            nn.Conv2d(32, 2, 3, padding=1),
        ])

    def features(self, distribution):
        outputs = []
        for index, layer in enumerate(self.layers):
            distribution = layer(distribution)
            outputs.append(distribution)
            if index < 2:
                distribution = F.relu(distribution)
        return outputs

    def forward(self, distribution):
        return self.features(distribution)[-1]


def binary_distribution(foreground):
    return torch.cat((1-foreground, foreground), dim=1)


def semantic_edges(target):
    """Edge iff one of eight neighbours differs; replicate labels at image border."""
    padded = F.pad(target, (1, 1, 1, 1), mode="replicate")
    upper = F.max_pool2d(padded, 3, 1)
    lower = -F.max_pool2d(-padded, 3, 1)
    return (upper != lower).squeeze(1).long()


def edge_cross_entropy(logits, edges):
    """Unweighted mean CE; avoid nondeterministic CUDA NLL2D reduction."""
    return -logits.log_softmax(1).gather(1, edges[:, None]).mean()


class SEMEDALoss(nn.Module):
    def __init__(self, network, layer_weights=(1., 1., 1.), reduction="mean"):
        super().__init__()
        if len(layer_weights) != 3 or any(not math.isfinite(w) or w < 0 for w in layer_weights):
            raise ValueError("SEMEDA requires three nonnegative layer weights")
        if reduction not in ("mean", "sum"):
            raise ValueError("SEMEDA reduction must be mean or sum")
        self.network, self.layer_weights, self.reduction = network, tuple(layer_weights), reduction
        self.network.requires_grad_(False)
        self.network.eval()

    def train(self, mode=True):
        super().train(mode)
        self.network.eval()
        return self

    def forward(self, logits, target):
        predicted = self.network.features(binary_distribution(logits.sigmoid()))
        with torch.no_grad():
            expected = self.network.features(binary_distribution(target))
        return sum(weight*F.l1_loss(p, g, reduction=self.reduction)
                   for weight, p, g in zip(self.layer_weights, predicted, expected))


class SobelLoss(nn.Module):
    """Uploaded-code diagnostic; deliberately has a different name from SEMEDA."""
    def __init__(self):
        super().__init__()
        horizontal = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer("filters", torch.stack((horizontal, horizontal.T))[:, None])

    def forward(self, logits, target):
        def gradient(mask):
            edges = F.conv2d(mask, self.filters.to(mask), padding=1)
            return torch.sqrt(edges.square().sum(1, keepdim=True)+1e-12)
        return F.l1_loss(gradient(logits.sigmoid()), gradient(target))


class WeightedLoss(nn.Module):
    """Each named active term is evaluated once; zero-weight terms are skipped."""
    NAMES = {"bce", "dice", "cldice", "soft_cts", "semeda", "sobel"}

    def __init__(self, weights, *, soft_parameters=None, edge_network=None,
                 edge_checkpoint=None, semeda_layer_weights=(1., 1., 1.),
                 semeda_reduction="mean", dice_smooth=1e-6):
        super().__init__()
        if set(weights)-self.NAMES or not weights or any(not math.isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError(f"Invalid loss weights: {weights}")
        self.weights = {key: float(value) for key, value in weights.items() if value > 0}
        if not math.isfinite(dice_smooth) or dice_smooth <= 0:
            raise ValueError("Dice smoothing must be positive and finite")
        self.dice_smooth = dice_smooth
        parameters = soft_parameters or {}
        self.terms = nn.ModuleDict()
        for name in self.weights:
            if name == "soft_cts":
                self.terms[name] = SoftCTSLoss(**parameters)
            elif name == "cldice":
                self.terms[name] = SoftCLDiceLoss(iterations=parameters.get("iterations", 25), eps=parameters.get("eps", 1e-6))
            elif name == "sobel":
                self.terms[name] = SobelLoss()
            elif name == "semeda":
                if edge_network is None:
                    if edge_checkpoint is None:
                        raise ValueError("SEMEDA needs a pretrained edge checkpoint; run pretrain-edge first")
                    state = torch.load(Path(edge_checkpoint), map_location="cpu", weights_only=True)
                    if state.get("kind") != "semeda-edge-v1" or state.get("training_split") != "train":
                        raise ValueError("Edge checkpoint must record SEMEDA training on train only")


                    with torch.random.fork_rng(devices=[]):
                        edge_network = EdgeNetwork()
                    edge_network.load_state_dict(state["model"], strict=True)
                self.terms[name] = SEMEDALoss(edge_network, semeda_layer_weights, semeda_reduction)

    def forward(self, logits, target):
        validate_pair(logits, target)
        target = target.to(logits)
        values = {}
        for name in self.weights:
            if name == "bce":
                values[name] = F.binary_cross_entropy_with_logits(logits, target)
            elif name == "dice":
                p = logits.sigmoid()
                dims = (1, 2, 3)
                values[name] = (1-(2*(p*target).sum(dims)+self.dice_smooth)
                               /(p.sum(dims)+target.sum(dims)+self.dice_smooth)).mean()
            else:
                values[name] = self.terms[name](logits, target)
        total = sum(self.weights[name]*value for name, value in values.items())
        return total, values
