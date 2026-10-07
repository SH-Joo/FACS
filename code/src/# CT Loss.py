import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.morphology import dilation, erosion

class SoftSkeleton(nn.Module):
    """
    Differentiable approximation of skeletonization via iterative morphological erosion.
    """
    def __init__(self, kernel_size: int = 3, iterations: int = 5):
        super().__init__()
        self.iterations = iterations
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        device = x.device

        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size, device=device)
        sk = x
        for _ in range(self.iterations):
            eroded = erosion(sk, kernel)
            diff = sk - eroded

            sk = eroded + diff * torch.sigmoid(diff * 10.0)
        return sk

class CTLoss(nn.Module):
    """
    Differentiable CT-Dice loss with approximate segment-level topology awareness.
    Segments are defined by a fixed grid over the image, and soft matching
    weights are computed per segment.
    """
    def __init__(self,
                 dilation_radius: int = 4,
                 grid_size: tuple = (4, 4),
                 sk_iter: int = 5,
                 sk_kernel: int = 3,
                 eps: float = 1e-6):
        super().__init__()
        self.dilation_radius = dilation_radius
        self.grid_size = grid_size
        self.eps = eps

        self.soft_skel = SoftSkeleton(kernel_size=sk_kernel, iterations=sk_iter)

    def forward(self, pred_logits: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:

        pred_prob = torch.sigmoid(pred_logits)
        gt_prob   = gt_mask.float().unsqueeze(1)


        skel_pred = self.soft_skel(pred_prob)
        skel_gt   = self.soft_skel(gt_prob)


        device = pred_logits.device
        k = 2 * self.dilation_radius + 1
        dil_kernel = torch.ones(1, 1, k, k, device=device)
        pred_dil = dilation(skel_pred, dil_kernel)
        gt_dil   = dilation(skel_gt,   dil_kernel)


        B, C, H, W = skel_pred.shape
        Gx, Gy = self.grid_size

        xs = torch.linspace(0, W, steps=W, device=device)
        ys = torch.linspace(0, H, steps=H, device=device)

        yy, xx = torch.meshgrid(ys, xs, indexing='ij')

        xx = xx / (W - 1)
        yy = yy / (H - 1)

        seg_masks = []
        for i in range(Gy):
            for j in range(Gx):

                x0, x1 = j / Gx, (j + 1) / Gx
                y0, y1 = i / Gy, (i + 1) / Gy
                mask = ((xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)).float()
                seg_masks.append(mask)

        seg_masks = torch.stack(seg_masks, dim=0).to(device)
        M = seg_masks.shape[0]


        seg_pred = skel_pred.unsqueeze(1) * seg_masks.unsqueeze(0)
        seg_gt   = skel_gt.unsqueeze(1)   * seg_masks.unsqueeze(0)


        sum_seg_pred = seg_pred.sum(dim=[2,3])
        tp_seg_pred  = (seg_pred * gt_dil.unsqueeze(1)).sum(dim=[2,3])
        ICS_pred     = tp_seg_pred / (sum_seg_pred + self.eps)
        weights_pred = sum_seg_pred / (sum_seg_pred.sum(dim=1, keepdim=True) + self.eps)
        PCS_seg      = (weights_pred * ICS_pred).sum(dim=1)


        sum_seg_gt = seg_gt.sum(dim=[2,3])
        tp_seg_gt  = (seg_gt * pred_dil.unsqueeze(1)).sum(dim=[2,3])
        ICS_gt     = tp_seg_gt / (sum_seg_gt + self.eps)
        weights_gt = sum_seg_gt / (sum_seg_gt.sum(dim=1, keepdim=True) + self.eps)
        RCS_seg    = (weights_gt * ICS_gt).sum(dim=1)


        CTDice_seg = 2 * PCS_seg * RCS_seg / (PCS_seg + RCS_seg + self.eps)

        return 1.0 - CTDice_seg.mean()
