


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.morphology import thin
from skimage.measure import label
from scipy.ndimage import binary_dilation
import config


ALPHA = 0.8
GAMMA = 2


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, weight=0.5):
        inputs_sigmoid = F.sigmoid(inputs)
        inputs_flat = inputs.view(-1)
        inputs_sig = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_sig * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_sig.sum() + targets_flat.sum() + smooth)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        return weight * BCE + (1 - weight) * dice_loss


class clIoU_class(nn.Module):
    def __init__(self, dilation_radius=4):
        super(clIoU_class, self).__init__()
        self.dilation_radius = dilation_radius

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        inputs_bin = (inputs > 0.5).float().squeeze(1)
        targets_bin = (targets > 0.5).float().squeeze(1)
        batch_size = inputs_bin.shape[0] if inputs_bin.dim() == 3 else 1
        cliou_total = 0.0

        for i in range(batch_size):
            with torch.no_grad():
                if batch_size > 1:
                    pred_img = inputs_bin[i].cpu().numpy()
                    gt_img   = targets_bin[i].cpu().numpy()
                else:
                    pred_img = inputs_bin.cpu().numpy()[0]
                    gt_img   = targets_bin.cpu().numpy()[0]


                pred_skel = thin(pred_img.astype(bool)).astype(np.float32)
                gt_skel   = thin(gt_img.astype(bool)).astype(np.float32)
                structure = np.ones((2 * self.dilation_radius + 1, 2 * self.dilation_radius + 1))
                pred_skel_dilated = binary_dilation(pred_skel, structure=structure)
                gt_skel_dilated   = binary_dilation(gt_skel,   structure=structure)

            pred_flat = torch.tensor(pred_skel_dilated, dtype=torch.float32, device=inputs.device).flatten()
            gt_flat   = torch.tensor(gt_skel_dilated,   dtype=torch.float32, device=inputs.device).flatten()

            intersection = (pred_flat * gt_flat).sum()
            sum_pred = pred_flat.sum()
            sum_gt   = gt_flat.sum()
            union    = sum_pred + sum_gt - intersection

            if sum_gt == 0:
                cliou_img = torch.tensor(1.0 if sum_pred == 0 else 0.0, device=inputs.device)
            else:
                cliou_img = intersection / (union + 1e-6)

            cliou_total += cliou_img

        return cliou_total / batch_size

class clDice_class(nn.Module):
    def __init__(self, dilation_radius=4):
        super(clDice_class, self).__init__()
        self.dilation_radius = dilation_radius

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        inputs_bin = (inputs > 0.5).float().squeeze(1)
        targets_bin = (targets > 0.5).float().squeeze(1)
        batch_size = inputs_bin.shape[0] if inputs_bin.dim() == 3 else 1
        cldice_total = 0.0

        for i in range(batch_size):
            with torch.no_grad():
                if batch_size > 1:
                    pred_img = inputs_bin[i].cpu().numpy()
                    gt_img   = targets_bin[i].cpu().numpy()
                else:
                    pred_img = inputs_bin.cpu().numpy()[0]
                    gt_img   = targets_bin.cpu().numpy()[0]

                pred_skel = thin(pred_img.astype(bool)).astype(np.float32)
                gt_skel   = thin(gt_img.astype(bool)).astype(np.float32)
                structure = np.ones((2 * self.dilation_radius + 1, 2 * self.dilation_radius + 1))
                pred_skel_dil = binary_dilation(pred_skel, structure=structure)
                gt_skel_dil   = binary_dilation(gt_skel,   structure=structure)

            pred_flat = torch.tensor(pred_skel_dil, dtype=torch.float32, device=inputs.device).flatten()
            gt_flat   = torch.tensor(gt_skel_dil,   dtype=torch.float32, device=inputs.device).flatten()
            intersection = (pred_flat * gt_flat).sum()
            sum_pred = pred_flat.sum()
            sum_gt   = gt_flat.sum()
            cldice_total += (2.0 * intersection) / (sum_pred + sum_gt + 1e-6)

        return cldice_total / batch_size

def clIoU(pred, y, px=4):
    fn = clIoU_class(dilation_radius=px).to(pred.device)
    return fn(pred, y)

def clDice(pred, y, px=4):
    fn = clDice_class(dilation_radius=px).to(pred.device)
    return fn(pred, y)


def min_pool2d(x, kernel_size=3):
    return -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

def soft_skeleton(x, iterations=25, kernel_size=3, alpha=100):
    I0 = F.max_pool2d(min_pool2d(x, kernel_size=kernel_size),
                      kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    S = F.relu(x - I0)
    I_current = x
    for _ in range(iterations):
        I_current = min_pool2d(I_current, kernel_size=kernel_size)
        I0 = F.max_pool2d(min_pool2d(I_current, kernel_size=kernel_size),
                          kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        S = S + (1 - S) * F.relu(I_current - I0)
    return S

class SoftCLDice_class(nn.Module):
    def __init__(self, iterations=25, kernel_size=3, eps=1e-6):
        super(SoftCLDice_class, self).__init__()
        self.iterations = iterations
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, pred, target):
        pred = F.sigmoid(pred)
        target = F.sigmoid(target)
        SP = soft_skeleton(pred, iterations=self.iterations, kernel_size=self.kernel_size)
        SL = soft_skeleton(target, iterations=self.iterations, kernel_size=self.kernel_size)
        Tprec = torch.sum(SP * target, dim=[1,2,3]) / (torch.sum(SP, dim=[1,2,3]) + self.eps)
        Tsens = torch.sum(SL * pred,  dim=[1,2,3]) / (torch.sum(SL, dim=[1,2,3]) + self.eps)
        clDice = 2 * Tprec * Tsens / (Tprec + Tsens + self.eps)
        return clDice.mean()

def SoftclDice(pred, y):
    fn = SoftCLDice_class().to(pred.device)
    return fn(pred, y)
def torch_dilation(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Morphological dilation via max-pooling."""
    pad = kernel_size // 2
    return F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)

def torch_erosion(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Morphological erosion via inverted max-pooling."""
    pad = kernel_size // 2
    return -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=pad)

class SoftSkeleton(nn.Module):
    """
    Differentiable approximation of skeletonization via iterative morphological erosion.
    """
    def __init__(self, kernel_size: int = 3, iterations: int = 5):
        super().__init__()
        self.iterations = iterations
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sk = x
        for _ in range(self.iterations):
            eroded = torch_erosion(sk, self.kernel_size)
            diff = sk - eroded
            sk = eroded + diff * torch.sigmoid(diff * 10.0)
        return sk

class ct_dice(nn.Module):
    """
    Differentiable CT-Dice loss with approximate segment-level topology awareness.
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


        k = 2 * self.dilation_radius + 1
        pred_dil = torch_dilation(skel_pred, k)
        gt_dil   = torch_dilation(skel_gt,   k)


        B, C, H, W = skel_pred.shape
        Gx, Gy = self.grid_size
        xs = torch.linspace(0, W, steps=W, device=pred_logits.device)
        ys = torch.linspace(0, H, steps=H, device=pred_logits.device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        xx, yy = xx/(W-1), yy/(H-1)

        seg_masks = []
        for i in range(Gy):
            for j in range(Gx):
                x0, x1 = j/Gx, (j+1)/Gx
                y0, y1 = i/Gy, (i+1)/Gy
                mask = ((xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)).float()
                seg_masks.append(mask)
        seg_masks = torch.stack(seg_masks, dim=0).to(pred_logits.device)

        seg_pred = skel_pred.unsqueeze(1) * seg_masks.unsqueeze(0)
        seg_gt   = skel_gt.unsqueeze(1)   * seg_masks.unsqueeze(0)

        sum_p = seg_pred.sum(dim=[2,3])
        tp_p  = (seg_pred * gt_dil.unsqueeze(1)).sum(dim=[2,3])
        PCS   = (tp_p / (sum_p + self.eps) *
                 (sum_p / (sum_p.sum(1,keepdim=True)+self.eps))).sum(1)

        sum_g = seg_gt.sum(dim=[2,3])
        tp_g  = (seg_gt * pred_dil.unsqueeze(1)).sum(dim=[2,3])
        RCS   = (tp_g / (sum_g + self.eps) *
                 (sum_g / (sum_g.sum(1,keepdim=True)+self.eps))).sum(1)

        ct_dice = 2*PCS*RCS / (PCS + RCS + self.eps)
        return 1.0 - ct_dice.mean()


def torch_dilation(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Morphological dilation via max-pooling."""
    pad = kernel_size // 2
    return F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)

def torch_erosion(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Morphological erosion via inverted max-pooling."""
    pad = kernel_size // 2
    return -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=pad)

class ct_dice(nn.Module):
    """
    Differentiable CT-Dice loss with approximate segment-level topology awareness.
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


        k = 2 * self.dilation_radius + 1
        pred_dil = torch_dilation(skel_pred, k)
        gt_dil   = torch_dilation(skel_gt,   k)


        B, C, H, W = skel_pred.shape
        Gx, Gy = self.grid_size
        xs = torch.linspace(0, W, steps=W, device=pred_logits.device)
        ys = torch.linspace(0, H, steps=H, device=pred_logits.device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        xx, yy = xx/(W-1), yy/(H-1)

        seg_masks = []
        for i in range(Gy):
            for j in range(Gx):
                x0, x1 = j/Gx, (j+1)/Gx
                y0, y1 = i/Gy, (i+1)/Gy
                mask = ((xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)).float()
                seg_masks.append(mask)
        seg_masks = torch.stack(seg_masks, dim=0).to(pred_logits.device)

        seg_pred = skel_pred.unsqueeze(1) * seg_masks.unsqueeze(0)
        seg_gt   = skel_gt.unsqueeze(1)   * seg_masks.unsqueeze(0)

        sum_p = seg_pred.sum(dim=[2,3])
        tp_p  = (seg_pred * gt_dil.unsqueeze(1)).sum(dim=[2,3])
        PCS   = (tp_p / (sum_p + self.eps) *
                 (sum_p / (sum_p.sum(1,keepdim=True)+self.eps))).sum(1)

        sum_g = seg_gt.sum(dim=[2,3])
        tp_g  = (seg_gt * pred_dil.unsqueeze(1)).sum(dim=[2,3])
        RCS   = (tp_g / (sum_g + self.eps) *
                 (sum_g / (sum_g.sum(1,keepdim=True)+self.eps))).sum(1)

        ct_dice = 2*PCS*RCS / (PCS + RCS + self.eps)
        return 1.0 - ct_dice.mean()


class EdgeAwareLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(EdgeAwareLoss, self).__init__()
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32)
        self.register_buffer('sobel_kernel_x', sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_kernel_y', sobel_y.unsqueeze(0).unsqueeze(0))
        self.reduction = reduction

    def forward(self, logits, targets):
        pred = F.sigmoid(logits)
        targets = targets.float()
        kx = self.sobel_kernel_x.to(pred.dtype)
        ky = self.sobel_kernel_y.to(pred.dtype)
        pred_ex = F.conv2d(pred, kx, padding=1)
        pred_ey = F.conv2d(pred, ky, padding=1)
        tgt_ex  = F.conv2d(targets, kx, padding=1)
        tgt_ey  = F.conv2d(targets, ky, padding=1)
        pred_edge = torch.sqrt(pred_ex**2 + pred_ey**2 + 1e-6)
        tgt_edge  = torch.sqrt(tgt_ex**2 + tgt_ey**2   + 1e-6)
        return F.l1_loss(pred_edge, tgt_edge, reduction=self.reduction)

def EdgeLoss(pred, y):
    return EdgeAwareLoss().to(config.DEVICE)(pred, y)

class EdgeIoU(nn.Module):
    def __init__(self, dilation_radius=3):
        super(EdgeIoU, self).__init__()
        self.dilation_radius = dilation_radius
        self.kernel_size = 2*dilation_radius+1

    def extract_edge(self, mask):
        eroded = 1 - F.max_pool2d(1-mask, kernel_size=3, stride=1, padding=1)
        return mask - eroded

    def forward(self, pred, gt):
        pred_bin = (F.sigmoid(pred)>=0.5).float()
        gt_bin   = (gt>=0.5).float()
        pe = self.extract_edge(pred_bin)
        ge = self.extract_edge(gt_bin)
        pd = F.max_pool2d(pe, kernel_size=self.kernel_size,
                          stride=1, padding=self.dilation_radius)
        gd = F.max_pool2d(ge, kernel_size=self.kernel_size,
                          stride=1, padding=self.dilation_radius)
        inter = (pd*gd).sum(dim=[1,2,3])
        uni   = ((pd+gd)>0).float().sum(dim=[1,2,3])
        iou   = torch.where(uni==0, torch.ones_like(inter), inter/uni)
        return iou.mean()

def edgeIoU(pred, y, px=3):
    return EdgeIoU(dilation_radius=px).to(pred.device)(pred, y)

class edgeDice(nn.Module):
    def __init__(self, dilation_radius=3):
        super(edgeDice, self).__init__()
        self.dilation_radius = dilation_radius
        self.kernel_size = 2*dilation_radius+1

    def extract_edge(self, mask):
        eroded = 1 - F.max_pool2d(1-mask, kernel_size=3, stride=1, padding=1)
        return mask - eroded

    def forward(self, pred, gt):
        pred_bin = (F.sigmoid(pred)>=0.5).float()
        gt_bin   = (gt>=0.5).float()
        pe = self.extract_edge(pred_bin)
        ge = self.extract_edge(gt_bin)
        pd = F.max_pool2d(pe, kernel_size=self.kernel_size,
                          stride=1, padding=self.dilation_radius)
        gd = F.max_pool2d(ge, kernel_size=self.kernel_size,
                          stride=1, padding=self.dilation_radius)
        pf = pd.view(pd.size(0), -1)
        gf = gd.view(gd.size(0), -1)
        inter = (pf*gf).sum(dim=1)
        dice  = (2*inter+1e-6)/(pf.sum(dim=1)+gf.sum(dim=1)+1e-6)
        return dice.mean()

def edgeDiceMetric(pred, y, px=3):
    return edgeDice(dilation_radius=px).to(pred.device)(pred, y)


def compute_multi_px_metrics(pred, y, px_list=[0,2,4,8], overlap_threshold=0.5):
    cliou_list  = []
    cldice_list = []

    for px in px_list:
        cliou_list.append(clIoU(pred, y, px=px))
        cldice_list.append(clDice(pred, y, px=px))

    return cliou_list, cldice_list
