import torch
import os
from torch import nn
from einops import rearrange
from math import sqrt
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
import torchmetrics
import torchmetrics as Metric
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import config
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from metric import DiceBCELoss, DiceLoss, \
    clDice, clIoU, SoftclDice, soft_skeleton, EdgeLoss, ct_dice, Soft_ct_dice
import torchmetrics
from torchmetrics.classification \
    import BinaryJaccardIndex, BinaryRecall, BinaryAccuracy, \
        BinaryPrecision, BinaryF1Score, Dice
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import json

DEVICE = config.DEVICE

# Layer Normalisation
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x
    
# Depth-wise CNN
class DepthWiseConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, padding, stride=1, bias=True):
        super(DepthWiseConv, self).__init__()
        # Depthwise Convolution
        self.DW_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,
                                 kernel_size=kernel, stride=stride, 
                                 padding=padding, groups=in_dim, bias=bias, padding_mode='reflect')
        # Pointwise Convolution
        self.PW_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                                 kernel_size=1, bias=bias, padding_mode='reflect')
    
    def forward(self, x):
        x = self.DW_conv(x)
        x = self.PW_conv(x)

        return x
        
class DoubleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DoubleConv, self).__init__()

        hidden_dim = int((in_dim + out_dim)/2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        output = self.conv_block(x)

        return output

class OverlapPatchEmbedding(nn.Module):
    def __init__(self, kernel, stride, padding, in_dim, out_dim):
        super(OverlapPatchEmbedding, self).__init__()
        self.overlap_patches = nn.Unfold(kernel_size=kernel, stride=stride, padding=padding)
        self.embedding = nn.Conv2d(in_dim*kernel**2, out_dim, 1, padding_mode='reflect')

    def forward(self, x):
        h, w = x.shape[-2:]
        x = self.overlap_patches(x)
        n_patches = x.shape[-1]
        divider = int(sqrt(h*w / n_patches))
        x = rearrange(x, 'b c (h w) -> b c h w', h = h//divider)
        x = self.embedding(x)

        return x

class EfficientMSA(nn.Module):
    # same size of input and output
    def __init__(self, dim, n_heads, reduction_ratio):
        super(EfficientMSA, self).__init__()
        self.reshaping_k = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio, padding_mode='reflect')
        self.reshaping_v = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio, padding_mode='reflect')
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)

    def forward(self, x):
        n, c, h, w = x.shape
        LN = LayerNorm2d(c).to(device=DEVICE)
        x = LN(x)
        reshaped_k = self.reshaping_k(x)
        reshaped_v = self.reshaping_v(x)
        reshaped_k = rearrange(reshaped_k, "b c h w -> b (h w) c") # reshape (batch, sequence_length, channels) for attention
        reshaped_v = rearrange(reshaped_v, "b c h w -> b (h w) c") # reshape (batch, sequence_length, channels) for attention
        q = rearrange(x, "b c h w -> b (h w) c")
        output, output_weights = self.attention(q, reshaped_k, reshaped_v)
        output = rearrange(output, "b (h w) c -> b c h w", h=h, w=w)

        return output


class MixFFN(nn.Module):
    # same size of inputs and outputs
    def __init__(self, dim, expansion_factor):
        super(MixFFN, self).__init__()
        latent_dim = dim*expansion_factor
        self.mixffn = nn.Sequential(
            nn.Conv2d(dim, latent_dim, 1, padding_mode='reflect'),
            DepthWiseConv(latent_dim, latent_dim, kernel=3, padding=1),
            nn.GELU(),
            nn.Conv2d(latent_dim, dim, 1, padding_mode='reflect')
        )
    def forward(self, x):
        n, c, h, w = x.shape
        LN = LayerNorm2d(c).to(device=DEVICE)
        x = LN(x)
        x = self.mixffn(x)
        return x
    
class MiT(nn.Module):
    def __init__(self, channels, dims, n_heads, expansion, reduction_ratio, n_layers):
        super(MiT, self).__init__()
        kernel_stride_pad = ((3, 2, 1), (3, 2, 1), (3, 2, 1), (3, 2, 1), (3, 2, 1))
        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        self.stages = nn.ModuleList([])

        for (in_dim, out_dim), (kernel, stride, padding), n_layers, expansion, n_heads, reduction_ratio in zip(dim_pairs, kernel_stride_pad, n_layers, expansion, n_heads, reduction_ratio):
            overlapping = OverlapPatchEmbedding(kernel, stride, padding, in_dim, out_dim)
            layers = nn.ModuleList([])
            
            for _ in range(n_layers):
                layers.append(nn.ModuleList([EfficientMSA(dim=out_dim, n_heads=n_heads, reduction_ratio=reduction_ratio),
                              MixFFN(dim=out_dim, expansion_factor=expansion)]))
            self.stages.append(nn.ModuleList([overlapping, layers]))

    def forward(self, x):
        # h, w = x.shape[-2:]
        layer_outputs = []
        for overlapping, layers in self.stages:
            x = overlapping(x)  # (b, c x kernel x kernel, num_patches)
            for (attension, ffn) in layers:  # attention, feed forward
                x = attension(x) + x  # skip connection
                x = ffn(x) + x

            layer_outputs.append(x)  # multi scale features

        return layer_outputs
    
class conv_upsample(nn.Module):
    def __init__(self, scale, in_dim, out_dim=32):
        super(conv_upsample, self).__init__()
        self.conv = DoubleConv(in_dim, out_dim)
        self.upscale = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)

    def forward(self, x):
        output = self.upscale(self.conv(x))
        return output
        
#### ResNet50 ##########################################################################################################
resnet_encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
# resnet_encoder = resnet50()
class ResNetEncoder(nn.Module):
    def __init__(self, encoder = resnet_encoder):
        super(ResNetEncoder, self).__init__()
        self.encoder1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu) # 64x128x128
        self.mp = encoder.maxpool
        self.encoder2 = encoder.layer1 # 256x64x64
        self.encoder3 = encoder.layer2 # 512x32x32
        self.encoder4 = encoder.layer3 # 1024x16x16
        # self.encoder5 = encoder.layer4 # 2048x8x8

    def forward(self,x):
        output1 = self.encoder1(x)
        output2 = self.mp(output1)
        output2 = self.encoder2(output2)
        output3 = self.encoder3(output2)
        output4 = self.encoder4(output3)
        # output5 = self.encoder5(output4)

        return output1, output2, output3, output4 #, output5
    
#### CBAM ##########################################################################################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False, padding_mode='reflect')
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False, padding_mode='reflect')
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False, padding_mode='reflect')
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

# CBAM Module combining Channel and Spatial Attention
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out
    
#### FPCM ##########################################################################################################
class FPCM(nn.Module):
    def __init__(self, in_channels=96, cutoff_ratio=0.25):
        super(FPCM, self).__init__()
        self.cutoff_ratio = nn.Parameter(torch.tensor(cutoff_ratio, dtype=torch.float32))
        
        # Low frequency branch 처리
        self.low_processor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # High frequency branch 처리
        self.high_processor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )

        # Weight
        self.cbam = CBAM(in_channels, 8)      
        
        # 좀 더 복잡한 Segmentation Head (Residual block 형태 일부 포함)
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1, padding_mode='reflect')
        )
        
    def forward(self, x):
        B, C, H, W = x.shape

        # Convert input to float32 for stable FFT operations (especially under mixed precision)
        x_fp32 = x.float()
        F_x = torch.fft.fft2(x_fp32)
        F_x = torch.fft.fftshift(F_x)
        
        # Create frequency grid
        u = torch.arange(H, device=x.device).float() - (H // 2)
        v = torch.arange(W, device=x.device).float() - (W // 2)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='ij')
        D = torch.sqrt(grid_u ** 2 + grid_v ** 2)
        D_norm = D / D.max()
        
        # Gaussian low-pass filter with clamped exponent to avoid overflow
        cutoff = torch.clamp(self.cutoff_ratio, min=1e-3)
        exp_val = (D_norm / cutoff) ** 2
        exp_val = torch.clamp(exp_val, max=100)  # Clamp exponent to avoid overflow in exp
        H_filter = torch.exp(-exp_val)
        H_filter = H_filter.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)
        
        # Extract low frequency component
        F_low = F_x * H_filter
        F_low = torch.fft.ifftshift(F_low)
        x_low = torch.fft.ifft2(F_low).real
        
        # High frequency component as residual
        x_high = x - x_low
        
        # Process each branch
        low_feat = self.low_processor(x_low)
        high_feat = self.high_processor(x_high)
        
        # Fuse features with CBAM attention
        fused = torch.concat((low_feat, high_feat), dim=1)
        fused = self.cbam(fused)
        
        # Final segmentation head to produce mask logits
        seg_logits = self.seg_head(fused)
        return seg_logits

#### MAIN ##########################################################################################################
class Model(pl.LightningModule):
    def __init__(self, args, channels=3, dims=(64, 256, 512, 1024), n_heads=(1, 2, 8, 8), expansion=(8, 8, 4, 4), reduction_ratio=(8, 4, 2, 1), n_layers=(2, 2, 2, 2), learning_rate = 1e-4):
        super(Model, self).__init__()
        self.args = args
        
        # HEAD
        ori_channels = 16
        
        self.head1 =  nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=ori_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(ori_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ori_channels, out_channels=ori_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(ori_channels),
            nn.ReLU(inplace=True)
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(in_channels=ori_channels, out_channels=3, kernel_size=1, stride=1, padding=0, padding_mode='reflect'),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # Body (Encoder)
        self.mix_transformer = MiT(channels, dims, n_heads, expansion, reduction_ratio, n_layers)
        self.cnn_encoder     = ResNetEncoder()
        
        # Body (Decoder): Progressive Growing Structure
        self.upsampling_4 = conv_upsample(2, dims[3]*2, dims[2]) 
        self.upsampling_3 = conv_upsample(2, dims[2]*3, dims[1])
        self.upsampling_2 = conv_upsample(2, dims[1]*3, dims[0])
        self.upsampling_1 = conv_upsample(2, dims[0]*3, ori_channels)
        
        self.CBAM4 = CBAM(dims[3]*2     , ratio=16)
        self.CBAM3 = CBAM(dims[2]*3     , ratio=16)
        self.CBAM2 = CBAM(dims[1]*3     , ratio=16)
        self.CBAM1 = CBAM(dims[0]*3     , ratio=8 )
        self.CBAM0 = CBAM(ori_channels*2, ratio=4 )
        
        # Tail
        self.fpcm_tail = FPCM(ori_channels*2, 0.25)

        # loss function
        self.loss_fn = DiceBCELoss()
        self.accuracy = BinaryAccuracy()
        self.f1_score = BinaryF1Score()
        self.recall = BinaryRecall()
        self.precision = BinaryPrecision()
        
        # Overlapped area metrics (Ignore Backgrounds)
        self.jaccard_ind = BinaryJaccardIndex()
        self.dice = Dice()

        # LR
        self.lr = args.lr

        # temp
        self.val_samples = []
        self.val_step_metrics = []
    

    def forward(self, x):
        # Head
        cnn_0 = self.head1(x)       # 16 channels 
        x       = self.head2(cnn_0) # 3  channels
        
        # Body - Encoder
        mit_1, mit_2, mit_3, mit_4 = self.mix_transformer(x)
        cnn_1, cnn_2, cnn_3, cnn_4 = self.cnn_encoder(x)

        # Body - Progressive Growing
        x = torch.cat((mit_4, cnn_4), dim=1)
        x = self.CBAM4(x)
        x = self.upsampling_4(x)
        
        x = torch.cat((mit_3, cnn_3, x), dim=1)
        x = self.CBAM3(x)
        x = self.upsampling_3(x)
        
        x = torch.cat((mit_2, cnn_2, x), dim=1)
        x = self.CBAM2(x)
        x = self.upsampling_2(x)
        
        x = torch.cat((mit_1, cnn_1, x), dim=1)
        x = self.CBAM1(x)
        x = self.upsampling_1(x)
        
        x = torch.cat((cnn_0, x), dim=1)
        x = self.CBAM0(x)
        
        # Head
        out = self.fpcm_tail(x)
        
        return out, (None)
    def criterion(self, pred, y):
        # BCE Loss
        loss_BCE = self.loss_fn(pred, y, weight=self.args.w_dice)

        # Edge Aware Loss
        loss_Edge = EdgeLoss(pred, y)

        # Center Line Loss with epsilon for numerical stability
        s_cl_dice = SoftclDice(pred, y)
        epsilon = 1e-6  # Small constant to avoid division by zero or sqrt(0)
        loss_CL = torch.sqrt(torch.clamp(1 - s_cl_dice, min=0) + epsilon)  

        # ct_dice Loss with epsilon for numerical stability
        s_ct_dice = Soft_ct_dice(pred, y)
        loss_ct_dice = torch.sqrt(torch.clamp(1 - s_ct_dice, min=0) + epsilon)

        # Total Loss
        loss = (loss_BCE * self.args.w_bce +
                loss_Edge * self.args.w_edge +
                loss_CL * self.args.w_cl +
                loss_ct_dice * self.args.w_ct_dice)

        return loss, loss_BCE, loss_Edge, s_cl_dice, s_ct_dice

    def _common_step(self, batch, batch_idx):  ##################################################################
        x, y = batch
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        y = (y > 0).float().unsqueeze(1) 

        # Pred
        pred_lst = self.forward(x)
        mask = pred_lst[0]

        # Loss
        LOSS = self.criterion(mask, y)

        # Activate Mask
        pred = torch.sigmoid(mask)
        pred = (pred > 0.5).float()
        
        return LOSS, pred, y, (y, x, pred, mask, pred_lst[1]), mask

    def training_step(self, batch, batch_idx): #################################
        x, y = batch
        LOSS, pred, y, _ , mask = self._common_step(batch, batch_idx)

        loss, loss_BCE, loss_Edge, s_cl_dice, s_ct_dice = LOSS

        # Log
        accuracy = self.accuracy(pred, y)
        f1_score = self.f1_score(pred, y)
        re = self.recall(pred, y)
        precision = self.precision(pred, y)
        jaccard = self.jaccard_ind(pred, y)
        y = y.clone().detach().to(torch.int32)
        dice = self.dice(pred, y)
        ct_d = ct_dice(pred, y)

        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score, 
                      'train_precision': precision,  'train_recall': re, 'train_IOU': jaccard, 'train_dice': dice, 'train_ct_dice':ct_d,
                      'train_soft_CL_Dice':s_cl_dice, 'train_BCE':loss_BCE, 'train_Edge':loss_Edge, 'train_soft_ct_dice':s_ct_dice},
                      on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        LOSS, pred, y, datas, mask = self._common_step(batch, batch_idx)

        loss, loss_BCE, loss_Edge, s_cl_dice, s_ct_dice = LOSS

        # Log
        accuracy = self.accuracy(pred, y)
        f1_score = self.f1_score(pred, y)
        re = self.recall(pred, y)
        precision = self.precision(pred, y)
        jaccard = self.jaccard_ind(pred, y)
        y = y.clone().detach().to(torch.int32)
        dice = self.dice(pred, y)
        ct_d = ct_dice(pred, y)

        cl_IoU_   = clIoU(pred, y)
        cl_Dice_  = clDice(pred, y)

        self.log_dict({
            'val_loss': loss, 
            'val_accuracy': accuracy, 
            'val_f1_score': f1_score, 
            'val_precision': precision,  
            'val_recall': re, 
            'val_IOU': jaccard, 
            'val_dice': dice,
            'val_ct_dice':ct_d,
            'val_CL_IoU':cl_IoU_, 
            'val_CL_Dice':cl_Dice_, 
            'val_soft_CL_Dice':s_cl_dice,
            'val_BCE':loss_BCE,
            'val_Edge':loss_Edge,
            'val_soft_ct_dice':s_ct_dice
            },
            on_step=False, on_epoch=True, prog_bar=True)

        self.val_step_metrics.append({ #hist
            'val_loss': loss.item(),
            'val_accuracy': accuracy.item(),
            'val_f1_score': f1_score.item(),
            'val_precision': precision.item(),
            'val_recall': re.item(),
            'val_IOU': jaccard.item(),
            'val_dice': dice.item(),
            'val_ct_dice':ct_d.item(),
            'val_CL_IoU':cl_IoU_.item(), 
            'val_CL_Dice':cl_Dice_.item(), 
            'val_soft_CL_Dice':s_cl_dice.item(),
            'val_BCE':loss_BCE.item(),
            'val_Edge':loss_Edge.item(),
            'val_soft_ct_dice':s_ct_dice.item()
        })
    
        self.val_samples.append(datas) #plot

    def on_validation_epoch_end(self):
        self.plot_val_comparison(self.val_samples)
        
        current_epoch = self.current_epoch
        if current_epoch != 0:
            if self.val_step_metrics:
                avg_metrics = {}
                keys = self.val_step_metrics[0].keys()
                for key in keys:
                    values = [m[key] for m in self.val_step_metrics]
                    avg_metrics[key] = sum(values) / len(values)
            else:
                avg_metrics = {}
            self.update_and_plot_history(avg_metrics, self.args.ckpt, self.logger.version, current_epoch, data_name="val", plot_line=True)
        
        self.val_step_metrics = []
        self.val_samples = []
    
    def test_step(self, batch, batch_idx):
        x,y,_ = batch
        batch = (x, y)
        LOSS, pred, y, datas, _ = self._common_step(batch, batch_idx)
        loss, loss_BCE, loss_Edge, s_cl_dice, s_ct_dice = LOSS

        accuracy = self.accuracy(pred, y)
        f1_score = self.f1_score(pred, y)
        re = self.recall(pred, y)
        precision = self.precision(pred, y)
        jaccard = self.jaccard_ind(pred, y)
        y = y.clone().detach().to(torch.int32)
        dice = self.dice(pred, y)
        ct_d = ct_dice(pred,y)
        cl_IoU_   = clIoU(pred, y)
        cl_Dice_  = clDice(pred, y)
        self.log_dict({'test_loss': loss,
                       'test_accuracy': accuracy, 
                       'test_f1_score': f1_score, 
                       'test_precision': precision,  
                       'test_recall': re, 
                       'test_IOU': jaccard, 
                       'test_dice': dice,
                       'test_ct_dice': ct_d,
                       'test_CL_IoU':cl_IoU_, 
                       'test_CL_Dice':cl_Dice_, 
                       'test_soft_CL_Dice':s_cl_dice,
                       'test_BCE':loss_BCE,
                       'test_Edge':loss_Edge.item(),
                       'test_soft_ct_dice':s_ct_dice

                       },
                      on_step=False, on_epoch=True, prog_bar=False) 
        self.val_step_metrics.append({ #hist
            'test_loss': loss.item(),
            'test_accuracy': accuracy.item(),
            'test_f1_score': f1_score.item(),
            'test_precision': precision.item(),
            'test_recall': re.item(),
            'test_IOU': jaccard.item(),
            'test_dice': dice.item(),
            'test_ct_dice': ct_d.item(),
            'test_CL_IoU':cl_IoU_.item(), 
            'test_CL_Dice':cl_Dice_.item(), 
            'test_soft_CL_Dice':s_cl_dice.item(),
            'test_BCE':loss_BCE,
            'test_Edge':loss_Edge.item(),
            'test_soft_ct_dice':s_ct_dice.item()
        })
        self.val_samples.append(datas) #plot
        
        return loss      

    def on_test_epoch_end(self):
        self.plot_val_comparison(self.val_samples)
        current_epoch = self.current_epoch
        if self.val_step_metrics:
            avg_metrics = {}
            keys = self.val_step_metrics[0].keys()
            for key in keys:
                values = [m[key] for m in self.val_step_metrics]
                avg_metrics[key] = sum(values) / len(values)
        else:
            avg_metrics = {}
        
        self.update_and_plot_history(avg_metrics, self.args.ckpt, self.logger.version, current_epoch, data_name="test", plot_line=False)
        self.val_step_metrics = []
        self.val_samples = []
    
    def predict_step(self, batch, batch_idx):###################################
        x, y = batch
        x = x.to(config.DEVICE)
        y = y.float().unsqueeze(1).to(config.DEVICE)
        pred = self.forward(x)
        preds = torch.sigmoid(pred)
        preds = (preds > 0.5).float()
        return preds
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_schedule = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_schedule,
                "monitor": "val_loss",
                "frequency": 1,
            },
            "gradient_clip_val": 1.0,     
            "gradient_clip_algorithm": "value"
        }

    # Validate #################################################################
    def plot_val_comparison(self, samples, output_dir=None):
        print("\n\n ")

        if output_dir==None:
            output_dir = f"outputs/{self.args.ckpt}/v{self.logger.version}"
        os.makedirs(output_dir, exist_ok=True)
        idx = 0

        def denormalize(tensor):
            mean = torch.tensor([0.51789941, 0.51360926, 0.547762], device=DEVICE).view(1, 3, 1, 1)
            std = torch.tensor([0.1812099,  0.17746663, 0.20386334], device=DEVICE).view(1, 3, 1, 1)
            return tensor * std + mean

        for y, x, p, m_out, logits in samples:
            l_count = 0

            batch_size = p.shape[0]
            for b in range(batch_size):
                if idx >= 20:
                  print("Image Saved")
                  return

                device = x[b].device
                cmap = "hot"

                c = 4
                r = (l_count-1)//4 + 2
                fig, axes = plt.subplots(r, c, figsize=(4*c, 4*r))
                axes = axes.flatten() 

                for i in range(l_count+4):
                  axes[i].axis("off")

                # 0. Input (X)
                x_denorm = denormalize(x[b])
                if x_denorm.dim() == 4:
                    x_denorm = x_denorm.squeeze(0)
                x_denorm = x_denorm.permute(1, 2, 0).cpu().numpy()
                x_denorm = x_denorm.clip(0, 1)
                axes[0].imshow(x_denorm)
                axes[0].set_title("Input")

                # 1. Ground Truth (Y) + Skeleton (red)
                gt_mask = y[b].squeeze().cpu().numpy()  
                gt_rgb = np.stack([gt_mask, gt_mask, gt_mask], axis=-1)
                gt_skel = skeletonize(gt_mask.astype(bool))
                gt_rgb[gt_skel] = [1, 0, 0]
                axes[1].imshow(gt_rgb)
                axes[1].set_title("Ground Truth + Skeleton")

                # 2. Predict (P) + Skeleton (Hard: Red, Soft: Green)
                pred_mask = p[b].squeeze().cpu().numpy()  
                pred_rgb = np.stack([pred_mask, pred_mask, pred_mask], axis=-1)
                # Hard Skel
                pred_skel = skeletonize(pred_mask.astype(bool))
                pred_rgb[pred_skel] = [1, 0, 0]
                # Soft skel
                pred_tensor = torch.sigmoid(p[b].unsqueeze(0))  # shape: [1, H, W]
                soft_skel = soft_skeleton(pred_tensor)
                soft_skel = soft_skel.squeeze().cpu().numpy()
                soft_skel_bin = soft_skel > 0.5
                pred_rgb[soft_skel_bin] = [0, 1, 0]
                axes[2].imshow(pred_rgb)
                axes[2].set_title("Prediction + Skeleton (Hard: Red, Soft: Green)")

                # 3. Final Out
                axes[3].imshow(m_out[b].squeeze().cpu().detach().numpy(), cmap=cmap, vmin=0, vmax=1)
                axes[3].set_title("Prediction (map)")

                plt.tight_layout()
                save_path = os.path.join(output_dir, f"val_{idx}.png")
                plt.savefig(save_path)
                plt.close(fig)
                idx += 1
        print("Image Saved")
    
    def update_and_plot_history(self, current_metrics, config_name, logger_version, current_epoch, data_name="val", plot_line=True):
        output_dir = f"outputs/{config_name}/v{logger_version}/plt"
        os.makedirs(output_dir, exist_ok=True)
        
        history_path = os.path.join(output_dir, f"{data_name}_history_{config_name}_v{logger_version}.json")
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        else:
            history = {"epochs": []}
        
        history.setdefault("epochs", []).append(current_epoch)
        
        for key, val in current_metrics.items():
            if hasattr(val, "item"):
                history.setdefault(key, []).append(val.item())
            else:
                history.setdefault(key, []).append(val)
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        plt.figure(figsize=(10, 6))
        keys = list(current_metrics.keys())
        values = [val.item() if hasattr(val, "item") else val for val in current_metrics.values()]
        bars = plt.bar(keys, values, color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"{data_name.capitalize()} Metrics ({config_name} v{logger_version}) - Epoch {current_epoch}")
        plt.tight_layout()
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        bar_path = os.path.join(output_dir, f"{data_name}_{config_name}_v{logger_version}.png")
        plt.savefig(bar_path)
        plt.close()
        
        if plot_line:
            epochs = history.get("epochs", [])
            if not epochs:
                print("Error! No history data")
                return
            
            for key, values in history.items():
                if key == "epochs":
                    continue
                plt.figure(figsize=(8, 5))
                plt.plot(epochs, values, marker='o', linestyle='-', color='b')
                plt.xlabel("Epoch")
                plt.ylabel(key)
                plt.title(f"{data_name.capitalize()} {key} Evolution ({config_name} v{logger_version})")
                plt.grid(True)
                plt.tight_layout()
                
                line_path = os.path.join(output_dir, f"line_{key}_{config_name}_v{logger_version}.png")
                plt.savefig(line_path)
                plt.close()

