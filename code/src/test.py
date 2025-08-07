import torch
import os
import cv2
import numpy as np
import torch.nn.functional as F
from skimage.morphology import disk, thin  # for dilation kernel & thinning
from dataloader import get_loaders
import config
from data_init import getDataPath

# ----------------------------------------------------------------------------
# Utility: Tile-based Inference for Large Images
# ----------------------------------------------------------------------------
def tile_predict(x, model, tile_size=256):
    b, c, h, w = x.shape
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    # replicate padding to avoid reflect constraints
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    _, _, h_p, w_p = x_padded.shape

    output = torch.zeros((b, 1, h_p, w_p), device=x.device)
    count = torch.zeros_like(output)

    for i in range(0, h_p, tile_size):
        for j in range(0, w_p, tile_size):
            tile = x_padded[:, :, i:i+tile_size, j:j+tile_size]
            with torch.no_grad():
                pred = model(tile)
                if isinstance(pred, (tuple, list)):
                    pred = pred[0]
                prob = torch.sigmoid(pred).squeeze(0)                
            output[:, :, i:i+tile_size, j:j+tile_size] += prob
            count[:, :, i:i+tile_size, j:j+tile_size] += 1

    output = output / count
    return output[:, :, :h, :w]

# ----------------------------------------------------------------------------
# Helper: compute centerline IoU with tolerance (as in benchmark code)
# ----------------------------------------------------------------------------
def compute_cliou_per_pixel(true_mask, pred_mask, tol):
    """
    Compute centerline IoU with given tolerance:
    - true_mask, pred_mask: 2D numpy arrays with 0/1 values
    - tol: integer radius for disk structuring element
    """
    # 1-pixel skeletonization
    true_skel = thin(true_mask.astype(bool)).astype(np.uint8)
    pred_skel = thin(pred_mask.astype(bool)).astype(np.uint8)

    # dilation by disk(tol)
    kernel = disk(tol).astype(np.uint8)
    true_dil = cv2.dilate(true_skel, kernel, iterations=1)
    pred_dil = cv2.dilate(pred_skel, kernel, iterations=1)

    # true positives, false positives, false negatives
    tp = true_skel * pred_dil
    fp = pred_skel - (pred_skel * true_dil)
    fn = true_skel - tp

    # new masks after tolerance
    true_new = tp + fn
    pred_new = tp + fp

    # flatten and mask out background-only pixels
    ft = true_new.flatten()
    fp_flat = pred_new.flatten()
    mask = (ft == 1) | (fp_flat == 1)
    if not mask.any():
        return 1.0

    # manual IoU
    intersection = np.logical_and(ft[mask] == 1, fp_flat[mask] == 1).sum()
    union        = np.logical_or (ft[mask] == 1, fp_flat[mask] == 1).sum()
    if union == 0:
        return 1.0
    return intersection / union

# ----------------------------------------------------------------------------
# Evaluation Utility with Cropping and Metrics
# ----------------------------------------------------------------------------
def crop_eval_metrics(args, loader, model, device="cuda", tile_size=256):
    model.eval()
    eps = 1e-7
    threshold = 0.5

    # Global pixel metrics
    TP_tot = FP_tot = TN_tot = FN_tot = 0
    iou_list, dice_list = [], []

    # Centerline metrics per dilation
    px_list = [0, 2, 4, 8, 16, 32, 64]
    cliou_lists = {px: [] for px in px_list}

    # create dynamic save directory based on model name and test set name
    save_dir = os.path.join(args.model, args.test_set)
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for x, y, names in loader:
            # get base filename
            file_base = names[0] if isinstance(names, (list, tuple)) else 'result'
            file_base = os.path.splitext(os.path.basename(file_base))[0]

            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            # inference
            prob_map = tile_predict(x, model, tile_size)
            pred_bin = (prob_map > threshold).float()

            # save stitched images
            pred_img = (pred_bin[0,0].cpu().numpy() * 255).astype(np.uint8)
            gt_img   = (y[0,0]      .cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, f'{file_base}.png'),    pred_img)
            # cv2.imwrite(os.path.join(save_dir, f'{file_base}_gt.png'), gt_img)

            # compute metrics per image
            for b in range(y.shape[0]):
                yt = y[b, 0]
                pt = pred_bin[b, 0]

                # pixel-wise metrics
                TP = int(((pt == 1) & (yt == 1)).sum().item())
                FP = int(((pt == 1) & (yt == 0)).sum().item())
                TN = int(((pt == 0) & (yt == 0)).sum().item())
                FN = int(((pt == 0) & (yt == 1)).sum().item())
                TP_tot += TP; FP_tot += FP
                TN_tot += TN; FN_tot += FN

                gt_sum   = int(yt.sum().item())
                pred_sum = int(pt.sum().item())
                if gt_sum == 0:
                    img_iou  = 1.0 if pred_sum == 0 else 0.0
                    img_dice = 1.0 if pred_sum == 0 else 0.0
                else:
                    img_iou  = TP / (TP + FP + FN + eps)
                    img_dice = 2 * TP / (2 * TP + FP + FN + eps)
                iou_list.append(img_iou)
                dice_list.append(img_dice)

                # centerline IoU computations
                true_np = yt.cpu().numpy().astype(np.uint8)
                pred_np = pt.cpu().numpy().astype(np.uint8)
                for px in px_list:
                    img_cliou = compute_cliou_per_pixel(true_np, pred_np, px)
                    cliou_lists[px].append(img_cliou)

    # compute global metrics
    accuracy  = (TP_tot + TN_tot) / (TP_tot + FP_tot + TN_tot + FN_tot + eps)
    precision = TP_tot / (TP_tot + FP_tot + eps)
    recall    = TP_tot / (TP_tot + FN_tot + eps)
    mean_iou  = float(np.mean(iou_list))
    mean_dice = float(np.mean(dice_list))

    print(f'Global Accuracy : {accuracy:.4f} | Precision : {precision:.4f} | Recall : {recall:.4f}')
    print(f'Mean IoU  : {mean_iou:.4f} | Mean Dice : {mean_dice:.4f}')

    # print clIoU averages
    for px in px_list:
        avg_cliou = np.mean(cliou_lists[px])
        print(f'clIoU ({px}px): {avg_cliou:.4f}')

    return (accuracy, precision, recall, mean_iou, mean_dice), prob_map

# ----------------------------------------------------------------------------
# Test Routine (preserved format)
# ----------------------------------------------------------------------------
def test(args, model):
    TRAIN_IMG_DIR, TRAIN_MASK_DIR, TEST_IMG_DIR, TEST_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR = getDataPath(args)
    _, _, test_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        VAL_IMG_DIR, VAL_MASK_DIR,
        TEST_IMG_DIR, TEST_MASK_DIR,
        batch_size=1,
        num_workers=args.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # load model checkpoint
    ck_file = f'{args.ckpt_path}/{args.ckpt}.{args.save_type}'
    checkpoint = torch.load(ck_file, map_location=config.DEVICE, weights_only=False)
    if args.save_type == 'ckpt' and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        key = 'net' if args.model == 'benchmarks.DECS-Net.DECS-Net' else 'state_dict'
        model.load_state_dict(checkpoint.get(key, checkpoint))
    model.to(config.DEVICE).eval()

    # evaluate with tile-based crop and metrics
    print("\nComputing Metrics on Test Set (tile-based)")
    metrics, _ = crop_eval_metrics(args, test_loader, model,
                                   device=config.DEVICE, tile_size=256)
    print('-----------------------------')

    # optionally save additional visualizations
    if hasattr(args, 'save') and args.save:
        print('Also saving visualizations via save_predictions_as_imgs...')
        from utils import save_predictions_as_imgs
        save_predictions_as_imgs(test_loader, model,
                                 folder=args.ckpt,
                                 device=config.DEVICE,
                                 multiple_outputs=True)
        print('Saved additional images.')
