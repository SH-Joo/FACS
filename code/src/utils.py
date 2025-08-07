import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

result_save_ind = 0
threshold = 0.5

def eval_metrics(args, loader, model, device="cuda", multiple_outputs=False):
    model.eval()
    eps = 1e-7
    threshold = 0.5

    # 전역 집계 지표
    TP_tot = FP_tot = TN_tot = FN_tot = 0

    # 이미지별 IoU/Dice 저장
    iou_list, dice_list = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)      # (B,1,H,W)
            y_bin = (y > threshold).float()    # GT 이진화

            out = model(x)
            if multiple_outputs:
                out = out[result_save_ind]
            prob = torch.sigmoid(out)          # (B,1,H,W)
            pred_bin = (prob > threshold).float()               

            B = y_bin.shape[0]
            # 배치 내 각 이미지별로
            for b in range(B):
                yt = y_bin[b,0]    # (H,W)
                pt = pred_bin[b,0]
                
                if yt.shape != pt.shape:
                    print(f"yt: {yt.shape} / pt: {pt.shape}")

                # 픽셀 단위 TP/FP/TN/FN
                TP = int(((pt == 1) & (yt == 1)).sum().item())
                FP = int(((pt == 1) & (yt == 0)).sum().item())
                TN = int(((pt == 0) & (yt == 0)).sum().item())
                FN = int(((pt == 0) & (yt == 1)).sum().item())

                TP_tot += TP;  FP_tot += FP
                TN_tot += TN;  FN_tot += FN

                # 이미지별 IoU/Dice
                gt_sum = int(yt.sum().item())
                pred_sum = int(pt.sum().item())

                if gt_sum == 0:
                    # GT에 크랙이 없을 때
                    if pred_sum == 0:
                        img_iou  = 1.0
                        img_dice = 1.0
                    else:
                        img_iou  = 0.0
                        img_dice = 0.0
                else:
                    img_iou  = TP / (TP + FP + FN + eps)
                    img_dice = 2 * TP / (2*TP + FP + FN + eps)

                iou_list .append(img_iou)
                dice_list.append(img_dice)

    # 전역 지표
    accuracy  = (TP_tot + TN_tot) / (TP_tot + FP_tot + TN_tot + FN_tot + eps)
    precision = TP_tot / (TP_tot + FP_tot + eps)
    recall    = TP_tot / (TP_tot + FN_tot + eps)

    # 이미지별 평균 IoU/Dice
    mean_iou  = float(np.mean(iou_list))
    mean_dice = float(np.mean(dice_list))

    print(f'Global Accuracy : {accuracy:.4f} | Precision : {precision:.4f} | Recall : {recall:.4f}')
    print(f'Mean IoU  : {mean_iou:.4f} | Mean Dice : {mean_dice:.4f}')

    return accuracy, precision, recall, mean_iou, mean_dice

def eval_OIS(loader, model, device="cuda", multiple_outputs=False):
    best_OIS_lst = []
    best_thres_lst = []
    thres_list = [i for i in np.arange(0, 1, step=0.01)]
    eps = 1e-7

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            best_thres = 0
            best_OIS = 0
            for thres in thres_list:
                if multiple_outputs == True:
                    final_output = model(x)[result_save_ind]
                    preds_probability = torch.sigmoid(final_output)
                    preds = (preds_probability > threshold).float()
                else:
                    preds_probability = torch.sigmoid(model(x))
                    preds = (preds_probability > threshold).float()


                confusion_matirx = preds / y

                TP = torch.sum(confusion_matirx == 1).item()
                FP = torch.sum(confusion_matirx == float('inf')).item()
                TN = torch.sum(torch.isnan(confusion_matirx)).item()
                FN = torch.sum(confusion_matirx == 0).item()

                precision = (TP) / (TP+FP+eps)
                recall = (TP) / (TP+FN+eps) # TP rate
                f1_score = 2* (precision*recall)/(precision+recall+eps)

                if f1_score > best_OIS:
                    best_OIS = f1_score
                    best_thres = thres

            best_thres_lst.append(best_thres)
            best_OIS_lst.append(best_OIS)

    mean_OIS = np.mean(best_OIS_lst)
    mean_thres = np.mean(best_thres_lst)

    print(f'OIS F1 Score : {mean_OIS} / with the mean threshod : {mean_thres}')

    return mean_OIS, mean_thres

def eval_ODS(loader, model, device="cuda", multiple_outputs=False):
    model.eval()

    best_ODS = 0
    best_thres = 0
    thres_list = [i for i in np.arange(0, 1, step=0.01)]
    eps = 1e-7


    with torch.no_grad():
        for thres in thres_list:
            TP_total = 0
            FP_total = 0
            TN_total = 0
            FN_total = 0
            for x, y in loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)

                if multiple_outputs == True:
                    final_output = model(x)[result_save_ind]
                    preds_probability = torch.sigmoid(final_output)
                    preds = (preds_probability > threshold).float()
                else:
                    preds_probability = torch.sigmoid(model(x))
                    preds = (preds_probability > threshold).float()

                confusion_matirx = preds / y

                TP =  torch.sum(confusion_matirx == 1).item()
                FP = torch.sum(confusion_matirx == float('inf')).item()
                TN = torch.sum(torch.isnan(confusion_matirx)).item()
                FN = torch.sum(confusion_matirx == 0).item()

                TP_total += TP
                FP_total += FP
                TN_total += TN
                FN_total += FN

            precision = (TP_total) / (TP_total+FP_total+eps)
            recall = (TP_total) / (TP_total+FN_total+eps) # TP rate
            f1_score = 2* (precision*recall)/(precision+recall+eps)
            if f1_score > best_ODS:
                best_ODS = f1_score
                best_thres = thres

    print(f'ODS F1 Score : {best_ODS} / with the threshod : {best_thres}')

    return best_ODS, best_thres

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda", multiple_outputs=False):
    import matplotlib.pyplot as plt
    import torch
    import os

    os.makedirs(folder, exist_ok=True)

    mean = torch.tensor([0.51789941, 0.51360926, 0.547762], device=device).view(3, 1, 1)
    std = torch.tensor([0.1812099,  0.17746663, 0.20386334], device=device).view(3, 1, 1)

    model.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            if multiple_outputs:
                final_output = model(x)[result_save_ind]
                preds_probability = torch.sigmoid(final_output)
                preds = (preds_probability > threshold).float()
            else:
                output = model(x)
                preds_probability = torch.sigmoid(output)
                preds = (preds_probability > threshold).float()

            input_img = x[0].detach().cpu()
            input_img = input_img * std.cpu() + mean.cpu()
            input_img = torch.clamp(input_img, 0, 1)
            input_img_np = input_img.permute(1, 2, 0).numpy()

            gt = y[0].detach().cpu()
            if gt.ndim == 3 and gt.shape[0] == 1:
                gt = gt.squeeze(0)
            elif gt.ndim == 4:
                gt = gt.squeeze(0).squeeze(0)
            else:
                gt = gt.squeeze()
            gt_np = gt.numpy()

            pred_mask = preds[0].detach().cpu()
            if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
                pred_mask = pred_mask.squeeze(0)
            elif pred_mask.ndim == 4:
                pred_mask = pred_mask.squeeze(0).squeeze(0)
            else:
                pred_mask = pred_mask.squeeze()
            pred_mask_np = pred_mask.numpy()

            prob_map = preds_probability[0].detach().cpu()
            if prob_map.ndim == 3 and prob_map.shape[0] == 1:
                prob_map = prob_map.squeeze(0)
            elif prob_map.ndim == 4:
                prob_map = prob_map.squeeze(0).squeeze(0)
            else:
                prob_map = prob_map.squeeze()
            prob_map_np = prob_map.numpy()

            fig, axs = plt.subplots(1, 4, figsize=(16, 4))

            axs[0].imshow(input_img_np)
            axs[0].set_title("Input Image")
            axs[0].axis('off')

            axs[1].imshow(gt_np, cmap='gray')
            axs[1].set_title("Ground Truth")
            axs[1].axis('off')

            axs[2].imshow(pred_mask_np, cmap='gray')
            axs[2].set_title("Predicted Mask")
            axs[2].axis('off')

            axs[3].imshow(prob_map_np, cmap='hot')
            axs[3].set_title("Probability Map")
            axs[3].axis('off')

            plt.tight_layout()
            plt.savefig(f"{folder}/pred_{idx}.png")
            plt.close(fig)

    model.train()

def loss_plot(train_loss, val_loss):
    if len(train_loss) != len(val_loss):
        print('The number of losses are different')
    else:
        labels = [i for i in range(1, len(train_loss)+1)]
        plt.plot(train_loss)
        plt.plot(val_loss)
        # plt.xticks(range(0, len(train_loss), 10), labels[::9])
        ticks = [i for i in range(0, len(train_loss), 10)]  # Ticks at every 10th index
        tick_labels = [labels[i-1] for i in ticks]  # Corresponding labels for the ticks
        plt.xticks(ticks, tick_labels)
        plt.gca().get_xticklabels()[0].set_visible(False)
        plt.xlabel('Epoch', fontsize=17)
        plt.ylabel('Loss', fontsize=17)
        plt.show()
        plt.savefig('loss_output.png')