import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, test=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.test = test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_filename = self.images[index]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, img_filename)


        if not os.path.exists(mask_path):
            base, ext = os.path.splitext(img_filename)
            alternative_filename = base + '.png'
            mask_path = os.path.join(self.mask_dir, alternative_filename)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found for {img_filename}")


        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        mask = (mask > 0.5).astype(np.float32)
        mask[mask == 255.0] = 1.0


        h, w = image.shape[:2]

        new_h = (h // 32) * 32
        new_w = (w // 32) * 32


        if new_h != h or new_w != w:
            top = (h - new_h) // 2
            left = (w - new_w) // 2

            image = image[top: top + new_h, left: left + new_w]
            mask  = mask[top: top + new_h, left: left + new_w]


        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask  = augmentations["mask"]

        if self.test:
            return image, mask, img_filename
        else:
            return image, mask
