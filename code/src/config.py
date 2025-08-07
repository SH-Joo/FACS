import torch
import os 

### train on total dataset
# NUM_EPOCHS = 1000
DATASET_SIZE = {'train' : 9600, 'val' : 1200, 'test' : 1200}

mode = "train"

layer = 5

# Hyperparameters etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

ckptPath = '/content/drive/MyDrive/Edge_Crack/save/checkpoints'