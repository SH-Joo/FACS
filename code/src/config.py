import torch
import os


DATASET_SIZE = {'train' : 9600, 'val' : 1200, 'test' : 1200}

mode = "train"

layer = 5


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

ckptPath = '/content/drive/MyDrive/Edge_Crack/save/checkpoints'
