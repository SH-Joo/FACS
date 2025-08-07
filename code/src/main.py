import argparse
import torch
import config

from trainer import train
# from test_full import test
from test import test
import importlib

# Parser
parser = argparse.ArgumentParser(description='Crack Segmentation')

# Model Settings
parser.add_argument('--model', type=str, default="model_fft", help='Model Name')
parser.add_argument('--ckpt', type=str, default="default", help='ckpt name')
parser.add_argument('--mode', type=str, default="train", help='mode: train or Test')

# Train Settings
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
parser.add_argument('--BATCH_SIZE', type=int, default=12, help='Batch Size')
parser.add_argument('--NUM_WORKERS', type=int, default=2, help='NUMBER of WORKERS')
parser.add_argument('--NUM_EPOCHS', type=int, default=1000, help='MAX EPOCHS')
parser.add_argument('--patience', type=int, default=10, help='Early Stopping patience')

df = "default"
parser.add_argument('--pre_trained', type=str, default=df, help='Pretrained Model Path')

# Loss weights (works only for Our Model)
parser.add_argument('--w_bce', type=float, default=0.5, help='weights of Binary Cross Entropy')
parser.add_argument('--w_cl', type=float, default=0.0, help='weights of Soft-CL-Dice Loss')
parser.add_argument('--w_edge', type=float, default=0.25, help='weights of Edge-Aware Loss')
parser.add_argument('--w_ct_dice', type=float, default=0.25, help='weights of soft Crack Topology Score')
parser.add_argument('--w_dice', type=float, default=0.0, help='weights of Dice Loss')

# Data Settings
parser.add_argument('--data_path', type=str, default="../datas", help='Data Path')
parser.add_argument('--train_set', type=str, default="CrackVision12K", help='Train Dataset (Zip Name)')
parser.add_argument('--test_set' , type=str, default="CrackVision12K", help='Test Dataset (Zip Name)')
parser.add_argument('--val_set'  , type=str, default="CrackVision12K", help='Validation Dataset (Zip Name)')
parser.add_argument('--classes', type=int, default=1, help='num classes')

# Test Settings 
parser.add_argument('--ckpt_path', type=str, default="../ckpts", help='Path to CheckPoints')
parser.add_argument('--save', action='store_true', help='Save test Images')
parser.add_argument('--testmode', type=int, default=0, help='0:EverySet, 1:0~2, 2:2~4, 3:4~8, 4:8~16, 5:16~32, 6: 32+, 7: 0')
parser.add_argument('--save_type', type=str, default="ckpt", help='save type')
parser.add_argument('--tc', action='store_true', help='Deleted')

parser.add_argument('--load_benchmark', action='store_true', help='')

args = parser.parse_args()

# Model
args.device = "cuda" if torch.cuda.is_available() else "cpu"
model = importlib.import_module(f"models.{args.model}").Model(args).to(args.device)

# Load
ck_file_path = f'{args.ckpt_path}/{args.pre_trained}.{args.save_type}'

if args.pre_trained != df:
    checkpoint = torch.load(ck_file_path)
    model.load_state_dict(checkpoint['state_dict'])

if args.mode == "train":
  train(args, model)
elif args.mode == "test":
  test(args, model)