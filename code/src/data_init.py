import os
import importlib

def CrackVision12K(args, mode, path):
  folderName = "CrackVision12K"

  path = os.path.join(path, folderName)

  if mode == "test":
      if args.testmode == 0:
        mode = "test"
      else:
        mode = "split"

  path = os.path.join(path, mode)

  IMG_DIR = os.path.join(path, "IMG")
  GT_DIR  = os.path.join(path, "GT")

  m = "null"

  if mode =="split":
      if args.testmode == 1:
        m = "0_2"
      elif args.testmode == 2:
        m = "2_4"
      elif args.testmode == 3:
        m = "4_8"
      elif args.testmode == 4:
        m = "8_16"
      elif args.testmode == 5:
        m = "16_32"
      elif args.testmode == 6:
        m = "thick"
      elif args.testmode == 7:
        m = "zero"

      IMG_DIR = os.path.join(IMG_DIR, m)
      GT_DIR  = os.path.join(GT_DIR , m)

  return IMG_DIR, GT_DIR

def CrackTree260(args, mode, path):
  folderName = "CrackTree260"
  path = os.path.join(path, folderName)

  IMG_DIR = os.path.join(path, "IMG")
  GT_DIR  = os.path.join(path, "GT")

  return IMG_DIR, GT_DIR

def OmniCrack30K(args, mode, path):
  folderName = "OmniCrack30K"
  path = os.path.join(path, folderName)

  if mode == "test":
      if args.testmode == 0:
        mode = "val"
        IMG_DIR = os.path.join(path, "IMG", mode)
        GT_DIR  = os.path.join(path, "GT" , mode)
        print(GT_DIR)
        return IMG_DIR, GT_DIR
      else:
        mode = "split"

  m = "null"
  if mode =="split":
      if args.testmode == 1:
        m = "0_2"
      elif args.testmode == 2:
        m = "2_4"
      elif args.testmode == 3:
        m = "4_8"
      elif args.testmode == 4:
        m = "8_16"
      elif args.testmode == 5:
        m = "16_32"
      elif args.testmode == 6:
        m = "32+"
      elif args.testmode == 7:
        m = "0"

  IMG_DIR = os.path.join(path, "IMG", mode)
  GT_DIR  = os.path.join(path, "GT" , mode)

  if m != "null":
    IMG_DIR = os.path.join(IMG_DIR, m)
    GT_DIR  = os.path.join(GT_DIR , m)

  return IMG_DIR, GT_DIR

def ADE20K(args, mode, path):
  if mode == "train":
    folderName = f"ade20k/{mode}"
  else:
    folderName = f"ade20k/val"
  path = os.path.join(path, folderName)

  IMG_DIR = os.path.join(path, "IMG")
  GT_DIR  = os.path.join(path, "GT")

  return IMG_DIR, GT_DIR


def getDataPath(args):
  path = args.data_path

  Train_IMG_DIR, Train_MASK_DIR = globals()[args.train_set](args, "train", path)
  Test_IMG_DIR , Test_MASK_DIR  = globals()[args.test_set](args, "test", path)
  Val_IMG_DIR  , Val_MASK_DIR   = globals()[args.val_set](args, "val", path)

  return Train_IMG_DIR, Train_MASK_DIR, Test_IMG_DIR , Test_MASK_DIR, Val_IMG_DIR  , Val_MASK_DIR
