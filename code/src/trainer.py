from dataloader import get_loaders
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, Callback, ModelCheckpoint
from callback import MyPrintingCallBack, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchsummary import summary
import config

from data_init import getDataPath

import os

torch.set_float32_matmul_precision("medium")

def train(args, model):
    modelName = args.model
    logger = TensorBoardLogger("outputs/tb_logs", name=args.ckpt)
    print(f"\n\n<{modelName} - v{logger.version}>\n\n")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), 'outputs/checkpoints', f'{args.ckpt}_v{logger.version}'),
        filename=f'{args.ckpt}'+"-epoch{epoch:02d}-val_loss{val_loss:.4f}",
        verbose=True,
        save_last=True,
        save_top_k=2,
        monitor='val_loss',
        mode='min'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        verbose=True,
        mode='min'
    )

    TRAIN_IMG_DIR, TRAIN_MASK_DIR, TEST_IMG_DIR, TEST_MASK_DIR, VAL_IMG_DIR , VAL_MASK_DIR = getDataPath(args)

    train_loader, val_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        VAL_IMG_DIR, VAL_MASK_DIR,
        TEST_IMG_DIR, TEST_MASK_DIR,
        args.BATCH_SIZE, args.NUM_WORKERS, config.PIN_MEMORY,
   )


    trainer = pl.Trainer(
        logger=logger,
        accelerator="auto",
        min_epochs=1,
        max_epochs=args.NUM_EPOCHS,
        precision='16-mixed',
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, early_stopping]
    )


    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")
