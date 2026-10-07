from pytorch_lightning.callbacks import EarlyStopping, Callback, ModelCheckpoint
import os

class MyPrintingCallBack(Callback):
    def __init__(self):
        super(MyPrintingCallBack, self).__init__()

    def on_train_start(self, trainer, pl_module):
        print("Start Training")

    def on_train_end(self, trainer, pl_module):
        print("Training is done")
