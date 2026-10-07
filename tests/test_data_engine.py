"""Paired transforms, provenance guards and uninterrupted-vs-resumed optimization."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from facs.data import CV12Dataset
from facs.engine import build_loss, build_optimizer, model_state_hash, read_config, train


ROOT=Path(__file__).resolve().parents[1]


class TinySegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.ReLU(),nn.Conv2d(4,1,1))

    def forward(self,x):
        return self.layers(x)


class DataEngineTests(unittest.TestCase):
    def test_model_state_hash_covers_weights_scalar_buffers_and_shapes(self):
        model = nn.Sequential(nn.Conv2d(3,4,3),nn.BatchNorm2d(4))
        before = model_state_hash(model)
        self.assertEqual(before,model_state_hash(model))
        model[1].num_batches_tracked.add_(1)
        self.assertNotEqual(before,model_state_hash(model))
        before = model_state_hash(model)
        with torch.no_grad():model[0].weight[0,0,0,0].add_(1)
        self.assertNotEqual(before,model_state_hash(model))

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)/"dataset"
        (self.root/"manifests").mkdir(parents=True)
        rows=[]
        for split,ids in (("train",(1,2,3)),("val",(9601,9602,9603)),("test",(10801,))):
            split_rows=[]
            for sample_id in ids:
                name=f"{sample_id}.png"
                values=np.zeros((256,256),dtype=np.uint8)
                values[120:136,32:224]=255
                image=np.stack((values,np.full_like(values,100),np.full_like(values,200)),axis=-1)
                for kind,array in (("IMG",image),("GT",values)):
                    folder=self.root/split/kind
                    folder.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(array).save(folder/name)
                row={"sample_id":sample_id,"split":split,"filename":name,"height":256,"width":256,
                     "image_path":f"{split}/IMG/{name}","gt_path":f"{split}/GT/{name}","thickness_bin":"16_32",
                     "image_sha256":hashlib.sha256((self.root/split/"IMG"/name).read_bytes()).hexdigest(),
                     "gt_sha256":hashlib.sha256((self.root/split/"GT"/name).read_bytes()).hexdigest()}
                split_rows.append(row)
                rows.append(row)
            self.write_csv(self.root/"manifests"/f"{split}.csv",split_rows)
        self.write_csv(self.root/"manifests/all.csv",rows)
        (self.root/"source_archive.json").write_text(json.dumps({"sha256":"synthetic-fixture"}))
        (self.root/"preparation_summary.json").write_text(json.dumps({"protocol":"test-fixture"}))
        self.config=read_config(ROOT/"tests/fixtures/bce_pilot.json")
        self.config["data"]["root"]=str(self.root)
        self.config["data"]["train_ids"]=[1,2,3]
        self.config["data"]["val_ids"]=[9601,9602,9603]
        self.config["training"].update(batch_size=2,workers=0,max_epochs=2,seed=21,cpu_threads=2)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write_csv(path,rows):
        with path.open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_validation_is_identical_and_mask_is_b1hw(self):
        dataset=CV12Dataset(self.root,"val")
        first,second=dataset[0],dataset[0]
        torch.testing.assert_close(first["image"],second["image"],rtol=0,atol=0)
        torch.testing.assert_close(first["target"],second["target"],rtol=0,atol=0)
        self.assertEqual(tuple(first["image"].shape),(3,256,256))
        self.assertEqual(tuple(first["target"].shape),(1,256,256))
        self.assertEqual(set(first["target"].unique().tolist()),{0.,1.})

    def test_dihedral_transform_keeps_image_and_label_aligned(self):
        dataset=CV12Dataset(self.root,"train",mean=(0,0,0),std=(1,1,1),augmentation="dihedral")
        for seed in range(8):
            torch.manual_seed(seed)
            item=dataset[0]
            torch.testing.assert_close(item["image"][0],item["target"][0],rtol=0,atol=0)

    def test_changed_source_missing_gt_and_invalid_val_augmentation_fail(self):
        dataset=CV12Dataset(self.root,"train")
        (self.root/"train/GT/1.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"changed"):
            dataset[0]
        (self.root/"val/GT/9601.png").unlink()
        with self.assertRaisesRegex(ValueError,"Missing"):
            CV12Dataset(self.root,"val")
        with self.assertRaisesRegex(ValueError,"deterministic"):
            CV12Dataset(self.root,"val",augmentation="dihedral")

    def test_missing_or_repeated_selection_is_rejected(self):
        for ids in ([1,1],[100],[]):
            with self.assertRaises(ValueError):
                CV12Dataset(self.root,"train",sample_ids=ids)

    def test_nonfinite_normalization_is_rejected(self):
        with self.assertRaises(ValueError):
            CV12Dataset(self.root,"train",mean=(float("nan"),0,0))

    def test_optimizer_groups_cover_parameters_once_with_requested_rates(self):
        model = nn.Module()
        model.cnn_encoder = nn.Linear(2,2)
        model.mix_transformer = nn.Linear(2,2)
        model.fpcm_tail = nn.Module()
        model.fpcm_tail.seg_head = nn.Linear(2,1)
        model.decoder = nn.Linear(2,2)
        settings = {"lr":1e-4,"head_lr_multiplier":10,"encoder_lr_multiplier":0.5}
        optimizer = build_optimizer(model,settings)
        groups = {group["name"]:group for group in optimizer.param_groups}
        self.assertEqual({name:group["lr"] for name,group in groups.items()},
                         {"main":1e-4,"encoder":5e-5,"head":1e-3})
        assigned = [id(p) for group in groups.values() for p in group["params"]]
        self.assertEqual(len(assigned),len(set(assigned)))
        self.assertEqual(set(assigned),{id(p) for p in model.parameters()})
        self.assertEqual({id(p) for p in groups["head"]["params"]},
                         {id(p) for p in model.fpcm_tail.seg_head.parameters()})

    def test_semeda_checkpoint_with_wrong_train_provenance_is_rejected(self):
        checkpoint = Path(self.temp.name)/"wrong-edge.pt"
        torch.save({"kind":"semeda-edge-v1","training_split":"train", "training_ids":[9601]},checkpoint)
        self.config["loss"]={"weights":{"semeda":1},"edge_checkpoint":str(checkpoint)}
        with self.assertRaisesRegex(ValueError,"provenance"):
            build_loss(self.config)

    @patch("facs.engine.build_model",side_effect=lambda *args,**kwargs:TinySegmentation())
    def test_epoch_boundary_resume_matches_uninterrupted_training(self,build_model):
        full=Path(self.temp.name)/"full"
        resumed=Path(self.temp.name)/"resumed"
        train(self.config,full,device="cpu")
        train(self.config,resumed,device="cpu",stop_after_epoch=1)
        train(self.config,resumed,device="cpu",resume=resumed/"last.pt")
        first=torch.load(full/"last.pt",weights_only=True,map_location="cpu")
        second=torch.load(resumed/"last.pt",weights_only=True,map_location="cpu")
        self.assertEqual(first["epoch"],1)
        for key in first["model"]:
            torch.testing.assert_close(first["model"][key],second["model"][key],rtol=0,atol=0)
        for left,right in zip(first["history"],second["history"]):
            self.assertEqual(left["train"]["loss"],right["train"]["loss"])
            self.assertEqual(left["val"]["loss"],right["val"]["loss"])
            self.assertEqual(left["train"]["count"],3)
            self.assertEqual(left["train"]["steps"],2)
        self.assertEqual(first["scheduler"],second["scheduler"])
        torch.testing.assert_close(first["rng"]["torch"],second["rng"]["torch"],rtol=0,atol=0)
        torch.testing.assert_close(first["rng"]["loader_generator"],second["rng"]["loader_generator"],rtol=0,atol=0)
        self.assertFalse(json.loads((full/"result.json").read_text())["test_evaluated"])

    @patch("facs.engine.build_model",side_effect=lambda *args,**kwargs:TinySegmentation())
    def test_resume_rejects_changed_config_and_completed_run_is_not_overwritten(self,build_model):
        destination=Path(self.temp.name)/"run"
        train(self.config,destination,device="cpu",stop_after_epoch=1)
        changed=deepcopy(self.config)
        changed["training"]["lr"]*=2
        with self.assertRaisesRegex(ValueError,"does not match"):
            train(changed,destination,device="cpu",resume=destination/"last.pt")
        with self.assertRaises(FileExistsError):
            train(self.config,destination,device="cpu")


if __name__=="__main__":
    unittest.main()
