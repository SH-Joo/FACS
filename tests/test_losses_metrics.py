"""Equation and independent morphology checks; no trained models needed."""

import importlib.util
import ast
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.ndimage import binary_dilation, convolve, maximum_filter, minimum_filter
from skimage.morphology import disk, skeletonize
import torch
import torch.nn.functional as F

from facs.losses import EdgeNetwork, SEMEDALoss, SoftCTSLoss, WeightedLoss, semantic_edges, soft_skeleton
from facs.metrics import MetricAccumulator, centerline_counts, compute_cts, masks, score_image
from facs.model import FPCM, EfficientMSA, GlobalPool2d, LayerNorm2d, MiT, OverlapPatchEmbedding, ResNetEncoder

torch.set_num_threads(2)


class LossTests(unittest.TestCase):
    def test_loading_fixed_edge_weights_does_not_change_model_initialization_rng(self):
        network = EdgeNetwork()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"edge.pt"
            torch.save({"kind":"semeda-edge-v1","training_split":"train","model":network.state_dict()},path)
            torch.manual_seed(23)
            before = torch.get_rng_state().clone()
            loss = WeightedLoss({"bce":0.5,"semeda":0.5},edge_checkpoint=path)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            for name,value in network.state_dict().items():
                torch.testing.assert_close(value,loss.terms["semeda"].network.state_dict()[name],rtol=0,atol=0)

    def test_deterministic_edge_cross_entropy_matches_reference_and_gradient(self):
        from facs.losses import edge_cross_entropy
        generator = torch.Generator().manual_seed(17)
        logits = torch.randn(2,2,9,11,generator=generator,dtype=torch.float64,requires_grad=True)
        edges = torch.randint(0,2,(2,9,11),generator=generator)
        actual = edge_cross_entropy(logits,edges)
        expected = torch.nn.functional.cross_entropy(logits,edges)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        actual_grad = torch.autograd.grad(actual,logits,retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected,logits)[0]
        torch.testing.assert_close(actual_grad,expected_grad,rtol=1e-12,atol=1e-12)

    def setUp(self):
        torch.manual_seed(10)
        self.logits = torch.randn(2, 1, 12, 12, dtype=torch.float64, requires_grad=True)
        self.target = torch.zeros_like(self.logits)
        self.target[:, :, 6, 2:10] = 1

    def test_bce_is_exactly_bce_with_logits(self):
        actual, terms = WeightedLoss({"bce":1., "dice":0., "semeda":0.})(self.logits, self.target)
        expected = F.binary_cross_entropy_with_logits(self.logits, self.target)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(set(terms), {"bce"})
        actual.backward()
        torch.testing.assert_close(self.logits.grad, (self.logits.sigmoid()-self.target)/self.logits.numel())

    def test_weighted_terms_match_independent_equations(self):
        actual, values = WeightedLoss({"bce":0.5,"dice":0.25,"soft_cts":0.25},
                                      soft_parameters={"iterations":1,"sigma":1.,"kernel_size":3})(self.logits,self.target)
        expected = 0.5*values["bce"]+0.25*values["dice"]+0.25*values["soft_cts"]
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(self.logits.grad).all())
        self.assertGreater(float(self.logits.grad.abs().sum()), 0)

    def test_gt_is_not_sigmoided_and_wrong_shapes_are_rejected(self):
        objective = WeightedLoss({"bce":1.})
        for bad in (self.target.squeeze(1), self.target.unsqueeze(1), self.target+0.1, self.target*255):
            with self.assertRaises(ValueError):
                objective(self.logits,bad)

    def test_soft_skeleton_matches_independent_cpu_morphology(self):
        image = np.random.default_rng(3).random((12,12))
        cross = np.array([[0,1,0],[1,1,1],[0,1,0]],dtype=bool)
        def erosion(value):
            return minimum_filter(value, footprint=cross, mode="constant",cval=np.inf)
        def opening(value):
            return maximum_filter(erosion(value),size=3,mode="constant",cval=-np.inf)
        current=image.copy()
        expected=np.maximum(current-opening(current),0)
        for _ in range(3):
            current=erosion(current)
            residual=np.maximum(current-opening(current),0)
            expected+=np.maximum(residual*(1-expected),0)
        actual=soft_skeleton(torch.tensor(image)[None,None],3)[0,0].numpy()
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-15)

    def test_soft_cts_matches_gaussian_and_literal_equations(self):
        objective=SoftCTSLoss(iterations=0,sigma=1.,kernel_size=3,eps=1e-6)
        mask=self.target[0,0].numpy()
        kernel=np.exp(-(np.arange(-1,2)[:,None]**2+np.arange(-1,2)[None,:]**2)/2)
        kernel/=kernel.sum()
        field=convolve(mask,kernel,mode="constant",cval=0)
        overlap=float((field*field).sum())
        precision=overlap/(field.sum()+1e-6)
        expected=2*precision*precision/(precision+precision)
        actual=objective.score_from_probabilities(self.target[:1],self.target[:1]).item()
        self.assertAlmostEqual(actual,expected,places=13)
        self.assertGreater(actual,0)
        self.assertLess(actual,1)

    def test_soft_cts_gradient_matches_finite_differences(self):
        objective=SoftCTSLoss(iterations=0,sigma=1.,kernel_size=3)
        logits=self.logits[:1].detach().clone().requires_grad_()

        self.assertTrue(torch.autograd.gradcheck(lambda x:objective(x,self.target[:1]), (logits,), fast_mode=True))

    def test_every_active_auxiliary_has_finite_prediction_gradient(self):
        for name in ("dice","cldice","soft_cts","sobel"):
            logits=self.logits.detach().clone().requires_grad_()
            loss,_=WeightedLoss({name:1.},soft_parameters={"iterations":2,"sigma":1.,"kernel_size":3})(logits,self.target)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertGreater(float(logits.grad.abs().sum()),0,name)

    def test_empty_auxiliary_losses_are_finite(self):
        target=torch.zeros_like(self.target)
        for name in ("dice","cldice","soft_cts","sobel"):
            loss,_=WeightedLoss({name:1.},soft_parameters={"iterations":2})(self.logits,target)
            self.assertTrue(torch.isfinite(loss),name)

    def test_semeda_freezes_network_but_keeps_input_gradient(self):
        network=EdgeNetwork().double()
        objective=SEMEDALoss(network)
        objective.train()
        self.assertFalse(network.training)
        loss=objective(self.logits,self.target)
        loss.backward()
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in network.parameters()))
        self.assertGreater(float(self.logits.grad.abs().sum()),0)

    def test_semeda_cannot_silently_use_untrained_or_sobel_weights(self):
        with self.assertRaisesRegex(ValueError,"pretrained"):
            WeightedLoss({"bce":0.5,"semeda":0.5})

    def test_semantic_edges_include_both_sides_and_handle_border(self):
        target=torch.zeros(1,1,6,6)
        target[:,:,0:3,:]=1
        expected=torch.zeros(1,6,6,dtype=torch.long)
        expected[:,2:4,:]=1
        torch.testing.assert_close(semantic_edges(target),expected)
        self.assertEqual(int(semantic_edges(torch.ones_like(target)).sum()),0)

    def test_invalid_weights_and_gaussian_parameters(self):
        for weights in ({"bce":0.},{"bce":-1.},{"unknown":1.},{"bce":float("nan")}):
            with self.assertRaises(ValueError):
                WeightedLoss(weights)
        for kwargs in ({"sigma":float("nan")},{"sigma":0},{"kernel_size":2},{"iterations":-1}):
            with self.assertRaises(ValueError):
                SoftCTSLoss(**kwargs)


class HardMetricTests(unittest.TestCase):
    def setUp(self):
        self.line=np.zeros((32,32),dtype=bool)
        self.line[16,5:25]=True
        self.empty=np.zeros_like(self.line)

    def test_identical_and_empty_policies(self):
        identical=score_image(self.line,self.line)
        self.assertEqual(identical["iou"],1.)
        self.assertEqual(identical["cts"],1.)
        self.assertTrue(all(identical[f"cliou_{d}"]==1. for d in (0,2,4,8)))
        empty=score_image(self.empty,self.empty)
        self.assertEqual(empty["iou"],1.)
        self.assertEqual(empty["cliou_4"],1.)
        self.assertEqual(empty["cts"],0.)
        for prediction,target in ((self.empty,self.line),(self.line,self.empty)):
            actual=score_image(prediction,target)
            self.assertEqual(actual["iou"],0.)
            self.assertEqual(actual["cliou_4"],0.)
            self.assertEqual(actual["cts"],0.)

    def test_offset_tolerance_uses_matching_not_dilated_iou(self):
        shifted=np.roll(self.line,2,axis=0)
        actual=score_image(shifted,self.line)
        self.assertEqual(actual["iou"],0.)
        self.assertEqual(actual["cliou_0"],0.)
        self.assertEqual(actual["cliou_2"],1.)
        self.assertEqual(actual["cliou_4"],1.)

    def test_distance_implementation_matches_disk_dilation_reference(self):
        rng=np.random.default_rng(12)
        for _ in range(4):
            prediction=rng.random((32,32))<0.02
            target=rng.random((32,32))<0.02
            for delta in (0,1,2,4,8):
                gt_matched=target & binary_dilation(prediction,structure=disk(delta))
                false_positive=prediction & ~binary_dilation(target,structure=disk(delta))
                reference={"tp":int(gt_matched.sum()),"fp":int(false_positive.sum()),"fn":int(target.sum()-gt_matched.sum())}
                self.assertEqual(centerline_counts(prediction,target,delta),reference)

    def test_cliou_matches_preserved_standalone_evaluator(self):
        path=Path(__file__).resolve().parents[1]/"code/src/test.py"
        tree=ast.parse(path.read_text())
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="compute_cliou_per_pixel")
        namespace={"np":np,"thin":__import__("skimage.morphology",fromlist=["thin"]).thin,"disk":disk,
                   "cv2":SimpleNamespace(dilate=lambda x,kernel,iterations:binary_dilation(x,structure=kernel,iterations=iterations).astype(x.dtype))}
        exec(compile(ast.Module(body=[function],type_ignores=[]),"preserved-cliou","exec"),namespace)
        rng=np.random.default_rng(21)
        for _ in range(5):
            prediction=rng.random((32,32))<0.2
            target=rng.random((32,32))<0.2
            actual=score_image(prediction,target)
            for delta in (0,2,4,8):
                expected=namespace["compute_cliou_per_pixel"](target.astype(np.uint8),prediction.astype(np.uint8),delta)
                self.assertAlmostEqual(actual[f"cliou_{delta}"],expected,places=14)

    def test_offline_empty_exclusion_has_explicit_denominator(self):
        accumulator=MetricAccumulator(cliou_empty_policy="exclude",empty_cts=1.)
        accumulator.update("1",self.empty,self.empty)
        accumulator.update("2",self.line,self.line)
        summary=accumulator.summary()
        self.assertEqual(summary["count"],2)
        self.assertEqual(summary["macro_counts"]["cliou_4"],1)
        self.assertEqual(summary["macro"]["cliou_4"],1.)
        self.assertEqual(summary["macro"]["cts"],1.)

    def test_legacy_cts_matches_original_reference(self):
        path=Path(__file__).resolve().parents[1]/"code/src/CTS.py"
        spec=importlib.util.spec_from_file_location("legacy_cts_for_test",path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng=np.random.default_rng(5)
        for _ in range(5):
            prediction=rng.random((20,20))<0.03
            target=rng.random((20,20))<0.03
            reference=module.compute_cts(prediction,target,buffer_radius=2,threshold=0.5)
            actual=compute_cts(prediction,target,radius=2)
            for key in ("pcs","rcs","cts"):
                self.assertAlmostEqual(actual[key],reference[key.upper()],places=13)

    def test_cts_matching_variants_are_explicit(self):
        shifted=np.roll(self.line,2,axis=0)
        self.assertEqual(compute_cts(shifted,self.line,radius=4,mode="legacy_exact")["cts"],0.)
        self.assertEqual(compute_cts(shifted,self.line,radius=4,mode="buffered")["cts"],1.)
        far=np.roll(self.line,10,axis=0)
        self.assertEqual(compute_cts(far,self.line,radius=4,mode="buffered")["cts"],0.)

    def test_macro_micro_and_sample_order_are_independent(self):
        pairs=[("1",self.line,self.line),("2",self.empty,self.line),("3",self.empty,self.empty)]
        first,second=MetricAccumulator(),MetricAccumulator()
        for sample_id,prediction,target in pairs:
            first.update(sample_id,prediction,target)
        for sample_id,prediction,target in reversed(pairs):
            second.update(sample_id,prediction,target)
        self.assertEqual(first.summary(),second.summary())
        self.assertAlmostEqual(first.summary()["macro"]["iou"],2/3)
        self.assertEqual(first.summary()["micro"]["iou"],0.5)
        with self.assertRaises(ValueError):
            first.update("1",self.line,self.line)

    def test_raw_encoding_wrong_shape_and_no_sample_are_rejected(self):
        for bad in (self.line.astype(float)*255,np.zeros((4,4)),np.full((32,32),np.nan)):
            with self.assertRaises(ValueError):
                masks(bad,self.line)
        with self.assertRaises(ValueError):
            MetricAccumulator().summary()


class ModelComponentTests(unittest.TestCase):
    def test_fixed_affine_norm_preserves_original_kernel_without_new_keys(self):
        original=LayerNorm2d(8,elementwise_affine=True)
        fixed=LayerNorm2d(8,elementwise_affine=False)
        image=torch.randn(2,8,12,16)
        torch.testing.assert_close(fixed(image),original(image),rtol=0,atol=0)
        self.assertEqual(list(fixed.state_dict()),[])
        self.assertEqual(list(fixed.parameters()),[])

    def test_deterministic_global_pool_values_and_gradients_match_adaptive_pool(self):
        for maximum in (False,True):
            image=torch.randn(2,8,12,16,dtype=torch.float64,requires_grad=True)
            pool=GlobalPool2d(maximum)
            expected=(F.adaptive_max_pool2d(image,1) if maximum else F.adaptive_avg_pool2d(image,1))
            actual=pool(image)
            torch.testing.assert_close(actual,expected,rtol=0,atol=2e-15)
            reference_grad=torch.autograd.grad(expected.sum(),image,retain_graph=True)[0]
            actual_grad=torch.autograd.grad(actual.sum(),image)[0]
            torch.testing.assert_close(actual_grad,reference_grad,rtol=0,atol=0)
            pool.eval()
            torch.testing.assert_close(pool(image),expected,rtol=0,atol=0)
    def test_overlap_embedding_uses_actual_rectangular_grid(self):
        embedding=OverlapPatchEmbedding(3,2,1,3,8)
        self.assertEqual(tuple(embedding(torch.randn(2,3,65,81)).shape),(2,8,33,41))

    def test_attention_norm_is_registered_and_has_gradient(self):
        attention=EfficientMSA(8,2,2,norm_affine=True)
        loss=attention(torch.randn(2,8,16,16)).square().mean()
        loss.backward()
        self.assertIn("norm.weight",attention.state_dict())
        self.assertGreater(float(attention.norm.weight.grad.abs().sum()),0)

    def test_mit_has_four_expected_resolutions(self):
        model=MiT(dims=(8,16,32,64),n_heads=(1,2,4,8),expansion=(2,2,2,2),n_layers=(1,1,1,1))
        outputs=model(torch.randn(1,3,64,80))
        self.assertEqual([tuple(x.shape) for x in outputs],[(1,8,32,40),(1,16,16,20),(1,32,8,10),(1,64,4,5)])

    def test_cnn_instances_do_not_share_parameters(self):
        first,second=ResNetEncoder(),ResNetEncoder()
        self.assertNotEqual(first.encoder1[0].weight.data_ptr(),second.encoder1[0].weight.data_ptr())
        before=second.encoder1[0].weight.clone()
        with torch.no_grad():
            first.encoder1[0].weight.zero_()
        torch.testing.assert_close(before,second.encoder1[0].weight,rtol=0,atol=0)

    def test_fft_decomposition_is_per_batch_and_channel(self):
        module=FPCM(8,filter_mode="normalized").double()
        data=torch.randn(2,8,16,20,dtype=torch.float64)
        low,high=module.decompose(data)
        torch.testing.assert_close(low+high,data,rtol=0,atol=5e-16)
        single_low,_=module.decompose(data[1:2,3:4])
        torch.testing.assert_close(single_low,low[1:2,3:4],rtol=0,atol=0)
        constant=torch.full_like(data,2)
        constant_low,constant_high=module.decompose(constant)
        torch.testing.assert_close(constant_low,constant,rtol=0,atol=0)
        self.assertEqual(float(constant_high.detach().abs().max()),0.)

    def test_filter_formulas_and_cutoff_trainability_are_explicit(self):
        literal=FPCM(8,filter_mode="paper")
        normalized=FPCM(8,filter_mode="normalized")
        self.assertNotIn("cutoff_ratio",dict(literal.named_parameters()))
        self.assertIn("cutoff_ratio",dict(normalized.named_parameters()))
        self.assertAlmostEqual(float(literal.frequency_filter(256,256).min()),0.99996948,places=7)
        self.assertAlmostEqual(float(normalized.frequency_filter(256,256).detach().min()),np.exp(-16),places=12)


if __name__=="__main__":
    unittest.main()
