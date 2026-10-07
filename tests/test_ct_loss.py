"""Numerical and reference-formula checks for soft-CTS."""

import unittest

import torch

from facs.losses import SoftCTSLoss


class SoftCTSContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_literal_formula_can_prefer_a_deleted_gt_branch(self):
        target = torch.zeros(1, 1, 64, 64)
        target[:, :, 8:32:2, 8:40] = 1
        target[:, :, 52, 8:56] = 1
        incomplete = target.clone()
        incomplete[:, :, 52, 8:56] = 0
        loss = SoftCTSLoss()
        self.assertGreater(float(loss.score_from_probabilities(incomplete, target)),
                           float(loss.score_from_probabilities(target, target)))

    def test_literal_empty_target_has_no_false_positive_gradient(self):
        logits = torch.randn(1, 1, 16, 16, generator=torch.Generator().manual_seed(3), requires_grad=True)
        loss = SoftCTSLoss(iterations=3, kernel_size=5)(logits, torch.zeros_like(logits))
        gradient = torch.autograd.grad(loss, logits)[0]
        self.assertEqual(float(loss.detach()), 1.)
        self.assertEqual(int(torch.count_nonzero(gradient)), 0)

    def test_literal_gradient_matches_finite_difference(self):
        logits = (torch.randn(1, 1, 8, 8, generator=torch.Generator().manual_seed(73), dtype=torch.float64)*.3).requires_grad_()
        target = torch.zeros_like(logits)
        target[:, :, 3, 1:7] = 1
        loss = SoftCTSLoss(iterations=2, sigma=.8, kernel_size=3)
        self.assertTrue(torch.autograd.gradcheck(lambda values: loss(values, target), (logits,),
                                                eps=1e-6, atol=1e-5, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()
