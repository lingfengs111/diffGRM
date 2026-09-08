import unittest

import torch

from scripts.train_domino_causal_corrector import (
    frozen_target_ranks,
    hard_candidate_batch,
    rank_aware_topk_objective,
    select_configuration,
    standardized_residuals,
)


class ARResidualCandidateTest(unittest.TestCase):
    def test_rank_aware_loss_only_selects_handoff_zone(self):
        base = torch.tensor([
            [5.0, 4.0, 3.0, 2.0],
            [5.0, 4.0, 3.0, 2.0],
        ])
        ranks = frozen_target_ranks(base, torch.tensor([0, 2]))
        scores = torch.tensor([
            [4.0, 3.0, 2.0, 1.0],
            [1.5, 4.0, 2.0, 1.0],
        ])
        residual = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, -0.1, 0.0],
        ])

        promotion, preservation, eligible = rank_aware_topk_objective(
            scores,
            residual,
            ranks,
            target_in_pool=torch.tensor([True, True]),
            min_rank=2,
            max_rank=3,
            target_top_k=2,
            preserve_head_k=2,
        )

        self.assertEqual(ranks.tolist(), [1, 3])
        self.assertEqual(eligible.tolist(), [False, True])
        self.assertGreater(promotion.item(), 0.0)
        self.assertAlmostEqual(preservation.item(), 0.025, places=6)

    def test_hard_candidates_do_not_inject_out_of_pool_target(self):
        scores = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0]])
        catalog = torch.arange(10).reshape(5, 2)

        candidates, candidate_scores, rows, target_in_pool = hard_candidate_batch(
            scores,
            target_rows=torch.tensor([4]),
            catalog=catalog,
            pool_k=3,
            n_candidates=3,
        )

        self.assertFalse(target_in_pool.item())
        self.assertEqual(rows.tolist(), [[0, 1, 2]])
        self.assertEqual(candidate_scores.tolist(), [[9.0, 8.0, 7.0]])
        self.assertTrue(torch.equal(candidates, catalog[rows]))

    def test_boundary_hard_candidates_mix_head_and_proposal_boundary(self):
        scores = torch.arange(12.0, 0.0, -1.0).unsqueeze(0)
        catalog = torch.arange(24).reshape(12, 2)

        _, _, rows, target_in_pool = hard_candidate_batch(
            scores,
            target_rows=torch.tensor([1]),
            catalog=catalog,
            pool_k=10,
            n_candidates=5,
            proposal_k=7,
            boundary_fraction=0.5,
        )

        self.assertTrue(target_in_pool.item())
        self.assertEqual(rows[0, 0].item(), 1)
        self.assertEqual(set(rows[0, 1:3].tolist()), {0, 2})
        self.assertLessEqual(set(rows[0, 3:].tolist()), {5, 6, 7})

    def test_standardized_residual_uses_base_scale_and_zero_stays_zero(self):
        base = torch.tensor([[1.0, 2.0, 4.0]])
        ar = torch.tensor([[2.0, 4.0, 8.0]])
        residual = torch.zeros_like(base)

        base_z, ar_z, student_z = standardized_residuals(
            base, ar, residual, temperature=2.0
        )

        self.assertTrue(torch.allclose(base_z, ar_z))
        self.assertTrue(torch.equal(student_z, torch.zeros_like(student_z)))
        self.assertTrue(
            torch.allclose(base_z.mean(dim=1), torch.zeros(1), atol=1e-7)
        )

    def test_recall_guard_rejects_higher_ndcg_with_lower_proposal_recall(self):
        metrics = {
            'proposal_k': 72,
            'domino_b0_recall@72': 0.30,
            'domino_b0_ndcg@10': 0.10,
            'domino_b0_recall@10': 0.20,
            'domino_b1_recall@72': 0.29,
            'domino_b1_ndcg@10': 0.20,
            'domino_b1_recall@10': 0.25,
        }

        selected = select_configuration(
            metrics,
            betas=(0.0, 1.0),
            alphas=(),
            selection_metric='drafter_ndcg10',
            has_ar=False,
            recall72_tolerance=0.0,
        )

        self.assertAlmostEqual(selected['beta'], 0.0)
        self.assertAlmostEqual(selected['recall_at_72'], 0.30)


if __name__ == '__main__':
    unittest.main()
