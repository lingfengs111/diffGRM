import itertools
import math

import numpy as np
import torch

from genrec.models.DIFF_GRM.path_objective import (
    build_hard_negative_table,
    order_marginal_dp,
)


def test_order_marginal_dp_matches_explicit_permutations() -> None:
    n_digit = 3
    transition = torch.randn(2, (1 << n_digit) - 1, n_digit)
    score = order_marginal_dp(transition)

    explicit = []
    for permutation in itertools.permutations(range(n_digit)):
        state = 0
        path_score = torch.zeros(2)
        for digit in permutation:
            path_score = path_score + transition[:, state, digit]
            state |= 1 << digit
        explicit.append(path_score)
    expected = torch.logsumexp(torch.stack(explicit), dim=0) - math.log(math.factorial(n_digit))
    assert torch.allclose(score, expected, atol=1e-6)


def test_hard_negative_table_prefers_shared_codes() -> None:
    codes = np.asarray(
        [[0, 0, 0], [0, 0, 1], [0, 2, 2], [3, 3, 3]], dtype=np.int64
    )
    negatives = build_hard_negative_table(codes, num_negatives=1, min_shared_digits=2)
    assert negatives[0, 0] == 1
    assert negatives[1, 0] == 0
