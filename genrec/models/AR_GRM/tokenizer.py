"""Autoregressive tokenizer backed by the shared SID construction pipeline.

AR and diffusion experiments must consume byte-for-byte identical catalog
codes.  Keeping a second copy of OPQ/RQ and collision logic had already caused
normalization and cache-tag drift, so only the model-specific collators remain
different here.
"""

import torch

from genrec.models.DIFF_GRM.tokenizer import DIFF_GRMTokenizer


class AR_GRMTokenizer(DIFF_GRMTokenizer):
    """Use the canonical SID tokenizer and raw-code collate representation."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        if self.sid_prefix_strategy == 'random_latent':
            base_train_collate = self.collate_fn['train']

            def collate_random_latent(batch):
                collated = base_train_collate(batch)
                latent = torch.randint(
                    0,
                    int(self.config.get('n_latent_tokens', 8)),
                    (collated['decoder_labels'].shape[0],),
                    dtype=torch.long,
                )
                collated['decoder_input_ids'][:, 0] = latent
                collated['decoder_labels'][:, 0] = latent
                return collated

            self.collate_fn['train'] = collate_random_latent
