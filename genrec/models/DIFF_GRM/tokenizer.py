# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import json
import hashlib
import pickle
import re
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer
from genrec.models.DIFF_GRM.collision import (
    append_dedup_digit,
    collision_stats,
    repair_code_digit,
    repair_product_codes,
)
from genrec.models.DIFF_GRM.opq_regularizer import regularize_product_centroids


class DIFF_GRMTokenizer(AbstractTokenizer):
    """
    DIFF_GRM Tokenizer for Diffusion-based Generative Recommendation Model
    
    Special tokens:
    - PAD=0, BOS=1, EOS=2, SID_OFFSET=3
    
    SID Configuration:
    - n_digit: configurable (e.g., 4, 8, 12), codebook_size=256
    - vocab_size = 3 + n_digit * codebook_size
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        # 兜底，避免 KeyError
        config.setdefault('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        config.setdefault('num_proc', 1)

        self.n_codebook_bits = self._get_codebook_bits(config['codebook_size'])
        self.quantizer_n_digit = int(config.get('quantizer_n_digit', config['n_digit']))
        self.sid_prefix_strategy = str(
            config.get('sid_prefix_strategy', 'none')
        ).lower()
        if self.sid_prefix_strategy not in ('none', 'behavior_route', 'random_latent'):
            raise ValueError(
                "sid_prefix_strategy must be one of none|behavior_route|random_latent, "
                f"got {self.sid_prefix_strategy}"
            )
        self.sid_prefix_digits = 1 if self.sid_prefix_strategy != 'none' else 0
        self.sid_alias_path = config.get('sid_alias_path')
        if self.sid_alias_path and self.sid_prefix_strategy != 'none':
            raise ValueError(
                'sid_alias_path currently requires sid_prefix_strategy=none; '
                'compose aliases and route/latent prefixes in a separate experiment'
            )
        # Canonical paths remain in item2tokens. Optional PIT-style aliases
        # live separately so histories and held-out labels stay unchanged.
        self.item2alias_tokens = {}
        self.sid_collision_strategy = str(
            config.get('sid_collision_strategy', 'none')
        ).lower()
        if self.sid_collision_strategy not in ('none', 'append_dedup', 'hungarian'):
            raise ValueError(
                "sid_collision_strategy must be one of none|append_dedup|hungarian, "
                f"got {self.sid_collision_strategy}"
            )
        expected_model_digits = self.sid_prefix_digits + self.quantizer_n_digit + (
            1 if self.sid_collision_strategy == 'append_dedup' else 0
        )
        if int(config['n_digit']) != expected_model_digits:
            raise ValueError(
                f"n_digit={config['n_digit']} is incompatible with "
                f"quantizer_n_digit={self.quantizer_n_digit} and "
                f"sid_collision_strategy={self.sid_collision_strategy}; "
                f"expected n_digit={expected_model_digits}"
            )

        # 选择量化器：opq_pq(默认) | rq_kmeans | rq_opq | none(随机)
        self.sid_quantizer = config.get('sid_quantizer', 'opq_pq')
        assert self.sid_quantizer in ('opq_pq', 'rq_kmeans', 'rq_opq', 'none'), \
            ("sid_quantizer must be one of "
             "['opq_pq','rq_kmeans','rq_opq','none'], "
             f"got {self.sid_quantizer}")

        # 🚀 兼容旧配置：仅在 opq_pq 模式下使用 disable_opq/index_factory
        if self.sid_quantizer == 'opq_pq':
            use_opq = not config.get('disable_opq', False)
            if use_opq:
                self.index_factory = f'OPQ{self.quantizer_n_digit},IVF1,PQ{self.quantizer_n_digit}x{self.n_codebook_bits}'
            else:
                self.index_factory = f'IVF1,PQ{self.quantizer_n_digit}x{self.n_codebook_bits}'
        elif self.sid_quantizer == 'rq_kmeans':
            self.index_factory = f'RQKMEANS{self.quantizer_n_digit}x{self.n_codebook_bits}'
        elif self.sid_quantizer == 'rq_opq':
            self.rq_n_digit = int(config.get('rq_opq_rq_digits', 2))
            self.opq_n_digit = int(
                config.get(
                    'rq_opq_opq_digits',
                    self.quantizer_n_digit - self.rq_n_digit,
                )
            )
            if self.rq_n_digit <= 0 or self.opq_n_digit <= 0:
                raise ValueError('RQ-OPQ requires positive RQ and OPQ digit counts')
            if self.rq_n_digit + self.opq_n_digit != self.quantizer_n_digit:
                raise ValueError(
                    'rq_opq_rq_digits + rq_opq_opq_digits must equal '
                    f'quantizer_n_digit={self.quantizer_n_digit}'
                )
            self.index_factory = (
                f'RQ{self.rq_n_digit}x{self.n_codebook_bits}_'
                f'OPQ{self.opq_n_digit},IVF1,'
                f'PQ{self.opq_n_digit}x{self.n_codebook_bits}'
            )
        else:  # 'none'
            self.index_factory = f'RAND{self.quantizer_n_digit}x{self.n_codebook_bits}'
        # 先初始化父类，保证 self.config / self.logger 等字段可用
        super(DIFF_GRMTokenizer, self).__init__(config, dataset)
        
        # 现在再写日志
        self.log(f'[TOKENIZER] Index factory: {self.index_factory}')
        self.log(
            f'[TOKENIZER] SID collision strategy: {self.sid_collision_strategy}; '
            f'prefix strategy={self.sid_prefix_strategy}; '
            f'quantizer digits={self.quantizer_n_digit}, model digits={self.n_digit}'
        )
        self.dataset = dataset  # 添加dataset引用
        self.item2id = dataset.item2id
        self.id2item = dataset.id_mapping['id2item']
        
        # Special tokens - 简化token ID分配
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.mask_token = -1  # MASK token用于推理，不在vocab中
        self.sid_offset = 3  # SID token从3开始
        
        self.item2tokens = self._init_tokenizer(dataset)
        
        # Create reverse mapping for inference (如果还没有创建的话)
        if not hasattr(self, 'tokens2item'):
            self.tokens2item = self._create_reverse_mapping()
        
        # Set collate functions
        from genrec.models.DIFF_GRM.collate import collate_fn_train, collate_fn_val, collate_fn_test
        self.collate_fn = {
            'train': collate_fn_train,
            'val': collate_fn_val,
            'test': collate_fn_test
        }

    @property
    def n_digit(self):
        return self.config['n_digit']

    @property
    def codebook_size(self):
        return self.config['codebook_size']

    @property
    def max_token_seq_len(self) -> int:
        return 1 + self.n_digit  # [BOS] + n_digit SID tokens

    @property
    def vocab_size(self) -> int:
        return 3 + self.n_digit * self.codebook_size  # PAD(0) + BOS(1) + EOS(2) + SID tokens

    def _get_codebook_bits(self, n_codebook):
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)

    def _quant_tag(self) -> str:
        """Stable cache tag including post-quantization collision handling."""
        quant_tag = self.index_factory
        if self.sid_quantizer == 'rq_kmeans':
            quant_tag += f'_seed{self.config.get("rq_kmeans_seed",1234)}_it{self.config.get("rq_kmeans_niters",20)}'
        elif self.sid_quantizer == 'rq_opq':
            quant_tag += (
                f'_seed{self.config.get("rq_opq_seed",2026)}'
                f'_it{self.config.get("rq_opq_niters",25)}'
                f'_norm{int(bool(self.config.get("rq_opq_normalize_residual",False)))}'
            )
        elif self.sid_quantizer == 'none':
            quant_tag += f'_seed{self.config.get("sid_random_seed",12345)}'
        if self.sid_collision_strategy == 'append_dedup':
            quant_tag += '_appenddedup'
        elif self.sid_collision_strategy == 'hungarian':
            if self.sid_quantizer == 'rq_opq':
                repair_digit = (
                    f'opq{self.config.get("rq_opq_repair_opq_digit", -1)}'
                )
            else:
                repair_digit = self.config.get('collision_repair_digit', 'auto')
            quant_tag += f'_cfhungarian-{repair_digit}'
        regularizer_steps = int(self.config.get('opq_regularizer_steps', 0))
        if regularizer_steps > 0:
            balance = str(self.config.get('opq_balance_weight', 0.0)).replace('.', 'p')
            mi = str(self.config.get('opq_mi_weight', 0.0)).replace('.', 'p')
            hard = str(self.config.get('opq_hardness_weight', 0.0)).replace('.', 'p')
            st = int(bool(self.config.get('opq_straight_through', False)))
            anchor = str(self.config.get('opq_anchor_weight', 0.0)).replace('.', 'p')
            quant_tag += (
                f'_reg-s{regularizer_steps}-bal{balance}-mi{mi}'
                f'-hard{hard}-st{st}-a{anchor}'
            )
        return quant_tag

    def _mapping_tag(self, model_basename: str) -> str:
        tag = (
            f'{model_basename}_pca{self.config["sent_emb_pca"]}_'
            f'{self._quant_tag()}'
        )
        if self.sid_prefix_strategy == 'behavior_route':
            tag += (
                f'_behaviorroute-k{int(self.config.get("route_codebook_size", 64))}'
                f'-svd{int(self.config.get("route_svd_dim", 32))}'
                f'-seed{int(self.config.get("route_seed", 2026))}'
            )
        elif self.sid_prefix_strategy == 'random_latent':
            tag += f'_randomlatent-k{int(self.config.get("n_latent_tokens", 8))}'
        sid_override_path = self.config.get('sid_override_path')
        if sid_override_path:
            override_tag = self.config.get(
                'sid_override_tag', os.path.splitext(os.path.basename(sid_override_path))[0]
            )
            override_tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(override_tag))
            tag += f'_override-{override_tag}'
        if self.sid_alias_path:
            alias_path = os.path.abspath(os.path.expanduser(self.sid_alias_path))
            with open(alias_path, 'rb') as handle:
                alias_hash = hashlib.sha256(handle.read()).hexdigest()[:12]
            alias_tag = self.config.get(
                'sid_alias_tag', os.path.splitext(os.path.basename(alias_path))[0]
            )
            alias_tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(alias_tag))
            tag += f'_aliases-{alias_tag}-{alias_hash}'
        return f'{tag}_{self.n_digit}d'

    def _embedding_cache_basename(self) -> str:
        """Name embedding/SID caches without mixing different text views."""
        model_basename = os.path.basename(self.config["sent_emb_model"])
        metadata_tag = self.config.get("metadata_cache_tag")
        if not metadata_tag:
            return model_basename
        metadata_tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(metadata_tag)).strip('._')
        if not metadata_tag:
            raise ValueError("metadata_cache_tag must contain a filename-safe character")
        return f'{model_basename}_meta-{metadata_tag}'

    @property
    def has_item_aliases(self) -> bool:
        return bool(self.item2alias_tokens)

    def _raw_path_to_tokens(self, path):
        return tuple(
            int(code) + self.sid_offset + digit * self.codebook_size
            for digit, code in enumerate(path)
        )

    def _load_item_aliases(self, canonical_raw: dict) -> None:
        """Load globally unique alternative paths while preserving canonical SIDs."""
        if not self.sid_alias_path:
            self.item2alias_tokens = {}
            return

        alias_path = os.path.abspath(os.path.expanduser(self.sid_alias_path))
        with open(alias_path, encoding='utf-8') as handle:
            payload = json.load(handle)
        item_to_paths = payload.get('item_to_paths', payload)
        if not isinstance(item_to_paths, dict):
            raise ValueError('sid_alias_path must contain an item_to_paths dictionary')

        expected = set(canonical_raw)
        actual = set(item_to_paths)
        if expected != actual:
            raise ValueError(
                'sid_alias_path catalog mismatch: '
                f'missing={list(expected - actual)[:5]}, '
                f'extra={list(actual - expected)[:5]}'
            )

        seen = {}
        token_aliases = {}
        counts = []
        for item in canonical_raw:
            paths = item_to_paths[item]
            if not isinstance(paths, list) or not paths:
                raise ValueError(f'No SID paths provided for item {item!r}')
            normalized = []
            for path in paths:
                if len(path) != self.n_digit:
                    raise ValueError(
                        f'Alias for {item!r} has width {len(path)}, expected {self.n_digit}'
                    )
                raw = tuple(int(code) for code in path)
                if any(code < 0 or code >= self.codebook_size for code in raw):
                    raise ValueError(f'Alias for {item!r} contains out-of-range code: {raw}')
                owner = seen.get(raw)
                if owner is not None:
                    raise ValueError(
                        f'sid_alias_path collision: path={raw}, owners={owner!r},{item!r}'
                    )
                seen[raw] = item
                normalized.append(raw)

            canonical = tuple(int(code) for code in canonical_raw[item])
            if canonical not in normalized:
                raise ValueError(
                    f'Canonical SID {canonical} is absent from aliases for {item!r}'
                )
            normalized = [canonical] + [path for path in normalized if path != canonical]
            token_aliases[item] = tuple(
                self._raw_path_to_tokens(path) for path in normalized
            )
            counts.append(len(normalized))

        self.item2alias_tokens = token_aliases
        self.log(
            '[TOKENIZER] Loaded collision-free item aliases from '
            f'{alias_path}: paths={len(seen)}, per_item={min(counts)}..{max(counts)}'
        )

    def iter_item_token_paths(self, item, canonical_tokens=None):
        """Return every legal token path for an item in stable order."""
        if self.has_item_aliases:
            return self.item2alias_tokens[item]
        tokens = canonical_tokens if canonical_tokens is not None else self.item2tokens[item]
        if self.sid_prefix_strategy == 'random_latent':
            n_latent = int(self.config.get('n_latent_tokens', 8))
            paths = []
            for latent in range(n_latent):
                variant = list(tokens)
                variant[0] = self.sid_offset + latent
                paths.append(tuple(variant))
            return tuple(paths)
        return (tuple(tokens),)

    def alias_codebook_candidates(self, canonical_labels: torch.Tensor) -> torch.Tensor:
        """Resolve canonical label rows to their PIT alias codebook paths."""
        if not self.has_item_aliases:
            return canonical_labels.unsqueeze(1)
        device = canonical_labels.device
        rows = []
        for label in canonical_labels.detach().cpu().tolist():
            item_id = self.codebooks_to_item_id(label)
            if item_id is None:
                raise ValueError(f'Canonical training label is not a legal item SID: {label}')
            item = self.id2item[item_id]
            raw_paths = []
            for tokens in self.item2alias_tokens[item]:
                raw_paths.append([
                    int(token) - (self.sid_offset + digit * self.codebook_size)
                    for digit, token in enumerate(tokens)
                ])
            rows.append(raw_paths)
        n_paths = {len(paths) for paths in rows}
        if len(n_paths) != 1:
            raise ValueError(
                'Dynamic batch alias selection requires the same number of paths per item'
            )
        return torch.tensor(rows, dtype=torch.long, device=device)

    def _prepend_random_latent(self, item2sem_ids: dict) -> dict:
        """Add a canonical latent-0 prefix; training samples it dynamically.

        ``item2tokens`` keeps a single canonical row for history encoding and
        labels.  The reverse mapping and AR decoding trie expand every payload
        to all latent prefixes, so item identity remains the OPQ payload only.
        """
        n_latent = int(self.config.get('n_latent_tokens', 8))
        if n_latent <= 1 or n_latent > self.codebook_size:
            raise ValueError(
                f'n_latent_tokens must be in [2,{self.codebook_size}], got {n_latent}'
            )
        return {
            item: [0, *[int(code) for code in codes]]
            for item, codes in item2sem_ids.items()
        }

    def _prepend_behavior_route(self, item2sem_ids: dict, cache_dir: str) -> dict:
        """Prepend a train-only transition-cluster route to collision-free SIDs.

        Rows of the sparse target-by-predecessor matrix encode which items tend
        to precede each target.  Clustering its low-rank representation groups
        targets that are predictable from similar one-hop histories.  The
        held-out validation and test interactions never participate in route
        construction.
        """
        route_size = int(self.config.get('route_codebook_size', 64))
        svd_dim = int(self.config.get('route_svd_dim', 32))
        seed = int(self.config.get('route_seed', 2026))
        if route_size <= 1 or route_size > self.codebook_size:
            raise ValueError(
                f'route_codebook_size must be in [2,{self.codebook_size}], got {route_size}'
            )

        cache_stem = f'behavior_route_k{route_size}_svd{svd_dim}_seed{seed}'
        route_path = os.path.join(cache_dir, f'{cache_stem}.npy')
        report_path = os.path.join(cache_dir, f'{cache_stem}.json')
        force = bool(self.config.get('force_regenerate_route', False))
        n_items = self.dataset.n_items

        if os.path.exists(route_path) and not force:
            route_codes = np.load(route_path)
            if route_codes.shape != (n_items,):
                raise ValueError(
                    f'cached route shape {route_codes.shape} != expected {(n_items,)}'
                )
            self.log(f'[TOKENIZER] Loading behavior routes from {route_path}')
        else:
            from scipy import sparse
            from sklearn.cluster import MiniBatchKMeans
            from sklearn.decomposition import TruncatedSVD
            from sklearn.preprocessing import normalize

            rows, cols = [], []
            train_edges = []
            for seq in self.dataset.all_item_seqs.values():
                train_seq = seq[:-2]
                for previous, target in zip(train_seq, train_seq[1:]):
                    previous_id = self.dataset.item2id.get(previous)
                    target_id = self.dataset.item2id.get(target)
                    if previous_id and target_id:
                        rows.append(target_id - 1)
                        cols.append(previous_id - 1)
                        train_edges.append((previous_id, target_id))
            matrix = sparse.coo_matrix(
                (np.ones(len(rows), dtype=np.float32), (rows, cols)),
                shape=(n_items - 1, n_items - 1),
            ).tocsr()
            matrix.sum_duplicates()
            effective_dim = min(svd_dim, matrix.shape[0] - 1, matrix.shape[1] - 1)
            route_features = TruncatedSVD(
                n_components=effective_dim,
                n_iter=7,
                random_state=seed,
            ).fit_transform(matrix)
            route_features = normalize(route_features, norm='l2', copy=False)
            labels = MiniBatchKMeans(
                n_clusters=route_size,
                batch_size=2048,
                max_iter=200,
                n_init=10,
                random_state=seed,
            ).fit_predict(route_features)

            route_codes = np.zeros(n_items, dtype=np.int64)
            route_codes[1:] = labels
            zero_rows = np.asarray(matrix.getnnz(axis=1) == 0).reshape(-1)
            if zero_rows.any():
                # Only a handful of catalog items lack a train predecessor.
                # Give them a deterministic content fallback rather than using
                # held-out interactions to construct their route.
                for item, item_id in self.dataset.item2id.items():
                    if item_id and zero_rows[item_id - 1]:
                        route_codes[item_id] = int(item2sem_ids[item][0]) % route_size
            np.save(route_path, route_codes)

            counts = np.bincount(route_codes[1:], minlength=route_size)
            probs = counts[counts > 0] / counts.sum()
            train_route_counts = {}
            for previous_id, target_id in train_edges:
                if previous_id not in train_route_counts:
                    train_route_counts[previous_id] = np.zeros(route_size, dtype=np.int64)
                train_route_counts[previous_id][route_codes[target_id]] += 1
            global_route = int(counts.argmax())
            previous_prediction = {
                item_id: int(values.argmax())
                for item_id, values in train_route_counts.items()
            }

            def route_accuracy(edges):
                if not edges:
                    return 0.0
                hits = sum(
                    previous_prediction.get(previous_id, global_route)
                    == int(route_codes[target_id])
                    for previous_id, target_id in edges
                )
                return hits / len(edges)

            val_edges, test_edges = [], []
            for seq in self.dataset.all_item_seqs.values():
                if len(seq) >= 3:
                    val_edges.append((
                        self.dataset.item2id[seq[-3]],
                        self.dataset.item2id[seq[-2]],
                    ))
                if len(seq) >= 2:
                    test_edges.append((
                        self.dataset.item2id[seq[-2]],
                        self.dataset.item2id[seq[-1]],
                    ))
            report = {
                'route_size': route_size,
                'svd_dim': effective_dim,
                'seed': seed,
                'train_edges': len(train_edges),
                'zero_predecessor_items': int(zero_rows.sum()),
                'used_routes': int((counts > 0).sum()),
                'normalized_entropy': float(
                    -(probs * np.log(probs)).sum() / math.log(route_size)
                ),
                'max_bucket_ratio': float(counts.max() / counts.sum()),
                'one_hop_majority_train_accuracy': route_accuracy(train_edges),
                'one_hop_majority_val_accuracy': route_accuracy(val_edges),
                'one_hop_majority_test_accuracy': route_accuracy(test_edges),
            }
            with open(report_path, 'w') as handle:
                json.dump(report, handle, indent=2)
            self.log(f'[TOKENIZER] Behavior route report: {report}')

        routed = {}
        for item, codes in item2sem_ids.items():
            item_id = self.dataset.item2id[item]
            routed[item] = [int(route_codes[item_id]), *[int(code) for code in codes]]
        return routed

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """编码句子嵌入：支持任意 Hugging Face SentenceTransformer 模型 id，并做向量归一化"""
        assert self.config['metadata'] == 'sentence', \
            'DIFF_GRMTokenizer only supports sentence metadata.'

        meta_sentences = []
        for i in range(1, dataset.n_items):
            meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])

        # 接受任意HF模型id（如 Alibaba-NLP/gte-large-en-v1.5 或 BAAI/bge-large-en-v1.5）
        model_id = self.config['sent_emb_model']
        sent_emb_model = SentenceTransformer(model_id, trust_remote_code=True).to(self.config['device'])

        # 直接encode（GTE/BGE无需前缀），并进行L2归一化
        sent_embs = sent_emb_model.encode(
            meta_sentences,
            convert_to_numpy=True,
            batch_size=self.config['sent_emb_batch_size'],
            show_progress_bar=True,
            device=self.config['device'],
            normalize_embeddings=True,
        )

        # 按模型basename分别落盘，避免不同模型冲突
        sent_embs.tofile(output_path)
        return sent_embs

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        """获取训练用的商品"""
        items_for_training = set()
        
        # 首先触发数据集分割（如果还没有分割）
        split_data = dataset.split()
        
        # 从训练集中收集所有items
        if 'train' in split_data:
            train_dataset = split_data['train']
            # train_dataset是Hugging Face Dataset对象
            if hasattr(train_dataset, 'column_names') and 'item_seq' in train_dataset.column_names:
                # 遍历所有item_seq
                for item_seq in train_dataset['item_seq']:
                    if isinstance(item_seq, (list, tuple)):
                        items_for_training.update(item_seq)
                    else:
                        items_for_training.add(item_seq)
        
        # 修复：确保mask大小与sent_embs匹配
        # sent_embs只包含item_id从1到n_items-1的商品
        n_sent_embs = dataset.n_items - 1  # 与_encode_sent_emb中的range(1, dataset.n_items)匹配
        self.log(f'[TOKENIZER] Items for training: {len(items_for_training)} of {n_sent_embs}')
        self.log(f'[TOKENIZER] Training items sample: {list(items_for_training)[:10]}')
        
        mask = np.zeros(n_sent_embs, dtype=bool)
        for item in items_for_training:
            item_id = dataset.item2id[item]
            if 1 <= item_id < dataset.n_items:  # 确保item_id在有效范围内
                mask[item_id - 1] = True  # 转换为0-based索引
        
        self.log(f'[TOKENIZER] Mask shape: {mask.shape}, True count: {np.sum(mask)}')
        return mask

    def _generate_semantic_id_opq(self, sent_embs, sem_ids_path, train_mask):
        """使用OPQ/PQ生成语义ID（兼容 disable_opq），并用 invlists 的 ids 对齐。"""
        import faiss

        # 调试信息
        self.log(f'[TOKENIZER] sent_embs shape: {sent_embs.shape}')
        self.log(f'[TOKENIZER] train_mask shape: {train_mask.shape}')
        self.log(f'[TOKENIZER] train_mask True count: {np.sum(train_mask)}')

        # 构建索引
        if self.config['opq_use_gpu']:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.n_digit >= 56
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
        index = faiss.index_factory(
            sent_embs.shape[1],
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT
        )
        self.log(f'[TOKENIZER] Training index...')
        if self.config['opq_use_gpu']:
            index = faiss.index_cpu_to_gpu(res, self.config['opq_gpu_id'], index, co)
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        if self.config['opq_use_gpu']:
            index = faiss.index_gpu_to_cpu(index)

        # 兼容 IndexPreTransform 与非 PreTransform
        if isinstance(index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(index.index)
        else:
            ivf_index = faiss.downcast_index(index)

        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        # 取 codes 与 ids，并保持同序对齐
        codes_ptr = invlists.get_codes(0)
        ids_ptr = invlists.get_ids(0)
        pq_codes_u8 = faiss.rev_swig_ptr(codes_ptr, ls * invlists.code_size)
        ids = faiss.rev_swig_ptr(ids_ptr, ls).copy()
        pq_codes_u8 = pq_codes_u8.reshape(-1, invlists.code_size)

        # 解析 PQ Code
        faiss_sem_ids = []
        n_bytes = invlists.code_size
        for u8code in pq_codes_u8:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for _ in range(self.quantizer_n_digit):
                code.append(bs.read(self.n_codebook_bits))
            faiss_sem_ids.append(code)

        # 用 ids 对齐 item 顺序。Faiss 的 inverted-list 顺序不保证与 item 顺序一致。
        native_codes = np.full(
            (sent_embs.shape[0], self.quantizer_n_digit), -1, dtype=np.int64
        )
        for pos, iid0 in enumerate(ids):
            native_codes[int(iid0)] = np.asarray(faiss_sem_ids[pos], dtype=np.int64)
        if np.any(native_codes < 0):
            raise RuntimeError("Faiss OPQ/PQ index did not return codes for every catalog item")

        regularizer_report = None
        regularizer_steps = int(self.config.get('opq_regularizer_steps', 0))
        if regularizer_steps > 0:
            if not hasattr(ivf_index, 'pq'):
                raise ValueError('OPQ regularization requires an IVF-PQ index')
            pq_inputs = np.ascontiguousarray(sent_embs.astype(np.float32, copy=False))
            if isinstance(index, faiss.IndexPreTransform):
                for transform_idx in range(index.chain.size()):
                    transform = faiss.downcast_VectorTransform(index.chain.at(transform_idx))
                    pq_inputs = transform.apply_py(pq_inputs)
            if bool(getattr(ivf_index, 'by_residual', False)):
                coarse_centroid = faiss.downcast_index(ivf_index.quantizer).reconstruct(0)
                pq_inputs = pq_inputs - np.asarray(coarse_centroid, dtype=np.float32)[None, :]
            pq = ivf_index.pq
            initial_centroids = faiss.vector_to_array(pq.centroids).reshape(
                pq.M, pq.ksub, pq.dsub
            )
            pq_subvectors = pq_inputs.reshape(
                pq_inputs.shape[0], self.quantizer_n_digit, pq.dsub
            )
            native_before_regularizer = native_codes.copy()
            native_codes, learned_centroids, regularizer_report = (
                regularize_product_centroids(
                    pq_subvectors,
                    initial_centroids,
                    train_mask,
                    steps=regularizer_steps,
                    batch_size=int(self.config.get('opq_regularizer_batch_size', 1024)),
                    learning_rate=float(self.config.get('opq_regularizer_lr', 1e-3)),
                    temperature=float(self.config.get('opq_regularizer_temperature', 0.05)),
                    balance_weight=float(self.config.get('opq_balance_weight', 0.0)),
                    mi_weight=float(self.config.get('opq_mi_weight', 0.0)),
                    hardness_weight=float(self.config.get('opq_hardness_weight', 0.0)),
                    anchor_weight=float(self.config.get('opq_anchor_weight', 0.01)),
                    device=str(self.config.get('opq_regularizer_device', self.config['device'])),
                    seed=int(self.config.get('opq_regularizer_seed', 2026)),
                    straight_through=bool(self.config.get('opq_straight_through', False)),
                )
            )
            regularizer_report['changed_native_code_rate'] = float(
                np.any(native_codes != native_before_regularizer, axis=1).mean()
            )
            centroids_for_repair = learned_centroids
            self.log(
                '[TOKENIZER] Regularized OPQ/PQ centroids: '
                f'changed native code rate={regularizer_report["changed_native_code_rate"]:.4f}'
            )
        else:
            pq_inputs = None
            pq_subvectors = None
            centroids_for_repair = None

        if self.sid_collision_strategy == 'append_dedup':
            final_codes, collision_report = append_dedup_digit(
                native_codes, codebook_size=self.codebook_size
            )
        elif self.sid_collision_strategy == 'hungarian':
            # PQ quantizes the OPQ-rotated residual after the IVF coarse centroid.
            # Reconstruct that exact space so assignment costs match the tokenizer.
            if pq_inputs is None:
                pq_inputs = np.ascontiguousarray(sent_embs.astype(np.float32, copy=False))
                if isinstance(index, faiss.IndexPreTransform):
                    for transform_idx in range(index.chain.size()):
                        transform = faiss.downcast_VectorTransform(index.chain.at(transform_idx))
                        pq_inputs = transform.apply_py(pq_inputs)
                if bool(getattr(ivf_index, 'by_residual', False)):
                    coarse_centroid = faiss.downcast_index(ivf_index.quantizer).reconstruct(0)
                    pq_inputs = pq_inputs - np.asarray(coarse_centroid, dtype=np.float32)[None, :]
                pq = ivf_index.pq
                centroids_for_repair = faiss.vector_to_array(pq.centroids).reshape(
                    pq.M, pq.ksub, pq.dsub
                )
                pq_subvectors = pq_inputs.reshape(
                    pq_inputs.shape[0], self.quantizer_n_digit, pq.dsub
                )
            final_codes, collision_report = repair_product_codes(
                native_codes,
                pq_subvectors=pq_subvectors,
                centroids=centroids_for_repair,
                repair_digit=self.config.get('collision_repair_digit', 'auto'),
            )
        else:
            final_codes = native_codes
            collision_report = {
                'strategy': 'none',
                'before': collision_stats(native_codes),
                'after': collision_stats(native_codes),
            }

        if final_codes.shape[1] != self.n_digit:
            raise RuntimeError(
                f"final SID width {final_codes.shape[1]} != model n_digit {self.n_digit}"
            )

        item2sem_ids = {}
        for iid0, row in enumerate(final_codes):
            item = self.id2item[iid0 + 1]
            item2sem_ids[item] = tuple(int(v) for v in row)

        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)
        with open(f'{sem_ids_path}.collision.json', 'w') as f:
            json.dump(
                {
                    **collision_report,
                    'opq_regularizer': regularizer_report,
                },
                f,
                indent=2,
            )
        self.log(f'[TOKENIZER] Collision report: {collision_report}')

    def _generate_semantic_id_random(self, sem_ids_path, n_items, seed=12345):
        """为每个商品随机生成 n_digit 个 codebook ID（均匀[0, K-1]）。"""
        rng = np.random.default_rng(seed)
        item2sem_ids = {}
        for i in range(1, n_items):
            item = self.id2item[i]
            codes = rng.integers(low=0, high=self.codebook_size, size=self.quantizer_n_digit, endpoint=False, dtype=np.int64)
            item2sem_ids[item] = tuple(int(c) for c in codes.tolist())
        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _generate_semantic_id_rq_kmeans(self, sent_embs, sem_ids_path, train_mask):
        """使用 Residual Quantization（KMeans）生成语义ID。"""
        import faiss
        d = sent_embs.shape[1]
        K = self.codebook_size
        niter = int(self.config.get('rq_kmeans_niters', 20))
        seed = int(self.config.get('rq_kmeans_seed', 1234))

        # 初始化残差为原始向量
        residuals = sent_embs.copy().astype(np.float32, copy=False)
        codes_all = np.zeros((sent_embs.shape[0], self.quantizer_n_digit), dtype=np.int64)
        # Keep each stage's residual and codebook so the same minimum-cost
        # collision repair used by product codes can be applied to RQ as a
        # controlled, fixed-width alternative to an appended identity digit.
        stage_inputs = []
        stage_centroids = []

        for stage in range(self.quantizer_n_digit):
            stage_inputs.append(residuals.copy())
            kmeans = faiss.Kmeans(d=d, k=K, niter=niter, verbose=False, seed=seed + stage)
            kmeans.train(residuals[train_mask])
            # In current Faiss Python, Kmeans.centroids is already a numpy array
            centroids = np.asarray(kmeans.centroids, dtype=np.float32)
            if centroids.ndim == 1:
                centroids = centroids.reshape(K, d)
            elif centroids.shape == (d, K):
                centroids = centroids.T
            assert centroids.shape == (K, d), f"centroids shape {centroids.shape} != {(K, d)}"
            stage_centroids.append(centroids.copy())
            
            # 为全部样本分配最近质心
            index = faiss.IndexFlatL2(d)
            index.add(centroids)
            D, I = index.search(residuals, 1)  # I: [N, 1]
            codes_all[:, stage] = I[:, 0].astype(np.int64)
            
            # 更新残差
            residuals = residuals - centroids[I[:, 0]]
        
        if self.sid_collision_strategy == 'append_dedup':
            codes_all, collision_report = append_dedup_digit(
                codes_all, codebook_size=self.codebook_size
            )
        elif self.sid_collision_strategy == 'hungarian':
            codes_all, collision_report = repair_product_codes(
                codes_all,
                pq_subvectors=np.stack(stage_inputs, axis=1),
                centroids=np.stack(stage_centroids, axis=0),
                repair_digit=self.config.get('collision_repair_digit', 'auto'),
            )
        else:
            collision_report = {
                'strategy': 'none',
                'before': collision_stats(codes_all),
                'after': collision_stats(codes_all),
            }

        # 转成 dict
        item2sem_ids = {}
        for i in range(codes_all.shape[0]):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(int(v) for v in codes_all[i].tolist())
        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)
        with open(f'{sem_ids_path}.collision.json', 'w') as f:
            json.dump(collision_report, f, indent=2)

    def _generate_semantic_id_rq_opq(self, sent_embs, sem_ids_path, train_mask):
        """Generate OneSearch-style RQ prefixes plus OPQ residual digits.

        RQ stages encode coarse-to-fine full-vector residuals.  OPQ/PQ is fit
        only on the final RQ residual and contributes parallel fine-grained
        coordinates.  Quantizer fitting uses the canonical training-item mask;
        all catalog items are encoded afterward.
        """
        import faiss

        vectors = np.ascontiguousarray(sent_embs.astype(np.float32, copy=False))
        residuals = vectors.copy()
        n_items, dimension = residuals.shape
        codebook_size = self.codebook_size
        niter = int(self.config.get('rq_opq_niters', 25))
        seed = int(self.config.get('rq_opq_seed', 2026))
        normalize_residual = bool(
            self.config.get('rq_opq_normalize_residual', False)
        )
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])

        rq_codes = np.zeros((n_items, self.rq_n_digit), dtype=np.int64)
        rq_reports = []
        initial_norm = np.linalg.norm(residuals, axis=1)
        for stage in range(self.rq_n_digit):
            stage_input_norm = np.linalg.norm(residuals, axis=1)
            kmeans = faiss.Kmeans(
                d=dimension,
                k=codebook_size,
                niter=niter,
                verbose=False,
                seed=seed + stage,
                max_points_per_centroid=10000,
            )
            kmeans.train(np.ascontiguousarray(residuals[train_mask]))
            centroids = np.asarray(kmeans.centroids, dtype=np.float32).reshape(
                codebook_size, dimension
            )
            search = faiss.IndexFlatL2(dimension)
            search.add(centroids)
            distances, assignments = search.search(residuals, 1)
            assigned = assignments[:, 0].astype(np.int64)
            rq_codes[:, stage] = assigned
            residuals = residuals - centroids[assigned]
            if normalize_residual:
                norms = np.linalg.norm(residuals, axis=1, keepdims=True)
                residuals = residuals / np.maximum(norms, 1e-8)
            counts = np.bincount(assigned, minlength=codebook_size)
            probs = counts[counts > 0].astype(np.float64) / n_items
            rq_reports.append(
                {
                    'stage': stage,
                    'used_codes': int((counts > 0).sum()),
                    'normalized_entropy': float(
                        -(probs * np.log(probs)).sum() / math.log(codebook_size)
                    ),
                    'mean_input_norm': float(stage_input_norm.mean()),
                    'mean_output_norm': float(
                        np.linalg.norm(residuals, axis=1).mean()
                    ),
                    'mean_squared_assignment_error': float(distances.mean()),
                }
            )

        opq_input = np.ascontiguousarray(residuals.astype(np.float32, copy=False))
        opq_factory = (
            f'OPQ{self.opq_n_digit},IVF1,'
            f'PQ{self.opq_n_digit}x{self.n_codebook_bits}'
        )
        index = faiss.index_factory(
            dimension, opq_factory, faiss.METRIC_INNER_PRODUCT
        )
        self.log(
            '[TOKENIZER] Training residual OPQ after RQ: '
            f'rq_digits={self.rq_n_digit}, opq_digits={self.opq_n_digit}'
        )
        index.train(np.ascontiguousarray(opq_input[train_mask]))
        index.add(opq_input)

        ivf_index = faiss.downcast_index(
            index.index if isinstance(index, faiss.IndexPreTransform) else index
        )
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        list_size = invlists.list_size(0)
        raw_codes = faiss.rev_swig_ptr(
            invlists.get_codes(0), list_size * invlists.code_size
        ).reshape(-1, invlists.code_size)
        ids = faiss.rev_swig_ptr(invlists.get_ids(0), list_size).copy()
        opq_codes = np.full((n_items, self.opq_n_digit), -1, dtype=np.int64)
        for position, packed in enumerate(raw_codes):
            reader = faiss.BitstringReader(
                faiss.swig_ptr(packed), invlists.code_size
            )
            opq_codes[int(ids[position])] = [
                reader.read(self.n_codebook_bits)
                for _ in range(self.opq_n_digit)
            ]
        if np.any(opq_codes < 0):
            raise RuntimeError('Residual OPQ did not encode every catalog item')

        # Recover the exact rotated PQ space used for assignment and collision
        # repair on an OPQ residual digit.
        pq_inputs = opq_input
        if isinstance(index, faiss.IndexPreTransform):
            for transform_idx in range(index.chain.size()):
                transform = faiss.downcast_VectorTransform(
                    index.chain.at(transform_idx)
                )
                pq_inputs = transform.apply_py(pq_inputs)
        if bool(getattr(ivf_index, 'by_residual', False)):
            coarse = faiss.downcast_index(ivf_index.quantizer).reconstruct(0)
            pq_inputs = pq_inputs - np.asarray(coarse, dtype=np.float32)[None]
        pq = ivf_index.pq
        opq_centroids = faiss.vector_to_array(pq.centroids).reshape(
            pq.M, pq.ksub, pq.dsub
        )
        opq_subvectors = pq_inputs.reshape(n_items, pq.M, pq.dsub)

        native_codes = np.concatenate([rq_codes, opq_codes], axis=1)
        if self.sid_collision_strategy == 'append_dedup':
            final_codes, collision_report = append_dedup_digit(
                native_codes, codebook_size=codebook_size
            )
        elif self.sid_collision_strategy == 'hungarian':
            local_digit = int(
                self.config.get('rq_opq_repair_opq_digit', self.opq_n_digit - 1)
            )
            if local_digit < 0:
                local_digit += self.opq_n_digit
            if not 0 <= local_digit < self.opq_n_digit:
                raise ValueError(
                    'rq_opq_repair_opq_digit must index an OPQ residual digit'
                )
            global_digit = self.rq_n_digit + local_digit
            final_codes, collision_report = repair_code_digit(
                native_codes,
                digit_vectors=opq_subvectors[:, local_digit],
                digit_centroids=opq_centroids[local_digit],
                repair_digit=global_digit,
            )
        else:
            final_codes = native_codes
            collision_report = {
                'strategy': 'none',
                'before': collision_stats(native_codes),
                'after': collision_stats(native_codes),
            }

        if final_codes.shape[1] != self.n_digit:
            raise RuntimeError(
                f'RQ-OPQ SID width {final_codes.shape[1]} != n_digit={self.n_digit}'
            )
        reconstruction = np.stack(
            [
                opq_centroids[digit, opq_codes[:, digit]]
                for digit in range(self.opq_n_digit)
            ],
            axis=1,
        ).reshape(n_items, -1)
        quantizer_report = {
            'quantizer': 'rq_opq',
            'rq_digits': self.rq_n_digit,
            'opq_digits': self.opq_n_digit,
            'codebook_size': codebook_size,
            'rq_stages': rq_reports,
            'mean_initial_norm': float(initial_norm.mean()),
            'mean_final_rq_residual_norm': float(
                np.linalg.norm(opq_input, axis=1).mean()
            ),
            'opq_rotated_residual_mse': float(
                np.mean((pq_inputs - reconstruction) ** 2)
            ),
        }
        item2sem_ids = {
            self.id2item[index + 1]: tuple(int(value) for value in row)
            for index, row in enumerate(final_codes)
        }
        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, 'w') as handle:
            json.dump(item2sem_ids, handle)
        with open(f'{sem_ids_path}.collision.json', 'w') as handle:
            json.dump(
                {**collision_report, 'rq_opq': quantizer_report},
                handle,
                indent=2,
            )
        self.log(f'[TOKENIZER] RQ-OPQ report: {quantizer_report}')
        self.log(f'[TOKENIZER] Collision report: {collision_report}')

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        """将语义ID转换为token"""
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            # 修复：重新引入offset，避免与PAD/BOS冲突
            # 每个digit的codebook ID加上对应的offset
            tokens = [t + self.sid_offset + d * self.codebook_size 
                     for d, t in enumerate(tokens)]
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids

    def _init_tokenizer(self, dataset: AbstractDataset):
        """初始化tokenizer"""
        # 构建路径 - 修复：使用类名和category
        dataset_name = dataset.__class__.__name__  # 使用类名，如"AmazonReviews2014"
        
        # 如果有category属性，加入路径中
        if hasattr(dataset, 'category') and dataset.category:
            cache_dir = os.path.join(
                dataset.cache_dir, 'processed'
            )
        else:
            cache_dir = os.path.join(
                'data', dataset_name, 'processed'
            )
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)

        # 加载语义ID（在文件名中加入 PCA 维度与量化器标签，避免配置冲突）
        model_basename = self._embedding_cache_basename()
        quant_tag = self._quant_tag()
        sem_ids_path = os.path.join(
            cache_dir,
            f'{model_basename}_pca{self.config["sent_emb_pca"]}_{quant_tag}.sem_ids'
        )

        # 🚀 新增：检查是否需要强制重新生成量化结果
        force_regenerate = self.config.get('force_regenerate_opq', False)
        
        # 两份嵌入文件：raw 和 pca 版本，避免命名歧义与冲突
        model_basename = self._embedding_cache_basename()
        raw_path = os.path.join(
            cache_dir,
            f'{model_basename}_raw_d{self.config["sent_emb_dim"]}.sent_emb'
        )
        pca_path = os.path.join(
            cache_dir,
            f'{model_basename}_pca{self.config["sent_emb_pca"]}.sent_emb'
        )

        # 如果量化器需要句向量，则准备句向量；none 模式无需
        sent_embs = None
        if self.sid_quantizer in ('opq_pq', 'rq_opq'):
            # OPQ and RQ-OPQ share the exact PCA-normalized embedding view so
            # tokenizer comparisons do not silently change item features.
            if self.config['sent_emb_pca'] > 0 and os.path.exists(pca_path):
                self.log(f'[TOKENIZER] Loading PCA-ed sentence embeddings from {pca_path}...')
                sent_embs = np.fromfile(pca_path, dtype=np.float32).reshape(
                    -1, self.config['sent_emb_pca']
                )
            elif os.path.exists(raw_path):
                self.log(f'[TOKENIZER] Loading RAW sentence embeddings from {raw_path}...')
                raw_embs = np.fromfile(raw_path, dtype=np.float32).reshape(
                    -1, self.config['sent_emb_dim']
                )
                if self.config['sent_emb_pca'] > 0:
                    self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                    training_item_mask = self._get_items_for_training(dataset)
                    pca.fit(raw_embs[training_item_mask])
                    sent_embs = pca.transform(raw_embs)
                    sent_embs = sent_embs.astype(np.float32, copy=False)
                    if self.config.get('normalize_after_pca', True):
                        norms = np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-12
                        sent_embs = sent_embs / norms
                    sent_embs.tofile(pca_path)
                else:
                    sent_embs = raw_embs
            else:
                self.log(f'[TOKENIZER] Encoding sentence embeddings...')
                raw_embs = self._encode_sent_emb(dataset, raw_path)
                if self.config['sent_emb_pca'] > 0:
                    self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                    training_item_mask = self._get_items_for_training(dataset)
                    pca.fit(raw_embs[training_item_mask])
                    sent_embs = pca.transform(raw_embs)
                    sent_embs = sent_embs.astype(np.float32, copy=False)
                    if self.config.get('normalize_after_pca', True):
                        norms = np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-12
                        sent_embs = sent_embs / norms
                    sent_embs.tofile(pca_path)
                else:
                    sent_embs = raw_embs
            self.log(f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')
        elif self.sid_quantizer == 'rq_kmeans':
            # rq_kmeans：不进行 PCA，直接使用 RAW（满足你的需求）
            if os.path.exists(raw_path):
                self.log(f'[TOKENIZER] Loading RAW sentence embeddings from {raw_path}...')
                sent_embs = np.fromfile(raw_path, dtype=np.float32).reshape(
                    -1, self.config['sent_emb_dim']
                )
            else:
                self.log(f'[TOKENIZER] Encoding sentence embeddings (RAW, no PCA for RQ-KMeans)...')
                sent_embs = self._encode_sent_emb(dataset, raw_path)
            self.log(f'[TOKENIZER] Sentence embeddings shape (RAW): {sent_embs.shape}')

        sid_override_path = self.config.get('sid_override_path')
        if sid_override_path and not os.path.isfile(sid_override_path):
            raise FileNotFoundError(
                f'sid_override_path does not exist: {sid_override_path}'
            )

        # 🚀 生成或加载量化结果.  An explicit override is
        # already the complete quantizer output, so do not train an unrelated
        # fallback quantizer merely because its cache is absent.
        if not sid_override_path and (force_regenerate or not os.path.exists(sem_ids_path)):
            if force_regenerate:
                self.log(f'[TOKENIZER] Force regenerating quantization results ({self.sid_quantizer})...')
            else:
                self.log(f'[TOKENIZER] Quantization results not found, generating ({self.sid_quantizer})...')
            training_item_mask = self._get_items_for_training(dataset)
            if self.sid_quantizer == 'opq_pq':
                self._generate_semantic_id_opq(sent_embs, sem_ids_path, training_item_mask)
            elif self.sid_quantizer == 'rq_opq':
                self._generate_semantic_id_rq_opq(
                    sent_embs, sem_ids_path, training_item_mask
                )
            elif self.sid_quantizer == 'rq_kmeans':
                self._generate_semantic_id_rq_kmeans(sent_embs, sem_ids_path, training_item_mask)
            else:  # 'none'
                self._generate_semantic_id_random(
                    sem_ids_path, n_items=self.dataset.n_items,
                    seed=int(self.config.get('sid_random_seed', 12345))
                )
        elif not sid_override_path:
            self.log(f'[TOKENIZER] Using existing quantization results from {sem_ids_path}')
        else:
            self.log(
                f'[TOKENIZER] Skipping native quantizer; using SID override '
                f'{sid_override_path}'
            )

        load_sem_ids_path = sid_override_path or sem_ids_path
        self.log(f'[TOKENIZER] Loading semantic IDs from {load_sem_ids_path}...')
        with open(load_sem_ids_path, 'r') as sem_ids_file:
            item2sem_ids = json.load(sem_ids_file)
        if sid_override_path:
            expected_items = {
                item for item, item_id in self.dataset.item2id.items()
                if int(item_id) > 0
            }
            actual_items = set(item2sem_ids)
            if expected_items != actual_items:
                missing = list(expected_items - actual_items)[:5]
                extra = list(actual_items - expected_items)[:5]
                raise ValueError(
                    f'sid_override_path catalog mismatch: missing={missing}, extra={extra}'
                )
            override_codes = np.asarray(list(item2sem_ids.values()), dtype=np.int64)
            if override_codes.shape != (len(expected_items), self.quantizer_n_digit):
                raise ValueError(
                    'sid_override_path must contain one quantizer-width SID per item, '
                    f'got {override_codes.shape}'
                )
            if len(np.unique(override_codes, axis=0)) != len(override_codes):
                raise ValueError('sid_override_path must be collision free')
            self.log(
                f'[TOKENIZER] Loaded collision-free SID override '
                f'({len(override_codes)} items)'
            )
        if self.sid_prefix_strategy == 'behavior_route':
            item2sem_ids = self._prepend_behavior_route(item2sem_ids, cache_dir)
        elif self.sid_prefix_strategy == 'random_latent':
            item2sem_ids = self._prepend_random_latent(item2sem_ids)
        self._load_item_aliases(item2sem_ids)
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)

        # 🚀 映射文件名：复用前面构造的 quant_tag
        map_tag = self._mapping_tag(model_basename)
        fwd_path = os.path.join(cache_dir, f'item_id2tokens_{map_tag}.npy')
        inv_path = os.path.join(cache_dir, f'tokens2item_{map_tag}.pkl')
        
        # 🚀 修复①：处理映射文件的一致性
        if force_regenerate:
            # 强制重新生成时，直接忽略旧文件，让下面逻辑走"重新保存"
            fwd_exists = inv_exists = False
            self.log(f'[TOKENIZER] Force regenerate enabled, ignoring existing mapping files')
        else:
            fwd_exists = os.path.exists(fwd_path)
            inv_exists = os.path.exists(inv_path)
        
        if fwd_exists and inv_exists:
            # ---------- ① 文件已存在 ----------
            self.log(f'[TOKENIZER] Loading existing mappings for tag: {map_tag} from {fwd_path}')
            
            # 重新构建item2tokens映射
            item_id2tokens = np.load(fwd_path)
            item2tokens = {}
            for iid, toks in enumerate(item_id2tokens):
                if iid == 0:  # PAD行全0，跳过
                    continue
                item2tokens[self.id2item[iid]] = tuple(toks.tolist())
            
            # 加载倒排索引
            with open(inv_path, 'rb') as f:
                self.tokens2item = pickle.load(f)
                
            self.log(f'[TOKENIZER] Successfully loaded {len(item2tokens)} item mappings')
        else:
            # ---------- ② 文件不存在或强制重新生成，需要重新生成 ----------
            if force_regenerate:
                self.log(f'[TOKENIZER] Force regenerate enabled, generating new mappings')
            else:
                self.log(f'[TOKENIZER] No existing mappings found for {self.n_digit}-digit, will generate new ones')
            
            # 无论是文件不存在还是 forceRegenerate，都按新的 item2tokens 保存
            self.item2tokens = item2tokens
            self.tokens2item = self._create_reverse_mapping()
            self._save_mappings()  # 只在"新建"时真正落盘

        # ---- ③ 统一：把映射挂到实例属性再返回 ----
        # 注意：在"文件已存在"分支中，需要设置self.item2tokens
        if not hasattr(self, 'item2tokens'):
            self.item2tokens = item2tokens
        return item2tokens

    def _create_reverse_mapping(self):
        """创建反向映射用于推理"""
        tokens2item = {}
        for item, tokens in self.item2tokens.items():
            item_id = self.dataset.item2id[item]
            for path in self.iter_item_token_paths(item, tokens):
                previous = tokens2item.get(tuple(path))
                if previous is not None and previous != item_id:
                    raise ValueError(
                        f'Non-unique item path {tuple(path)} maps to {previous} and {item_id}'
                    )
                tokens2item[tuple(path)] = item_id
        return tokens2item

    def _save_mappings(self):
        """保存映射文件"""
        # 构建路径 - 修复：使用类名和category
        dataset_name = self.dataset.__class__.__name__  # 使用类名，如"AmazonReviews2014"
        
        # 如果有category属性，加入路径中
        if hasattr(self.dataset, 'category') and self.dataset.category:
            cache_dir = os.path.join(
                self.dataset.cache_dir, 'processed'
            )
        else:
            cache_dir = os.path.join(
                'data', dataset_name, 'processed'
            )
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # 🚀 文件名包含：模型+PCA+量化器标签(+种子/iters)+n_digit，完全避免不同配置冲突
        model_basename = self._embedding_cache_basename()
        map_tag = self._mapping_tag(model_basename)
        
        # 保存正排索引：item_id → SID-tokens
        item_id2tokens = np.zeros((self.dataset.n_items, self.n_digit), dtype=np.int64)
        for item, tokens in self.item2tokens.items():
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = np.array(tokens)
        
        np.save(os.path.join(cache_dir, f'item_id2tokens_{map_tag}.npy'), item_id2tokens)
        
        # 保存倒排索引：SID-tokens → item_id
        with open(os.path.join(cache_dir, f'tokens2item_{map_tag}.pkl'), 'wb') as f:
            pickle.dump(self.tokens2item, f)
        
        self.log(f'[TOKENIZER] Saved mappings with tag: {map_tag} to {cache_dir}')
        self.log(f'[TOKENIZER] Files: item_id2tokens_{map_tag}.npy, tokens2item_{map_tag}.pkl')

    def encode_history(self, item_seq, max_len=None):
        """编码用户历史序列"""
        if max_len is None:
            max_len = self.config.get('max_history_len', 20)
        if len(item_seq) > max_len:
            item_seq = item_seq[-max_len:]
        
        history_sid = []
        for item in item_seq:
            if item in self.item2tokens:
                # 将带offset的token ID转换为codebook ID (0..K-1)
                tokens = list(self.item2tokens[item])  # 带offset的token ID
                codebook_ids = []
                for digit, token_id in enumerate(tokens):
                    codebook_id = token_id - (self.sid_offset + digit * self.codebook_size)
                    codebook_ids.append(codebook_id)
                history_sid.append(codebook_ids)
            else:
                # 未知商品用PAD填充（使用-1作为PAD的哨兵值，避免与codebook_id=0混淆）
                history_sid.append([-1] * self.n_digit)
        
        # 填充到固定长度
        while len(history_sid) < max_len:
            history_sid.append([-1] * self.n_digit)
        
        return history_sid  # 返回list，让datasets.map自动张量化
    
    def encode_history_with_mask(self, item_seq, max_len=None):
        """编码用户历史序列，同时返回padding mask"""
        if max_len is None:
            max_len = self.config.get('max_history_len', 20)
        if len(item_seq) > max_len:
            item_seq = item_seq[-max_len:]
        
        history_sid = []
        history_mask = []  # True=有效位置，False=PAD位置
        
        for item in item_seq:
            if item in self.item2tokens:
                # 将带offset的token ID转换为codebook ID (0..K-1)
                tokens = list(self.item2tokens[item])  # 带offset的token ID
                codebook_ids = []
                for digit, token_id in enumerate(tokens):
                    codebook_id = token_id - (self.sid_offset + digit * self.codebook_size)
                    codebook_ids.append(codebook_id)
                history_sid.append(codebook_ids)
                history_mask.append(True)  # 有效位置
            else:
                # 未知商品用PAD填充（使用-1作为PAD的哨兵值，避免与codebook_id=0混淆）
                history_sid.append([-1] * self.n_digit)
                history_mask.append(False)  # PAD位置
        
        # 填充到固定长度
        while len(history_sid) < max_len:
            history_sid.append([-1] * self.n_digit)
            history_mask.append(False)  # PAD位置
        
        return history_sid, history_mask  # 返回list，让datasets.map自动张量化

    def encode_decoder_input(self, target_item):
        """编码decoder输入 - 与RPG_ED保持一致"""
        if target_item in self.item2tokens:
            tokens = list(self.item2tokens[target_item])  # 4个token ID（带offset）
            
            # 将token ID转换为codebook ID
            codebook_tokens = []
            for digit, token_id in enumerate(tokens):
                codebook_id = token_id - (self.sid_offset + digit * self.codebook_size)
                codebook_tokens.append(codebook_id)
            
            # decoder输入和标签都是codebook IDs
            decoder_input = codebook_tokens  # [cb0, cb1, cb2, cb3]
            decoder_labels = codebook_tokens  # [cb0, cb1, cb2, cb3]
        else:
            # 未知商品
            decoder_input = [self.pad_token] * self.n_digit  # 长度n_digit
            decoder_labels = [self.pad_token] * self.n_digit  # 长度n_digit
        
        return decoder_input, decoder_labels

    def decode_tokens_to_item(self, tokens):
        """将token序列解码为商品ID"""
        if len(tokens) != self.n_digit:
            return None
        
        token_tuple = tuple(tokens)
        return self.tokens2item.get(token_tuple)

    def codebooks_to_item_id(self, cb_ids):
        """
        将codebook ID序列转换为item_id，检查合法性
        
        Args:
            cb_ids: List[int] 长度 n_digit, 原始 codebook ID (0-255)
            
        Returns:
            item_id(int) 或 None（如果非法）
        """
        if len(cb_ids) != self.n_digit:
            return None
        
        # 将codebook ID转换为token ID
        token_ids = [
            cb_ids[d] + self.sid_offset + d * self.codebook_size
            for d in range(self.n_digit)
        ]
        
        # 查找对应的item_id
        return self.tokens2item.get(tuple(token_ids))

    def tokenize_function(self, example: dict, split: str) -> dict:
        """tokenize函数 - 修复数据泄露问题"""
        item_seq = example['item_seq']  # Python list
        target_item = item_seq[-1]  # 原始字符串
        
        # 修复：所有split都应该用item_seq[:-1]作为历史，避免数据泄露
        history_sid, history_mask = self.encode_history_with_mask(item_seq[:-1])
        
        if split == 'train':
            # 训练时编码decoder输入
            decoder_input, decoder_labels = self.encode_decoder_input(target_item)
            return {
                'history_sid': history_sid,  # 直接list
                'history_mask': history_mask,  # 直接list
                'decoder_input_ids': decoder_input,  # 直接list
                'decoder_labels': decoder_labels  # 直接list
            }
        else:
            # 验证/测试时生成真标签
            _, decoder_labels = self.encode_decoder_input(target_item)
            return {
                'history_sid': history_sid,  # 直接list
                'history_mask': history_mask,  # 直接list
                'labels': decoder_labels  # 新增：真标签序列
            }

    def tokenize(self, datasets: dict) -> dict:
        """tokenize数据集"""
        tokenized_datasets = {}
        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=False,  # 关闭批处理，避免数据结构混乱
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: '
            )

        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')

        return tokenized_datasets 

    # ====== 新增：SID→items 映射与工具 ======
    def _sid_tokens_to_cb_tuple(self, tokens):
        """
        把带 offset 的 SID-token（长度 n_digit）转换为 codebook 索引 tuple（每位 0..K-1）。
        例如 [sid_offset + 0*K + a, sid_offset + 1*K + b, ...] → (a,b,...)
        """
        assert len(tokens) == self.n_digit
        cb = []
        for d, tok in enumerate(tokens):
            cb.append(int(tok) - (self.sid_offset + d * self.codebook_size))
        return tuple(cb)

    def _build_cb2items_map(self):
        """
        基于 self.item2tokens 构建 SID 组合 → items 的倒排表。
        注意：允许一对多（发生“冲突”时都放进去）。
        """
        from collections import defaultdict
        cb2items = defaultdict(list)
        for item, toks in self.item2tokens.items():
            for path in self.iter_item_token_paths(item, toks):
                cb = self._sid_tokens_to_cb_tuple(path)
                cb2items[cb].append(item)
        return cb2items

    @property
    def cb2items(self):
        """
        惰性缓存：首次访问时构建 SID→items 映射并缓存到 _cb2items
        """
        if not hasattr(self, "_cb2items") or self._cb2items is None:
            self._cb2items = self._build_cb2items_map()
        return self._cb2items

    def cb_tuple_to_item_ids(self, cb):
        """
        给定一个 codebook tuple，返回对应的 item_id 列表（按构建顺序稳定）。
        """
        items = self.cb2items.get(cb, [])
        out = []
        for it in items:
            iid = self.item2id.get(it, 0)
            if iid > 0:
                out.append(iid)
        return out
