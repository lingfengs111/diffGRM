# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np


class AR_GRMEvaluator:
    """
    AR_GRM 模型的评估器（顺序自回归 beam-search 输出）
    """
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'hit': self.hit_at_k,
            'ndcg': self.ndcg_at_k
        }

        self.pad_token = self.tokenizer.pad_token
        self.maxk = max(config['topk'])
        self.item_identity_start_digit = int(
            config.get('item_identity_start_digit', 0)
        )
        
        # 仅统计每个 batch 的 Top-10 合法率
        self.batch_legal_at10 = []
        
        # 调试信息：确认使用了DIFF_GRMEvaluator
        print(f'>> Using evaluator = {self.__class__.__name__} (AR beam)')
        print(f'>> Recall: any() 按首次命中计分')
        print(f'>> NDCG: first-hit-only 计算')



    def calculate_pos_index(self, preds, labels):
        """
        计算预测结果与真标签的匹配情况（beam search已保证合法性）
        
        Args:
            preds: (batch_size, maxk, n_digit) - 生成的SID序列（已过滤为合法）
            labels: (batch_size, n_digit) - 真标签序列
        
        Returns:
            pos_index: (batch_size, maxk) - 每个位置是否匹配真标签
        """
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        
        B, maxk, n_digit = preds.shape
        
        # 🚀 简化：beam search已保证返回的序列都是合法的
        # 直接计算匹配情况，无需再过滤
        pos_index = torch.zeros((B, maxk), dtype=torch.bool)
        
        for i in range(B):
            # 获取真标签
            cur_label = labels[i, self.item_identity_start_digit:].tolist()
            target_item_id = None
            if self.tokenizer.has_item_aliases:
                # PIT aliases are different legal paths to one concrete item.
                # Evaluation must therefore compare item identities, never the
                # literal canonical label path.
                target_item_id = self.tokenizer.codebooks_to_item_id(
                    labels[i].tolist()
                )
                if target_item_id is None:
                    raise ValueError(
                        f'Ground-truth SID is absent from item mapping: {labels[i].tolist()}'
                    )
            
            for j in range(maxk):
                # 获取预测序列
                cur_pred = preds[i, j, self.item_identity_start_digit:].tolist()
                
                # 比较codebook IDs（0-255范围）
                if self.tokenizer.has_item_aliases:
                    is_match = (
                        self.tokenizer.codebooks_to_item_id(preds[i, j].tolist())
                        == target_item_id
                    )
                else:
                    is_match = cur_pred == cur_label
                if is_match:
                    pos_index[i, j] = True
                    break  # 找到第一个匹配就停止，避免重复计分
        
        return pos_index

    def recall_at_k(self, pos_index, k):
        """计算Recall@k（修复：重复命中只计1分）"""
        # pos_index: (batch_size, maxk) - 已经过滤为合法序列
        # 修复：使用any()避免重复计分，只要top-k中有≥1个匹配就得1分
        return pos_index[:, :k].any(dim=1).cpu().float()

    def hit_at_k(self, pos_index, k):
        """单目标 next-item 评测下，Hit@k 与 Recall@k 完全相同。"""
        return self.recall_at_k(pos_index, k)

    def ndcg_at_k(self, pos_index, k):
        """计算NDCG@k（修复：重复命中只计第一次的DCG）"""
        # pos_index: (batch_size, maxk) - 已经过滤为合法序列
        batch_size, maxk = pos_index.shape
        device = pos_index.device
        
        # 创建rank权重：1/log2(rank+1)
        ranks = torch.arange(1, maxk + 1, device=device).float()
        dcg_weights = 1.0 / torch.log2(ranks + 1)
        
        # 修复：只计算第一个匹配位置的DCG，避免重复计分（向量化实现）
        # 创建位置索引矩阵
        position_matrix = torch.arange(maxk, device=device).expand(batch_size, -1)
        
        # 找到每个样本第一个匹配的位置（向量化）
        # 将非匹配位置设为一个很大的数，这样min操作会忽略它们
        masked_positions = torch.where(pos_index, position_matrix, torch.full_like(position_matrix, maxk))
        first_hit_positions = masked_positions.min(dim=1).values  # [batch_size]
        
        # 计算DCG：只有在top-k内且有匹配时才得分
        # 修复索引越界问题：确保only真正有匹配且在top-k内的样本才计分
        has_hit = first_hit_positions < maxk  # 是否有匹配
        in_topk = first_hit_positions < k  # 匹配是否在top-k内
        valid_mask = has_hit & in_topk  # 既有匹配又在top-k内
        
        # 安全的索引访问：先限制索引范围，再计算得分
        safe_positions = torch.clamp(first_hit_positions, 0, maxk - 1)
        dcg_scores = torch.where(valid_mask, dcg_weights[safe_positions], torch.tensor(0.0, device=device))
        
        # 对于单标签推荐任务，IDCG=1.0，所以DCG就是NDCG
        return dcg_scores.cpu().float()

    def _dup_ratio_per_user(self, preds, k=10):
        """
        计算一个 batch 内"用户内部"的重复率。
        preds: [B, maxk, n_digit]（已保证 maxk ≥ k）
        返回:  [B] 每个用户自己的重复率
        """
        B, _, n_digit = preds.shape
        dup_ratios = []

        for b in range(B):
            # 仅取前 k 条
            seqs = preds[b, :k]                       # [k, n_digit]
            k_seqs = [tuple(s.tolist()) for s in seqs]
            unique_cnt = len(set(k_seqs))
            dup_ratios.append(1 - unique_cnt / k)     # in [0,1]

        return torch.tensor(dup_ratios, dtype=torch.float32)

    def calculate_weighted_score(self, preds, labels):
        """
        计算加权综合分数：NDCG@10 * 0.8 + RECALL@10 * 0.2
        
        Args:
            preds: (batch_size, beam_size, n_digit) - 生成的SID序列
            labels: (batch_size, n_digit) - 真标签序列
        
        Returns:
            weighted_score: 加权综合分数
        """
        pos_index = self.calculate_pos_index(preds, labels)
        
        # 计算NDCG@10
        ndcg_10 = self.ndcg_at_k(pos_index, k=10)
        
        # 计算RECALL@10
        recall_10 = self.recall_at_k(pos_index, k=10)
        
        # 计算加权分数：NDCG@10 * 0.8 + RECALL@10 * 0.2
        weighted_score = 0.8 * ndcg_10 + 0.2 * recall_10
        
        return weighted_score

    def calculate_metrics(self, preds, labels, suffix=""):
        """计算所有指标"""
        results = {}
        pos_index = self.calculate_pos_index(preds, labels)
        
        # ---------- 仅统计 Top-10 合法率 ----------
        B, maxk, n_digit = preds.shape
        K = min(10, maxk)
        legal_mask = []
        for b in range(B):
            cnt = 0
            for j in range(K):
                if self.tokenizer.codebooks_to_item_id(preds[b, j].tolist()) is not None:
                    cnt += 1
            legal_mask.append(cnt / float(K))
        legal_at10 = torch.tensor(legal_mask, dtype=torch.float32)
        results['legal@10'] = legal_at10
        self.batch_legal_at10.append(legal_at10.mean().item())
        
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}{suffix}"] = self.metric2func[metric](pos_index, k)

        # 始终显式输出 Hit@K，便于与报告 Hit Rate 的工作直接比较。
        for k in self.config['topk']:
            results.setdefault(f"hit@{k}{suffix}", self.hit_at_k(pos_index, k))
        
        # 添加加权综合分数（只在confidence模式下计算）
        if suffix == "":  # 仅confidence模式才算weighted_score
            weighted_score = self.calculate_weighted_score(preds, labels)
            results['weighted_score'] = weighted_score
        
        return results
    
    def print_final_stats(self):
        """打印最终统计结果（仅 Top-10 合法率）"""
        if self.batch_legal_at10:
            avg_legal_at10 = sum(self.batch_legal_at10) / len(self.batch_legal_at10)
            print(f"[SID_STATS] Top-10 合法率: {avg_legal_at10:.3f}")
