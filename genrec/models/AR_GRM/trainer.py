# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
from collections import defaultdict, OrderedDict

from genrec.utils import get_total_steps, get_file_name, config_for_log
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.diagnostics import (
    catalog_diagnostics,
    conditional_diagnostics,
    generation_diagnostics,
)


class AR_GRMTrainer:
    """
    AR_GRM 模型训练器：标准 next-token 自回归训练与顺序 beam-search 验证
    """

    def __init__(self, config: dict, model: AbstractModel, tokenizer: AbstractTokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        # Use the same Accelerator created by Pipeline so logging / is_main_process stay consistent
        self.accelerator = config.get('accelerator', None) or Accelerator()
        
        # 设置保存路径
        self.saved_model_ckpt = os.path.join(
            'saved', get_file_name(config), 'pytorch_model.bin'
        )
        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)

    def _select_pit_alias_targets(self, batch, epoch):
        """Hard-EM target step: choose the highest-scoring path per history/item."""
        strategy = str(self.config.get('pit_alias_selection', 'none')).lower()
        if strategy == 'none':
            return batch, None
        if strategy != 'min_nll':
            raise ValueError(
                f'pit_alias_selection must be none|min_nll, got {strategy}'
            )
        if not self.tokenizer.has_item_aliases:
            raise ValueError('pit_alias_selection=min_nll requires sid_alias_path')

        candidates = self.tokenizer.alias_codebook_candidates(
            batch['decoder_labels']
        )
        warmup_epochs = int(self.config.get('pit_alias_warmup_epochs', 0))
        if epoch < warmup_epochs:
            # A canonical checkpoint has never seen the new branch and would
            # immediately collapse hard-EM to path 0. Balanced exposure gives
            # every legal path a learned score before assignments become hard.
            best_index = torch.randint(
                candidates.shape[1],
                (candidates.shape[0],),
                device=candidates.device,
            )
            scores = None
        else:
            was_training = self.model.training
            self.model.eval()
            score_model = self.accelerator.unwrap_model(self.model)
            with torch.no_grad():
                scores = score_model.score_candidate_paths(
                    batch,
                    candidates,
                    chunk_size=self.config.get('pit_score_chunk_size'),
                )
                best_index = scores.argmax(dim=1)
            if was_training:
                self.model.train()

            exploration = float(self.config.get('pit_alias_exploration_rate', 0.0))
            if exploration > 0:
                explore = torch.rand(
                    best_index.shape, device=best_index.device
                ).lt(exploration)
                random_index = torch.randint(
                    candidates.shape[1],
                    best_index.shape,
                    device=best_index.device,
                )
                best_index = torch.where(explore, random_index, best_index)

        row = torch.arange(candidates.shape[0], device=candidates.device)
        selected = candidates[row, best_index]

        selected_batch = dict(batch)
        selected_batch['decoder_input_ids'] = selected
        selected_batch['decoder_labels'] = selected
        stats = {
            'n': int(best_index.numel()),
            'canonical': int(best_index.eq(0).sum().item()),
            'score_gain': (
                float((scores[row, best_index] - scores[:, 0]).sum().item())
                if scores is not None else 0.0
            ),
            'warmup': scores is None,
        }
        return selected_batch, stats

    def fit(self, train_dataloader, val_dataloader):
        """标准训练流程（自回归损失在 model.forward 内实现）"""
        trainable_parameters = [
            parameter for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError('AR_GRM has no trainable parameters')
        self.log(
            '[TRAINING] Trainable parameters: '
            f'{sum(parameter.numel() for parameter in trainable_parameters):,}'
        )
        optimizer = AdamW(
            trainable_parameters,
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        # 先不计算 num_training_steps，先 prepare
        self.model, optimizer, train_dataloader, val_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader
        )

        # 每个进程"看见"的 steps/epoch
        steps_per_epoch = len(train_dataloader)

        # 正确的总步数（以"每个进程"视角）：epochs * steps_per_epoch
        num_training_steps = self.config['epochs'] * steps_per_epoch

        if num_training_steps == 0:
            self.log('No training steps needed.')
            return None, None

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=num_training_steps,
        )

        scheduler = self.accelerator.prepare(scheduler)
        
        self.accelerator.init_trackers(
            project_name=get_file_name(self.config, suffix=''),
            config=config_for_log(self.config),
            init_kwargs={"tensorboard": {"flush_secs": 60}},
        )

        # 训练轮数直接用配置值，不要再除以进程数
        n_epochs = int(self.config['epochs'])
        best_epoch = 0
        best_val_score = -1
        
        # 新增：跟踪评估次数和无提升的评估次数
        eval_count = 0
        no_improve_count = 0
        
        # 显示评估配置
        eval_start_epoch = self.config.get('eval_start_epoch', 1)
        eval_interval = self.config['eval_interval']
        self.log(f'[TRAINING] Evaluation config: start from epoch {eval_start_epoch}, interval: {eval_interval}')
        if str(self.config.get('pit_alias_selection', 'none')).lower() != 'none':
            self.log(
                '[TRAINING] PIT-lite hard target selection enabled: '
                f'{self.config.get("pit_alias_selection")}, '
                f'item aggregation={self.config.get("item_path_aggregation", "logsumexp")}'
            )

        for epoch in range(n_epochs):
            # Training
            self.model.train()
            total_loss = 0.0
            pit_examples = 0
            pit_canonical = 0
            pit_score_gain = 0.0
            pit_warmup = False
            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
            )
            
            for batch in train_progress_bar:
                optimizer.zero_grad()

                batch, pit_stats = self._select_pit_alias_targets(batch, epoch)
                if pit_stats is not None:
                    pit_examples += pit_stats['n']
                    pit_canonical += pit_stats['canonical']
                    pit_score_gain += pit_stats['score_gain']
                    pit_warmup = pit_warmup or pit_stats['warmup']

                outputs = self.model(batch, return_loss=True)
                loss = outputs.loss
                
                self.accelerator.backward(loss)
                if self.config['max_grad_norm'] is not None:
                    clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                total_loss = total_loss + loss.item()

            self.accelerator.log({"Loss/train_loss": total_loss / len(train_dataloader)}, step=epoch + 1)
            self.log(f'[Epoch {epoch + 1}] Train Loss: {total_loss / len(train_dataloader):.6f}')
            if pit_examples:
                pit_log = {
                    'PIT/canonical_rate': pit_canonical / pit_examples,
                    'PIT/mean_selected_score_gain': pit_score_gain / pit_examples,
                }
                self.accelerator.log(pit_log, step=epoch + 1)
                self.log(
                    f'[Epoch {epoch + 1}] PIT target selection: '
                    f'phase={"balanced-warmup" if pit_warmup else "hard-EM"}, '
                    f'canonical={pit_log["PIT/canonical_rate"]:.4f}, '
                    f'mean_logp_gain={pit_log["PIT/mean_selected_score_gain"]:.4f}'
                )

            # Evaluation - 修改早停逻辑
            eval_start_epoch = self.config.get('eval_start_epoch', 1)  # 默认从第1个epoch开始评估
            if (epoch + 1) >= eval_start_epoch and (epoch + 1) % self.config['eval_interval'] == 0:
                eval_count += 1  # 增加评估次数
                all_results = self.evaluate(val_dataloader, split='val')
                if self.accelerator.is_main_process:
                    for key in all_results:
                        self.accelerator.log({f"Val_Metric/{key}": all_results[key]}, step=epoch + 1)
                    self.log(f'[Epoch {epoch + 1}] Val Results: {all_results}')
                    
                    # 新增：显示加权分数计算详情
                    if 'weighted_score' in all_results:
                        ndcg_10 = all_results.get('ndcg@10', 0)
                        recall_10 = all_results.get('recall@10', 0)
                        weighted_score = all_results['weighted_score']
                        self.log(f'[Epoch {epoch + 1}] Weighted Score Details: NDCG@10={ndcg_10:.4f}*0.8 + RECALL@10={recall_10:.4f}*0.2 = {weighted_score:.4f}')
                    
                    # 新增：显示评估进度信息
                    self.log(f'[Epoch {epoch + 1}] Evaluation #{eval_count}, Best score: {best_val_score:.4f} (Epoch {best_epoch})')
                
                # 确保早停和最佳模型选择仍然基于confidence模式的weighted_score
                val_score = all_results[self.config['val_metric']]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    no_improve_count = 0  # 重置无提升计数
                    if self.accelerator.is_main_process:
                        if self.config.get('use_ddp', False): # unwrap model for saving
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
                        else:
                            torch.save(self.model.state_dict(), self.saved_model_ckpt)
                        self.log(f'[Epoch {epoch + 1}] 🎉 New best score! Saved model checkpoint to {self.saved_model_ckpt}')
                else:
                    no_improve_count += 1  # 增加无提升计数
                    if self.accelerator.is_main_process:
                        self.log(f'[Epoch {epoch + 1}] No improvement for {no_improve_count}/{self.config["patience"]} evaluations')

                # 修改：基于评估次数的早停判断
                if self.config['patience'] is not None and no_improve_count >= self.config['patience']:
                    self.log(f'🛑 Early stopping at epoch {epoch + 1} (after {eval_count} evaluations, {no_improve_count} without improvement)')
                    break
                    
        self.log(f'Best epoch: {best_epoch}, Best val score: {best_val_score:.4f}')
        self.log(f'Training completed after {eval_count} evaluations (eval every {self.config["eval_interval"]} epochs)')
        
        # 🚀 修复：在训练结束前加载最佳模型权重
        if self.accelerator.is_main_process:
            self.log(f'Loading best checkpoint ({self.saved_model_ckpt}) for final test')
            state_dict = torch.load(self.saved_model_ckpt, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model.to(next(self.model.parameters()).device)  # 保险：放回正确device
        
        return best_epoch, best_val_score

    def evaluate(self, dataloader, split='test'):
        """评估模型：顺序自回归 beam search，返回指标字典"""
        self.model.eval()

        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )
        
        from .evaluator import AR_GRMEvaluator
        evaluator = AR_GRMEvaluator(self.config, self.tokenizer)
        diag_cfg = self.config.get('decoder_diagnostics', {}) or {}
        diag_splits = set(diag_cfg.get('splits', ['test']))
        run_diag = bool(diag_cfg.get('enabled', False)) and split in diag_splits
        
        for batch in val_progress_bar:
            with torch.no_grad():
                # 当前 split 信息（如需）
                self.config["current_split"] = split
                maxk = max(self.config['topk'])
                preds = self.model.generate(batch, n_return_sequences=maxk)  # [B, maxk, n_digit]
                labels = batch['labels']  # [B, n_digit]
                batch_results = evaluator.calculate_metrics(preds, labels)
                for key, values in batch_results.items():
                    all_results[key].extend(values.tolist())

                if run_diag:
                    identity_start = int(
                        self.config.get('item_identity_start_digit', 0)
                    )
                    generated = generation_diagnostics(
                        preds[:, :, identity_start:],
                        labels[:, identity_start:],
                        prefix=(
                            'diag/free_ar_identity'
                            if identity_start else 'diag/free_ar'
                        ),
                    )
                    for key, values in generated.items():
                        all_results[key].extend(values.detach().cpu().tolist())

                    if bool(diag_cfg.get('conditional', True)):
                        # Validation collators do not carry decoder inputs.  The
                        # target path supplies the teacher-forced prefix here.
                        diagnostic_batch = dict(batch)
                        diagnostic_batch['decoder_input_ids'] = labels
                        diagnostic_batch['decoder_labels'] = labels
                        oracle = self.model(diagnostic_batch, return_loss=True)
                        conditional = conditional_diagnostics(
                            oracle.logits,
                            labels,
                            prefix='diag/oracle_prefix_ar',
                        )
                        for key, values in conditional.items():
                            all_results[key].extend(values.detach().cpu().tolist())

        # 计算平均指标
        final_results = OrderedDict()
        for key, values_list in all_results.items():
            final_results[key] = np.mean(values_list)
        if run_diag and bool(diag_cfg.get('catalog', True)):
            final_results.update(catalog_diagnostics(self.tokenizer, self.config['codebook_size']))

        # 🚀 打印最终统计结果
        evaluator.print_final_stats()

        self.model.train()
        return final_results

    def end(self):
        """结束训练"""
        self.accelerator.end_training()

    def log(self, message, level='info'):
        """输出日志"""
        if self.accelerator is not None:
            if self.accelerator.is_main_process:
                print(message)
        else:
            print(message)
