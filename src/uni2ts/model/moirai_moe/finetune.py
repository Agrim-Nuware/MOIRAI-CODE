#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Fine-tuning wrapper for MoiraiMoEModule.

The upstream uni2ts release only ships MoiraiFinetune (for the dense Moirai
model) plus inference-only code for Moirai-MoE. MoiraiMoEModule shares the
exact same forward() signature and building blocks (RMSNorm, BinaryAttentionBias,
MultiInSizeLinear, ...) as MoiraiModule, so MoiraiFinetune's training_step,
validation_step, configure_optimizers, and transform_map logic all apply
unchanged -- this class only swaps in MoiraiMoEModule for construction.
"""

import math
from typing import Any, Optional

import torch
from torch import nn

from uni2ts.loss.packed import PackedDistributionLoss, PackedLoss, PackedNLLLoss
from uni2ts.model.moirai.finetune import MoiraiFinetune
from uni2ts.module.norm import RMSNorm
from uni2ts.module.position import BinaryAttentionBias, LearnedEmbedding, LearnedProjection
from uni2ts.module.ts_embed import FeatLinear, MultiInSizeLinear, MultiOutSizeLinear
from uni2ts.optim import SchedulerType, get_scheduler

from .module import MoiraiMoEModule


class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with a trainable low-rank delta:
        y = base(x) + scaling * (x @ lora_A^T @ lora_B^T)

    lora_B is zero-initialized, so the wrapped layer is an exact no-op at
    step 0 -- training starts *at* the zero-shot solution and can only move
    away from it by a small, rank-constrained amount, unlike full/freeze_ffn
    fine-tuning which leave whichever parameters are trainable completely
    unconstrained (full-rank). This directly targets the overfitting pattern
    seen with those patterns (validation loss degrading while training loss
    keeps improving), without touching the base pretrained weights at all.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scaling = alpha / rank
        # Match the device/dtype of the layer being wrapped -- injection
        # happens inside configure_optimizers, which Lightning calls *after*
        # moving the model to its target device, so freshly-created
        # parameters here would otherwise default to CPU/fp32 and never get
        # swept up in a later .to(device) call.
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * (x @ self.lora_A.T @ self.lora_B.T)


def inject_lora_attention(module: nn.Module, rank: int = 8, alpha: float = 16.0) -> int:
    """
    Recursively replace attention projection layers (q_proj/k_proj/v_proj/
    out_proj -- the standard nn.Linear layers in GroupedQueryAttention and its
    subclasses) with LoRALinear wrappers, in place. Returns the number of
    layers wrapped. FFN/MoE-expert layers, embeddings, and the output head
    are untouched (left however finetune_pattern="lora"'s caller configured
    them -- see MoiraiMoEFinetune.configure_optimizers).
    """
    target_names = ("q_proj", "k_proj", "v_proj", "out_proj")
    count = 0
    for name, child in list(module.named_children()):
        if name in target_names and isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            count += 1
        else:
            count += inject_lora_attention(child, rank=rank, alpha=alpha)
    return count


class MoiraiMoEFinetune(MoiraiFinetune):
    def __init__(
        self,
        min_patches: int,
        min_mask_ratio: float,
        max_mask_ratio: float,
        max_dim: int,
        num_training_steps: int,
        num_warmup_steps: int,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[MoiraiMoEModule] = None,
        num_samples: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.98,
        loss_func: PackedDistributionLoss = PackedNLLLoss(),
        val_metric: Optional[PackedLoss | list[PackedLoss]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        log_on_step: bool = False,
        context_length: Optional[int | list[int]] = None,
        prediction_length: Optional[int | list[int]] = None,
        patch_size: Optional[int] = None,
        finetune_pattern: str | list[str] = "full",
        use_8bit_adam: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
    ):
        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"
        if num_training_steps is not None:
            assert (
                num_warmup_steps <= num_training_steps
            ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."

        # Bypass MoiraiFinetune.__init__ (it hardcodes MoiraiModule for the
        # module_kwargs fallback path) and replicate it here with
        # MoiraiMoEModule instead. Signature/hparams mirror MoiraiFinetune
        # exactly so the inherited training_step/validation_step/
        # configure_optimizers (which read self.hparams.*) keep working.
        super(MoiraiFinetune, self).__init__()
        self.save_hyperparameters(ignore=["module"])
        self.module = MoiraiMoEModule(**module_kwargs) if module is None else module

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_size = patch_size
        self.finetune_pattern = finetune_pattern
        self.use_8bit_adam = use_8bit_adam
        self._lora_injected = False

    def configure_optimizers(self) -> dict:
        # Full copy of MoiraiFinetune.configure_optimizers with FeatLinear
        # (a Moirai-MoE-specific layer not present in dense Moirai) added to
        # the weight-decay whitelist, and optional 8-bit AdamW for memory.
        decay = set()
        no_decay = set()

        if self.finetune_pattern == "full":
            pass
        elif self.finetune_pattern == "freeze_ffn":
            for pn, p in self.named_parameters():
                if "ffn" in pn:
                    p.requires_grad = False
        elif self.finetune_pattern == "head_only":
            for pn, p in self.named_parameters():
                if "param_proj" not in pn:
                    p.requires_grad = False
        elif self.finetune_pattern == "lora":
            if not self._lora_injected:
                for p in self.parameters():
                    p.requires_grad = False
                n_wrapped = inject_lora_attention(
                    self.module,
                    rank=self.hparams.lora_rank,
                    alpha=self.hparams.lora_alpha,
                )
                self._lora_injected = True
                print(
                    f"LoRA: wrapped {n_wrapped} attention projection layers "
                    f"(rank={self.hparams.lora_rank}, alpha={self.hparams.lora_alpha}); "
                    "everything else (FFN/MoE experts, embeddings, output head) stays frozen."
                )
        else:
            raise ValueError(
                "Unsupported finetune pattern {}".format(self.finetune_pattern)
            )

        whitelist_params = (
            LearnedProjection,
            MultiInSizeLinear,
            MultiOutSizeLinear,
            FeatLinear,
            nn.Linear,
        )
        blacklist_params = (
            BinaryAttentionBias,
            LearnedEmbedding,
            RMSNorm,
            nn.Embedding,
            nn.LayerNorm,
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn
                if isinstance(m, LoRALinear) and pn in ("lora_A", "lora_B"):
                    # LoRA's own low-rank matrices -- not caught by the
                    # weight/bias suffix rules below since they aren't named
                    # "weight"/"bias" (m.base, the frozen wrapped Linear, is
                    # skipped above since its params have requires_grad=False).
                    decay.add(fpn)
                elif pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_params):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_params):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        optim_groups = [
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(decay))],
                ),
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(no_decay))],
                ),
                "weight_decay": 0.0,
            },
        ]

        if self.use_8bit_adam:
            # Full fp32 AdamW state for a ~935M param model is ~15GB, right
            # at the limit of a free-tier 16GB GPU. bitsandbytes' 8-bit
            # AdamW cuts optimizer state memory ~4x with negligible impact.
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                optim_groups,
                lr=self.hparams.lr,
                betas=(self.hparams.beta1, self.hparams.beta2),
                eps=1e-6,
            )
        else:
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=self.hparams.lr,
                betas=(self.hparams.beta1, self.hparams.beta2),
                eps=1e-6,
            )
        scheduler = get_scheduler(
            SchedulerType.CONSTANT,
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step",
            },
        }


class MoiraiMoELinearProbe(MoiraiMoEFinetune): ...
