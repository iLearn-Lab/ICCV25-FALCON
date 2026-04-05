import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer, CLIPImageProcessor
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional

from jiutian.train.utils import (
    maybe_zero_3,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
)


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class JiutianTrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            lr = self.args.learning_rate
            if getattr(self.args, "mm_projector_lr", None) is None:
                self.args.mm_projector_lr = lr

            if getattr(self.args, "vision_query_lr", None) is None:
                self.args.vision_query_lr = lr

            if getattr(self.args, "crop_embedding_lr", None) is None:
                self.args.crop_embedding_lr = lr

            # === deprecated ===
            if getattr(self.args, "vision_adapter_lr", None) is None:
                self.args.vision_adapter_lr = lr
            # ==================

            # The lr settings of the components of vision encoder have higher priority
            # i.e. overwrite vision_tower_lr when different, or keep consistent when do not specify
            if getattr(self.args, "vision_tower_lr", None) is None:
                self.args.vision_tower_lr = lr

            if getattr(self.args, "interact_attn_lr", None) is None:
                self.args.interact_attn_lr = self.args.vision_tower_lr

            projector_parameters = [name for name, _ in opt_model.named_parameters() if "vision2text" in name]  # MODIFY NAME FOR OTHER MODEL
            vision_queries_parameters = [name for name, _ in opt_model.named_parameters() if "vision_queries" in name]  # MODIFY NAME FOR OTHER MODEL
            crop_embedding_parameters = [name for name, _ in opt_model.named_parameters() if "crop_embedding" in name]  # MODIFY NAME FOR OTHER MODEL

            vision_adapter_parameters = [name for name, _ in opt_model.named_parameters() if "gate" in name]  # MODIFY NAME FOR OTHER MODEL
            interact_attn_parameters = [name for name, _ in opt_model.named_parameters() if "interact_attn" in name]  # MODIFY NAME FOR OTHER MODEL
            vision_tower_parameters = [
                name for name, _ in opt_model.named_parameters()
                if "vision_tower" in name and "interact_attn" not in name
            ]  # MODIFY NAME FOR OTHER MODEL

            all_special_params = (
                    projector_parameters + vision_queries_parameters +
                    crop_embedding_parameters + vision_adapter_parameters +
                    interact_attn_parameters + vision_tower_parameters
            )

            optimizer_grouped_parameters = [
                # normal
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in all_special_params and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in all_special_params and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
                # vision tower
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if
                        (n in decay_parameters and n in vision_tower_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.vision_tower_lr,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if
                        (n not in decay_parameters and n in vision_tower_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.vision_tower_lr,
                },
                # projector
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.mm_projector_lr,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.mm_projector_lr,
                },
                # vision queries
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in vision_queries_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.vision_query_lr,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in vision_queries_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.vision_query_lr,
                },
                # crop embedding
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if
                        (n in decay_parameters and n in crop_embedding_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.crop_embedding_lr,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if
                        (n not in decay_parameters and n in crop_embedding_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.crop_embedding_lr,
                },
            ]

            if len(vision_adapter_parameters) > 0:
                optimizer_grouped_parameters += [
                    # vision adapter
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n in vision_adapter_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_adapter_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n in vision_adapter_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_adapter_lr,
                    },
                ]

            if len(interact_attn_parameters) > 0:
                optimizer_grouped_parameters += [
                    # vision interactive attention
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n in interact_attn_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.interact_attn_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n in interact_attn_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.interact_attn_lr,
                    },
                ]

            # optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if not getattr(self.args, "lora_enable", False):
            # FIXME: high version of transformers would cause error without manually setting these arguments
            #  ValueError: The generation config instance is invalid
            model.generation_config.do_sample = False
            model.generation_config.temperature = 1.0
            model.generation_config.top_p = 1.0

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if getattr(self.args, "lora_enable", False):
            state_dict = get_peft_state_maybe_zero_3(
                self.model.named_parameters(), self.args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters()
            )
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                self.model.save_pretrained(output_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))

            super(JiutianTrainer, self)._save_checkpoint(model, trial, metrics)
        elif getattr(self.args, 'save_trainable', False):
            state_dict = {k: t for k, t in self.model.named_parameters() if t.requires_grad}
            state_dict = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in state_dict.items()}

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(state_dict, os.path.join(output_dir, f'model_trainables.bin'))
        else:
            super(JiutianTrainer, self)._save_checkpoint(model, trial, metrics)
