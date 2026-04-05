# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from icecream import ic
import random
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

import torch
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn

import transformers
import tokenizers
from transformers import CLIPImageProcessor

from jiutian.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from jiutian.train.jiutian_trainer import JiutianTrainer
from jiutian.train.utils import (
    maybe_zero_3,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
)

from jiutian import conversation as conversation_lib
from jiutian.model import *
from jiutian.mm_utils import tokenizer_image_token, expand2square
from jiutian.processor import AdaptiveCropProcessor

from jiutian.registry import registry
from jiutian.datasets import ConcatDataset, InterleaveDateset, SubSet


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def setup_seeds(seed):
    seed = seed + local_rank

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    visual_feature_type: Optional[str] = field(default='patch')
    visual_enable_interact_attn: bool = field(default=True)
    num_vision_queries: int = field(default=64)
    projector_type: Optional[str] = field(default='linear')
    pretrained_weights: Optional[str] = field(default=None)
    processor_anchors: Optional[str] = field(default='grid_9')
    processor_enable_low_res: bool = field(default=False)
    processor_image_size: int = field(default=336)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data."}
    )
    data_config: str = field(
        default=None,
        metadata={"help": "Path to the config of data."}
    )
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    tune_vision2text: bool = field(default=True)
    freeze_vision_queries: bool = field(default=False)
    freeze_vision_tower: bool = field(default=True)
    unfreeze_vision_interact_attn: bool = field(default=False)
    unfreeze_vision_self_attn: bool = field(default=False)
    unfreeze_vision_adapter: bool = field(default=False)
    save_trainable: Optional[bool] = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    adapter_enable: bool = False
    adapter_hidden_size: int = 32
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_weight_path: str = ""
    mm_projector_lr: Optional[float] = None
    vision_query_lr: Optional[float] = None
    crop_embedding_lr: Optional[float] = None
    vision_adapter_lr: Optional[float] = None
    interact_attn_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    # FIXME: Interleave dataset do not support group_by_modality_length
    group_by_modality_length: bool = field(default=False)


def find_all_linear_names(model, excluded_names=None):
    cls = torch.nn.Linear
    lora_module_names = set()

    if excluded_names is None:
        excluded_names = []

    for name, module in model.named_modules():
        if any(keyword in name for keyword in excluded_names):
            continue
        if isinstance(module, cls):
            # names = name.split('.')
            # lora_module_names.add(names[0] if len(names) == 1 else names[-1])

            lora_module_names.add(name)

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=batch_first,
            padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    # new code for support various LLMs
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "id"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0  # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels,
                     attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        # batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), ids=ids)

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            batch['images'] = torch.cat(images)  # Sum(Crop+1) x c x h x w

        if 'patch_position' in instances[0]:
            patch_positions = [instance['patch_position'] for instance in instances]
            batch['patch_positions'] = torch.cat(patch_positions)  # Sum(Crop+1) x 2

        if 'num_image' in instances[0]:
            num_images = []
            for instance in instances:
                num_image = instance['num_image'] if isinstance(instance['num_image'], list) else [instance['num_image']]  # list or int
                num_images.extend(num_image)
            batch['num_images'] = num_images

        return batch


def count_params(model):
    total_param = 0
    trainable_param = 0
    trainable_param_list = []
    for name, param in model.named_parameters():
        total_param += param.numel()
        if param.requires_grad == True:
            trainable_param_list.append(name)
            trainable_param += param.numel()

    rank0_print(trainable_param_list)
    rank0_print(f'total params: {total_param / 1e6} M\ntrainable params: {trainable_param / 1e6} M')
    rank0_print("--- NOTE: When using deepspeed zero3, the params count maybe inaccurate ---")


def get_interleave_dataset(datasets_cfg, tokenizer, image_processor, seed, data_args):
    if datasets_cfg is None:
        return None

    datasets = []
    ratios = []

    for ds_name in datasets_cfg:
        ds_cfg = datasets_cfg[ds_name]
        builder_func = registry.get_builder_func(ds_cfg.type)

        rank0_print(f"=== Loading {ds_name}, type: {ds_cfg.type} ===")

        # additional argument dataset class
        ds_args = ds_cfg.get('args', None)
        if ds_args is not None and OmegaConf.is_dict(ds_args):
            ds_args = OmegaConf.to_container(ds_args)
        else:
            ds_args = {}
        rank0_print(f"Additional arguments: {ds_args}")

        ds_args.update(dict(
            image_aspect_ratio=data_args.image_aspect_ratio,
            is_multimodal=data_args.is_multimodal,
        ))

        # build dataset
        ds = builder_func(
            data_path=ds_cfg.data_path,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_folder=ds_cfg.image_folder,
            **ds_args
        )

        # subsample if needed
        subsample = ds_cfg.get('subsample', None)
        if subsample is not None:
            ds = SubSet(ds, subsample, seed=seed)

        datasets.append(ds)
        ratios.append(ds_cfg.get('sample_ratio', None))

    combined_dataset = None
    if all(item is None for item in ratios):
        ratios = [len(ds) for ds in datasets]
        ratios = [r/sum(ratios) for r in ratios]
        combined_dataset = ConcatDataset(datasets)
    elif all(item is not None for item in ratios):
        ratios = [r/sum(ratios) for r in ratios]
        combined_dataset = InterleaveDateset(datasets, ratios, seed=seed)
    else:
        raise ValueError(
            f'Only setting the sampling rate of all datasets to empty for automatic calculation '
            f'or manually setting the sampling rate for all datasets is acceptable. But received ratios={ratios}'
        )

    rank0_print("--- Finish loading ---")
    # rank0_print(combined_dataset)

    rank0_print(f"Using {combined_dataset.__class__.__name__}")
    rank0_print("Total number:", sum(len(ds) for ds in datasets))
    for (ds_name, ds, ratio) in zip(datasets_cfg, datasets, ratios):
        rank0_print(f"{ds_name}: number {len(ds)}, ratio {ratio:.3f}")

    return combined_dataset


""" 
======= NOTE =======
If you want to write a train() method for the new model, 
you only need to focus on and modify the highlighted part of the function. 
In theory, the rest of the code generally doesn't need to be changed, except for some special cases.
"""
def train(attn_implementation=None):
    global local_rank

    # Parsing arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    setup_seeds(training_args.seed)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["vision2text"],  # MODIFY FOR NEW MODEL
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    # ==========================  MODIFY FOR NEW MODEL ==========================
    # Initialize model
    cfg = JiutianConfig.from_pretrained(model_args.model_name_or_path)

    cfg.visual_config["model_name_or_path"] = model_args.vision_tower
    cfg.visual_config["feature_type"] = model_args.visual_feature_type
    cfg.visual_config["num_vision_queries"] = model_args.num_vision_queries
    cfg.visual_config["anchor_max"] = int(model_args.processor_anchors.split('_')[-1])
    cfg.visual_config["enable_interactive_attn"] = model_args.visual_enable_interact_attn

    cfg.projector_config["projector_type"] = model_args.projector_type
    cfg.projector_config["hidden_size"] = cfg.projector_config["output_dim"] = cfg.hidden_size

    cfg.processor_config["anchors"] = model_args.processor_anchors
    cfg.processor_config["enable_low_res"] = model_args.processor_enable_low_res
    cfg.processor_config["image_size"] = model_args.processor_image_size

    model, msg = JiutianLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=cfg,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,  # This argument can be used by transformers>=4.37
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),  # when using `flash_attention_2`, this should be either 'bf16', 'fp16' or None
        output_loading_info=True,
        **bnb_model_from_pretrained_args
    )
    model.config.use_cache = False

    # copy weights for interactive attn
    if model_args.visual_enable_interact_attn and any('interact_attn' in k for k in msg['missing_keys']):
        print("\n==== coping weights for interactive attention ====\n")
        for layer in model.get_vision_model().vision_tower.vision_model.encoder.layers:
            layer.interact_attn.load_state_dict(layer.self_attn.state_dict())

    # ic(msg['missing_keys'])
    # ic(msg['unexpected_keys'])
    # ic(msg['mismatched_keys'])

    # ==========================  MODIFY FOR NEW MODEL ==========================

    if model_args.freeze_backbone:
        model.requires_grad_(False)

    # Quantization
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        # FIXME: When the CLIPVISionModel is unfreezed, use_reentrant=True will cause the error below
        #  "AssertionError: The parameter 577 has already been reduced. Gradient computed twice for this partition.
        #  Multiple gradient reduction is currently not supported"
        training_args.gradient_checkpointing_kwargs = {'use_reentrant': False}

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Apply Lora
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(
                model, excluded_names=['vision_model', 'vision2text']),  # MODIFY FOR NEW MODEL
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # ==========================  MAYBE DELETE FOR OTHER MODEL ==========================

    # Pretrained models that are not initialized by "from_pretrained" should be
    # loaded with a delay to avoid errors of size mismatch when using deepspeed zero3,
    model.get_model().initialize_vision_modules(
        fsdp=training_args.fsdp
    )

    # load pretrained weights
    if model_args.pretrained_weights is not None:
        print("Loading pretrained weights from {} ...".format(model_args.pretrained_weights))
        mm_weights = torch.load(model_args.pretrained_weights, map_location='cpu')

        def get_w(weights, keyword):
            return {k.split(keyword + '.', 1)[1]: v for k, v in weights.items() if keyword in k}

        v2t_msg = model.get_model().get_vision2text().load_state_dict(get_w(mm_weights, "vision2text"))
        ic(v2t_msg.unexpected_keys)

        vision_msg = model.get_model().get_vision_model().load_state_dict(get_w(mm_weights, "vision_model"), strict=False)
        ic(vision_msg.unexpected_keys)

        # msg = model.load_state_dict(mm_weights, strict=False)
        # ic(msg.unexpected_keys)

    # set dtype
    vision_model = model.get_model().get_vision_model()
    vision_model.vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    vision2text = model.get_model().get_vision2text()
    if training_args.tune_vision2text and training_args.bits in [4, 8]:
        vision2text.to(dtype=compute_dtype, device=training_args.device)
    else:
        vision2text.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    # Initialze image processor
    # data_args.image_processor = CLIPImageProcessor.from_pretrained(model_args.vision_tower)


    data_args.image_processor = AdaptiveCropProcessor(
        cfg.processor_config["image_size"],
        anchors=cfg.processor_config["anchors"],
        add_global_img=cfg.processor_config["add_global_img"],
        add_textual_crop_indicator=cfg.processor_config["add_textual_crop_indicator"],
        enable_low_res=cfg.processor_config["enable_low_res"]
    )

    # set trainable params
    vision_model = model.get_model().get_vision_model()

    if not training_args.tune_vision2text:
        for p in model.get_model().get_vision2text().parameters():
            p.requires_grad = False

    if training_args.freeze_vision_queries:
        vision_model.vision_queries.requires_grad = False

    if training_args.freeze_vision_tower:
        for p in vision_model.vision_tower.parameters():
            p.requires_grad = False
    else:
        model.config.vision_delay_load = False

    if training_args.unfreeze_vision_interact_attn:
        model.config.vision_delay_load = False
        for name, p in vision_model.vision_tower.named_parameters():
            if 'interact_attn' in name:
                p.requires_grad = True

    if training_args.unfreeze_vision_self_attn:
        model.config.vision_delay_load = False
        for name, p in vision_model.vision_tower.named_parameters():
            if 'self_attn' in name:
                p.requires_grad = True

    if training_args.unfreeze_vision_adapter:
        model.config.vision_delay_load = False
        for name, p in vision_model.vision_tower.named_parameters():
            if 'gate_adapter_proj' in name or 'gate_alpha' in name:
                p.requires_grad = True

    if not training_args.save_trainable and not training_args.lora_enable:
        model.config.vision_delay_load = False

    # =========== for window attention ablation ===========
    for name, p in vision_model.vision_tower.named_parameters():
        if 'window_attention' in name:
            p.requires_grad = True
    # =====================================================

    # # set query to trainable
    # model.query_tokens.requires_grad = True

    # set config
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.mm_projector_lr = training_args.mm_projector_lr

    model_args.tune_vision2text = training_args.tune_vision2text

    # ==========================  MAYBE DELETE FOR OTHER MODEL ==========================

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # initialize dataset
    datasets_cfg = OmegaConf.load(data_args.data_config)
    train_dataset = get_interleave_dataset(
        datasets_cfg.train_datasets, tokenizer, data_args.image_processor, training_args.seed, data_args)
    eval_dataset = get_interleave_dataset(
        datasets_cfg.eval_datasets, tokenizer, data_args.image_processor, training_args.seed, data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    count_params(model)

    trainer = JiutianTrainer(model=model,
                             tokenizer=tokenizer,
                             args=training_args,
                             **data_module)

    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    #     trainer.train()

    saved_ckpt_list = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if saved_ckpt_list:
        non_lora_trainable_path = os.path.join(saved_ckpt_list[-1], 'non_lora_trainables.bin')
        if os.path.exists(non_lora_trainable_path):
            non_lora_trainables = torch.load(non_lora_trainable_path, map_location='cpu')
            # non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            # if any(k.startswith('model.model.') for k in non_lora_trainables):
            #     non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            msg = model.load_state_dict(non_lora_trainables, strict=False)
            rank0_print(f"Unexpected_keys of resume ckpt: {msg.unexpected_keys}")

        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    # FIXME: high version of transformers would cause error without manually setting these arguments
    #  ValueError: The generation config instance is invalid
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    elif training_args.save_trainable:
        state_dict = {k: t for k, t in model.named_parameters() if t.requires_grad}
        state_dict = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in state_dict.items()}

        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            torch.save(state_dict, os.path.join(training_args.output_dir, 'model_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
