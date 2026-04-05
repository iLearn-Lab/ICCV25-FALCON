import os
import copy
import json
from PIL import Image
from typing import Dict
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import transformers
from datasets import load_from_disk, load_dataset

from jiutian.mm_utils import tokenizer_image_token, expand2square, process_video_with_decord
from jiutian.constants import DEFAULT_IMAGE_TOKEN
from jiutian.datasets.grounding_utils import renorm_bbox

from .utils import preprocess, preprocess_multimodal

try:
    import av
    from decord import VideoReader, cpu
except ImportError:
    print("Please install pyav to use video processing functions.")


class LlavaDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        image_processor,
        image_folder,
        image_aspect_ratio='pad',
        is_multimodal=True,
        **kwargs,
    ):
        super().__init__()

        # list_data_dict = json.load(open(data_path, "r"))

        if isinstance(data_path, str):
            data_path = [data_path]

        list_data_dict = []
        for path in data_path:
            if os.path.isdir(path):
                if os.path.exists(os.path.join(path, 'dataset_info.json')):
                    list_data_dict += load_from_disk(path)["train"]
                else:
                    list_data_dict += load_dataset(path, split="train")
            else:
                list_data_dict += json.load(open(path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.image_processor = image_processor
        self.image_folder = image_folder
        self.is_multimodal = is_multimodal
        self.image_aspect_ratio = image_aspect_ratio

        # # checking
        # for i in range(len(list_data_dict)):
        #     it = list_data_dict[i]
        #     n_img = 0
        #     conversations = it['conversations']
        #     for idx in range(0, len(conversations)):
        #         if '<image>' in conversations[idx]['value']:
        #             n_img += 1
        #
        #     if 'image' in it:
        #         assert n_img == 1
        #     else:
        #         assert n_img == 0

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source = self.list_data_dict[i]
        assert isinstance(source, dict)
        if 'image' in source:
            image_file = source['image']
            image_folder = self.image_folder
            processor = self.image_processor

            if isinstance(image_file, str):
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            else:
                image = image_file

            if self.image_aspect_ratio == 'pad':
                image = expand2square(image, (255, 255, 255))
                image = processor(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor(image, return_tensors='pt')['pixel_values'][0]
            source = preprocess_multimodal(
                copy.deepcopy([source["conversations"]])
            ) if self.is_multimodal else source
        else:
            source = copy.deepcopy([source["conversations"]])

        data_dict = preprocess(
            source,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))

        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0]
        )

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            size = self.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, size['height'], size['width'])
        return data_dict


class LlavaHRDataset(LlavaDataset):
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        while True:
            try:
                source = self.list_data_dict[i]
                assert isinstance(source, dict)
                if 'image' in source:
                    # load image
                    image_file = source['image']
                    image_folder = self.image_folder
                    processor = self.image_processor

                    if isinstance(image_file, str):
                        image = [Image.open(os.path.join(image_folder, image_file)).convert('RGB')]
                    elif isinstance(image_file, list):
                        image = [
                            Image.open(os.path.join(image_folder, fname)).convert('RGB')
                            for fname in image_file
                        ]
                    else:
                        image = [image_file]

                    # old_w, old_h = image.size

                    # preprocess text
                    source = preprocess_multimodal(
                        copy.deepcopy([source["conversations"]])
                    ) if self.is_multimodal else source

                    # preprocess image
                    processed_data = processor(
                        images=image, query=source[0][0]['value']
                    )
                    global_image = processed_data['global_image']
                    image = processed_data['cropped_images']
                    patch_position = processed_data['patch_positions']
                    source[0][0]['value'] = processed_data['text']
                else:
                    source = copy.deepcopy([source["conversations"]])

                data_dict = preprocess(
                    source,
                    self.tokenizer,
                    has_image=('image' in self.list_data_dict[i]))

                data_dict = dict(
                    input_ids=data_dict["input_ids"][0],
                    labels=data_dict["labels"][0]
                )

                # if 'image' in self.list_data_dict[i]:
                #     assert (data_dict['input_ids'] == -200).sum() == image.shape[0]

                # image exist in the data
                if 'image' in self.list_data_dict[i]:
                    data_dict['image'] = image
                    data_dict['patch_position'] = patch_position
                    data_dict['num_image'] = patch_position.shape[0]
                elif self.is_multimodal:
                    # image does not exist in the data, but the model is multimodal
                    size = self.image_processor.crop_size
                    data_dict['image'] = torch.zeros(1, 3, size['height'], size['width'])
                    data_dict['patch_position'] = torch.zeros(1, 2).long()
                    data_dict['num_image'] = 1

                return data_dict
            except Exception as e:
                print(e)
                i = (i + 1) % len(self.list_data_dict)
                continue