from os import path
from jiutian.registry import registry


from jiutian.datasets.llava_dataset import LlavaHRDataset
@registry.register_builder("llava_hr")
def build_llava_hr_dataset(data_path, tokenizer, image_processor, image_folder, **kwargs):
    dataset = LlavaHRDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_folder=image_folder,
        **kwargs
    )
    return dataset


from jiutian.datasets.llava_dataset import LlavaDataset
@registry.register_builder("llava")
def build_llava_dataset(data_path, tokenizer, image_processor, image_folder, **kwargs):
    dataset = LlavaDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_folder=image_folder,
        **kwargs
    )
    return dataset