import os
# from jiutian.train.llama_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )
# replace_llama_attn_with_flash_attn()

from jiutian.train.train import train

attn_implementation = 'flash_attention_2'
if __name__ == "__main__":
    train(attn_implementation=attn_implementation)