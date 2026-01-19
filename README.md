<div align="center">

<!-- <h1>JiuTian (九天) </h1> -->
<h2 class="papername"> <img src="./assets/logo.png" style="vertical-align: middle; height: 1em; padding: 0 0.2em;"> FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers<br>ICCV 2025</h2>
<div>
<div>
    <a href="https://scholar.google.com/citations?user=iMJYtvwAAAAJ" target="_blank">Renshan Zhang</a><sup>1</sup>,
    <a href="https://rshaojimmy.github.io/OrionLab/" target="_blank">Rui Shao</a><sup>1</sup>†,
    <a href="https://scholar.google.com/citations?user=Mpg0w3cAAAAJ" target="_blank">Gongwei Chen</a><sup>1</sup>,
    <a href="https://faculty.hitsz.edu.cn/zhangmiao">Miao Zhang</a><sup>1</sup>,
    <a href="https://faculty.hitsz.edu.cn/guanweili" target="_blank">Weili Guan</a><sup>1</sup>,
    <a href="https://jnhujnhu.github.io/" target="_blank">Kaiwen Zhou</a><sup>2</sup>,
    <a href="https://liqiangnie.github.io/index.html" target="_blank">Liqiang Nie</a><sup>1</sup>†
</div>

<sup>1</sup>Harbin Institute of Technology, Shenzhen<br>
<sup>2</sup>Huawei Noah's Ark Lab<br>
†Corresponding author


[![arXiv](https://img.shields.io/badge/arXiv-2501.16297-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2501.16297)
[![project page](https://img.shields.io/badge/Project-FALCON-9cf)](https://jiutian-vl.github.io/FALCON.github.io/)
[![FALCON-8B](https://img.shields.io/badge/HF_Model-FALCON_8B-yellow)](https://huggingface.co/renns/Falcon-8B)
[![falcon++](https://img.shields.io/badge/TechRxiv-FALCON++-21bcee.svg)](https://www.techrxiv.org/users/1015861/articles/1376224-falcon-enabling-elastic-efficiency-and-robust-perception-for-high-resolution-multimodal-large-language-model-via-visual-registers)

</div>

</div>

## If you find this work useful for your research, please kindly cite our paper and star our repo.

## Updates
- [01/2026] :fire: The extended paper of **FALCON++** is released on [TechRxiv](https://www.techrxiv.org/users/1015861/articles/1376224-falcon-enabling-elastic-efficiency-and-robust-perception-for-high-resolution-multimodal-large-language-model-via-visual-registers).
- [12/2025] :fire: [Checkpoint](https://huggingface.co/renns/Falcon-8B) released. Enjoy it!
- [07/2025] :fire: The code and [project page](https://jiutian-vl.github.io/FALCON.github.io/) are released. Enjoy it!
- [06/2025] :fire: The [arXiv paper](https://arxiv.org/abs/2501.16297) is updated.
- [06/2025] FALCON is accepted to **ICCV 2025**!
- [01/2025] [arXiv paper](https://arxiv.org/abs/2501.16297) released.

## Introduction

This is the github repository of *FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers*. In this work, we propose the FALCON model, which introduces a novel visual register technique to simultaneously address the issues of visual redundancy and fragmentation in the high-resolution visual encoding of MLLMs.

<div align="center">
<img src='assets/FALCON_arch.png' width='100%'>
</div>

## Installation

1. Clone this repository and navigate to the folder
```bash
git clone git@github.com:JiuTian-VL/JiuTian-FALCON.git
cd falcon
```

2. Install Package
```Shell
conda create -n falcon python=3.10 -y
conda activate falcon
pip install --upgrade pip
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Quick Start

We have developed a well-encapsulated class `JiutianHDInfer` specifically designed for model inference in `jiutian/eval/model_infer.py`.

Below is an example of how to use the `JiutianHDInfer` class. By calling the `inference` method, you can easily obtain the model's inference results.

```python
from jiutian.eval.model_infer import JiutianHDInfer

model_infer = JiutianHDInfer(
    model_path='/path/to/ckpt',
    model_base='/path/to/base_ckpt or None',
    conv_mode='llama_3_1',
)

image_file = '/path/to/image'
question = 'question'
model_infer.inference(image_file, question)
```

## Evaluations

See `docs/Evaluation.md` for details.

## Citation

If you find this work useful for your research, please kindly cite our paper:

```
@inproceedings{zhang2025falcon,
  title={Falcon: Resolving visual redundancy and fragmentation in high-resolution multimodal large language models via visual registers},
  author={Zhang, Renshan and Shao, Rui and Chen, Gongwei and Zhang, Miao and Zhou, Kaiwen and Guan, Weili and Nie, Liqiang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={23530--23540},
  year={2025}
}
```