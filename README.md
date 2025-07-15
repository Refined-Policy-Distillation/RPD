# Refined Policy Distillation (RPD)
This repo contains the code used in the [RPD Paper](https://refined-policy-distillation.github.io/) to distill and refine a VLA ([Octo](https://octo-models.github.io/) or [OpenVLA](https://openvla.github.io/)) using PPO on the [maniskill3](https://github.com/haosulab/ManiSkill) manipulation tasks.

Also checkout our [paper on arXiv](https://arxiv.org/abs/2503.05833), [openvla weights](https://huggingface.co/Juelg/openvla-7b-finetuned-maniskill) and [octo weights](https://huggingface.co/Juelg/octo-base-1.5-finetuned-maniskill) on hugging face and the [maniskill dataset](https://huggingface.co/datasets/Juelg/RPD-maniskill) in RLDS format that we used to train them.

## Installation
If you clone the repos into folders with different names, these need to be adapted in the following guide.

Create a fresh virtual/conda environment and
```shell
conda create -n rpd python=3.11 # should also work with later python versions
conda activate rpd
git clone https://github.com/Refined-Policy-Distillation/RPD.git
cd RPD
pip install -ve .
```
This should already install all required dependencies.
If you need GPU support for the simulation, install a GPU supported torch version and follow the [installation guidelines](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html) from maniskill.

Checkout the [agents repo](https://github.com/juelg/agents) for more details on the installation of specific teacher VLAs (Octo and OpenVLA).

## Training
Please note that we use the "human" camera perspective in maniksill which is not out of the box available and needs our custom `HumanCameraWrapper` available in [wrappers.py](src/rpd/wrappers.py).

### Dataset
- how it is generated
- how it is converted
- how to download it
- how to view it -> inspector

### Fine-tuning VLAs
- weights of finetune VLAs url and ids
- what needs to be implemented (configs etc.)

### Train RPD from fine-tuned VLAs
At this stage you should have a conda environment for RPD and for each VLA that you want to distill (checkout [the agents repo](https://github.com/juelg/agents) to install Octo or OpenVLA if you haven't already).
The name of the conda environments is important and needs to be lower case.
Example output:
```shell
conda env list
# conda environments:
#
base                     /home/juelg/miniconda3
rpd                      /home/juelg/miniconda3/envs/rpd
octo                     /home/juelg/miniconda3/envs/octo
openvla                  /home/juelg/miniconda3/envs/openvla
```

Checkout the [train.py](train.py) python script. It configures all hyperparameters for the RPD PPO training including what foundation model to use. You can also train the baseline PPO by switching `use_rpd=False`.
```shell
python train.py
```


## Citation
If you find RPD useful for your work, please consider citing it:
```
@inproceedings{juelg2025refinedpolicydistillationvla,
    title={{Refined Policy Distillation}: {F}rom {VLA} Generalists to {RL} Experts}, 
    author={Tobias Jülg and Wolfram Burgard and Florian Walter},
    year={2025},
    booktitle={Proc.~of the IEEE/RSJ Int.~Conf.~on Intelligent Robots and Systems (IROS)},
    note={Accepted for publication.}
}
```