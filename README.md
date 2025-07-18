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
Please note that we use the "human" camera perspective in maniksill which is not out of the box available and needs our custom `HumanCameraWrapper` available in [wrappers.py](https://github.com/juelg/agents/blob/master/src/agents/wrappers.py) in the agents repo.

### Dataset
First, the maniskill dataset needs to be down loaded from [huggingface](https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations).
Afterwards, you need to generate the camera data by replaying the recorded data in the simulation again.
Note that the [`HumanCameraWrapper`](https://github.com/juelg/agents/blob/master/src/agents/wrappers.py) needs to be added to the replay environment in order to optain the correct RPD views.
More information, on how to replay the data can be found on the [maniskill documentation page](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html).
We used the following command
```shell
python -m mani_skill.trajectory.replay_trajectory  --traj-path {path} --save-traj --target-control-mode pd_ee_delta_pose --obs-mode rgb+depth --num-procs 1 --reward-mode normalized_dense --record-rewards --shader default --use-env-states --max-retry 3
```
where path is `demos/*/rl/trajectory.none.pd_ee_delta_pose.cuda.h5`

The output will be data in hdf5 as described by the [maniskill documentation](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html).
In order to fine-tune Octo and OpenVLA you need to convert the data to RLDS for which you can use [this tool](https://github.com/kpertsch/rlds_dataset_builder) from Karl Pertsch.
We provide the already converted RLDS dataset [here on huggingface](https://huggingface.co/datasets/Juelg/RPD-maniskill).

You can [download it](https://huggingface.co/docs/hub/datasets-downloading) with git (or the huggingface cli)
```shell
git lfs install
git clone git@Juelg/RPD-maniskill
```
and use a tool such as [dlimp](https://github.com/kvablack/dlimp) to load and visualize it.

### Fine-tuning VLAs
To fine-tune Octo and OpenVLA with this dataset you need to add a new dataset mix containing only that dataset.

We release the fine-tuned checkpoint of [Octo](https://huggingface.co/Juelg/octo-base-1.5-finetuned-maniskill) and [OpenVLA](https://huggingface.co/Juelg/openvla-7b-finetuned-maniskill) on huggingface.

### Train RPD from fine-tuned VLAs
At this stage you should have a conda environment for RPD and for each VLA that you want to distill (checkout [the agents repo](https://github.com/juelg/agents) to install Octo or OpenVLA if you haven't already).

Checkout the [train.py](train.py) python script. It configures all hyperparameters for the RPD PPO training including what foundation model to use. You can also train the baseline PPO by switching `use_rpd=False`.
```shell
python train.py
```
The main code is located in [ppo_rgb_rpd.py](src/rpd/ppo_rgb_rpd.py).

Hint: If you train OpenVLA, you might consider checking its preprocessor.
By default that is running on CPU but can be ported to GPU which speeds up the training process especially if you spawn multiple training instances.

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