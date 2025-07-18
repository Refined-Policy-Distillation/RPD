import datetime
import glob
import os
import re
import shlex

use_rpd = True
# env options are "LiftPegUpright-v1", "PegInsertionSide-v1", "PickCube-v1", "PlugCharger-v1", "PokeCube-v1", "PullCube-v1", "PullCubeTool-v1", "PushCube-v1", "RollBall-v1", "StackCube-v1"
env = "PickCube-v1"
loss_type = "mse" # other options: "l1" (l1 loss), "bc" (behavior cloning loss)
vla_type = "openvla" # other options: "octo-base", "openvla", "openvla-base"
seed = 0 # this should be varied for mass experiments
checkpoint_fmt = None # PPO checkpoint path string with {seed} and {task} placeholders, e.g. "checkpoints/{task}/ckpt_{seed}.pt" (hint: VLA paths are changed in the src/rpd/ppo_rgb_rpd.py file at the bottom)
sparse = False
note = "some experiment note"
wandb_entity = "juelg"  # change this to your wandb entity if needed
vla_conda_path = "/home/juelg/miniconda3/envs/rpd_openvla/bin/python" # adapt this to the conda path of the vla that you train, you can look it up with `conda env list`


if checkpoint_fmt is not None:
    checkpoint_pattern = checkpoint_fmt.format(seed=seed, task=env)
    files = glob.glob(checkpoint_pattern)
    ckpt_files = []
    for f in files:
        base = os.path.basename(f)
        match = re.match(r'ckpt_(\d+)\.pt$', base)
        if match:
            num = int(match.group(1))
            ckpt_files.append((num, f))

    if ckpt_files:
        max_ckpt = max(ckpt_files, key=lambda x: x[0])
        checkpoint = max_ckpt[1]
    else:
        raise ValueError(f"No checkpoint found")
    print(checkpoint)
else:
    checkpoint = None


algo = f"PPO_{loss_type}_{'sparse' if sparse else 'dense'}" if not use_rpd else f"RPD_{loss_type}_{'sparse' if sparse else 'dense'}"
uid = f"{algo}_{env}_{seed}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
args = [
    "python",
    f"src/rpd/ppo_rgb{'' if not use_rpd else '_rpd'}.py",
    f"--exp_name={uid}",
    f"--env_id={env}",
    "--num_envs=128",
    "--update_epochs=8",
    "--num_minibatches=48",
    "--total_timesteps=30_000_000",
    "--eval_freq=10",
    "--num-steps=50",
    "--control-mode=pd_ee_delta_pose",
    f"--wandb-group={algo}",
    "--track",
    f"--notes={note}",
    f"--seed={seed}",
    "--num_eval_envs=500",
    f"--gamma={0.99 if sparse else 0.8}",
    # "--wandb_project_name=ManiSkill_different_camera_angle",
    "--dense" if not sparse else "--no-dense",
    f"--wandb-entity={wandb_entity}",
]
if use_rpd:
    args.append("--ppd-lambda=1.0")
    args.append("--num-vla-samples=1") # increasing this will lead to action sampling from the vla and take much more time
    args.append(f"--vlad_loss_type={loss_type}")
    args.append(f"--vla_type={vla_type}")
    args.append(f"--vla_conda_path={vla_conda_path}")
    # args.append("--max-ppd-anneal-steps=6_000_000")

if checkpoint is not None:
    args.append(f"--checkpoint={checkpoint}")

# run the command
cmd = shlex.join(args)
print("Starting training with command:")
print(cmd)

# if you run this on a cluster, just replace the line with a slurm command e.g. by
# using simple_slurm
os.system(cmd)