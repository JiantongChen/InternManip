# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import warnings

import numpy as np

from internmanip.dataset.base import LeRobotSingleDataset
from internmanip.configs.dataset.data_config import DATA_CONFIG_MAP
from internmanip.agent.gr00t.Gr00tPolicy import BasePolicy, Gr00tPolicy
from scripts.eval.eval import calc_mse_for_single_trajectory
from internmanip.model.basemodel.gr00t.gr00t import GR00T_N1
from internmanip.dataset.transform.base import ComposedModalityTransform

warnings.simplefilter("ignore", category=FutureWarning)

"""
Example command:

python scripts/eval_policy.py --host localhost --port 5555 --plot
    --modality_keys delta_ee_pos delta_ee_rot gripper
    --keys delta_ee_pos_x delta_ee_pos_y delta_ee_pos_z delta_ee_rot_x delta_ee_rot_y delta_ee_rot_z gripper
    --trajs 10
    --action_horizon 16
    --video_backend torchvision_av
    --dataset_path /PATH/TO/YOUR/DATA/Isaac-GR00T/Banana_Real_Data
    --embodiment_tag new_embodiment
    --data_config franka_gripper
    --model_path /PATH/TO/YOUR/GR00T_FINETUNED_CHECKPOINT
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost", help="host")
    parser.add_argument("--port", type=int, default=5555, help="port")
    parser.add_argument("--plot", action="store_true", help="plot images")
    parser.add_argument("--modality_keys", nargs="+", type=str, default=["right_arm", "right_hand"])
    parser.add_argument("--keys", nargs="+", type=str, default=["right_arm", "right_hand"])
    parser.add_argument(
        "--data_config",
        type=str,
        default="gr1_arms_waist",
        choices=list(DATA_CONFIG_MAP.keys()),
        help="data config name",
    )
    parser.add_argument("--steps", type=int, default=150, help="number of steps to run")
    parser.add_argument("--trajs", type=int, default=1, help="trajectories to run")
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--video_backend", type=str, default="decord")
    parser.add_argument("--dataset_path", type=str, default="demo_data/robot_sim.PickNPlace/")
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="gr1",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="[Optional] Path to the model checkpoint directory, this will disable client server mode.",
    )
    args = parser.parse_args()

    data_config = DATA_CONFIG_MAP[args.data_config]
    if args.model_path is not None:
        import torch

        model = GR00T_N1.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
        )

        # modality_config = data_config.modality_config()
        # modality_transform = data_config.transform()

        # modality configs and transforms
        data_config_cls = DATA_CONFIG_MAP[args.data_config]
        modality_config = data_config_cls.modality_config()
        transforms = data_config_cls.transform()
        transforms.append(model.config.transform())
        transforms = ComposedModalityTransform(transforms=transforms)

        modality_transform = transforms

        policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            device="cuda" if torch.cuda.is_available() else "cpu",
            data_path=args.dataset_path,
        )
    else:
        raise NotImplementedError(
            "Client server mode is not implemented yet. Please provide a model path."
        )

    all_gt_actions = []
    all_pred_actions = []

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print(modality)

    # Create the dataset
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=args.embodiment_tag,
        # filter_={'1000':(-1,0),'0506':(-1,0),'sim':(0,400)}
    )

    print(len(dataset))
    # Make a prediction
    obs = dataset[0]
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    for k, v in dataset.get_step_data(0, 0).items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    print("Total trajectories:", len(dataset.trajectory_lengths))
    print("All trajectories:", dataset.trajectory_lengths)
    print("Running on all trajs with modality keys:", args.modality_keys)

    all_mse = []
    all_mse_ = []
    for traj_id in range(args.trajs):
        print("Running trajectory:", traj_id)
        steps = dataset.trajectory_lengths[traj_id]
        mse, mse_ = calc_mse_for_single_trajectory(
            policy,
            dataset,
            traj_id,
            modality_keys=args.modality_keys,
            steps=steps,
            action_horizon=args.action_horizon,
            plot=args.plot,
            keys=args.keys,
        )
        all_mse.append(mse)
        all_mse_.append(mse_)
    import ipdb
    ipdb.set_trace()
    print("Average MSE across all trajs:", np.mean(all_mse))
    print("Average Group MSE across all trajs:", np.mean(all_mse_,axis=0))
    print("Done")
    exit()
