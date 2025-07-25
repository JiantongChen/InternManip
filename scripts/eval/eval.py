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

import matplotlib.pyplot as plt
import numpy as np
import os

from internmanip.dataset.base import LeRobotSingleDataset
from internmanip.agent.gr00t.Gr00tPolicy import BasePolicy

# numpy print precision settings 3, dont use exponential notation
np.set_printoptions(precision=3, suppress=True)


def download_from_hg(repo_id: str, repo_type: str) -> str:
    """
    Download the model/dataset from the hugging face hub.
    return the path to the downloaded
    """
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download(repo_id, repo_type=repo_type)
    return repo_path


def calc_mse_for_single_trajectory(
    policy: BasePolicy,
    dataset: LeRobotSingleDataset,
    traj_id: int,
    modality_keys: list,
    keys: list,
    steps=300,
    action_horizon=16,
    plot=False,
):
    gt_action_joints_across_time = []
    pred_action_joints_across_time = []
    for step_count in range(steps):
        data_point = dataset.get_step_data(traj_id, step_count)

        concat_gt_action = np.concatenate(
            [data_point[f"action.{key}"][0] for key in modality_keys], axis=0
        )

        gt_action_joints_across_time.append(concat_gt_action)

        if step_count % action_horizon == 0: 
            print("inferencing at step: ", step_count)
            action_chunk = policy.get_action(data_point)
            for j in range(action_horizon):
                # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
                # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
                concat_pred_action = np.concatenate(
                    [np.atleast_1d(action_chunk[f"action.{key}"][j]) for key in modality_keys],
                    axis=0,
                )
                if 'gripper' in keys:
                    idx = keys.index('gripper')
                    concat_pred_action[idx] = 1 if concat_pred_action[idx] > 0.5 else -1
                pred_action_joints_across_time.append(concat_pred_action)

    # plot the joints
    gt_action_joints_across_time = np.array(gt_action_joints_across_time)
    pred_action_joints_across_time = np.array(pred_action_joints_across_time)[:steps]
    assert (
        gt_action_joints_across_time.shape
        == pred_action_joints_across_time.shape
    )

    # calc MSE across time
    se = (gt_action_joints_across_time - pred_action_joints_across_time) ** 2
    mse = np.mean(se)
    print("Unnormalized Action MSE across single traj:", mse)
    # calc MSE across time for different action step
    mse_=[]
    indices = np.arange(steps)
    for i in range(action_horizon):
        mask = (indices % action_horizon) == i
        group_loss = np.mean(se[mask])
        mse_.append(group_loss)

    print("Unnormalized Action Group MSE across single traj:", mse_)

    num_of_joints = gt_action_joints_across_time.shape[1]

    if plot:
        fig, axes = plt.subplots(nrows=num_of_joints, ncols=1, figsize=(8, 4 * num_of_joints))

        # Add a global title showing the modality keys
        fig.suptitle(
            f"Trajectory {traj_id} - Modalities: {', '.join(modality_keys)}",
            fontsize=16,
            color="blue",
        )

        for i, ax in enumerate(axes):
            ax.plot(gt_action_joints_across_time[:, i], label="gt action joints")
            ax.plot(pred_action_joints_across_time[:, i], label="pred action joints")

            # put a dot every ACTION_HORIZON
            for j in range(0, steps, action_horizon):
                if j == 0:
                    ax.plot(j, gt_action_joints_across_time[j, i], "ro", label="inference point")
                else:
                    ax.plot(j, gt_action_joints_across_time[j, i], "ro")

            ax.set_title(keys[i])
            ax.legend()
        plt.tight_layout()
        save_dir = "Checkpoints/tmp/{}/".format(
            str(policy.model_path).split("/")[-2]
        )
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{traj_id}.png"))

    return mse, mse_
