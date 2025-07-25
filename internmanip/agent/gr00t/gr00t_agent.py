"""
This file references codes from the open-source project SimplerEnv-OpenVLA.
Original repository link: https://github.com/DelinQu/SimplerEnv-OpenVLA/blob/main/simpler_env/policies/gr00t/gr00t_model.py

The copyright of the original code belongs to its respective authors. Please adhere to the open-source license requirements when using this code.
"""
from internmanip.agent.base import BaseAgent
from internmanip.configs import AgentCfg
from internmanip.configs.dataset.data_config import DATA_CONFIG_MAP
# from internmanip.benchmarks.SimplerEnv.simpler_env.utils.action.action_ensemble import ActionEnsembler
from internmanip.agent.utils.geometry import euler2mat, quat2mat, mat2euler
from internmanip.agent.gr00t.Gr00tPolicy import Gr00tPolicy, unsqueeze_dict_values, squeeze_dict_values
from internmanip.dataset.embodiment_tags import EmbodimentTag
from internmanip.dataset.transform.base import ComposedModalityTransform
from internmanip.dataset.schema import DatasetMetadata
import os
from typing import Dict, Any
from pathlib import Path
import json
import torch
import numpy as np
from collections import deque
from typing import Optional, List, Sequence
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from transforms3d.euler import euler2axangle


class Gr00t_N1_Agent(BaseAgent):
    def __init__(self, config: AgentCfg):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        agent_settings = config.agent_settings
        self.policy_setup = agent_settings.get("policy_setup", None)

        if self.policy_setup == "widowx_bridge":
            self.action_ensemble = False
            self.data_config = DATA_CONFIG_MAP["widowx"]
            self.image_size = [256, 256]
            self.sticky_gripper_num_repeat = 1
            # EE pose in Bridge data was relative to a top-down pose, instead of robot base
            self.default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203
        elif self.policy_setup == "google_robot":
            self.data_config = DATA_CONFIG_MAP["google"]
            self.action_ensemble = False
            self.image_size = [320, 256]
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(
                f"Policy setup {self.policy_setup} not supported. The other datasets can be found in the huggingface config.json file."
            )
        self.observation_indices = [0]
        self.action_indices = list(range(16))
        self._modality_config = self.data_config.modality_config(self.observation_indices, self.action_indices)

        # Convert string embodiment tag to EmbodimentTag enum if needed
        embodiment_tag = agent_settings.get("embodiment_tag", "new_embodiment")
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        self.device_id = agent_settings.get("device_id", None)
        if self.device_id is None or self.device_id >= torch.cuda.device_count():
            self.device_id = 0
        self.device = torch.device(f"cuda:{self.device_id}")

        # BaseAgent.__init__ will load the model
        super().__init__(config)

        # import ipdb;ipdb.set_trace()
        # data_config_cls = DATA_CONFIG_MAP[config.data_config]
        model_transform, observation_indices, action_indices = self.policy_model.config.transform()

        transforms = self.data_config.transform()

        if model_transform is not None:
            transforms.append(model_transform)

        self._modality_transform = ComposedModalityTransform(transforms=transforms)

        # self._modality_transform = ComposedModalityTransform(transforms=self.data_config.transform())
        self._modality_transform.eval()  # set this to eval mode

        self.policy_model.eval()  # Set model to eval mode
        self.policy_model.to(device=self.device)  # type: ignore

        # Load metadata for normalization stats
        metadata_path = Path(config.model_name_or_path) / "experiment_cfg" / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )
        else:
            # deserialize the ndarray
            def convert_lists_to_arrays(obj):
                if isinstance(obj, list):
                    return np.array(obj)
                if isinstance(obj, dict):
                    return {k: convert_lists_to_arrays(v) for k, v in obj.items()}
                return obj
            
            metadata_dict = convert_lists_to_arrays(metadata_dict)

        metadata = DatasetMetadata.model_validate(metadata_dict)

        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

        # Load the horizons needed for the model.
        # Get modality configs
        # Video horizons
        self._video_delta_indices = np.array(self._modality_config["video"].delta_indices)
        self._assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        # State horizons (if used)
        if "state" in self._modality_config:
            self._state_delta_indices = np.array(self._modality_config["state"].delta_indices)
            self._assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None

        denoising_steps = agent_settings.get("denoising_steps", None)
        if denoising_steps is not None:
            if hasattr(self.policy_model, "action_head") and hasattr(
                self.policy_model.action_head, "num_inference_timesteps"
            ):
                self.policy_model.action_head.num_inference_timesteps = denoising_steps
                print(f"Set action denoising steps to {denoising_steps}")

        self.action_scale = agent_settings.get("action_scale", 1.0)
        self.obs_horizon = 1
        self.obs_interval = 1
        self.pred_action_horizon = 5
        self.image_history = deque(maxlen=self.obs_horizon)
        self.exec_horizon = agent_settings.get("exec_horizon", 1)

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.action_ensemble_temp = agent_settings.get("action_ensemble_temp", -0.8)

        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(
                self.pred_action_horizon, self.action_ensemble_temp
            )
        else:
            self.action_ensembler = None

        self.task = None
        self.task_description = None

    def _assert_delta_indices(self, delta_indices: np.ndarray):
        """Assert that the delta indices are valid."""
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"

    def _check_state_is_batched(self, obs: Dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        # Ensure correct dimensions before applying transforms
        return self._modality_transform(obs)

    def _get_action_from_normalized_input(self, normalized_input: Dict[str, Any]) -> torch.Tensor:
        # Set up autocast context if needed
        normalized_input = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in normalized_input.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model_pred = self.policy_model.inference(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        return self._modality_transform.unapply(action)

    def _get_unnormalized_action(self, normalized_action: torch.Tensor) -> Dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()})

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
        }

        Returns:
            Dict[str, Any]: The predicted action.
        """
        # let the get_action handles both batch and single input
        is_batch = self._check_state_is_batched(observations)
        if not is_batch:
            observations = unsqueeze_dict_values(observations)

        # normalized_input = unsqueeze_dict_values # ?
        normalized_input = self.apply_transforms(observations)
        normalized_action = self._get_action_from_normalized_input(normalized_input)
        unnormalized_action = self._get_unnormalized_action(normalized_action)
        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    def reset(self, task_description: str) -> None:
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.task_description = task_description
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.action_plan = deque()

    def preprocess_widowx_proprio(self, eef_pos) -> np.array:
        """convert ee rotation to the frame of top-down
        https://github.com/allenzren/open-pi-zero/blob/c3df7fb062175c16f69d7ca4ce042958ea238fb7/src/agent/env_adapter/simpler.py#L167
        """
        # StateEncoding.POS_EULER: xyz + rpy + gripper(openness)
        proprio = eef_pos
        rm_bridge = quat2mat(proprio[3:7])
        rpy_bridge_converted = mat2euler(rm_bridge @ self.default_rot.T)
        gripper_openness = proprio[7] # from simpler, 0 for close, 1 for open
        raw_proprio = np.concatenate(
            [
                proprio[:3],
                rpy_bridge_converted,
                [gripper_openness],
            ]
        ).astype(np.float32)
        return raw_proprio

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images: List[Image.Image] = self._obtain_image_history()

        eef_pos = kwargs.get("eef_pos", None)
        if self.policy_setup == "widowx_bridge":
            state = self.preprocess_widowx_proprio(eef_pos)
            batch = {
                "video.image_0": np.array(images[0][None]), # numpy (b h w c)
                "state.ee_pos": state[0:3][None],
                "state.ee_rot": state[3:6][None],
                "state.gripper": state[6:7][None],
                "annotation.human.action.task_description": [task_description],
            }
            if not self.action_plan:
                actions = self.get_action(batch)
                # actions = self.policy_model.inference(batch)
                action_chunk = np.concatenate([
                    actions["action.delta_ee_pos"],
                    actions["action.delta_ee_rot"],
                    actions["action.gripper"][...,None],
                ], axis=-1)[:self.pred_action_horizon]
                self.action_plan.extend(action_chunk)

        elif self.policy_setup == "google_robot":
            # state = self.preprocess_widowx_proprio(eef_pos)
            batch = {
                "video.image": np.array(images[0][None]),
                "state.ee_pos": eef_pos[0:3][None],
                "state.ee_rot": eef_pos[3:6][None],
                "state.gripper": eef_pos[6:7][None],
                "annotation.human.action.task_description": [task_description],
            }

            if not self.action_plan:
                actions = self.get_action(batch)
                raw_actions = np.concatenate([
                    actions["action.delta_ee_pos"],
                    actions["action.delta_ee_rot"],
                    actions["action.gripper"][...,None],
                ], axis=-1)[:self.pred_action_horizon]
                self.action_plan.extend(raw_actions)
            
        raw_actions = self.action_plan.popleft()


        # if self.action_ensemble:
        #     raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        raw_action = {
            "world_vector": np.array(raw_actions[:3]),
            "rotation_delta": np.array(raw_actions[3:6]),
            "open_gripper": np.array(
                raw_actions[6:7]
            ),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(
            raw_action["rotation_delta"], dtype=np.float64
        )
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            action["gripper"] = 0
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            
            # fix a bug in the SIMPLER code here
            # self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def _add_image_to_history(self, image: np.ndarray) -> None:
        if len(self.image_history) == 0:
            self.image_history.extend([image] * self.obs_horizon)
        else:
            self.image_history.append(image)

    def _obtain_image_history(self) -> List[Image.Image]:
        image_history = list(self.image_history)
        images = image_history[:: self.obs_interval]
        # images = [Image.fromarray(image).convert("RGB") for image in images]
        return images

    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[np.ndarray],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate(
                    [a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1
                )
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(
                pred_actions[:, action_dim], label="predicted action"
            )
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)