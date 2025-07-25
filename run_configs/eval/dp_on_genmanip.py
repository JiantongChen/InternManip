from internmanip.configs import *
from internmanip.configs.env.genmanip_env import CameraEnable
from internmanip.configs.model.dp_cfg import DiffusionConfig


eval_cfg = EvalCfg(
    eval_type="genmanip",
    agent=AgentCfg(
        agent_type="DP",
        model_name_or_path="/PATH/TO/YOUR/DP_FINETUNED_CHECKPOINT",
        model_cfg=DiffusionConfig(),
        agent_settings={
            "embodiment_tag": "gr1",
            "data_config": "genmanip",
            "dataset_path": "/PATH/TO/YOUR/DATA/Learn_Bench_Instruction",
            "n_obs_steps": 1,
        },
        server_cfg=ServerCfg(
            server_host="localhost",
            server_port=5000
        ),
    ),
    env=EnvCfg(
        env_type="genmanip",
        env_settings=GenmanipEnvSettings(
            dataset_path="/PATH/TO/YOUR/DATA/benchmark_data/tasks/Learn_Bench_Instruction",
            eval_tasks=["instruction_16_1"],
            res_save_path="/PATH/TO/YOUR/EVAL_RESULTS",
            is_save_image=True,
            camera_enable=CameraEnable(realsense=True, obs_camera=True, obs_camera_2=False),
            depth_obs=False,
            gripper_type="panda",
            env_num=1,
            max_step=400,
            max_success_step=40,
            physics_dt=1/30,
            rendering_dt=1/30,
            headless=True,
            ray_distribution=None,
        )
    ),
    eval_settings={},
)
