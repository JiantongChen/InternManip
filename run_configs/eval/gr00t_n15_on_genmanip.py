from internmanip.configs import *
from internmanip.configs.env.genmanip_env import CameraEnable


eval_cfg = EvalCfg(
    eval_type="genmanip",
    agent=AgentCfg(
        agent_type="gr00t_n15_genmanip",
        model_name_or_path="/PATH/TO/YOUR/GR00T_N15_FINETUNED_CHECKPOINT",
        agent_settings={
            "dataset_path": "/PATH/TO/YOUR/Genmanip-mini",
            "data_config": "genmanip",
            "embodiment_tag": "gr1",
            "video_backend": "decord",
            "pred_steps": 16,
            "pred_action_horizon": 16,
            "adaptive_ensemble_alpha": 0.5,
        },
        server_cfg=ServerCfg(
            server_host="localhost",
            server_port=5000
        ),
    ),
    env=EnvCfg(
        env_type="genmanip",
        env_settings=GenmanipEnvSettings(
            dataset_path="/PATH/TO/YOUR/DATA/Learn_Bench_Instruction",
            eval_tasks=["instruction_16_1"],
            res_save_path="/PATH/TO/YOUR/EVAL_RESULTS",
            is_save_image=True,
            camera_enable=CameraEnable(realsense=True, obs_camera=True, obs_camera_2=True),
            depth_obs=False,
            gripper_type="panda",
            env_num=1,
            max_step=400,
            max_success_step=50,
            physics_dt=1/60,
            rendering_dt=1/60,
            headless=False,
            ray_distribution=None,
        )
    ),
)
