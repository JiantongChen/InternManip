from internmanip.configs import *
from internmanip.configs.model.gr00t_cfg import GR00T_N1_Config
from pathlib import Path


eval_cfg = EvalCfg(
    eval_type="simpler",
    agent=AgentCfg(
        agent_type="gr00t_n1",
        model_name_or_path="/PATH/TO/YOUR/GR00T_FINETUNED_CHECKPOINT",
        agent_settings={
            "policy_setup": "widowx_bridge",
            "action_scale": 1.0,
            "exec_horizon": 1,
            "action_ensemble_temp": -0.8,
            "embodiment_tag": "gr1",
            "denoising_steps": 16,
        },
        server_cfg=ServerCfg(
            server_host="localhost",
            server_port=5000,
        ),
    ),
    env=EnvCfg(
        env_type="simpler",
        device_id=0,
        episodes_config_path=[
                f"{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/widowx_bridge/visual_matching/put_carrot_on_plate.json",
                f"{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/widowx_bridge/visual_matching/put_eggplant_in_basket.json",
                f"{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/widowx_bridge/visual_matching/put_spoon_on_towel.json",
                f"{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/widowx_bridge/visual_matching/stack_cube.json",


            ]
    ),
    logging_dir=f"{Path(__file__).absolute().parents[2]}/logs/eval/gr00tn1_windowx_node2_bs16_acc8_meanstd",
    distributed_cfg=DistributedCfg(
        num_workers=4,
        ray_head_ip="10.150.91.18", # or "auto"
        include_dashboard=True,
        dashboard_port=8265,
    )
)