from internmanip.configs import *
from internmanip.configs.model.gr00t_cfg import GR00T_N1_Config
from pathlib import Path


eval_cfg = EvalCfg(
    eval_type="simpler",
    agent=AgentCfg(
        agent_type="gr00t_n1",
        model_name_or_path="/PATH/TO/YOUR/GR00T_FINETUNED_CHECKPOINT",
        agent_settings={
            "policy_setup": "google_robot",
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
                f"{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/google_robot/variant_aggregation/move_near.json",
                f"{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/google_robot/variant_aggregation/open_and_close_drawer.json",
                f"{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/google_robot/variant_aggregation/pick_coke_can.json",
                f"{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/google_robot/variant_aggregation/place_in_drawer.json",

                f"{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/google_robot/visual_matching/move_near.json",
                f"{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/google_robot/visual_matching/open_and_close_drawer.json",
                f"{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/google_robot/visual_matching/pick_coke_can.json",
                f"{Path(__file__).absolute().parents[2]}/internmanip/benchmarks/utils/SimplerEnv/google_robot/visual_matching/place_in_drawer.json",

            ]
    ),
    logging_dir=f"{Path(__file__).absolute().parents[2]}/logs/eval/simpler",
    distributed_cfg=DistributedCfg(
        num_workers=4,
        ray_head_ip="10.150.91.18", # or "auto"
        include_dashboard=True,
        dashboard_port=8265,
    )
)