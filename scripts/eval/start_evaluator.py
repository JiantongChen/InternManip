"""
This script is used to start the evaluator pipeline.

Usage:
python scripts/eval/start_evaluator.py --config run_configs/eval/pi0_on_simpler_widowx.py [--distributed] [--server]

Options:
    --config: the path to the eval config file
    --distributed: whether to use distributed evaluation
    --server: whether to use client-server evaluation mode

Note:
    - If you want to use distributed evaluation, you need to set the `distributed_cfg` in the eval config file.
    - If you want to use client-server evaluation mode, you need to set the `server_cfg` in the eval config file.
"""

from internmanip.evaluator import Evaluator
from internmanip.configs import EvalCfg
import argparse
import importlib.util
import sys

def load_eval_cfg(config_path):
    spec = importlib.util.spec_from_file_location("eval_config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["eval_config_module"] = config_module
    spec.loader.exec_module(config_module)
    return getattr(config_module, "eval_cfg")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="eval config file path, e.g. run_configs/eval/pi0_on_simpler_widowx.py"
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="whether to use distributed evaluation"
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="whether to use client-server evaluation mode"
    )

    args = parser.parse_args()

    eval_cfg: EvalCfg = load_eval_cfg(args.config)

    if not args.server:
        eval_cfg.agent.server_cfg = None
    else:
        # TODO: call start_policy_server.py
        pass

    if args.distributed:
        print(f"+++++ Distributed evaluation is enabled +++++")
        
        from internmanip.evaluator.utils.distributed import EvaluatorRayActorGroup

        evaluator_ray_actor_group = EvaluatorRayActorGroup(eval_cfg)
        evaluator_ray_actor_group.eval()
    else:
        eval_cfg.distributed_cfg = None
        evaluator = Evaluator.init(eval_cfg)
        evaluator.eval()

if __name__ == "__main__":
    main()