# matsushibadenki/snn4/snn4-main/run_evolution.py
# Script to run the self-evolving agent and execute a meta-evolution cycle.
# 修正: SelfEvolvingAgentMaster に対応。 type hint 追加。

import argparse
from app.containers import BrainContainer
from snn_research.agent.self_evolving_agent import SelfEvolvingAgentMaster

# --- ▼ 修正: -> None を追加 ▼ ---
def main() -> None:
# --- ▲ 修正 ▲ ---
    """
    Starts the process of giving an initial task to the self-evolving agent,
    making it reflect on its performance, and generate improvement proposals.
    """
    parser = argparse.ArgumentParser(
        description="SNN Self-Evolving Agent Execution Framework",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument( "--task_description", type=str, required=True, help="Task for self-assessment." )
    parser.add_argument( "--model_config", type=str, default="configs/models/small.yaml", help="Base model config for evolution." )
    parser.add_argument( "--training_config", type=str, default="configs/base_config.yaml", help="Base training config for evolution." )
    parser.add_argument( "--initial_accuracy", type=float, default=0.75, help="Initial accuracy." )
    parser.add_argument( "--initial_spikes", type=float, default=1500.0, help="Initial average spikes." )
    args = parser.parse_args()

    container = BrainContainer()
    container.config.from_yaml(args.training_config)
    container.config.from_yaml(args.model_config)

    agent: SelfEvolvingAgentMaster = container.self_evolving_agent()
    agent.model_config_path = args.model_config
    agent.training_config_path = args.training_config

    initial_metrics = { "accuracy": args.initial_accuracy, "avg_spikes_per_sample": args.initial_spikes }

    agent.run_evolution_cycle( task_description=args.task_description, initial_metrics=initial_metrics )

if __name__ == "__main__":
    main()
