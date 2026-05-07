import hydra
from omegaconf import DictConfig

from math_flow import math_flow                # from cookbooks/math/
from math_eval import math_evaluator           # from cookbooks/math/

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer

@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset = DatasetRegistry.load_dataset("gsm8k", "train")
    test_dataset = DatasetRegistry.load_dataset("gsm8k", "test")

    trainer = AgentTrainer(
        backend="verl",
        agent_flow=math_flow,
        evaluator=math_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()