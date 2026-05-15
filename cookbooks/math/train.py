import hydra
from omegaconf import DictConfig

from math_flow import math_flow                # from cookbooks/math/
from math_eval import math_evaluator           # from cookbooks/math/

from rllm.data.dataset import DatasetRegistry
from rllm.experimental.unified_trainer import AgentTrainer

@hydra.main(config_path="pkg://rllm.experimental.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    # ================================================================
    # 调试信息：打印配置概览
    # ================================================================
    print("\n" + "=" * 70)
    print("【train.py】配置信息")
    print("=" * 70)
    print(f"  使用的 backend: verl")
    print(f"  数据集: gsm8k")
    print(f"  主要配置命名空间:")
    for key in config.keys():
        if key not in ("hydra",):
            print(f"    - {key}")

    # ================================================================
    # 调试信息：打印数据集信息
    # ================================================================
    train_dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")
    
    print(f"\n  训练集样本数: {len(train_dataset)}")
    print(f"  测试集样本数: {len(test_dataset)}")

    # ================================================================
    # 调试信息：打印 agent_flow 和 evaluator
    # ================================================================
    print(f"\n  agent_flow 类型: {type(math_flow).__name__}")
    if hasattr(math_flow, "_name"):
        print(f"  agent_flow 名字: {math_flow._name}")
    print(f"  evaluator 类型: {type(math_evaluator).__name__}")

    print("\n" + "=" * 70)
    print("【train.py】开始创建 AgentTrainer")
    print("=" * 70 + "\n")

    trainer = AgentTrainer(
        backend="verl",
        agent_flow=math_flow,
        evaluator=math_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )

    print("\n" + "=" * 70)
    print("【train.py】AgentTrainer 创建完成，开始训练")
    print("=" * 70 + "\n")

    trainer.train()

if __name__ == "__main__":
    main()