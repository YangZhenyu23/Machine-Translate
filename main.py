"""
英译中翻译器 —— 基于 PyTorch Transformer（从零训练，不加载预训练权重）

执行流程：
  1. 下载 FLORES-200 数据集 & 构建 DataLoader
  2. 构建 Transformer Encoder-Decoder 模型
  3. 训练 & 验证（保存最优 checkpoint）
  4. 加载最优模型 → 计算 BLEU → 展示样例翻译
  5. 交互式翻译：输入英文句子实时翻译
"""

from train import train
from evaluation import (
    load_and_evaluate,
    translate,
)

# ======================== 统一超参数 ========================
CONFIG = dict(
    # 数据
    batch_size=32,
    max_len=128,
    augment_factor=2,
    # 模型
    d_model=256,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=512,
    dropout=0.3,
    # 训练
    num_epochs=100,
    lr=3e-4,
    warmup_epochs=5,
    weight_decay=0.01,
    label_smoothing=0.1,
    eta_min=1e-6,
    clip_grad=1.0,
    patience=15,
    min_delta=0.02,
    save_path="best_model.pt",
)


def interactive_translate(model, device):
    """交互式翻译：循环读取英文输入并输出中文翻译。"""
    print("\n" + "=" * 60)
    print("交互式翻译（输入英文句子，输出中文翻译；输入 q 退出）")
    print("=" * 60)
    while True:
        try:
            sentence = input("\n[EN] >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not sentence or sentence.lower() == "q":
            break
        result = translate(model, sentence, device=device,
                           max_len=CONFIG["max_len"])
        print(f"[ZH] {result}")


def main():
    # -------------------- 1. 训练 --------------------
    print("=" * 60)
    print("  阶段 1 / 3 ：训练模型")
    print("=" * 60)
    model = train(**CONFIG)

    # -------------------- 2. 评估 --------------------
    print("\n" + "=" * 60)
    print("  阶段 2 / 3 ：评估模型（BLEU + 样例翻译）")
    print("=" * 60)
    load_and_evaluate(
        checkpoint_path=CONFIG["save_path"],
        batch_size=CONFIG["batch_size"],
        max_len=CONFIG["max_len"],
        d_model=CONFIG["d_model"],
        nhead=CONFIG["nhead"],
        num_encoder_layers=CONFIG["num_encoder_layers"],
        num_decoder_layers=CONFIG["num_decoder_layers"],
        dim_feedforward=CONFIG["dim_feedforward"],
        n_samples=10,
    )

    # -------------------- 3. 交互翻译 --------------------
    print("\n" + "=" * 60)
    print("  阶段 3 / 3 ：交互式翻译")
    print("=" * 60)
    device = next(model.parameters()).device
    interactive_translate(model, device)

    print("\n完成！")


if __name__ == "__main__":
    main()
