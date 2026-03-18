import torch  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]

from dataset import load_flores_data, PAD_IDX, EOS_IDX
from model import build_model


def train(
    num_epochs=100,
    batch_size=32,
    max_len=128,
    augment_factor=2,
    d_model=256,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=512,
    dropout=0.3,
    lr=3e-4,
    weight_decay=0.01,
    label_smoothing=0.1,
    T_max=None,
    eta_min=1e-6,
    clip_grad=1.0,
    patience=15,
    min_delta=0.02,
    warmup_epochs=5,
    save_path="best_model.pt",
):
    # ==================== 设备 ====================
    device = torch.device("cuda")
    print(f"使用设备: {device}")

    # ==================== 数据 ====================
    train_loader, val_loader, en_tokenizer, zh_tokenizer = load_flores_data(
        batch_size=batch_size, max_len=max_len, augment_factor=augment_factor,
    )

    # ==================== 模型 ====================
    model = build_model(
        src_vocab_size=en_tokenizer.vocab_size,
        tgt_vocab_size=zh_tokenizer.vocab_size,
        device=device,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
    )

    # ==================== 损失 / 优化器 / 调度器 ====================
    loss_func = torch.nn.CrossEntropyLoss(
        ignore_index=PAD_IDX, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.98),
        weight_decay=weight_decay)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(T_max or num_epochs) - warmup_epochs,
        eta_min=eta_min)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs])

    # ==================== 记录曲线 ====================
    train_loss_curve = []
    val_loss_curve = []
    lr_curve = []
    best_val_loss = float("inf")
    no_improve = 0

    # ==================== 训练 ====================
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # ---------- 训练阶段 ----------
        model.train()
        loss_sum = 0.0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train")
        for step, (src, tgt) in pbar:
            src = src.to(device)
            tgt = tgt.to(device)

            pred = model(src, tgt[:, :-1])

            if step == len(train_loader) - 1:
                sample_ids = pred[0].argmax(dim=-1).tolist()
                if EOS_IDX in sample_ids:
                    sample_ids = sample_ids[: sample_ids.index(EOS_IDX)]
                print(f"  [src] {en_tokenizer.decode(src[0].tolist(), skip_special_tokens=True)}")
                print(f"  [ref] {zh_tokenizer.decode(tgt[0].tolist(), skip_special_tokens=True)}")
                print(f"  [pred] {zh_tokenizer.decode(sample_ids, skip_special_tokens=True)}")

            pred = pred.contiguous().view(-1, pred.size(-1))
            target = tgt[:, 1:].contiguous().view(-1)
            loss = loss_func(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()

            loss_sum += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_avg_loss = loss_sum / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss_curve.append(train_avg_loss)
        lr_curve.append(current_lr)

        scheduler.step()

        # ---------- 验证阶段 ----------
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)

                pred = model(src, tgt[:, :-1])
                pred = pred.contiguous().view(-1, pred.size(-1))
                target = tgt[:, 1:].contiguous().view(-1)
                loss = loss_func(pred, target)

                val_loss_sum += loss.item()

        val_avg_loss = val_loss_sum / len(val_loader)
        val_loss_curve.append(val_avg_loss)

        print(f"  Train Loss: {train_avg_loss:.4f} | "
              f"Val Loss: {val_avg_loss:.4f} | "
              f"LR: {current_lr:.6f}")

        # ---------- Early Stopping（带容忍度 + warmup 保护期）----------
        if val_avg_loss < best_val_loss - min_delta:
            best_val_loss = val_avg_loss
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  >> 保存最优模型 (val_loss={best_val_loss:.4f})")
        else:
            if epoch + 1 <= warmup_epochs:
                print(f"  -- warmup 保护期，不计入早停 "
                      f"(val_loss={val_avg_loss:.4f}, best={best_val_loss:.4f})")
            else:
                no_improve += 1
                print(f"  -- 验证集未改善 ({no_improve}/{patience}), "
                      f"需下降 >{min_delta:.3f} "
                      f"(当前={val_avg_loss:.4f}, best={best_val_loss:.4f})")
                if no_improve >= patience:
                    print(f"\n  Early Stopping: 连续 {patience} 个 epoch 无改善，停止训练")
                    break

    # ==================== 加载最优模型 ====================
    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True)
    )
    print(f"已加载最优模型 checkpoint: {save_path}")

    # ==================== 绘制曲线 ====================
    _plot_curves(train_loss_curve, val_loss_curve, lr_curve)

    return model


def _plot_curves(train_loss, val_loss, lr_curve):
    """绘制训练/验证损失曲线和学习率曲线。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_loss, label="Train Loss", color="blue")
    ax1.plot(val_loss,   label="Val Loss",   color="red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(lr_curve, label="Learning Rate", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("训练曲线已保存至 training_curves.png")


if __name__ == "__main__":
    train()
