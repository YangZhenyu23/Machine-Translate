import torch  # pyright: ignore[reportMissingImports]
import sacrebleu  # pyright: ignore[reportMissingImports]

from dataset import (
    en_tokenizer, zh_tokenizer,
    PAD_IDX, SOS_IDX, EOS_IDX,
    load_flores_data,
)
from model import build_model


# ====================================================================
#  Greedy Decoding（自回归贪心解码）
# ====================================================================

@torch.no_grad()
def greedy_decode(model, src, max_len=128, device="cpu"):
    """
    对一个 batch 的英文 token ids 执行贪心解码，返回中文 token ids。
    """
    model.eval()
    src = src.to(device)
    batch_size = src.size(0)

    src_pad_mask = model.make_pad_mask(src)
    memory = model.encode(src, src_key_padding_mask=src_pad_mask)

    ys = torch.full((batch_size, 1), SOS_IDX, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_mask = model.generate_square_subsequent_mask(ys.size(1), device)
        tgt_pad_mask = model.make_pad_mask(ys)

        out = model.decode(
            ys, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        logits = model.fc_out(out[:, -1, :])
        next_token = logits.argmax(dim=-1)

        next_token = next_token.masked_fill(finished, PAD_IDX)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

        finished = finished | (next_token == EOS_IDX)
        if finished.all():
            break

    return ys


def decode_ids(token_ids, tokenizer):
    """将 token id 列表解码为字符串，截断到 EOS。"""
    ids = token_ids if isinstance(token_ids, list) else token_ids.tolist()
    if EOS_IDX in ids:
        ids = ids[: ids.index(EOS_IDX)]
    return tokenizer.decode(ids, skip_special_tokens=True)


# ====================================================================
#  翻译单句
# ====================================================================

def translate(model, sentence, device="cpu", max_len=128):
    """翻译一条英文句子，返回中文字符串。"""
    src_ids = en_tokenizer.encode(sentence, add_special_tokens=True)
    src = torch.tensor([src_ids], dtype=torch.long)
    out_ids = greedy_decode(model, src, max_len=max_len, device=device)
    return decode_ids(out_ids[0], zh_tokenizer)


# ====================================================================
#  BLEU 评估
# ====================================================================

def evaluate_bleu(model, val_loader, device="cpu", max_len=128):
    """在验证集上计算 BLEU 分数（sacrebleu）。"""
    model.eval()
    hypotheses = []
    references = []

    for src, tgt in val_loader:
        out_ids = greedy_decode(model, src, max_len=max_len, device=device)

        for i in range(src.size(0)):
            hyp = decode_ids(out_ids[i], zh_tokenizer)
            ref = decode_ids(tgt[i], zh_tokenizer)
            hypotheses.append(hyp)
            references.append(ref)

    bleu = sacrebleu.corpus_bleu(
        hypotheses, [references], tokenize="zh"
    )
    return bleu, hypotheses, references


# ====================================================================
#  展示样例翻译
# ====================================================================

def show_samples(model, val_loader, device="cpu", n=10, max_len=128):
    """从验证集中取前 n 条，打印 src / ref / hyp 三行对照。"""
    model.eval()
    shown = 0
    for src, tgt in val_loader:
        out_ids = greedy_decode(model, src, max_len=max_len, device=device)
        for i in range(src.size(0)):
            if shown >= n:
                return
            src_text = decode_ids(src[i], en_tokenizer)
            ref_text = decode_ids(tgt[i], zh_tokenizer)
            hyp_text = decode_ids(out_ids[i], zh_tokenizer)

            print(f"\n--- 样例 {shown + 1} ---")
            print(f"  [EN]  {src_text}")
            print(f"  [REF] {ref_text}")
            print(f"  [HYP] {hyp_text}")
            shown += 1


# ====================================================================
#  加载检查点并评估
# ====================================================================

def load_and_evaluate(
    checkpoint_path="best_model.pt",
    batch_size=64,
    max_len=128,
    d_model=256,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=512,
    n_samples=10,
):
    device = torch.device("cuda")

    _, val_loader, _, _ = load_flores_data(batch_size=batch_size, max_len=max_len)

    model = build_model(
        src_vocab_size=en_tokenizer.vocab_size,
        tgt_vocab_size=zh_tokenizer.vocab_size,
        device=device,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        max_len=max_len,
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    print(f"已加载模型检查点: {checkpoint_path}\n")

    bleu, hypotheses, references = evaluate_bleu(
        model, val_loader, device=device, max_len=max_len)
    print(f"验证集 BLEU: {bleu.score:.2f}")
    print(bleu)

    print("\n" + "=" * 60)
    show_samples(model, val_loader, device=device, n=n_samples, max_len=max_len)

    return bleu


if __name__ == "__main__":
    load_and_evaluate()
