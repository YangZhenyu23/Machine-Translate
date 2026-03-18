import builtins
import functools
from contextlib import contextmanager

import torch  # pyright: ignore[reportMissingImports]
from torch.utils.data import Dataset, DataLoader  # pyright: ignore[reportMissingImports]
from torch.nn.utils.rnn import pad_sequence  # pyright: ignore[reportMissingImports]
from datasets import load_dataset  # pyright: ignore[reportMissingImports]
from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]


@contextmanager
def _force_utf8():
    """临时将所有文本模式的 open() 默认编码改为 utf-8（修复 Windows GBK 问题）。"""
    _original = builtins.open

    @functools.wraps(_original)
    def _open(*args, **kwargs):
        mode = args[1] if len(args) >= 2 else kwargs.get("mode", "r")
        if "b" not in str(mode) and "encoding" not in kwargs:
            kwargs["encoding"] = "utf-8"
        return _original(*args, **kwargs)

    builtins.open = _open
    try:
        yield
    finally:
        builtins.open = _original

# --------------- 加载预训练分词器（仅加载 tokenizer，不加载模型权重） ---------------
en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
zh_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# --------------- 特殊 token 索引（与 BERT tokenizer 对齐） ---------------
PAD_IDX = en_tokenizer.pad_token_id       # 0
UNK_IDX = en_tokenizer.unk_token_id       # 100
SOS_IDX = zh_tokenizer.cls_token_id       # 101  [CLS] 作为句首
EOS_IDX = zh_tokenizer.sep_token_id       # 102  [SEP] 作为句尾


class TranslationDataset(Dataset):
    def __init__(self, en_sentences, zh_sentences, max_len=128):
        self.en_sentences = en_sentences
        self.zh_sentences = zh_sentences
        self.max_len = max_len

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        en_ids = en_tokenizer.encode(
            self.en_sentences[idx],
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=True,
        )
        zh_ids = zh_tokenizer.encode(
            self.zh_sentences[idx],
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=True,
        )
        return (torch.tensor(en_ids, dtype=torch.long),
                torch.tensor(zh_ids, dtype=torch.long))


def collate_fn(batch):
    """将不等长序列 padding 到同一长度。"""
    en_batch, zh_batch = zip(*batch)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX, batch_first=True)
    zh_batch = pad_sequence(zh_batch, padding_value=PAD_IDX, batch_first=True)
    return en_batch, zh_batch


def load_flores_data(batch_size=32, max_len=128, augment_factor=4):
    """
    下载 FLORES-200 数据集并构建训练 / 验证 DataLoader。

    dev（997 句）→ 训练集（经数据增强扩充），devtest（1012 句）→ 验证集。
    augment_factor: 每条原始样本额外生成的增强副本数，0 表示不增强。
    """
    from augumentation import augment_parallel_data

    print("正在下载 FLORES 数据集 ...")
    with _force_utf8():
        en_data = load_dataset("facebook/flores", "eng_Latn", trust_remote_code=True)
        zh_data = load_dataset("facebook/flores", "zho_Hans", trust_remote_code=True)

    train_en = [item["sentence"] for item in en_data["dev"]]
    train_zh = [item["sentence"] for item in zh_data["dev"]]
    val_en   = [item["sentence"] for item in en_data["devtest"]]
    val_zh   = [item["sentence"] for item in zh_data["devtest"]]

    print(f"原始训练集: {len(train_en)} 句 | 验证集: {len(val_en)} 句")

    if augment_factor > 0:
        train_en, train_zh = augment_parallel_data(
            train_en, train_zh, augment_factor=augment_factor)
    print(f"英文词表大小: {en_tokenizer.vocab_size} | "
          f"中文词表大小: {zh_tokenizer.vocab_size}")

    train_dataset = TranslationDataset(train_en, train_zh, max_len)
    val_dataset   = TranslationDataset(val_en,   val_zh,   max_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    return train_loader, val_loader, en_tokenizer, zh_tokenizer


if __name__ == "__main__":
    train_loader, val_loader, en_tok, zh_tok = load_flores_data()
    for en, zh in train_loader:
        print(f"English batch shape: {en.shape}")
        print(f"Chinese batch shape: {zh.shape}")
        break
