"""
数据增强模块 —— 对英文源句进行 **温和** 增强，中文译文保持不变。

核心原则：增强后的句子必须保留原句语义和基本语法。
过于激进的增强（如大量删词、多次交换）会产生"乱码"英文，
导致训练集和验证集分布严重不匹配，引起过早早停。

支持的增强方法（按语义保留度排序）：
  1. Synonym Replacement（同义词替换）—— 最安全，语义完全保留
  2. Random Insertion（随机插入同义词）—— 较安全
  3. Random Swap（相邻词交换）—— 仅交换相邻词对，变化温和
  4. Random Deletion（随机删除）—— 极低概率，仅轻微扰动
"""

import random
import nltk  # pyright: ignore[reportMissingImports]
from nltk.corpus import wordnet  # pyright: ignore[reportMissingImports]

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# ====================================================================
#  辅助函数
# ====================================================================

def _get_synonyms(word):
    """从 WordNet 获取英文单词的同义词集合。"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ").lower()
            if candidate != word.lower():
                synonyms.add(candidate)
    return list(synonyms)


def _tokenize(sentence):
    """简单按空格分词，保留标点。"""
    return sentence.strip().split()


def _detokenize(words):
    return " ".join(words)


# ====================================================================
#  1. Synonym Replacement（同义词替换）—— 最推荐
# ====================================================================

def synonym_replacement(sentence, n=1):
    """
    随机选 n 个词，用 WordNet 同义词替换。
    语义完全保留，是最安全的增强方式。
    """
    words = _tokenize(sentence)
    if len(words) == 0:
        return sentence

    candidates = [i for i, w in enumerate(words) if _get_synonyms(w)]
    random.shuffle(candidates)

    replaced = 0
    for idx in candidates:
        if replaced >= n:
            break
        syns = _get_synonyms(words[idx])
        if syns:
            words[idx] = random.choice(syns)
            replaced += 1

    return _detokenize(words)


# ====================================================================
#  2. Random Insertion（随机插入同义词）
# ====================================================================

def random_insertion(sentence, n=1):
    """
    随机选一个词，找到其同义词并插入到句中随机位置，重复 n 次。
    """
    words = _tokenize(sentence)
    if len(words) == 0:
        return sentence

    new_words = list(words)
    for _ in range(n):
        source_word = random.choice(words)
        syns = _get_synonyms(source_word)
        if syns:
            synonym = random.choice(syns)
            insert_pos = random.randint(0, len(new_words))
            new_words.insert(insert_pos, synonym)

    return _detokenize(new_words)


# ====================================================================
#  3. Adjacent Swap（相邻词交换）—— 温和版 paraphrasing
# ====================================================================

def adjacent_swap(sentence, n=1):
    """
    随机交换 n 对 **相邻** 词的位置（而非任意两个词）。
    相邻交换对语法破坏最小。
    """
    words = _tokenize(sentence)
    if len(words) < 2:
        return sentence

    new_words = list(words)
    for _ in range(n):
        i = random.randint(0, len(new_words) - 2)
        new_words[i], new_words[i + 1] = new_words[i + 1], new_words[i]

    return _detokenize(new_words)


# ====================================================================
#  4. Random Deletion（随机删除）—— 极低概率
# ====================================================================

def random_deletion(sentence, p=0.05):
    """
    以概率 p 随机删除每个词。至少保留原句 80% 的长度。
    """
    words = _tokenize(sentence)
    if len(words) <= 3:
        return sentence

    remaining = [w for w in words if random.random() > p]
    min_keep = max(1, int(len(words) * 0.8))
    if len(remaining) < min_keep:
        remaining = random.sample(words, min_keep)

    return _detokenize(remaining)


# ====================================================================
#  统一增强接口
# ====================================================================

ALL_METHODS = [
    "synonym_replacement",
    "synonym_replacement",
    "random_insertion",
    "adjacent_swap",
    "random_deletion",
]

_METHOD_FN = {
    "synonym_replacement":  lambda s: synonym_replacement(s, n=1),
    "random_insertion":     lambda s: random_insertion(s, n=1),
    "adjacent_swap":        lambda s: adjacent_swap(s, n=1),
    "random_deletion":      lambda s: random_deletion(s, p=0.05),
}


def augment_sentence(sentence, method=None):
    """
    对单条英文句子执行一种增强。
    method 为 None 时按权重随机选取（同义词替换概率最高）。
    """
    if method is None:
        method = random.choice(ALL_METHODS)
    fn = _METHOD_FN.get(method)
    if fn is None:
        raise ValueError(f"未知增强方法: {method}")
    return fn(sentence)


def augment_parallel_data(en_sentences, zh_sentences, augment_factor=2):
    """
    对平行语料进行温和数据增强。

    Parameters
    ----------
    en_sentences : list[str]  英文原句
    zh_sentences : list[str]  中文译文（不做修改）
    augment_factor : int
        每条原始样本生成的增强副本数。
        最终训练集大小 = 原始数据 + 原始数据 × augment_factor

    Returns
    -------
    (aug_en, aug_zh) : 增强后的英文/中文句子列表（包含原始数据）
    """
    aug_en = list(en_sentences)
    aug_zh = list(zh_sentences)

    n_original = len(en_sentences)
    for _ in range(augment_factor):
        for i in range(n_original):
            new_en = augment_sentence(en_sentences[i])
            if new_en.strip():
                aug_en.append(new_en)
                aug_zh.append(zh_sentences[i])

    print(f"数据增强: {n_original} → {len(aug_en)} 句 "
          f"(×{len(aug_en) / n_original:.1f})")
    return aug_en, aug_zh


# ====================================================================
#  测试
# ====================================================================

if __name__ == "__main__":
    test_sentence = "The quick brown fox jumps over the lazy dog."

    print("原句:", test_sentence)
    print()
    for method in list(_METHOD_FN.keys()):
        result = augment_sentence(test_sentence, method=method)
        print(f"[{method}]")
        print(f"  → {result}")
    print()
    print("[随机增强 x5]")
    for _ in range(5):
        print(f"  → {augment_sentence(test_sentence)}")
