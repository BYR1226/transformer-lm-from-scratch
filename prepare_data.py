import json
from pathlib import Path
from cs336_basics.tokenizer import Tokenizer, train_bpe

train_txt_path = "data/TinyStoriesV2-GPT4-train.txt"
valid_txt_path = "data/TinyStoriesV2-GPT4-valid.txt"

def save_tokenizer(vocab, merges, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # vocab: token_id -> bytes
    # JSON 的键只能是字符串，因此保存成 token_hex -> token_id
    vocab_json = {
        token_bytes.hex(): token_id
        for token_id, token_bytes in vocab.items()
    }
    with open(
        output_dir / "vocab.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            vocab_json,
            f,
            ensure_ascii=False,
            indent=2,
        )
    # 每行保存一对待合并 token，用制表符分隔
    with open(
        output_dir / "merges.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for left, right in merges:
            f.write(f"{left.hex()}\t{right.hex()}\n")

#加载函数
def load_tokenizer_files(input_dir):
    input_dir = Path(input_dir)
    with open(
        input_dir / "vocab.json",
        "r",
        encoding="utf-8",
    ) as f:
        saved_vocab = json.load(f)

    vocab = {
        int(token_id): bytes.fromhex(token_hex)
        for token_hex, token_id in saved_vocab.items()
    }

    merges = []

    with open(
        input_dir / "merges.txt",
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.rstrip("\n")

            if not line:
                continue

            left_hex, right_hex = line.split("\t")
            merges.append(
                (
                    bytes.fromhex(left_hex),
                    bytes.fromhex(right_hex),
                )
            )

    return vocab, merges


vocab, merges = train_bpe(
    input_path = train_txt_path,
    vocab_size = 10000,
    special_tokens = ["<|endoftext|>"]
)

save_tokenizer(
    vocab=vocab,
    merges=merges,
    output_dir="data/tinystories_tokenizer",
)

vocab, merges = load_tokenizer_files(
    "data/tinystories_tokenizer"
)

tokenizer = Tokenizer(
    vocab = vocab,
    merges = merges,
    special_tokens = ["<|endoftext|>"]
)

'''
临时验证
original = "Once upon a time, there was a little cat."

ids = tokenizer.encode(original)
restored = tokenizer.decode(ids)

assert restored == original
print(ids)
print(restored)
'''


