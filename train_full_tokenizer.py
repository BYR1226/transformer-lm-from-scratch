import time
from pathlib import Path

from cs336_basics.tokenizer import train_bpe
from prepare_data import (
    save_tokenizer,
    load_tokenizer_files,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

train_txt_path = (
    DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
)

tokenizer_dir = (
    DATA_DIR / "tinystories_tokenizer"
)


def main():
    if not train_txt_path.exists():
        raise FileNotFoundError(train_txt_path)

    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"

    # 避免误操作导致重新训练
    if vocab_path.exists() and merges_path.exists():
        print("正式 tokenizer 已经存在：")
        print(vocab_path)
        print(merges_path)
        print("如需重新训练，请先手动删除该目录。")
        return

    print("开始训练正式 TinyStories tokenizer")
    print("训练文件：", train_txt_path)
    print("目标词表大小：10000")
    print("这个过程可能耗时较长", flush=True)

    start_time = time.perf_counter()

    vocab, merges = train_bpe(
        input_path = train_txt_path,
        vocab_size = 10_000,
        special_tokens=["<|endoftext|>"],
    )

    elapsed = time.perf_counter() - start_time

    print(
        f"BPE训练完成：vocab={len(vocab)}, "
        f"merges={len(merges)}"
    )
    print(
        f"耗时：{elapsed / 60:.2f} 分钟"
    )

    save_tokenizer(
        vocab=vocab,
        merges=merges,
        output_dir=tokenizer_dir,
    )

    # 立即重新加载并验证
    loaded_vocab, loaded_merges = (
        load_tokenizer_files(tokenizer_dir)
    )

    assert loaded_vocab == vocab
    assert loaded_merges == merges

    print("正式 tokenizer 保存并验证成功")
    print("保存目录：", tokenizer_dir)


if __name__ == "__main__":
    main()