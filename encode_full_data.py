import json
import os
import time

from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer
from prepare_data import load_tokenizer_files


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

TRAIN_TEXT_PATH = (
    DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
)

VALID_TEXT_PATH = (
    DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
)

TOKENIZER_DIR = (
    DATA_DIR / "tinystories_tokenizer"
)

TRAIN_OUTPUT_PATH = (
    DATA_DIR / "tinystories_train.bin"
)

VALID_OUTPUT_PATH = (
    DATA_DIR / "tinystories_valid.bin"
)

SPECIAL_TOKEN = "<|endoftext|>"


def normalize_newlines(data: bytes) -> str:
# 模拟Python文本模式的通用换行转换。
    text = data.decode("utf-8")
    return text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )


def encode_file_streaming(
    input_path,
    output_path,
    tokenizer,
    special_token,
    chunk_size=4 * 1024 * 1024,
    flush_token_count=1_000_000,
):
    """
    以special token为文档边界，流式编码数据。
    输出为原始uint16二进制文件，可以通过
    np.memmap直接访问。
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    temporary_path = Path(
        str(output_path) + ".tmp"
    )

    delimiter = special_token.encode("utf-8")
    special_id = tokenizer.special_ids[
        special_token
    ]

    pending_ids = []
    total_tokens = 0
    document_count = 0
    buffer = b""

    start_time = time.perf_counter()

    def flush_pending(output_file):
        nonlocal pending_ids

        if not pending_ids:
            return

        array = np.asarray(
            pending_ids,
            dtype=np.uint16,
        )

        array.tofile(output_file)
        pending_ids.clear()

    print(
        f"开始编码：{input_path}",
        flush=True,
    )

    with open(input_path, "rb") as input_file:
        with open(
            temporary_path,
            "wb",
        ) as output_file:
            while True:
                chunk = input_file.read(chunk_size)

                if not chunk:
                    break

                buffer += chunk
                parts = buffer.split(delimiter)

                # 最后一部分可能还未完整，留给下一轮
                for document_bytes in parts[:-1]:
                    if document_bytes:
                        document = normalize_newlines(
                            document_bytes
                        )

                        document_ids = (
                            tokenizer.encode(document)
                        )

                        pending_ids.extend(
                            document_ids
                        )
                        total_tokens += len(
                            document_ids
                        )

                    # 原始文本中存在一个特殊token，
                    # 因此把其ID写入数据集
                    pending_ids.append(special_id)
                    total_tokens += 1
                    document_count += 1

                    if (
                        len(pending_ids)
                        >= flush_token_count
                    ):
                        flush_pending(output_file)

                    if (
                        document_count % 100_000
                        == 0
                    ):
                        elapsed = (
                            time.perf_counter()
                            - start_time
                        )

                        print(
                            f"编码进度："
                            f"documents="
                            f"{document_count:,}, "
                            f"tokens="
                            f"{total_tokens:,}, "
                            f"elapsed="
                            f"{elapsed / 60:.1f}min",
                            flush=True,
                        )

                buffer = parts[-1]

            # 文件末尾可能没有特殊token
            if buffer:
                document = normalize_newlines(
                    buffer
                )

                document_ids = tokenizer.encode(
                    document
                )

                pending_ids.extend(document_ids)
                total_tokens += len(document_ids)

            flush_pending(output_file)

    # 只有完整编码结束后才替换正式文件
    os.replace(
        temporary_path,
        output_path,
    )

    elapsed = time.perf_counter() - start_time

    metadata = {
        "source": str(input_path),
        "output": str(output_path),
        "dtype": "uint16",
        "token_count": total_tokens,
        "document_count": document_count,
        "elapsed_seconds": elapsed,
    }

    metadata_path = Path(
        str(output_path) + ".json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    expected_bytes = total_tokens * 2
    actual_bytes = output_path.stat().st_size

    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"文件大小不正确："
            f"expected={expected_bytes}, "
            f"actual={actual_bytes}"
        )

    print(
        f"编码完成：{output_path}",
        flush=True,
    )
    print(
        f"documents={document_count:,}, "
        f"tokens={total_tokens:,}, "
        f"size={actual_bytes / 1024**3:.2f}GiB, "
        f"elapsed={elapsed / 60:.2f}min",
        flush=True,
    )

    return total_tokens


def main():
    vocab, merges = load_tokenizer_files(
        TOKENIZER_DIR
    )

    if len(vocab) != 10_000:
        raise ValueError(
            f"正式词表大小应为10000，"
            f"当前为{len(vocab)}"
        )

    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=[SPECIAL_TOKEN],
    )

    max_token_id = max(tokenizer.vocab)

    if max_token_id >= 65536:
        raise ValueError(
            "token ID超出uint16表示范围"
        )

    print("Tokenizer加载完成")
    print("vocab：", len(tokenizer.vocab))
    print("merges：", len(tokenizer.merges))

    train_token_count = encode_file_streaming(
        input_path=TRAIN_TEXT_PATH,
        output_path=TRAIN_OUTPUT_PATH,
        tokenizer=tokenizer,
        special_token=SPECIAL_TOKEN,
    )

    valid_token_count = encode_file_streaming(
        input_path=VALID_TEXT_PATH,
        output_path=VALID_OUTPUT_PATH,
        tokenizer=tokenizer,
        special_token=SPECIAL_TOKEN,
    )

    # 最终重新映射并进行基本验证
    train_data = np.memmap(
        TRAIN_OUTPUT_PATH,
        dtype=np.uint16,
        mode="r",
        shape=(train_token_count,),
    )

    valid_data = np.memmap(
        VALID_OUTPUT_PATH,
        dtype=np.uint16,
        mode="r",
        shape=(valid_token_count,),
    )

    assert len(train_data) == train_token_count
    assert len(valid_data) == valid_token_count

    assert int(train_data.min()) >= 0
    assert int(train_data.max()) < len(vocab)

    assert int(valid_data.min()) >= 0
    assert int(valid_data.max()) < len(vocab)

    print("训练集和验证集memmap验证通过")
    print(
        "train tokens：",
        f"{len(train_data):,}",
    )
    print(
        "valid tokens：",
        f"{len(valid_data):,}",
    )


if __name__ == "__main__":
    main()