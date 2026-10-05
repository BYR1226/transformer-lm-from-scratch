import gc
import heapq
import time
import regex as re

from collections import defaultdict
from typing import Iterable, Iterator

PAT = (
    r"""'(?:[sdmt]|ll|ve|re)|"""
    r""" ?\p{L}+|"""
    r""" ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|"""
    r"""\s+(?!\S)|"""
    r"""\s+"""
)

PRE_TOKEN_PATTERN = re.compile(PAT)


def merge_tokens(token_seq, pair, new_token):
    """
    在一个pre-token序列中，从左到右合并指定pair。
    """
    output = []
    left, right = pair
    index = 0
    seq_len = len(token_seq)

    while index < seq_len:
        if (
            index + 1 < seq_len
            and token_seq[index] == left
            and token_seq[index + 1] == right
        ):
            output.append(new_token)
            index += 2
        else:
            output.append(token_seq[index])
            index += 1

    return output


def _find_next_special(buffer, delimiters):
    """
    在二进制buffer中寻找最早出现的特殊token。
    如果多个特殊token从同一位置开始，优先选择最长的，
    与按长度降序构造正则的行为一致。
    """
    best_position = None
    best_delimiter = None

    for delimiter in delimiters:
        position = buffer.find(delimiter)

        if position == -1:
            continue

        if (
            best_position is None
            or position < best_position
            or (
                position == best_position
                and len(delimiter) > len(best_delimiter)
            )
        ):
            best_position = position
            best_delimiter = delimiter

    return best_position, best_delimiter

def _decode_text_with_universal_newlines(
    data: bytes,
) -> str:
    """
    模拟文本模式newline=None的通用换行行为。
    """
    text = data.decode("utf-8")

    return text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

def _iter_text_segments(
    input_path,
    special_tokens,
    chunk_size=4 * 1024 * 1024,
):  #如果不采用流式读取，一次性读入整个数据集在linux上极有可能因运行内存不够被强制OOM！
    """
    流式读取文本，并以特殊token作为安全边界。
    TinyStories使用<|endoftext|>分隔文档，因此内存中通常
    只保留一个未结束文档，而不是整个2.1GB文件。
    如果没有特殊token，则为了维持原始预分词语义，
    退回到一次性读取。TinyStories正式训练会提供
    <|endoftext|>，因此会走流式路径。
    """
    if not special_tokens:
        with open(
            input_path,
            "r",
            encoding="utf-8",
        ) as file:
            yield file.read()

        return

    delimiters = sorted(
        {
            token.encode("utf-8")
            for token in special_tokens
        },
        key=len,
        reverse=True,
    )

    buffer = b""

    with open(input_path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break

            buffer += chunk

            while True:
                position, delimiter = _find_next_special(
                    buffer,
                    delimiters,
                )
                if position is None:
                    break

                segment = buffer[:position]

                if segment:
                    yield _decode_text_with_universal_newlines(
                        segment
                    )
                # 丢弃特殊token本身
                buffer = buffer[
                    position + len(delimiter):
                ]

    if buffer:
        yield _decode_text_with_universal_newlines(
            buffer
        )


def _pair_multiplicities(token_seq):
    """
    返回一个pre-token内部每个相邻pair出现的次数。
    例如：
        (1, 1, 1) -> {(1, 1): 2}
    """
    result = defaultdict(int)

    for left, right in zip(
        token_seq,
        token_seq[1:],
    ):
        result[(left, right)] += 1

    return result

def train_bpe(
    input_path,
    vocab_size,
    special_tokens,
):
    """
    使用流式预分词与增量pair统计训练BPE。

    返回：
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
    """
    if special_tokens is None:
        special_tokens = []

    if vocab_size < 256 + len(special_tokens):
        raise ValueError(
            "vocab_size不能小于256加特殊token数量"
        )

    start_time = time.perf_counter()

    # 1. 初始化基础字节词表和特殊token
    vocab: dict[int, bytes] = {
        token_id: bytes([token_id])
        for token_id in range(256)
    }

    max_id = 255

    for special_token in special_tokens:
        max_id += 1
        vocab[max_id] = special_token.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []


    # 2. 流式预分词，只保存不同pre-token及其频率
    counts: dict[tuple[int, ...], int] = defaultdict(int)

    segment_count = 0

    print(
        "开始流式预分词...",
        flush=True,
    )

    for segment in _iter_text_segments(
        input_path,
        special_tokens,
    ):
        for match in PRE_TOKEN_PATTERN.finditer(segment):
            byte_tuple = tuple(
                match.group().encode("utf-8")
            )
            counts[byte_tuple] += 1

        segment_count += 1

        if segment_count % 100_000 == 0:
            elapsed = time.perf_counter() - start_time

            print(
                f"预分词进度："
                f"segments={segment_count:,}, "
                f"unique_pre_tokens={len(counts):,}, "
                f"elapsed={elapsed / 60:.1f}min",
                flush=True,
            )

    print(
        f"流式预分词完成："
        f"segments={segment_count:,}, "
        f"unique_pre_tokens={len(counts):,}",
        flush=True,
    )

    # --------------------------------------------------
    # 3. 将pre-token转换成稳定的word ID
    #
    # words[word_id]      当前token ID序列
    # word_frequencies    该pre-token在语料中的频率
    # --------------------------------------------------

    words: list[tuple[int, ...]] = []
    word_frequencies: list[int] = []

    for token_seq, frequency in counts.items():
        words.append(token_seq)
        word_frequencies.append(frequency)

    # 释放原counts字典，避免后续重复占用内存
    del counts
    gc.collect()

    # --------------------------------------------------
    # 4. 构造增量统计结构
    #
    # pair_counts:
    #     pair在整个语料中的加权出现次数
    #
    # pair_to_words:
    #     pair -> 包含该pair的word ID集合
    #
    # pairs_by_count:
    #     frequency -> 当前具有该频率的pair集合
    #
    # active_counts + count_heap:
    #     快速得到当前最大pair频率
    # --------------------------------------------------

    pair_counts: dict[tuple[int, int], int] = (
        defaultdict(int)
    )

    pair_to_words: dict[
        tuple[int, int],
        set[int],
    ] = defaultdict(set)

    print(
        "开始建立pair反向索引...",
        flush=True,
    )

    for word_id, token_seq in enumerate(words):
        frequency = word_frequencies[word_id]

        multiplicities = _pair_multiplicities(
            token_seq
        )

        for pair, multiplicity in multiplicities.items():
            pair_counts[pair] += (
                frequency * multiplicity
            )
            pair_to_words[pair].add(word_id)

    pairs_by_count: dict[
        int,
        set[tuple[int, int]],
    ] = defaultdict(set)

    active_counts: set[int] = set()
    count_heap: list[int] = []

    for pair, frequency in pair_counts.items():
        pairs_by_count[frequency].add(pair)

    for frequency in pairs_by_count:
        active_counts.add(frequency)
        heapq.heappush(count_heap, -frequency)

    print(
        f"pair反向索引建立完成："
        f"unique_pairs={len(pair_counts):,}",
        flush=True,
    )

    def remove_from_frequency_bucket(
        pair,
        frequency,
    ):
        """
        将pair从旧频率桶中移除。
        """
        bucket = pairs_by_count.get(frequency)

        if bucket is None:
            return

        bucket.discard(pair)

        if not bucket:
            del pairs_by_count[frequency]
            active_counts.discard(frequency)

    def add_to_frequency_bucket(
        pair,
        frequency,
    ):
        """
        将pair添加到新频率桶中。
        """
        if frequency not in active_counts:
            active_counts.add(frequency)
            heapq.heappush(
                count_heap,
                -frequency,
            )

        pairs_by_count[frequency].add(pair)

    def change_pair_count(pair, delta):
        """
        增量修改一个pair的全局频率，同时维护频率桶。
        """
        if delta == 0:
            return

        old_frequency = pair_counts.get(pair, 0)

        if old_frequency > 0:
            remove_from_frequency_bucket(
                pair,
                old_frequency,
            )

        new_frequency = old_frequency + delta

        if new_frequency < 0:
            raise RuntimeError(
                f"pair计数变成负数："
                f"pair={pair}, "
                f"old={old_frequency}, "
                f"delta={delta}"
            )

        if new_frequency == 0:
            pair_counts.pop(pair, None)
        else:
            pair_counts[pair] = new_frequency
            add_to_frequency_bucket(
                pair,
                new_frequency,
            )

    def get_best_pair():
        """
        取得当前频率最高的pair。

        频率相同时，使用token对应bytes的字典序，
        保持与原始实现相同的tie-break规则。
        """
        while count_heap:
            frequency = -count_heap[0]

            if frequency in active_counts:
                break

            heapq.heappop(count_heap)

        if not count_heap:
            return None

        max_frequency = -count_heap[0]
        candidates = pairs_by_count[max_frequency]

        return max(
            candidates,
            key=lambda pair: (
                vocab[pair[0]],
                vocab[pair[1]],
            ),
        )

    # --------------------------------------------------
    # 5. 增量BPE merge
    # --------------------------------------------------

    print(
        "开始增量BPE合并...",
        flush=True,
    )

    merge_number = 0

    while len(vocab) < vocab_size:
        best_pair = get_best_pair()

        if best_pair is None:
            print(
                "已经没有可合并pair，提前结束",
                flush=True,
            )
            break

        best_frequency = pair_counts[best_pair]

        left, right = best_pair

        left_bytes = vocab[left]
        right_bytes = vocab[right]

        # 先复制受影响word集合，因为更新反向索引时
        # pair_to_words[best_pair]会发生变化
        affected_word_ids = tuple(
            pair_to_words.get(best_pair, ())
        )

        if not affected_word_ids:
            raise RuntimeError(
                f"最佳pair没有对应word：{best_pair}"
            )

        max_id += 1
        new_token = max_id

        vocab[new_token] = (
            left_bytes + right_bytes
        )

        merges.append(
            (
                left_bytes,
                right_bytes,
            )
        )

        for word_id in affected_word_ids:
            old_tokens = words[word_id]
            word_frequency = word_frequencies[word_id]

            old_multiplicities = (
                _pair_multiplicities(old_tokens)
            )

            new_tokens = tuple(
                merge_tokens(
                    old_tokens,
                    best_pair,
                    new_token,
                )
            )

            new_multiplicities = (
                _pair_multiplicities(new_tokens)
            )

            affected_pairs = (
                old_multiplicities.keys()
                | new_multiplicities.keys()
            )

            for pair in affected_pairs:
                old_multiplicity = (
                    old_multiplicities.get(pair, 0)
                )
                new_multiplicity = (
                    new_multiplicities.get(pair, 0)
                )

                multiplicity_delta = (
                    new_multiplicity
                    - old_multiplicity
                )

                if multiplicity_delta != 0:
                    change_pair_count(
                        pair,
                        multiplicity_delta
                        * word_frequency,
                    )

                # 维护pair -> word反向索引
                if (
                    old_multiplicity > 0
                    and new_multiplicity == 0
                ):
                    word_set = pair_to_words.get(pair)

                    if word_set is not None:
                        word_set.discard(word_id)

                        if not word_set:
                            del pair_to_words[pair]

                elif (
                    old_multiplicity == 0
                    and new_multiplicity > 0
                ):
                    pair_to_words[pair].add(word_id)

            words[word_id] = new_tokens

        merge_number += 1

        if (
            merge_number == 1
            or merge_number % 100 == 0
            or len(vocab) == vocab_size
        ):
            elapsed = time.perf_counter() - start_time

            print(
                f"BPE进度："
                f"vocab={len(vocab)}/{vocab_size}, "
                f"merges={merge_number:,}, "
                f"best_frequency={best_frequency:,}, "
                f"active_pairs={len(pair_counts):,}, "
                f"elapsed={elapsed:.2f}s",
                flush=True,
            )

    elapsed = time.perf_counter() - start_time

    print(
        f"BPE训练结束："
        f"vocab={len(vocab)}, "
        f"merges={len(merges)}, "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )

    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab,
        merges,
        special_tokens=None,
    ):
        self.vocab = vocab
        self.merges = merges

        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens

        self.special_token_set = set(
            self.special_tokens
        )

        self.bytes_to_id: dict[bytes, int] = {
            token_bytes: token_id
            for token_id, token_bytes
            in self.vocab.items()
        }

        self.merge_ranks: dict[
            tuple[bytes, bytes],
            int,
        ] = {
            merge: rank
            for rank, merge in enumerate(merges)
        }

        self.special_ids: dict[str, int] = {}

        next_id = (
            max(self.vocab.keys()) + 1
            if self.vocab
            else 0
        )

        for token_str in self.special_tokens:
            token_bytes = token_str.encode("utf-8")

            if token_bytes in self.bytes_to_id:
                token_id = self.bytes_to_id[
                    token_bytes
                ]
            else:
                token_id = next_id
                next_id += 1

                self.vocab[token_id] = token_bytes
                self.bytes_to_id[token_bytes] = (
                    token_id
                )

            self.special_ids[token_str] = token_id

        if self.special_tokens:
            sorted_specials = sorted(
                self.special_tokens,
                key=len,
                reverse=True,
            )

            escaped_specials = [
                re.escape(token)
                for token in sorted_specials
            ]

            special_pattern = "|".join(
                escaped_specials
            )

            self.special_split_pattern = re.compile(
                f"({special_pattern})"
            )
        else:
            self.special_split_pattern = None

        # 重复pre-token直接复用编码结果
        self.encode_cache: dict[
            bytes,
            tuple[int, ...],
        ] = {}

    def _encode_pre_token(
        self,
        pre_token,
    ) -> tuple[int, ...]:
        bytes_data = pre_token.encode("utf-8")

        cached = self.encode_cache.get(bytes_data)

        if cached is not None:
            return cached

        tokens: list[bytes] = [
            bytes([byte])
            for byte in bytes_data
        ]

        while len(tokens) > 1:
            best_pair = None
            best_rank = None

            for left, right in zip(
                tokens,
                tokens[1:],
            ):
                pair = (left, right)
                rank = self.merge_ranks.get(pair)

                if rank is not None and (
                    best_rank is None
                    or rank < best_rank
                ):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            new_tokens = []
            index = 0

            while index < len(tokens):
                if (
                    index + 1 < len(tokens)
                    and tokens[index] == best_pair[0]
                    and tokens[index + 1]
                    == best_pair[1]
                ):
                    new_tokens.append(
                        tokens[index]
                        + tokens[index + 1]
                    )
                    index += 2
                else:
                    new_tokens.append(
                        tokens[index]
                    )
                    index += 1

            tokens = new_tokens

        result = tuple(
            self.bytes_to_id[token]
            for token in tokens
        )

        self.encode_cache[bytes_data] = result

        return result

    def encode(self, text: str) -> list[int]:
        final_ids = []

        if self.special_split_pattern is not None:
            parts = self.special_split_pattern.split(
                text
            )
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue

            if part in self.special_token_set:
                final_ids.append(
                    self.special_ids[part]
                )
                continue

            for match in PRE_TOKEN_PATTERN.finditer(
                part
            ):
                final_ids.extend(
                    self._encode_pre_token(
                        match.group()
                    )
                )

        return final_ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )

        return token_bytes.decode(
            "utf-8",
            errors="replace",
        )
