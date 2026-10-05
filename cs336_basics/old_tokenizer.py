import regex as re
from collections import defaultdict
from typing import Iterable, Iterator
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def merge_tokens(token_seq, pair, new_token):
    i = 0
    output = []
    a, b = pair
    seq_len = len(token_seq)
    while i < seq_len:
        # 如果当前位置下一个元素存在，且匹配pair
        if i < seq_len - 1 and token_seq[i] == a and token_seq[i + 1] == b:
            output.append(new_token)
            i += 2  # 匹配成功，跳过下一个
        else:
            output.append(token_seq[i])
            i += 1  # 不匹配，只前进一格
    return output

def train_bpe(input_path, vocab_size, special_tokens):
    #读入数据
    with open(input_path, 'r', encoding='utf-8') as f:
        target=f.read()
    # token ID → token 的字节内容
    max_id = 255
    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])

    # merge记录合并的先后顺序
    merge: list[tuple[bytes, bytes]] = []

    counts = defaultdict(int)

    if special_tokens:
        for special_token in special_tokens:
            max_id += 1
            vocab[max_id] = special_token.encode('utf-8')

        escaped_specials = [re.escape(tok) for tok in special_tokens]
        # 构造正则：匹配任意一个特殊token
        special_pat = "|".join(escaped_specials)
        split_pat = re.compile(f"({special_pat})")  # 括号：保留分隔符本身
        parts = split_pat.split(target)
    else:
        parts = [target]

    for part in parts:
        if part not in special_tokens:
            result=re.finditer(PAT, part)
            for match in result:
                pre_token = match.group()
                byte_tuple = tuple(pre_token.encode("utf-8"))
                counts[byte_tuple] += 1
    # print(counts)
    while len(vocab) < vocab_size:
        pairs = defaultdict(int)
        for k, v in counts.items():
            for i in range(len(k) - 1):
                cur = k[i];
                nxt = k[i + 1];
                pairs[(cur, nxt)] += v
        if not pairs: break
        # sorted_pairs = sorted(
        #     pairs.items(),
        #     key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]]),  # 先比较频次，然后比较字典序
        #     reverse=True
        # )#这里每次都排序会导致时间开销较大
        best_pair = max(
            pairs,
            key=lambda pair: (
                pairs[pair],
                vocab[pair[0]],
                vocab[pair[1]],
            ),
        )
        # print(sorted_pairs)
        new_counts = defaultdict(int)
        # best_pair = sorted_pairs[0][0]
        max_id += 1
        vocab[max_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merge.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        for k, v in counts.items():
            new_tokens = merge_tokens(k, best_pair, max_id)
            new_counts[tuple(new_tokens)] += v;

        counts = new_counts
        # print(counts)
        # print(best_pair)
        # print(vocab[max_id])
        # print(merge)
        # print(len(vocab))
    return (vocab,merge)

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            self.special_tokens = []
        else :
            self.special_tokens = special_tokens

        self.bytes_to_id: dict[bytes, int] = {}
        for token_id, token_bytes in self.vocab.items():
            self.bytes_to_id[token_bytes] = token_id

        self.merge_ranks:dict[tuple[bytes, bytes], int] = {}
        for idx,val in enumerate(merges):
            self.merge_ranks[val] = idx

        self.special_ids:dict[str, int] = {}
        for token_str in self.special_tokens:
            token_bytes = token_str.encode('utf-8')
            if token_bytes in self.bytes_to_id:
                self.special_ids[token_str] = self.bytes_to_id[token_bytes]
            else:
                self.special_ids[token_str] = max(self.vocab.keys()) + 1#新特殊token的id
                self.vocab[self.special_ids[token_str]] = token_bytes
                self.bytes_to_id[token_bytes] = self.special_ids[token_str]

    # def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

    def encode(self, text: str) -> list[int]:
        final_ids = []
        if len(self.special_tokens) > 0:#这里沿用之前bpe处理特殊token的逻辑，但是不用在vocab中追加id
            escaped_specials = [re.escape(tok) for tok in self.special_tokens]
            special_pat = "|".join(escaped_specials)
            split_pat = re.compile(f"({special_pat})")
            parts = split_pat.split(text)
        else:
            parts = [text]#part得到的是同时含有正常token和特殊token的字符串列表

        for part in parts:
            if part == "":
                continue
            if part in self.special_tokens:
                final_ids.append(self.special_ids[part])#遇到特殊token直接追加
            else:
                pre_tokens = re.findall(PAT, part)#先进行预分词
                for pre_token in pre_tokens:#对每一个分开的pretoken单独处理，保证不会越界合并
                    bytes_data = pre_token.encode('utf-8')
                    tokens:list[bytes]=[bytes([data]) for data in bytes_data]
                    #开始BPE合并
                    while True:
                        pair=[]
                        for i in range(len(tokens)-1):
                            pair.append((tokens[i],tokens[i+1]))
                        valid_pairs=[p for p in pair if p in self.merge_ranks]#把所有出现在merge_ranks里的pair提取出来
                        if not valid_pairs:#一个都没有时说明合并完成，结束循环
                            break

                        best_pair = min(valid_pairs, key=lambda p: self.merge_ranks[p])#在valid_pair中找排名最小的
                        new_tokens = []
                        i=0
                        n=len(tokens)
                        #开始合并
                        while i<n:
                            if i<n-1 and (tokens[i],tokens[i+1])==best_pair:
                                new_tokens.append(tokens[i]+tokens[i+1])
                                i+=2
                            else:
                                new_tokens.append(tokens[i])
                                i+=1
                        tokens = new_tokens
                    ids = [self.bytes_to_id[token_bytes] for token_bytes in tokens]
                    final_ids += ids
        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        parts=[self.vocab[token_id] for token_id in ids]
        token_bytes=b''.join(parts)
        token_str = token_bytes.decode('utf-8',errors="replace")#遇到不完整编码时产生替换字符而不是UnicodeDecodeError
        return token_str

