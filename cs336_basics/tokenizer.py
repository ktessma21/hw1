import json
from heapq import heappush, heappop
from itertools import pairwise # note that this could actually be done via zip(), but zip takes two args arr, arr[1:] - they are copies of each other. needs O(N) extra memory

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
import regex as re

# this is a good paper. https://dl.acm.org/doi/pdf/10.1145/3808330

class Tokenizer:
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """Load GPT-2 byte-encoded JSON vocabulary and space-separated merges."""
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        byte_decoder = {chr(byte): byte for byte in visible}
        for offset, byte in enumerate(byte for byte in range(256) if byte not in visible):
            byte_decoder[chr(256 + offset)] = byte

        def decode_token(token):
            return bytes(byte_decoder[character] for character in token)

        with open(vocab_filepath, encoding="utf-8") as vocab_file:
            vocab = {
                token_id: decode_token(token)
                for token, token_id in json.load(vocab_file).items()
            }
        merges = []
        with open(merges_filepath, encoding="utf-8") as merges_file:
            for line in merges_file:
                line = line.rstrip("\r\n")
                if not line or line.startswith("#version:"):
                    continue
                left, right = line.split(" ")
                merges.append((decode_token(left), decode_token(right)))
        return cls(vocab, merges, special_tokens)

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        for token in special_tokens or []:
            encoded = token.encode("utf-8")
            if encoded not in self.vocab.values():
                self.vocab[max(self.vocab, default=-1) + 1] = encoded
        vocab = self.vocab
        self.token_to_id = {token: token_id for token_id, token in vocab.items()}
        self.merge_rank = {
                        pair: rank
                        for rank, pair in enumerate(merges)
                    }
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

        self.special_pattern = (
            re.compile(
                "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
            )
            if self.special_tokens else None
        )


   

    # Change it to Text first approach. 
    
    def encode(self, text: str) -> list[int]:
        pieces = (
            self.special_pattern.split(text)
            if self.special_pattern else [text]
        )

        token_ids = []

        for piece in pieces:
            if not piece:
                continue

            if piece in self.special_tokens:
                token_ids.append(self.token_to_id[piece.encode("utf-8")])
                continue

            for pretoken in re.findall(PAT, piece):
                token_ids.extend(self.encode_pretoken(pretoken))

        return token_ids


    def encode_pretoken(self, pretoken: str) -> list[int]:
        sequence: list[bytes | None] = [
            bytes([b]) for b in pretoken.encode("utf-8")
        ]
        n = len(sequence)

        if n == 0:
            return []

        prev_idx = [i - 1 for i in range(n)]
        next_idx = [i + 1 for i in range(n)]
        next_idx[-1] = -1

        heap: list[tuple[int, int]] = []

        for i, (left, right) in enumerate(pairwise(sequence)):
            if left is None or right is None:
                continue

            rank = self.merge_rank.get((left, right))
            if rank is not None:
                heappush(heap, (rank, i))

        while heap:
            rank, i = heappop(heap)
            j = next_idx[i]

            if j == -1:
                continue

            left_token = sequence[i]
            right_token = sequence[j]

            if left_token is None or right_token is None:
                continue

            if self.merge_rank.get((left_token, right_token)) != rank:
                continue

            merged_token = left_token + right_token
            sequence[i] = merged_token
            sequence[j] = None

            k = next_idx[j]
            next_idx[i] = k

            if k != -1:
                prev_idx[k] = i

            left = prev_idx[i]

            if left != -1:
                left_token = sequence[left]

                if left_token is not None:
                    new_rank = self.merge_rank.get(
                        (left_token, merged_token)
                    )
                    if new_rank is not None:
                        heappush(heap, (new_rank, left))

            right = next_idx[i]

            if right != -1:
                right_token = sequence[right]

                if right_token is not None:
                    new_rank = self.merge_rank.get(
                        (merged_token, right_token)
                    )
                    if new_rank is not None:
                        heappush(heap, (new_rank, i))

        token_ids = []
        i = 0

        while i != -1:
            token = sequence[i]

            if token is not None:
                token_ids.append(self.token_to_id[token])

            i = next_idx[i]

        return token_ids


   
    def encode_iterable(self, iterable):
        buffer = ""
        max_len = max((len(s) for s in self.special_tokens), default=0)

        for chunk in iterable:
            buffer += chunk

            # Reserve enough characters for a split special token.
            limit = max(0, len(buffer) - max_len + 1)

            # Find special tokens in the full buffer.
            matches = (
                list(self.special_pattern.finditer(buffer))
                if self.special_pattern else []
            )

            # Never cut through a recognized special token.
            for match in matches:
                if match.start() < limit < match.end():
                    limit = match.start()
                    break

            # Keep the last ordinary pretoken unfinished.
            pretokens = list(re.finditer(PAT, buffer[:limit]))

            if not pretokens:
                continue

            safe_end = pretokens[-1].start()

            if safe_end > 0:
                yield from self.encode(buffer[:safe_end])
                buffer = buffer[safe_end:]

        if buffer:
            yield from self.encode(buffer)



    
    def decode(self, token_ids):
        # Reject unknown token IDs.
        if any(i not in self.vocab for i in token_ids):
            raise ValueError("Unknown token ID")

        return b"".join(self.vocab[i] for i in token_ids).decode(
                    "utf-8", errors="replace"
                )







# need to give attention what if we have 
        #       sequence = [b"a", b"a", b"a"]
        #       merge = (b"a", b"a")
        # then do you go with [b"aa", b"a"] vs [b"a", b"aa"]. actually simply apply your learned merges in their original order.
        # usually left to right scan. 


# my tokenizer should basically do the same thing as the training 


# def encode(self, text):
#         special_tokens = sorted(self.special_tokens or [], key=len, reverse=True)

#         # 1. Split text while preserving special tokens.
#         if special_tokens:
#             pattern = re.compile(
#                 "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
#             )
#             pieces = pattern.split(text)
#         else:
#             pieces = [text]

#         token_ids = []

#         for piece in pieces:
#             if not piece:
#                 continue

#             # 2. Special tokens already have IDs.
#             if piece in special_tokens:
#                 token_ids.append(self.token_to_id[piece.encode("utf-8")])
#                 continue

#             # 3. Pretokenize using the same regex as training.
#             for pretoken in re.findall(PAT, piece):

#                 # 4. Start with individual UTF-8 byte tokens.
#                 sequence = [bytes([b]) for b in pretoken.encode("utf-8")]

#                 # 5. Apply BPE merges in learned order.
#                 for merge in self.merges:
#                     new_sequence = []
#                     i = 0

#                     while i < len(sequence):
#                         if (
#                             i + 1 < len(sequence)
#                             and (sequence[i], sequence[i + 1]) == merge
#                         ):
#                             new_sequence.append(sequence[i] + sequence[i + 1])
#                             i += 2
#                         else:
#                             new_sequence.append(sequence[i])
#                             i += 1

#                     sequence = new_sequence

#                 # 6. Convert final tokens to IDs.
#                 token_ids.extend(self.token_to_id[token] for token in sequence)

#         return token_ids
# 
# Optimized. 
# implement it with a in-place two-pointer strategy. my thought was with a second array keeping track of the next idx and leaving the main sequence 
    # fragmented with None. But this is not good for CPU memory fetching. 
    # def encode(self, text):
    #     special_tokens = sorted(self.special_tokens or [], key=len, reverse=True)

    #     # 1. Split text while preserving special tokens.
    #     if special_tokens:
    #         pattern = re.compile(
    #             "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
    #         )
    #         pieces = pattern.split(text)
    #     else:
    #         pieces = [text]

    #     token_ids = []

    #     for piece in pieces:
    #         if not piece:
    #             continue

    #         # 2. Special tokens already have IDs.
    #         if piece in special_tokens:
    #             token_ids.append(self.token_to_id[piece.encode("utf-8")])
    #             continue

    #         # 3. Pretokenize using the same regex as training.
    #         for pretoken in re.findall(PAT, piece):

    #             # 4. Start with individual UTF-8 byte tokens.
    #             sequence = [bytes([b]) for b in pretoken.encode("utf-8")]

    #             # 5. Apply BPE merges in learned order.
    #             for merge in self.merges:
    #                 read_ptr = 0
    #                 write_ptr = 0
    #                 n = len(sequence)
                    
    #                 while read_ptr < n:
    #                     if (
    #                         read_ptr + 1 < n 
    #                         and (sequence[read_ptr], sequence[read_ptr + 1]) == merge
    #                     ):
    #                         # Overwrite directly at write_ptr
    #                         sequence[write_ptr] = sequence[read_ptr] + sequence[read_ptr + 1]
    #                         write_ptr += 1
    #                         read_ptr += 2  # Skip both merged elements
    #                     else:
    #                         # No match: copy the element forward to compress the list
    #                         sequence[write_ptr] = sequence[read_ptr]
    #                         write_ptr += 1
    #                         read_ptr += 1
                    
    #                 # Truncate the list to discard the leftover elements at the end
    #                 del sequence[write_ptr:]

    #             # 6. Convert final tokens to IDs.
    #             token_ids.extend(self.token_to_id[token] for token in sequence)

    #     return token_ids


from collections import Counter, defaultdict
import regex as re
from pathlib import Path
from time import perf_counter
import os



def find_max_pair_token(pair_to_frequencies):
    return max(
        pair_to_frequencies,
        key=lambda pair: (pair_to_frequencies[pair], pair), # first frequency, then lexographical ordering
        default=None,
    )

def remove_old(sequences, pair_to_sequences, pair_to_frequencies, affected_sequences):

    for sequence, freq in affected_sequences.items():

        for pair in zip(sequence, sequence[1:]):
            pair_to_frequencies[pair] -= freq
            pair_to_sequences[pair].discard(sequence)

            if pair_to_frequencies[pair] == 0:
                del pair_to_frequencies[pair]

            if not pair_to_sequences[pair]:
                del pair_to_sequences[pair]

        del sequences[sequence]

def build_new(merged_pair, affected_sequences, sequences):
    new_sequences = Counter()

    for sequence, freq in affected_sequences.items():
        new_sequence = []
        i = 0

        while i < len(sequence):
            if (
                i + 1 < len(sequence)
                and (sequence[i], sequence[i + 1]) == merged_pair
            ):
                new_sequence.append(sequence[i] + sequence[i + 1])
                i += 2
            else:
                new_sequence.append(sequence[i])
                i += 1

        new_sequences[tuple(new_sequence)] += freq

    sequences.update(new_sequences)
    return new_sequences


def add_new_pairs(new_sequences, pair_to_sequences, pair_to_frequencies):
    for sequence, freq in new_sequences.items():
        for pair in zip(sequence, sequence[1:]):
            pair_to_sequences[pair].add(sequence)
            pair_to_frequencies[pair] += freq

            
    
            
# remove old, build new and add new  
def update_pair_state(
    merged_pair, sequences, pair_to_sequences, pair_to_frequencies
):
    affected_sequences = {
        sequence: sequences[sequence]
        for sequence in pair_to_sequences[merged_pair]
    }

    # 1. Remove old sequences and their pair contributions.
    remove_old(
       sequences, pair_to_sequences, pair_to_frequencies, affected_sequences
    )

    # 2. Build replacement sequences, preserving their frequencies.
    new_sequences = build_new(
        merged_pair, affected_sequences, sequences
    )

    # 3. Register the new neighboring pairs and their frequencies.
    add_new_pairs(
        new_sequences, pair_to_sequences, pair_to_frequencies
    )

def train_bpe_merges(sequences, vocab, vocab_size, pair_to_sequences, pair_to_frequencies):
    """Add BPE merges to an initialized vocabulary until it reaches vocab_size."""
    merged_tokens = []
    finding_seconds = 0.0
    merging_seconds = 0.0
    training_start = perf_counter()

    while len(vocab) < vocab_size:
        start = perf_counter()
        pair = find_max_pair_token(pair_to_frequencies)
        finding_seconds += perf_counter() - start
        if pair is None:
            break

        start = perf_counter()
        merged_tokens.append(pair)
        vocab[pair[0] + pair[1]] = len(vocab)
        update_pair_state(pair, sequences, pair_to_sequences, pair_to_frequencies)
        merging_seconds += perf_counter() - start

    total_seconds = perf_counter() - training_start
    print(f"Merges performed: {len(merged_tokens):,}")
    print(f"Finding pairs:    {finding_seconds:.3f} seconds")
    print(f"Applying merges:  {merging_seconds:.3f} seconds")
    print(f"Total merge loop: {total_seconds:.3f} seconds")

    return vocab, merged_tokens


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    path = Path(input_path)

    if path.is_file():
        corpus = path.read_text(encoding="utf-8")
    else:
        raise FileNotFoundError(f"File not found: {path}")

    special_tokens = sorted(special_tokens or [], key=len, reverse=True)

    if special_tokens:
        special_pattern = re.compile(
            "|".join(re.escape(token) for token in special_tokens)
        )
        pieces = special_pattern.split(corpus)
    else:
        pieces = [corpus]

    pretoken_counts = Counter()
    for piece in pieces:
        pretoken_counts.update(re.findall(PAT, piece))

    sequences = Counter(
        {
            tuple(bytes([byte]) for byte in text.encode("utf-8")): count 
            # Split UTF-8 text into individual byte tokens; one character may span multiple bytes.            
            for text, count in pretoken_counts.items()
        }
    )

    vocab = {bytes([byte]): byte for byte in range(256)}
    for token in special_tokens:
        vocab[token.encode("utf-8")] = len(vocab)

    # create two mapping. 
    # 1. adjacent pair to sequences that contain them. 
    # 2. adjacent pair to their frequencies. 

    pair_to_sequences = defaultdict(set)
    pair_to_frequencies = defaultdict(int)

    for tokens, freq in sequences.items():
        for pair in zip(tokens, tokens[1:]):
            pair_to_sequences[pair].add(tokens)
            pair_to_frequencies[pair] += freq

    vocab, merges = train_bpe_merges(
        sequences, vocab, vocab_size, pair_to_sequences, pair_to_frequencies
    )
    return {token_id: token for token, token_id in vocab.items()}, merges
