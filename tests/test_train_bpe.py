#!/usr/bin/env python3
import json
import time
from collections import Counter

from cs336_basics.tokenizer import Tokenizer, build_new


from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode


# def test_train_bpe_merges_with_initialized_vocab():
#     sequences = Counter(
#         {
#             (b"a", b"b", b"a", b"b"): 2,
#             (b"a", b"b"): 1,
#         }
#     )
#     initial_vocab = {b"a": 0, b"b": 1}

#     vocab, merges = train_bpe_merges(sequences, initial_vocab, vocab_size=3)

#     assert merges == [(b"a", b"b")]
#     assert vocab == {b"a": 0, b"b": 1, b"ab": 2}
#     assert sequences == Counter({(b"ab", b"ab"): 2, (b"ab",): 1})


# def test_pair_index_lists_sequence_once_when_pair_repeats():
#     text = "moonshoots foot"
#     sequence = tuple(bytes([byte]) for byte in text.encode("utf-8"))
#     sequences = Counter({sequence: 1})

#     pair, pair_to_token = find_max_pair_token(sequences)

#     assert pair == (b"o", b"o")
#     assert pair_to_token[pair] == [sequence]

#     merge_tokens(pair, sequences, pair_to_token)

#     assert sequences == Counter(
#         {
#             (
#                 b"m",
#                 b"oo",
#                 b"n",
#                 b"s",
#                 b"h",
#                 b"oo",
#                 b"t",
#                 b"s",
#                 b" ",
#                 b"f",
#                 b"oo",
#                 b"t",
#             ): 1
#         }
#     )


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 1.5
    print(end_time - start_time)


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path) as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path) as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes(
                [gpt2_byte_decoder[token] for token in gpt2_vocab_item]
            )
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_build_new_combines_matching_replacements():
    affected_sequences = Counter({
        (b"a", b"b", b"ab"): 2,
        (b"ab", b"a", b"b"): 3,
    })
    sequences = Counter()  # Old sequences have already been removed.

    result = build_new(
        merged_pair=(b"a", b"b"),
        affected_sequences=affected_sequences,
        sequences=sequences,
    )

    expected = Counter({
        (b"ab", b"ab"): 5,
    })

    assert result == expected
    assert sequences == expected


def test_train_bpe_from_files(tmp_path):
    vocab, merges = run_train_bpe(FIXTURES_PATH / "corpus.en", 500, ["<|endoftext|>"])
    byte_encoder = gpt2_bytes_to_unicode()

    def serialize(token):
        return "".join(byte_encoder[byte] for byte in token)

    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"
    vocab_path.write_text(json.dumps({serialize(token): i for i, token in vocab.items()}), encoding="utf-8")
    merges_path.write_text("\n".join(f"{serialize(left)} {serialize(right)}" for left, right in merges), encoding="utf-8")
    tokenizer = Tokenizer.from_files(str(vocab_path), str(merges_path), ["<|endoftext|>"])
    assert tokenizer.vocab == vocab
    assert list(tokenizer.merge_rank) == merges
    text = "Hello café! <|endoftext|> 🌍"
    assert tokenizer.encode(text) == Tokenizer(vocab, merges, ["<|endoftext|>"]).encode(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text
