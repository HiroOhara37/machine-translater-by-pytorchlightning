import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch import bool, float32, long
from torchtyping import TensorType
from train import TrainArguments
from Transformer_model import Transformer, generate_square_subsequent_mask
from transformers import AutoTokenizer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args() -> TrainArguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_mode", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--max_len", default=500, type=int)
    parser.add_argument("--epoch_num", default=10, type=int)
    parse_args: argparse.Namespace = parser.parse_args()
    args: TrainArguments = TrainArguments(
        data_mode=parse_args.data_mode,
        batch_size=parse_args.batch_size,
        max_len=parse_args.max_len,
        epoch_num=parse_args.epoch_num,
    )

    return args


def greedy_decode(
    model: Transformer,
    src: TensorType["batch_size", "src_len", "d_model", float32],
    src_mask: TensorType["src_len", "src_len", bool],
    bos: int,
    eos: int,
) -> list[int]:
    src_len: int = src.size(1)
    src.to(DEVICE)
    src_mask.to(DEVICE)

    memory: TensorType[1, src_len, float32] = model.encode(src, src_mask)
    tgt: TensorType[1, 1, long] = torch.ones(1, 1).fill_(bos).type(long).to(DEVICE)

    tgt_tokens: list[int] = []
    for i in range(src_len - 1):
        memory.to(DEVICE)
        tgt_mask: TensorType[1, 1, bool] = (
            generate_square_subsequent_mask(1).type(bool)
        ).to(DEVICE)

        model_output: TensorType[1, 1, "d_model"] = model.decode(tgt, memory, tgt_mask)
        print(model_output.shape)
        prob = model.out(model_output[:, -1])
        print(prob.shape)
        _, max_index = torch.max(prob, dim=1)
        next_word: int = max_index.item()
        tgt_tokens.append(next_word)
        tgt = torch.ones(1, 1).fill_(next_word).type(long)
        sys.exit()


args = get_args()

read_data_path: Path = Path("data/NTT/persona.txt")
data_frame: pd.DataFrame = pd.read_table(read_data_path)

# tokenzer設定
ja_tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
ja_vocab_size: int = len(ja_tokenizer.vocab)
en_vocab_size: int = len(en_tokenizer.vocab)
en_bos_token: int = en_tokenizer.convert_tokens_to_ids("[CLS]")
en_eos_token: int = en_tokenizer.convert_tokens_to_ids("[SEP]")

# モデル生成
model: Transformer = Transformer(
    src_vocab_size=ja_vocab_size,
    tgt_vocab_size=en_vocab_size,
    max_len=args.max_len,
)
model_file: Path = Path(
    "ja2en_translate/model/", "ja2en_" + args.data_mode + "_model_cpu.pth"
)
model.load_state_dict(torch.load(model_file))
model.to(DEVICE)

write_data_file: Path = Path(
    "ja2en_translate/result_data/", args.data_mode + "_trans.txt"
)
with open(file=write_data_file, mode="w", encoding="utf-8") as fw:
    for idx, text in enumerate(list(data_frame["text"])):
        if idx > 5:
            break
        src_tokens: list[int] = ja_tokenizer.encode(text)
        src_len: int = len(src_tokens)
        src: TensorType[1, src_len, long] = torch.LongTensor(src_tokens).unsqueeze(0)
        src_mask: TensorType[src_len, src_len, bool] = (
            torch.zeros(src_len, src_len)
        ).type(bool)

        tgt_tokens: list[int] = greedy_decode(
            model, src, src_mask, en_bos_token, en_eos_token
        )
        print(tgt_tokens)
        print(en_tokenizer.decode(tgt_tokens))
        print(text)
