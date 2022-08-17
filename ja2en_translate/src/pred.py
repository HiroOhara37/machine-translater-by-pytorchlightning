from pathlib import Path

import pandas as pd
import torch
from torch import Tensor, bool, long
from torchtyping import TensorType
from train import EN_TOKENIZER, JA_TOKENIZER, TrainArguments, get_args
from Transformer_model import DEVICE, Transformer, generate_square_subsequent_mask

INPUT_FILE_PATH: Path = Path("data/sample_directory/sample_file.txt")


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: Tensor,  # TensorType[1, src_len, long]
    src_mask: Tensor,  # TensorType[src_len, src_len, bool]
    src_len: int,
    bos: int,
    eos: int,
) -> list[int]:
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    # TensorType[1, src_len, d_model, float32]
    memory: Tensor = model.encode(src, src_mask).to(DEVICE)
    # TensorType[1, tgt_len, long]
    tgt: Tensor = torch.ones(1, 1).fill_(bos).type(long).to(DEVICE)

    for _ in range(src_len + 5):
        # TensorType[tgt_len, tgt_len, bool]
        tgt_mask: Tensor = (
            (generate_square_subsequent_mask(tgt.size(1))).type(bool).to(DEVICE)
        )

        # TensorType[1, tgt_len, d_model, float32]
        out: Tensor = model.decode(tgt, memory, tgt_mask)
        # TensorType[1, tgt_vocab_size, float32]
        prob: Tensor = model.out(out[:, -1])
        _, max_index = torch.max(prob, dim=1)
        next_word: int = max_index.item()

        if next_word == eos:
            break
        tgt = torch.cat([tgt, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

    tgt_token: list[int] = tgt.squeeze().tolist()

    return tgt_token[1:]


def predict() -> None:
    # 学習データ、設定値の取得
    args: TrainArguments = get_args()
    data_frame: pd.DataFrame = pd.read_table(INPUT_FILE_PATH, names=["text"])
    print(data_frame["text"].head(3))

    # tokenizerの取得
    ja_tokenizer = JA_TOKENIZER
    en_tokenizer = EN_TOKENIZER
    ja_vocab_size: int = len(ja_tokenizer.vocab)  # 32000
    en_vocab_size: int = len(en_tokenizer.vocab)  # 30522

    # ja_bos_token: int = ja_tokenizer.convert_tokens_to_ids("[CLS]")  # 2
    # ja_eos_token: int = ja_tokenizer.convert_tokens_to_ids("[SEP]")  # 3
    en_bos_token: int = en_tokenizer.convert_tokens_to_ids("[CLS]")  # 101
    en_eos_token: int = en_tokenizer.convert_tokens_to_ids("[SEP]")  # 102

    # モデル生成
    model: Transformer = Transformer(
        src_vocab_size=ja_vocab_size,
        tgt_vocab_size=en_vocab_size,
    )
    model_file: Path = Path(
        "ja2en_translate/model/", "ja2en_" + args.data_mode + "_model_cpu.pth"
    )
    model.load_state_dict(torch.load(model_file))
    model.to(DEVICE)
    model.eval()

    write_data_file: Path = Path(
        "ja2en_translate/result_data/", args.data_mode + "_trans.txt"
    )
    with open(file=write_data_file, mode="w", encoding="utf-8") as fw:
        for idx, text in enumerate(list(data_frame["text"])):
            print(idx)
            token: list[int] = ja_tokenizer.encode(text)
            src_len: int = len(token)
            src: TensorType[1, src_len, long] = torch.LongTensor(token).unsqueeze(0)
            src_mask: TensorType[src_len, src_len, bool] = (
                torch.zeros(src_len, src_len)
            ).type(torch.bool)

            tgt_tokens: list[int] = greedy_decode(
                model, src, src_mask, src_len, en_bos_token, en_eos_token
            )
            src_text: str = text.replace("\n", "")
            tgt_text: str = en_tokenizer.decode(tgt_tokens).replace("\t", "")
            fw.write(f"{src_text}\t{tgt_text}\n")


if __name__ == "__main__":
    predict()
