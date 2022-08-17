import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from dataset import CustomDataset
from sklearn.model_selection import train_test_split
from torch import FloatTensor, Tensor, float32, long
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchtyping import TensorType
from tqdm import tqdm
from Transformer_model import DEVICE, Transformer
from transformers import AutoTokenizer

print(f"device:{DEVICE}")
# Tokenizerの読み込み
JA_TOKENIZER = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
EN_TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
PAD_IDX: int = JA_TOKENIZER.pad_token_id


@dataclass
class TrainArguments:
    data_mode: str  # 学習に用いるデータのフォルダ名
    batch_size: int
    max_len: int
    epoch_num: int


def get_args() -> TrainArguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_mode", default=None, type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--max_len", default=500, type=int)
    parser.add_argument("--epoch_num", default=100, type=int)
    parse_args: argparse.Namespace = parser.parse_args()

    assert parse_args.data_mode is not None, "data_mode must have some input."

    args: TrainArguments = TrainArguments(
        data_mode=parse_args.data_mode,
        batch_size=parse_args.batch_size,
        max_len=parse_args.max_len,
        epoch_num=parse_args.epoch_num,
    )

    return args


def generate_batch(
    batch_data: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    """
    batch_data: list[tuple[TensorType[src_len, long], TensorType[tgt_len, long]]]
    return: tuple[TensorType["batch_size", "max_src_len", float32],
                  TensorType["batch_size", "max_tgt_len", float32],
            ]
    """
    # TensorType[src_len, long]
    batch_src_list: list[Tensor] = []
    # TensorType[tgt_len, long]
    batch_tgt_list: list[Tensor] = []
    for src, tgt in batch_data:
        batch_src_list.append(src)
        batch_tgt_list.append(tgt)

    # TensorType["batch_size", "max_src_len", long]
    batch_src: Tensor = pad_sequence(batch_src_list, batch_first=True).type(long)
    # TensorType["batch_size", "max_tgt_len", long]
    batch_tgt: Tensor = pad_sequence(batch_tgt_list, batch_first=True).type(long)

    return batch_src, batch_tgt


def train(
    model: Transformer,
    loss_func: nn.CrossEntropyLoss,
    optimizer: Adam,
    train_loader: DataLoader,
) -> float:
    model.train()
    losses: float = 0
    for _, batch in enumerate(tqdm(train_loader)):
        batch_size: int = batch[0].size(0)
        src_len: int = batch[0].size(1)
        tgt_len: int = batch[1].size(1)

        batch_src: TensorType[batch_size, src_len, long] = batch[0].to(DEVICE)
        batch_tgt: TensorType[batch_size, tgt_len, long] = batch[1].to(DEVICE)
        input_tgt: TensorType[batch_size, tgt_len - 1, long] = batch_tgt[:, :-1]

        # TensorType[batch_size, tgt_len - 1, "tgt_vocab_size", float32]
        output: Tensor = model(src=batch_src, tgt=input_tgt)
        assert output.size() == torch.Size(
            [batch_size, tgt_len - 1, en_vocab_size]
        ), f"output size is {output.size()}. It is not expected size."

        optimizer.zero_grad()

        # lossの計算
        targets: TensorType[batch_size * (tgt_len - 1), long] = batch_tgt[:, 1:].reshape(
            -1
        )
        preds: TensorType[
            batch_size * (tgt_len - 1), "tgt_vocab_size", float32
        ] = output.reshape(-1, output.shape[-1])

        loss: FloatTensor = loss_func(preds, targets)
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_loader)


@torch.no_grad()
def evaluate(
    model: Transformer, loss_func: nn.CrossEntropyLoss, dev_loader: DataLoader
) -> float:
    model.eval()
    losses: float = 0
    for _, batch in enumerate(dev_loader):
        batch_size: int = batch[0].size(0)
        src_len: int = batch[0].size(1)
        tgt_len: int = batch[1].size(1)

        batch_src: TensorType[batch_size, src_len, long] = batch[0].to(DEVICE)
        batch_tgt: TensorType[batch_size, tgt_len, long] = batch[1].to(DEVICE)
        input_tgt: TensorType[batch_size, tgt_len - 1, long] = batch_tgt[:, :-1]

        # TensorType[batch_size, tgt_len - 1, "tgt_vocab_size", float32]
        output: Tensor = model(src=batch_src, tgt=input_tgt)
        # TensorType[batch_size * (tgt_len - 1), long]
        targets: Tensor = batch_tgt[:, 1:].reshape(-1)
        # TensorType[batch_size * (tgt_len - 1), "tgt_vocab_size", float32]
        preds: Tensor = output.reshape(-1, output.shape[-1])
        loss: FloatTensor = loss_func(preds, targets)
        losses += loss.item()

    return losses / len(dev_loader)


if __name__ == "__main__":
    print("-" * 40 + "start run" + "-" * 40)
    args: TrainArguments = get_args()

    # データの読み込み
    read_data_path: Path = Path("data", args.data_mode, "data.tsv")
    data_frame: pd.DataFrame = pd.read_table(read_data_path, names=["ja", "en"])
    # train, dev, testに分割
    train_data, dev_test_data = train_test_split(
        data_frame, test_size=0.2, shuffle=True, random_state=0
    )
    dev_data, test_data = train_test_split(dev_test_data, test_size=0.5)
    print(f"{'-'*40}finish read datafile.{'-'*40}\n{train_data.head(3)}\n")

    ja_vocab_size: int = len(JA_TOKENIZER.vocab)
    en_vocab_size: int = len(EN_TOKENIZER.vocab)

    # dataset作成
    train_dataset = CustomDataset(
        df=train_data,
        ja_tokenizer=JA_TOKENIZER,
        en_tokenizer=EN_TOKENIZER,
    )
    dev_dataset = CustomDataset(
        df=dev_data,
        ja_tokenizer=JA_TOKENIZER,
        en_tokenizer=EN_TOKENIZER,
    )
    test_dataset = CustomDataset(
        df=test_data,
        ja_tokenizer=JA_TOKENIZER,
        en_tokenizer=EN_TOKENIZER,
    )
    print(f"{'-'*40}finish make dataset.{'-'*40}\n")
    print(f"train dataset num = {train_dataset.__len__()}")
    print(f"dev dataset num = {dev_dataset.__len__()}")
    print(f"test dataset num = {test_dataset.__len__()}")

    # dataloaderの作成
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=generate_batch,
    )
    dev_loader = DataLoader(
        dataset=dev_dataset,
        batch_size=args.batch_size,
        collate_fn=generate_batch,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        collate_fn=generate_batch,
    )
    print(f"{'-'*40}finish make dataloader.{'-'*40}\n")

    # モデル生成
    model = Transformer(
        src_vocab_size=ja_vocab_size,
        tgt_vocab_size=en_vocab_size,
        max_len=args.max_len,
    )
    model.to(DEVICE)

    # 損失関数、最適化関数の定義
    loss_func = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    best_loss: float = float("inf")
    best_model: Optional[Transformer] = None
    counter: int = 0
    patience: int = 5
    write_data_path: Path = Path(
        "ja2en_translate/model", "ja2en_" + args.data_mode + "_result.txt"
    )

    print(f"{'-'*40}start training.{'-'*40}\n")
    with open(file=write_data_path, mode="w", encoding="utf-8") as fw:
        for epoch in range(1, args.epoch_num + 1):

            start_time: float = time.time()
            train_loss: float = train(model, loss_func, optimizer, train_loader)
            valid_loss: float = evaluate(model, loss_func, dev_loader)
            elapsed_time: float = time.time() - start_time

            print(
                f"epoch:{epoch}/{args.epoch_num}"
                f"train loss:{train_loss:.4f} valid loss:{valid_loss:.4f}"
                f"time:{elapsed_time // 60}m{elapsed_time % 60:.0f}s\n"
            )
            fw.write(
                f"epoch:{epoch}/{args.epoch_num}\n"
                f"train loss:{train_loss:.4f} valid loss:{valid_loss:.4f}\n"
            )

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = model
                counter = 0
            if patience < counter:
                break
            counter += 1

        # テストの実行
        test_loss: float = evaluate(best_model, loss_func, test_loader)
        fw.write(f"\n\ntest loss:{test_loss:.4f}")

    # モデルの保存
    if best_model:
        best_model.to("cpu")
        torch.save(
            best_model.state_dict(),
            Path("ja2en_translate/model", "ja2en_" + args.data_mode + "_model.pth"),
        )

    print("-" * 40 + "finish run" + "-" * 40)
