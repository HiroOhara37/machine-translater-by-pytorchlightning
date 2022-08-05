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
from torch import float32, long
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchtyping import TensorType
from tqdm import tqdm
from Transformer_model import Transformer
from transformers import AutoTokenizer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device:{DEVICE}")


@dataclass
class TrainArguments:
    data_mode: str
    batch_size: int
    max_len: int
    epoch_num: int


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


def generate_batch(
    batch_data: list[tuple[TensorType["length", long], TensorType["length", long]]],
) -> tuple[
    TensorType["batch_size", "max_text_len", float32],
    TensorType["batch_size", "max_text_len", float32],
]:
    batch_src: list[torch.Tensor] = []
    batch_tgt: list[torch.Tensor] = []
    for src, tgt in batch_data:
        batch_src.append(src)
        batch_tgt.append(tgt)

    batch_src = pad_sequence(batch_src, batch_first=True)  # type: ignore
    batch_tgt = pad_sequence(batch_tgt, batch_first=True)  # type: ignore

    return batch_src, batch_tgt


def train(
    model: Transformer,
    loss_func: nn.CrossEntropyLoss,
    optimizer: Adam,
    train_loader: DataLoader,
) -> float:
    model.train()
    losses: float = 0
    for idx, batch in enumerate(tqdm(train_loader)):
        batch_size: int = batch[0].size(0)
        src_len: int = batch[0].size(1)
        tgt_len: int = batch[1].size(1)

        batch_src: TensorType[batch_size, src_len, float32] = batch[0].to(DEVICE)
        batch_tgt: TensorType[batch_size, tgt_len, float32] = batch[1].to(DEVICE)

        input_tgt: TensorType[batch_size, tgt_len - 1, long] = batch_tgt[:, :-1]
        output: TensorType[batch_size, tgt_len - 1, tgt_vocab_size, float32] = model(
            src=batch_src, tgt=input_tgt
        )
        assert output.size() == torch.Size(
            [batch_size, tgt_len - 1, tgt_vocab_size]
        ), f"output size is {output.size()}. It is not expected size."

        optimizer.zero_grad()

        # lossの計算
        targets: TensorType[batch_size * (tgt_len - 1), long] = batch_tgt[:, 1:].reshape(
            -1
        )
        preds: TensorType[
            batch_size * (tgt_len - 1), tgt_vocab_size, float32
        ] = output.reshape(-1, output.shape[-1])

        loss: torch.FloatTensor = loss_func(preds, targets)
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
    for idx, batch in enumerate(dev_loader):
        batch_size: int = batch[0].size(0)
        src_len: int = batch[0].size(1)
        tgt_len: int = batch[1].size(1)

        batch_src: TensorType[batch_size, src_len, long] = batch[0].to(DEVICE)
        batch_tgt: TensorType[batch_size, tgt_len, long] = batch[1].to(DEVICE)

        input_tgt: TensorType[batch_size, tgt_len - 1, long] = batch_tgt[:, :-1]
        output: TensorType[batch_size, tgt_len - 1, tgt_vocab_size, float32] = model(
            src=batch_src, tgt=input_tgt
        )

        targets: TensorType[batch_size * (tgt_len - 1), long] = batch_tgt[:, 1:].reshape(
            -1
        )
        preds: TensorType[
            batch_size * (tgt_len - 1), "tgt_vocab_size", float32
        ] = output.reshape(-1, output.shape[-1])
        loss: torch.FloatTensor = loss_func(preds, targets)
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

    # Tokenizerの読み込み
    ja_tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    src_vocab_size: int = len(en_tokenizer.vocab)
    tgt_vocab_size: int = len(ja_tokenizer.vocab)

    # dataset作成
    train_dataset = CustomDataset(
        df=train_data,
        max_len=args.max_len,
        ja_tokenizer=ja_tokenizer,
        en_tokenizer=en_tokenizer,
    )
    dev_dataset = CustomDataset(
        df=dev_data,
        max_len=args.max_len,
        ja_tokenizer=ja_tokenizer,
        en_tokenizer=en_tokenizer,
    )
    test_dataset = CustomDataset(
        df=test_data,
        max_len=args.max_len,
        ja_tokenizer=ja_tokenizer,
        en_tokenizer=en_tokenizer,
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
    model: Transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_len=args.max_len,
    )
    model.to(DEVICE)

    # 損失関数、最適化関数の定義
    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    best_loss: float = float("inf")
    best_model: Optional[Transformer] = None
    counter: int = 0
    patience: int = 5
    write_data_path: Path = Path(
        "en2ja_translate/model", "en2ja_" + args.data_mode + "_result.txt"
    )

    print(f"{'-'*40}start training.{'-'*40}\n")
    print(torch.cuda.get_device_properties(device=DEVICE).total_memory)

    with open(file=write_data_path, mode="w", encoding="utf-8") as fw:
        for epoch in range(1, args.epoch_num + 1):

            start_time: float = time.time()
            print(torch.cuda.memory_reserved(device=DEVICE))
            train_loss: float = train(model, loss_func, optimizer, train_loader)
            print(torch.cuda.memory_reserved(device=DEVICE))
            valid_loss: float = evaluate(model, loss_func, dev_loader)
            print(torch.cuda.memory_reserved(device=DEVICE))
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
        torch.save(
            best_model.state_dict(),
            Path("en2ja_translate/model", "en2ja_" + args.data_mode + "_model.pth"),
        )

    print("-" * 40 + "finish run" + "-" * 40)
