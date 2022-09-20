import argparse
import os
from typing import Optional

import pytorch_lightning as pl
import torch
from models import DataModule, LitTransformer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from utils import TrainArgs

# dataloaderのnum_worker>1にしているので、設定しないと警告が出る
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args() -> TrainArgs:
    parser = argparse.ArgumentParser()
    # 学習パラメータ
    parser.add_argument(
        "--train_mode",
        default="ja2en",
        type=str,
        help="define trasnlate mode 'ja2en' or 'en2ja'",
    )
    parser.add_argument("--train_data_file", default=None, type=str)
    parser.add_argument("--pred_data_file", default=None, type=str)
    parser.add_argument("--model_path", default=None, type=str, help="predict時に用いるモデルのパス")
    parser.add_argument(
        "--pred_save_file", default="./pl_translate/result/predict.tsv", type=str
    )
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_length", default=256, type=int)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--num_warmup_steps", default=0, type=int)
    # Transformer Config
    parser.add_argument("--d_model", default=256, type=int)
    parser.add_argument("--nhead", default=8, type=int)
    parser.add_argument("--dim_feedforward", default=512, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--num_layers", default=4, type=int)
    # Trainer Config
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="バッチ処理を何ループ蓄積したら勾配処理をするか",
    )
    parser.add_argument("--opt_level", default="O2", type=str, help="ampのオプション選択")
    # 解説記事 : https://qiita.com/sakaia/items/f9cf91af7c83a8542631
    parser.add_argument("--max_grad_norm", default=0.1, type=float)
    parser.add_argument("--use_auto_lr_find", default=False, type=bool)
    parser.add_argument("--use_scheduler", default=True, type=bool)
    parse_args: argparse.Namespace = parser.parse_args()

    assert parse_args.train_mode is not None, "data_mode must have some input."
    assert (
        parse_args.train_data_file is not None or parse_args.pred_data_file is not None
    ), "data_file train or pred must have some input."
    assert parse_args.train_mode in [
        "ja2en",
        "en2ja",
    ], "data_mode only acccept 'ja2en' or 'en2ja'"

    args = TrainArgs(
        train_mode=parse_args.train_mode,
        train_data_file=parse_args.train_data_file,
        pred_data_file=parse_args.pred_data_file,
        model_path=parse_args.model_path,
        pred_save_file=parse_args.pred_save_file,
        batch_size=parse_args.batch_size,
        max_length=parse_args.max_length,
        num_epochs=parse_args.num_epochs,
        lr=parse_args.lr,
        num_warmup_steps=parse_args.num_warmup_steps,
        d_model=parse_args.d_model,
        nhead=parse_args.nhead,
        dim_feedforward=parse_args.dim_feedforward,
        dropout=parse_args.dropout,
        num_layers=parse_args.num_layers,
        gradient_accumulation_steps=parse_args.gradient_accumulation_steps,
        opt_level=parse_args.opt_level,
        max_grad_norm=parse_args.max_grad_norm,
        use_auto_lr_find=parse_args.use_auto_lr_find,
        use_scheduler=parse_args.use_scheduler,
    )

    return args


if __name__ == "__main__":
    print("-" * 40 + "start run" + "-" * 40)
    assert torch.cuda.is_available()
    args: TrainArgs = get_args()
    data_module = DataModule(args)
    model = LitTransformer(args, args.lr)
    print("-" * 40 + "finish creat model & data module" + "-" * 40)

    trainer = pl.Trainer(
        logger=[TensorBoardLogger(save_dir="./pl_translate/", name="logs")],
        callbacks=[
            ModelCheckpoint(
                dirpath="./pl_translate/model/",
                filename=f"{args.train_mode}_best_model",
                monitor="val_loss",
                mode="min",
            ),
            EarlyStopping(
                monitor="val_loss", min_delta=0.05, patience=3, verbose=False, mode="min"
            ),
        ],
        gradient_clip_val=args.max_grad_norm,
        accelerator="gpu",
        devices=1,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        max_epochs=args.num_epochs,
        precision=32,
        auto_lr_find=args.use_auto_lr_find,
    )
    print("-" * 40 + "finish creat trainer" + "-" * 40)

    print("-" * 40 + "start training" + "-" * 40)
    if args.train_data_file:
        data_module.setup("fit")
        if args.use_auto_lr_find:
            lr_finder = trainer.tuner.lr_find(model, data_module)
            if lr_finder is not None:
                suggest_lr: Optional[float] = lr_finder.suggestion()
                if suggest_lr is not None:
                    model.lr = suggest_lr
                    print(suggest_lr)

        trainer.fit(model, datamodule=data_module)
        if args.pred_data_file:
            data_module.setup("predict")
            trainer.predict(model, datamodule=data_module)
    elif args.pred_data_file and args.model_path:
        data_module.setup("predict")
        model = model.load_from_checkpoint(args.model_path, args=args, learning_late=None)
        # outputs: list[Tensor] = trainer.predict(model, datamodule=data_module)
        sentences_list: list[list[str]] = trainer.predict(model, data_module)
        for sentences in sentences_list:
            for sentence in sentences:
                print(sentence)

    print("-" * 40 + "finish run" + "-" * 40)
