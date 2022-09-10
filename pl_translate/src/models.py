import math
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor, bool, float32, long
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from utils import TrainArgs

JA_TOKENIZER = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
EN_TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")


class LitTransformer(pl.LightningModule):
    def __init__(self, args: TrainArgs, learning_late: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr: float = learning_late
        self.args: TrainArgs = args
        self.pad_idx: int = JA_TOKENIZER.pad_token_id
        self.src_vocab_size: int
        self.tgt_vocab_size: int
        if args.train_mode == "ja2en":
            self.src_vocab_size = JA_TOKENIZER.vocab_size
            self.tgt_vocab_size = EN_TOKENIZER.vocab_size
        else:
            self.src_vocab_size = EN_TOKENIZER.vocab_size
            self.tgt_vocab_size = JA_TOKENIZER.vocab_size
        self.src_embedding = nn.Embedding(self.src_vocab_size, args.d_model, self.pad_idx)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, args.d_model, self.pad_idx)
        self.pos_encoder = PositionalEncoding(
            args.d_model, args.max_length, dropout=args.dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=args.num_layers,
            norm=nn.LayerNorm(args.d_model),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=args.num_layers,
            norm=nn.LayerNorm(args.d_model),
        )
        self.lm_head = nn.Linear(args.d_model, self.tgt_vocab_size, bias=False)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def step(self, batch: dict[str, Tensor]) -> Tensor:
        """trainとvalidationに共通の処理(lossの計算)を実行"""
        # TensorType[batch_size, max_length, d_model, long]
        src: Tensor = batch["src"]
        tgt: Tensor = batch["tgt"]
        input_tgt: Tensor = tgt[:, :-1]
        # TensorType[batch_size, max_length, d_model, float32]
        src_emb: Tensor = self.pos_encoder(self.src_embedding(src))
        tgt_emb: Tensor = self.pos_encoder(self.tgt_embedding(input_tgt))

        all_masks: dict[str, Tensor] = self.create_mask(src, input_tgt)

        # TensorType[batch_size, max_length, d_model, float32]
        memory: Tensor = self.encoder(
            src=src_emb,
            mask=all_masks["src_mask"],
            src_key_padding_mask=all_masks["src_pad_mask"],
        )
        # TensorType[batch_size, max_length, d_model, float32]
        model_output: Tensor = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=all_masks["tgt_mask"],
            tgt_key_padding_mask=all_masks["tgt_pad_mask"],
            memory_key_padding_mask=all_masks["src_pad_mask"],
        )
        # tie_word_embedding(正規化)
        model_output = model_output * (self.args.d_model**-0.5)
        # TensorType[batch_size, max_length, tgt_vocab_size, float32]
        output: Tensor = self.lm_head(model_output)

        # lossの計算
        # TensorType[batch_size * (tgt_len - 1), long]
        targets: Tensor = tgt[:, 1:].reshape(-1)
        # TensorType[batch_size * (tgt_len - 1), "tgt_vocab_size", float32]
        preds: Tensor = output.reshape(-1, output.shape[-1])
        loss: Tensor = self.loss_func(preds, targets)

        return loss

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch: dict[str, Tensor]
            batch_idx: int
        Return:
            dict[str, Tensor]
        """
        loss: Tensor = self.step(batch)
        self.log("train_loss", loss)
        current_lr: float = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True, on_step=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        Args:
            batch: dict[str, Tensor]
            batch_idx: int
        Return:
            dict[str, Tensor]
        """
        loss: Tensor = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        """
        Args:
            outputs: dict[str, Tensor]
        Return:
            dict[str, Tensor]
        """
        step_losses: list[Tensor] = [x["val_loss"] for x in outputs]
        epoch_loss: Tensor = torch.stack(step_losses).mean()
        self.log("val_loss", epoch_loss, prog_bar=True, on_epoch=True)

        return {"val_loss": epoch_loss}

    def configure_optimizers(self):
        """
        Return: optimizer | tuple[list[optimizer], list[scheduler]]
        """
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.args.use_scheduler:
            num_training_steps: int = self.calc_training_steps()
            print(f"total_training_steps: {num_training_steps}")
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.num_warmup_steps,
                num_training_steps=num_training_steps,
            )

            return [self.optimizer], [
                {"scheduler": self.scheduler, "interval": "step", "frequency": 1}
            ]
        else:
            return self.optimizer

    def forward(self, batch):
        """
        predict処理
        Args:
            batch: dict[str, Tensor]
        Return:
            list[str]
        """
        # TensorType[batch_size, max_length, long]
        src: Tensor = batch["src"]
        batch_size: int = src.size(0)
        length: int = src.size(1)
        # TensorType[batch_size, max_length, d_model, float32]
        src_emb: Tensor = self.pos_encoder(self.src_embedding(src))
        # TensorType[max_length, max_length, bool]
        src_mask: Tensor = ((torch.zeros(length, length)).type(bool)).to(self.device)
        # TensorType["batch_size", "src_length", bool]
        src_padding_mask: Tensor = src == self.pad_idx

        # TensorType[batch_size, max_length, d_model, float32]
        memory: Tensor = self.encoder(
            src=src_emb,
            mask=src_mask,
            src_key_padding_mask=src_padding_mask,
        )
        tgt_bos_id: int
        if self.args.train_mode == "ja2en":
            tgt_bos_id = EN_TOKENIZER.cls_token_id
        else:
            tgt_bos_id = JA_TOKENIZER.cls_token_id

        # TensorType[batch_size, length, long] length: 1 to max_length
        tgt: Tensor = (
            torch.ones(batch_size, 1).fill_(tgt_bos_id).type(long).to(self.device)
        )
        for _ in range(self.args.max_length - 1):
            # TensorType[batch_size, tgt_length, d_model, float32]
            tgt_emb: Tensor = self.pos_encoder(self.tgt_embedding(tgt))
            # TensorType[tgt_length, tgt_length, bool]
            tgt_mask: Tensor = (
                self.generate_square_subsequent_mask(tgt.size(1))
                .type(bool)
                .to(self.device)
            )
            # TensorType[batch, tgt_length, bool]
            tgt_padding_mask: Tensor = tgt == self.pad_idx
            # TensorType[batch_size, length, d_model, float32]
            model_output: Tensor = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask,
            )
            # TensorType[batch_size, tgt_vocab_size, float32]
            prob: Tensor = self.lm_head(model_output[:, -1])
            _, max_index = torch.max(prob, dim=1)
            # TensorType[batch]
            next_word: Tensor = max_index.unsqueeze(-1)
            # TensorType[batch_size, length+1, long]
            tgt = torch.cat([tgt, next_word], dim=1)

        sentences: list[str]
        if self.args.train_mode == "ja2en":
            sentences = EN_TOKENIZER.batch_decode(tgt)
        else:
            sentences = JA_TOKENIZER.batch_decode(tgt)
        return_sentences: list[str] = []
        for sentence in sentences:
            sentence = (
                sentence.split("[SEP]")[0].replace("[CLS]", "").replace("[UNK]", "")
            )
            return_sentences.append(sentence)

        return return_sentences

    def generate_square_subsequent_mask(
        self,
        length: int,
    ) -> Tensor:
        # TensorType["length", "length", float32]
        mask: Tensor = torch.triu(torch.full((length, length), float("-inf")), diagonal=1)

        return mask

    def create_mask(
        self,
        src: Tensor,  # TensorType["batch_size", "src_length", long]
        tgt: Tensor,  # TensorType["batch_size", "tgt_length", long]
    ) -> dict[str, Tensor]:
        src_len: int = src.size(1)
        tgt_len: int = tgt.size(1)
        # TensorType[src_length, src_length, bool]
        src_mask: Tensor = torch.zeros((src_len, src_len), dtype=bool)
        # TensorType[tgt_length, tgt_length, float32]
        tgt_mask: Tensor = self.generate_square_subsequent_mask(tgt_len)
        # TensorType["batch_size", "src_length", bool]
        src_pad_mask: Tensor = src == self.pad_idx
        # TensorType["batch_size", "tgt_length", bool]
        tgt_pad_mask: Tensor = tgt == self.pad_idx

        return {
            "src_mask": src_mask.to(self.device),
            "tgt_mask": tgt_mask.to(self.device),
            "src_pad_mask": src_pad_mask.to(self.device),
            "tgt_pad_mask": tgt_pad_mask.to(self.device),
        }

    def calc_training_steps(self) -> int:
        """schedulerに用いるtraining_total_stepsを求める"""
        # self.trainer.max_steps: (default = -1)
        if 0 < self.trainer.max_steps:
            return self.trainer.max_steps
        loader_size: int = len(
            self.trainer._data_connector._train_dataloader_source.dataloader()
        )
        step_size: int = loader_size // self.trainer.accumulate_grad_batches

        return step_size * self.trainer.max_epochs


class DataModule(pl.LightningDataModule):
    def __init__(self, args: TrainArgs) -> None:
        super().__init__()
        self.args = args

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" and self.args.train_data_file:
            dataset = CustomDataset(
                self.args.train_data_file, self.args.max_length, self.args.train_mode
            )
            all_size: int = len(dataset)
            train_size: int = int((0.9 * all_size))
            valid_size: int = all_size - train_size
            print(f"train size : {train_size}\nvalid size : {valid_size}")
            self.train_ds, self.valid_ds = random_split(
                dataset=dataset, lengths=[train_size, valid_size]
            )
        if stage == "predict" and self.args.pred_data_file:
            self.pred_dataset = PredDataset(
                self.args.pred_data_file,
                self.args.max_length,
                self.args.train_mode,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.args.batch_size,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_ds,
            batch_size=self.args.batch_size,
            num_workers=4,
            pin_memory=True,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.pred_dataset,
            batch_size=self.args.batch_size,
            num_workers=4,
            pin_memory=True,
        )


class CustomDataset(Dataset):
    def __init__(self, file: str, max_length: int, train_mode: str) -> None:
        super().__init__()
        self.max_length: int = max_length
        self.file: Path = Path(file)
        self.train_mode = train_mode
        self.src_ids: list[Tensor] = []
        self.tgt_ids: list[Tensor] = []

        self.build()

    def build(self) -> None:
        with open(file=self.file, mode="r", encoding="utf-8") as fr:
            for idx, line in enumerate(fr):
                texts: list[str] = line.strip().split("\t")
                assert len(texts) == 2, f"{idx}"
                assert len(texts[0]) > 0, f"{idx}"
                assert len(texts[1]) > 0, f"{idx}"
                ja_text: str = texts[0]
                en_text: str = texts[1]

                ja_ids: Tensor = JA_TOKENIZER.encode(
                    ja_text,
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                en_ids: Tensor = EN_TOKENIZER.encode(
                    en_text,
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                if self.train_mode == "ja2en":
                    self.src_ids.append(ja_ids)
                    self.tgt_ids.append(en_ids)
                else:
                    self.src_ids.append(en_ids)
                    self.tgt_ids.append(ja_ids)

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        # TensorType["text_len", long]
        src: Tensor = self.src_ids[index].squeeze()
        # TensorType["text_len", long]
        tgt: Tensor = self.tgt_ids[index].squeeze()

        return {"src": src, "tgt": tgt}


class PredDataset(Dataset):
    def __init__(self, file: str, max_length: int, train_mode: str) -> None:
        super().__init__()
        self.max_length: int = max_length
        self.file: Path = Path(file)
        self.train_mode = train_mode
        self.src_ids: list[Tensor] = []

        self.build()

    def build(self) -> None:
        with open(file=self.file, mode="r", encoding="utf-8") as fr:
            for _, line in enumerate(fr):
                text: str = line.strip()
                input_ids: Tensor
                if self.train_mode == "ja2en":
                    input_ids = JA_TOKENIZER.encode(
                        text,
                        padding="max_length",
                        max_length=self.max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                else:
                    input_ids = EN_TOKENIZER.encode(
                        text,
                        padding="max_length",
                        max_length=self.max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                self.src_ids.append(input_ids)

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        # TensorType["text_len", long]
        src: Tensor = self.src_ids[index].squeeze()
        return {"src": src}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model: int = d_model

        # TensorType[max_len, 1, float32]
        position: Tensor = torch.arange(0, max_len, dtype=float32).unsqueeze(1)
        # TensorType[d_model / 2, float32]
        div_term: Tensor = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # TensorType[max_len, d_model, float32]
        pos_enc: Tensor = torch.zeros(max_len, d_model)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        # TensorType[1, max_len, d_model, float32]
        positional_encoding: Tensor = pos_enc.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, token_emb: Tensor) -> Tensor:
        """
        token_emb: TensorType["batch_size", "token_len", "d_model", float32]
        return: TensorType["batch_size", "token_len", "d_model", float32]
        """
        seq_len: int = token_emb.size(1)
        token_emb = token_emb * math.sqrt(self.d_model)
        token_emb = token_emb + self.positional_encoding[:, :seq_len]  # type: ignore

        return self.dropout(token_emb)
