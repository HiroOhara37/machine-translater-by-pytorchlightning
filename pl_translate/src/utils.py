from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainArgs:
    train_mode: str
    train_data_file: Optional[str]
    pred_data_file: Optional[str]
    model_path: str
    pred_save_file: str
    batch_size: int
    max_length: int
    num_epochs: int
    lr: float
    num_warmup_steps: int

    d_model: int
    nhead: int
    dim_feedforward: int
    dropout: float
    num_layers: int

    gradient_accumulation_steps: int
    opt_level: str
    max_grad_norm: float
    use_auto_lr_find: bool
    use_scheduler: bool
