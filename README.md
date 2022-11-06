# machine-translater-by-pytorchlightning
## 説明
Transformerの翻訳モデル(machine-translater-by-pytorch)をpytorch lightningで再実装した。  

## 使い方
### 【学習データ】
pl_translate/dataディレクトリに学習データのディレクトリを追加し、その中に日本語データと英語データがタブで区切られたtsvファイル`data.tsv`を用意する。  

### 【推論データ】
dataディレクトリに推論データのディレクトリを追加し、その中にtxtファイルを用意する。  

### shファイルに引数を書き込む 
#### 学習時：
最低限、学習データのファイルパス`(--train_data_file)`と学習モード`(--train_mode)`を指定する。  
学習モードでは、日英翻訳(ja2en)か英日翻訳(en2ja)を指定  
#### 推論時：
推論用のデータのファイルパス`(--pred_data_file)`と学習したモデルのパス`(--model_path)`と推論結果を書き込むファイルのパス`(--pred_save_file)`を指定  
#### その他、引数で指定できるもの  
- --batch_size
- --max_len
- --epoch_num
など(詳細はtrain.pyのargsを参照)
## Dockerでの実行
```
docker build . -t translate -f ./Dockerfile && docker run --gpus all -it --rm --mount type=bind,source="$(pwd)",target=/usr/src/app translate
```
