# Transformer_translate
## 説明
[pytorchの公式チュートリアル](https://torch.classcat.com/2021/04/25/pytorch-1-8-tutorials-beginner-translation-transformer/)を参考に、翻訳モデルを構築した。  
公式チュートリアルのコードでは`batch_first`になっていないため、
`batch_first`で動くよう各部を変更し、各Tensorのサイズを型ヒントで追加した。  

## 使い方
dataディレクトリに学習データのディレクトリを追加し、その中に日本語データと英語データがタブで区切られたtsvファイル`data.tsv`を用意する。  
例）  
data/  
┣ JESC/  
┃        ┣ data.tsv  

DockerfileのCMDコマンドを任意に設定する。  
学習時：
```
python3 ja2en_translate/src/train.py --train_mode [学習データディレクトリ名]  
```
推論時：
```
python3 ja2en_translate/src/pred.py --train_mode [学習データディレクトリ名]  
```
その他、引数で指定できるもの  
- --batch_size
- --max_len
- --epoch_num

## Dockerでの実行
```
docker build . -t translate -f ./Dockerfile && docker run --gpus all -it --rm --mount type=bind,source="$(pwd)",target=/usr/src/app translate
```
## 生成物
trainではmodelディレクトリに、validation lossが最も低かったbest model(`ja2en_[学習データディレクトリ名]_model.pth`)が保存される。  
predではresult_dataディレクトリに、翻訳結果ファイル(`[学習データディレクトリ名]_trans.txt`)が保存される。  
