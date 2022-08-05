# Transformer_translate
## 説明
[pytorchの公式チュートリアル](https://torch.classcat.com/2021/04/25/pytorch-1-8-tutorials-beginner-translation-transformer/)を参考に、翻訳モデルを構築した。  
公式チュートリアルのコードでは`batch_first`になっていないため、
`batch_first`で動くよう各部を変更し、各Tensorのサイズを型ヒントで追加した。  

## 使い方
dataディレクトリに学習データのフォルダを追加し、その中に日本語データと英語データがタブで区切られたtsvファイル`data.tsv`を用意する。  
例）
data/  
    ┣ JESC/  
        ┣ data.tsv  
+ /data
    + /JESC
        + .data.tsv    