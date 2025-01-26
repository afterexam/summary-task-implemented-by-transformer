

## 2024.1.25

使用标准transformer, 取few-shot(10%)训练不收敛,猜测是数据集质量问题

## 1.26
https://arxiv.org/pdf/1910.10683
* t5架构
* batch=128
* N=12
* seq_len=512
* d_model=768
* h=12
* 2**35  tokens 一轮
* lr : During pre-training, we use an “inverse square root” learning rate schedule: 1 max(n,k)
where n is the current training iteration and k is the number of warm-up steps (set to 104
in all of our experiments).
* vocab(wordPiece) : Since we ultimately fine-tune our model on English to German, French, and
 Romanian translation, we also require that our vocabulary covers these non-English languages.
 To address this, we classified pages from the Common Crawl scrape used in C4 as German,
 French, and Romanian.
* limit : Note that our vocabulary makes it so that our model can only process a predetermined, fixed set of
languages.

没有中文语料啊..

