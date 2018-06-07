sentence-embedding
=====
This project compare five approachs of sentences embedding using data introduced by `ICLR2017 paper`"A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS" (https://openreview.net/pdf?id=SyK00v5xx) We only test on `sentiment task` in this paper.<br>
Five approaches including:<br>
1、implemention of paper "A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS"<br>
2、tf-idf<br>
3、average glove word vector<br>
4、A brilliant method for Engineering which combine the bag-of-word and word vector,can also be considered as a variety of bag-of-word<br>
<br>
`pearsonr's coefficient` as follow：<br>
![](https://github.com/wenrui2015/sentence-embedding/raw/master/image.png)
<br>
<br>
`quick start`<br>
1、first you should download the glove word2vector `glove_model.txt` and place it into dir data/res<br>
2、`glove_model.txt` iS too big you can contact me 1174950106@qq.com for the data<br>
3、python main.py  "approach"    #approach is optional in \["ICLR2017","variety-of-bow","tf-idf","ave-glove-vector"\]<br>
