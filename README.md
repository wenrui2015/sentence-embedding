sentence-embedding
=====
This project compare five approachs of sentences embedding using data introduced by `ICLR2017 paper`[A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS] (https://openreview.net/pdf?id=SyK00v5xx) We only test on `sentiment task` in this paper.<br>
Five approachs including:<br>
1、implemention of paper "A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS"<br>
2、tf-idf<br>
3、average glove word vector<br>
4、A brilliant method for Engineering which combine the bag-of-word and word vector,can also be considered as a variety of bag-of-word<br>
<br>
`pearsonr's coefficient` as follow：<br>
      test-data   approach_1"ICLR2017 paper" approach_2  approach_3                    approach_4<br>
  sick-test                    0.72                      0.6036         0.6918            0.6746<br>
  sick-train                   0.73                      0.6044         0.694             0.682<br>
<br>
<br>
`quick start`<br>
python main.py  "approach"    #approach is option on "ICLR2017"、"variety-of-bow"、"tf-idf"、"ave-glove-vector"<br>
