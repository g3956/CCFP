# Cross Camera Feature Prediction for Intra Camera Supervised Person Re identification across Distant Scenes
Codes of ACM MM 2021 paper: Cross-Camera Feature Prediction for Intra-Camera Supervised Person Re-identification across Distant Scenes. 

## Environment

The code is based on [fastreid](https://github.com/JDAI-CV/fast-reid). See [INSTALL.md](https://github.com/JDAI-CV/fast-reid/blob/master/INSTALL.md).

For Compiling with cython to accelerate evalution
```bash
cd fastreid/evaluation/rank_cylib; make all
```

## Dataset Preparation

1. Download Market-1501 and DukeMTMC-reID
2. Split Market-1501 and DukeMTMC-reID to Market-sct and DukeMTMC-sct according to the file names in the market-sct.txt and duke_sct.txt
3. Make new directories in data and organize them as follows:
<pre>
+-- data
|   +-- market
|       +-- market_sct
|       +-- query
|       +-- boudning_box_test
|   +-- duke
|       +-- duke_sct
|       +-- query
|       +-- boudning_box_test
</pre>
