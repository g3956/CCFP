# Cross-Camera Feature Prediction (CCFP)
The official implementation for the [Cross-Camera Feature Prediction for Intra-Camera Supervised Person Re-identification across Distant Scenes](https://arxiv.org/abs/2107.13904) which is accepted by ACMMM-2021.

## Environment

The code is based on [fastreid](https://github.com/JDAI-CV/fast-reid). See [INSTALL.md](https://github.com/JDAI-CV/fast-reid/blob/master/INSTALL.md).

For Compiling with cython to accelerate evalution
```bash
cd fastreid/evaluation/rank_cylib; make all
```

## Dataset Preparation

1. Download Market-1501 and DukeMTMC-reID
2. Split Market-1501 and DukeMTMC-reID to Market-sct and DukeMTMC-sct according to the file names in the market_sct.txt and duke_sct.txt
3. ```vim fastreid/data/build.py``` change the ```_root``` to your own data folder
4. Make new directories in data and organize them as follows:
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

## Train and test

To train market-sct with CCFP, simply run
```bash
sh run.sh
```
To train duke-sct with CCFP, simply run
```bash
sh run_d.sh
```

## Experiments

![image](https://github.com/g3956/Cross-Camera-Feature-Prediction-for-Intra-Camera-Supervised-Person-Re-identification-across-Distant-/blob/main/results.png)


## Citation

If you find this code useful, please kindly cite the following paper:
<pre>
@inproceedings{ge2021cross,
  title={Cross-Camera Feature Prediction for Intra-Camera Supervised Person Re-identification across Distant Scenes},
  author={Ge, Wenhang and Pan, Chunyan and Wu, Ancong and Zheng, Hongwei and Zheng, Wei-Shi},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={3644--3653},
  year={2021}
}
</pre>

