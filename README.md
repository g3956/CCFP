# Cross Camera Feature Prediction for Intra Camera Supervised Person Re identification across Distant Scenes
Codes of ACM MM 2021 paper: Cross-Camera Feature Prediction for Intra-Camera Supervised Person Re-identification across Distant Scenes. 



# Dataset Preparation

1. Download Market-1501 and DukeMTMC-reID
2. Split Market-1501 and DukeMTMC-reID to Market-sct and DukeMTMC-sct according to the file names in the market-sct.txt and duke_sct.txt
3. Make new directories in data and organize them as follows:
+-- data
|   +-- market
|       +-- bounding_box_train_sct
|       +-- query
|       +-- boudning_box_test
|   +-- duke
|       +-- bounding_box_train_sct
|       +-- query
|       +-- boudning_box_test
