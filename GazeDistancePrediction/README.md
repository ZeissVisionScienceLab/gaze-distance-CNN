## [CNN](/CNNET)
For the CNN training pipeline to run, please download the training data from [10.5281/zenodo.15439043](https://zenodo.org/records/15439043). The CNN requires the file DistanceData.zip to be extracted and saved under /data/distancedata/indoor and /outdoor, respectively. Training the model additionally requires the file [et_data_cnn_rollingmedian.feather](<data/et_data_cnn_rollingmedian.feather>), which contains the preprocessed eye tracking data.  

Run the file [main.py](<CNNET/main.py>) to start training. The training parameters can also be adjusted within that file.  

## [Center and Vergence](<Center and Vergence>)
The naive methods based on Vergence and the depth at the center of the gaze point are evaluated within a [Jupyter Notebook](</GazeDistancePrediction/Center and Vergence/center_estimation.ipynb>) and require the file [mlp_training_data_10percent_ang2.feather](<data/mlp_training_data_10percent_ang2.feather>) which contains the preprocessed eye tracking data.  

For more details on the preprocessing, please see the corresponding publication.  
