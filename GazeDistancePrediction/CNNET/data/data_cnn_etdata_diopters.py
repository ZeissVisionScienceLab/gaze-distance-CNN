import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from scripts import utils, params
import mmap
from data.data_cnn import ETData

max = 20
max_diopters = 1 / 0.1

class CNNDatasetETDiopters(Dataset):
    def __init__(self, et_data: pd.DataFrame, depth_path_indoor: str, depth_path_outdoor: str, permutation=ETData.NONE, seed=42):
        self.et_data = et_data
        self.depth_path_indoor = depth_path_indoor
        self.depth_path_outdoor = depth_path_outdoor

        size_of_float32 = 4
        self.frame_size = params.kernel_height * params.kernel_height * size_of_float32
        self.bytes_to_read = self.frame_size

        np.random.seed(seed)

        # create permutation
        if permutation == ETData.ECCENTRICITY:
            print('Eccentricity not used here, nothing permuted.')
        elif permutation == ETData.VERGENCE:
            print('Vergence not used here, nothing permuted.')
        elif permutation == ETData.DEPTH:
            depth = np.random.permutation(self.et_data[['scene_id', 'participant_id', 'frame_number']].to_numpy())
            self.et_data[['scene_id', 'participant_id', 'frame_number']] = depth
            print('Depth maps permuted.')
        elif permutation == ETData.ET:
            et = np.random.permutation(self.et_data[['l_gaze.x', 'l_gaze.y', 'r_gaze.x', 'r_gaze.y']].to_numpy())
            self.et_data[['l_gaze.x', 'l_gaze.y', 'r_gaze.x', 'r_gaze.y']] = et
            print('Eye-tracking data permuted.')
        elif permutation == ETData.IPD:
            self.et_data['ipd'] = np.random.permutation(self.et_data['ipd'].to_numpy())
            print('IPD permuted.')
        

    def __len__(self):
        # return length of dataframe et_data
        return len(self.et_data)
    
    def __getitem__(self, idx):
        # get relevant indices
        sid, pid, frame_number = self.idx_to_ids(idx)
        depth_path = self.depth_path_indoor if sid == 1 else self.depth_path_outdoor

        # reconstruct filename and path
        filename = "distance_data_" + str(sid) + "_" + str(pid) + ".bin"
        filepath = os.path.join(depth_path, filename)

        offset = frame_number * self.frame_size

        with open(filepath, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                mm.seek(offset)
                # Read the specific frame's data
                data = mm.read(self.bytes_to_read)
                # Convert to numpy array and reshape
                frame_data_np = np.frombuffer(data, dtype=np.float32).reshape(params.kernel_height, params.kernel_height)
                depthmap = torch.tensor(frame_data_np, dtype=torch.float32)



        target_distance = self.et_data.loc[idx, 'distance']
        ipd = self.et_data.loc[idx, 'ipd']
        l_gaze_x = self.et_data.loc[idx, 'l_gaze.x']
        l_gaze_y = self.et_data.loc[idx, 'l_gaze.y']
        r_gaze_x = self.et_data.loc[idx, 'r_gaze.x']
        r_gaze_y = self.et_data.loc[idx, 'r_gaze.y']

        # denormalize
        depthmap = depthmap * max
        target_distance = target_distance * max
        
        target_distance = torch.tensor(target_distance, dtype=torch.float32)

        # clip to 0.1, max
        depthmap = torch.clamp(depthmap, 0.1, max)
        target_distance = torch.clamp(target_distance, 0.1, max)

        # meters to diopters
        depthmap = 1 / depthmap
        target_distance = 1 / target_distance

        # normalize with max_diopters
        depthmap = depthmap / max_diopters
        target_distance = target_distance / max_diopters

        # create 3d tensor with depthmap in the first channel and ecc and vergence in the other two
        output = torch.zeros((6, params.kernel_height, params.kernel_height))
        output[0] = depthmap
        output[1] = ipd
        output[2] = l_gaze_x
        output[3] = l_gaze_y
        output[4] = r_gaze_x
        output[5] = r_gaze_y
        
        label = torch.tensor([target_distance], dtype=torch.float32)

        #print(1/(label * max_diopters))

        return output, label

    def ids_to_idx(self, scene_id, participant_id, frame_nr):
        """
        Get the index of the frame in the data based on the scene_id, participant_id and frame_number.

        Parameters
        ----------
            scene_id: int
                The index of the scene
            participant_id: int
                The id of the participant
            frame_nr: int
                The number of the frame
        """

        idx = self.et_data.loc[(self.et_data['scene_id'] == scene_id) & (self.et_data['participant_id'] == participant_id) & (self.et_data['frame_number'] == frame_nr)].index
        return idx

    def idx_to_ids(self, idx):
        """
        Get scene_id, participant_id and frame_number of the frame based on the index.

        Parameters
        ----------
            idx: int
                The index of the data within the dataframe
        """

        scene_id = self.et_data.loc[idx, 'scene_id']
        participant_id = self.et_data.loc[idx, 'participant_id']
        frame_number = self.et_data.loc[idx, 'frame_number']
        return scene_id, participant_id, frame_number

class DataCNNETDiopters:
    def __init__(self, test_only_one_subj=False, subj_id=3, permutation=ETData.NONE, seed=42):
        self.ds, self.train_indices, self.val_indices, self.test_indices = self.init_dataset(test_only_one_subj, subj_id, permutation=permutation, seed=seed)

    
    def init_dataset(self, test_only_one_subj, subj_id, permutation, seed):
        current_path = os.getcwd()

        path = os.path.abspath(os.path.join(current_path, '..', 'data', 'et_data_cnn_rollingmedian.feather')) #'et_data_artificial.feather'))#'et_data_cnn_rollingmedian.feather')) ##'et_data_cnn_var_rollingmedian.feather'))
        data = pd.read_feather(path)

        # depth-data
        distance_indoor_path = os.path.abspath(os.path.join(current_path, '..', 'data', 'distancedata_32bit', 'indoor'))
        distance_outdoor_path = os.path.abspath(os.path.join(current_path, '..', 'data', 'distancedata_32bit', 'outdoor'))

        # instantiate the dataset
        #ds = CNNDatasetETDiopters(data, distance_indoor_path, distance_outdoor_path, permutation, seed)

        # auf 30 subjects ods cross validation -> test auf den 11 mit den parametern
        subjs = range(3, 44)
        np.random.seed(42)

        if test_only_one_subj:
            train_percentage = 0.5
            val_percentage = 0.5

            not_ts = np.setdiff1d(subjs, [subj_id])
            print(f'Number Train + val subjects: {len(not_ts)}')
            val_subjs = np.random.choice(not_ts, int(val_percentage * len(not_ts)), replace=False)
            print(f'Number Validation subjects: {len(val_subjs)}')
            train_subjs = np.setdiff1d(not_ts, val_subjs)
            print(f'Number Train subjects: {len(train_subjs)}')
            test_subjs = [subj_id]
        else:
            train_percentage = 0.5
            val_percentage = 0.25

            train_subjs = np.random.choice(subjs, int(train_percentage * len(subjs)), replace=False)
            not_ts = np.setdiff1d(subjs, train_subjs)
            val_subjs = np.random.choice(not_ts, int(val_percentage * len(subjs)), replace=False)
            test_subjs = np.setdiff1d(not_ts, val_subjs)
    

        train_set = data[data['participant_id'].isin(train_subjs)]
        val_set = data[data['participant_id'].isin(val_subjs)]
        test_set = data[data['participant_id'].isin(test_subjs)]

        print(f'Train subjects: {train_subjs}')
        print(f'Validation subjects: {val_subjs}')
        print(f'Test subjects: {test_subjs}')

        # get indices of the datasets for the dataloaders
        train_indices = train_set.index
        val_indices = val_set.index
        test_indices = test_set.index

        
        # standardize gaze and ipd
        data = utils.standardize(data, train_indices, val_indices, test_indices)

        ds = CNNDatasetETDiopters(data, distance_indoor_path, distance_outdoor_path, permutation, seed)

        return ds, train_indices, val_indices, test_indices
    

