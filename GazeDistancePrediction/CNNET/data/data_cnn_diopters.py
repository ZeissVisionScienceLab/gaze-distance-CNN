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

max = 20
max_diopters = 1 / 0.1

class CNNDatasetDiopters(Dataset):
    def __init__(self, et_data: pd.DataFrame, depth_path_indoor: str, depth_path_outdoor: str):
        self.et_data = et_data
        self.depth_path_indoor = depth_path_indoor
        self.depth_path_outdoor = depth_path_outdoor

        size_of_float32 = 4
        self.frame_size = params.kernel_height * params.kernel_height * size_of_float32
        self.bytes_to_read = self.frame_size
        

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


        # get eccentricity and vergence
        ecc = self.et_data.loc[idx, 'eccentricity']
        vergence = self.et_data.loc[idx, 'vergence']
        target_distance = self.et_data.loc[idx, 'distance']

        label = torch.tensor([target_distance], dtype=torch.float32)
        print(1 / label)

        # denormalize
        vergence = vergence * max
        depthmap = depthmap * max
        label = label * max

        # to torch
        ecc = torch.tensor(ecc, dtype=torch.float32)
        vergence = torch.tensor(vergence, dtype=torch.float32)

        # clip to 0.1, max
        vergence = torch.clamp(vergence, 0.1, max)
        depthmap = torch.clamp(depthmap, 0.1, max)
        label = torch.clamp(label, 0.1, max)

        # meters to diopters
        vergence = 1 / vergence
        depthmap = 1 / depthmap
        label = 1 / label

        # normalize with max_diopters
        vergence = vergence / max_diopters
        depthmap = depthmap / max_diopters
        label = label / max_diopters

        # create 3d tensor with depthmap in the first channel and ecc and vergence in the other two
        output = torch.zeros((3, params.kernel_height, params.kernel_height))
        output[0] = depthmap
        output[1] = ecc
        output[2] = vergence

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

class DataCNNDiopters:
    def __init__(self):
        self.ds, self.train_indices, self.val_indices, self.test_indices = self.init_dataset()

    
    def init_dataset(self):
        current_path = os.getcwd()

        path = os.path.abspath(os.path.join(current_path, '..', 'data', 'et_data_cnn_rollingmedian.feather'))#et_data_cnn.feather'))
        data = pd.read_feather(path)

        # depth-data
        distance_indoor_path = os.path.abspath(os.path.join(current_path, '..', 'data', 'distancedata_32bit', 'indoor'))
        distance_outdoor_path = os.path.abspath(os.path.join(current_path, '..', 'data', 'distancedata_32bit', 'outdoor'))

        # instantiate the dataset
        ds = CNNDatasetDiopters(data, distance_indoor_path, distance_outdoor_path)

        # auf 30 subjects ods cross validation -> test auf den 11 mit den parametern
        train_percentage = 0.5
        val_percentage = 0.25

        subjs = range(3, 44)

        # take train_percentage of the subjects
        np.random.seed(42)
        train_subjs = np.random.choice(subjs, int(train_percentage * len(subjs)), replace=False)
        not_ts = np.setdiff1d(subjs, train_subjs)
        val_subjs = np.random.choice(not_ts, int(val_percentage * len(subjs)), replace=False)
        test_subjs = np.setdiff1d(not_ts, val_subjs)

        train_set = data[data['participant_id'].isin(train_subjs)]
        val_set = data[data['participant_id'].isin(val_subjs)]
        test_set = data[data['participant_id'].isin(test_subjs)]

        # get indices of the datasets for the dataloaders
        train_indices = train_set.index
        val_indices = val_set.index
        test_indices = test_set.index

        return ds, train_indices, val_indices, test_indices
    

