import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

class DataMLP():
    def __init__(self, train_percentage=0.5, val_percentage=0.25, test_percentage=0.25, test_only_one_subj=False, subj_id=3):
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        self.init_dataset(train_percentage, val_percentage, test_percentage, test_only_one_subj, subj_id)

    def init_dataset(self, train_percentage=0.5, val_percentage=0.25, test_percentage=0.25, test_only_one_subj=False, subj_id=3):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print('Using device:', device)

        #print('Splitting data into train, validation and test sets according to subjects...')
    
        current_path = os.getcwd()

        path = os.path.abspath(os.path.join(current_path, '..', 'data', 'mlp_training_data_10percent_ang2.feather'))
        data = pd.read_feather(path)

        # clip vergence estimates to 20
        data['vergence'] = data['vergence'].clip(0, 20)
        
        # auf 30 subjects ods cross validation -> test auf den 11 mit den parametern
        #train_percentage = 0.5
        #val_percentage = 0.25
        #test_percentage = 0.25

        subjs = range(3, 44)

        # take train_percentage of the subjects
        np.random.seed(42)

        if test_only_one_subj:
            not_ts = np.setdiff1d(subjs, [subj_id])
            print(f'Number Train + val subjects: {len(not_ts)}')
            val_subjs = np.random.choice(not_ts, int(val_percentage * len(not_ts)), replace=False)
            print(f'Number Validation subjects: {len(val_subjs)}')
            train_subjs = np.setdiff1d(not_ts, val_subjs)
            print(f'Number Train subjects: {len(train_subjs)}')
            test_subjs = [subj_id]
        else:
            train_subjs = np.random.choice(subjs, int(train_percentage * len(subjs)), replace=False)
            not_ts = np.setdiff1d(subjs, train_subjs)
            val_subjs = np.random.choice(not_ts, int(val_percentage * len(subjs)), replace=False)
            test_subjs = np.setdiff1d(not_ts, val_subjs)

        print(f'Train subjects: {train_subjs}')
        print(f'Validation subjects: {val_subjs}')
        print(f'Test subjects: {test_subjs}')

        # convert to strings 
        train_subjs_str = [str(subj) for subj in train_subjs]
        val_subjs_str = [str(subj) for subj in val_subjs]
        test_subjs_str = [str(subj) for subj in test_subjs]

        train_set = data[data['participant_id'].isin(train_subjs_str)]
        val_set = data[data['participant_id'].isin(val_subjs_str)]
        test_set = data[data['participant_id'].isin(test_subjs_str)]

        # train set
        X_train_subj = train_set[['center', 'eccentricity', 'vergence']].to_numpy()
        samples_train_subj = np.array([np.array(row) for row in train_set['samples'].to_numpy()])

        X_train_subj = np.concatenate((X_train_subj, samples_train_subj), axis=1)
        y_train_subj = train_set['distance'].to_numpy()

        # validation set
        X_val_subj = val_set[['center', 'eccentricity', 'vergence']].to_numpy()
        samples_val_subj = np.array([np.array(row) for row in val_set['samples'].to_numpy()])
        X_val_subj = np.concatenate((X_val_subj, samples_val_subj), axis=1)
        y_val_subj = val_set['distance'].to_numpy()

        # test set
        X_test_subj = test_set[['center', 'eccentricity', 'vergence']].to_numpy()
        samples_test_subj = np.array([np.array(row) for row in test_set['samples'].to_numpy()])

        X_test_subj = np.concatenate((X_test_subj, samples_test_subj), axis=1)
        y_test_subj = test_set['distance'].to_numpy()

        # to correct dimensions
        y_train_subj = y_train_subj.reshape(-1, 1)
        y_val_subj = y_val_subj.reshape(-1, 1)
        y_test_subj = y_test_subj.reshape(-1, 1)

        # convert to torch tensors
        self.X_train = torch.tensor(X_train_subj, dtype=torch.float32)#.to(device)
        self.X_val = torch.tensor(X_test_subj, dtype=torch.float32)#.to(device)
        self.X_test = torch.tensor(X_test_subj, dtype=torch.float32)#.to(device)
        self.y_train = torch.tensor(y_train_subj, dtype=torch.float32)#.to(device)
        self.y_val = torch.tensor(y_test_subj, dtype=torch.float32)#.to(device)
        self.y_test = torch.tensor(y_test_subj, dtype=torch.float32)#.to(device)

        print(f'Data loading done.\nSplit into {len(train_subjs_str)} train subjects, {len(val_subjs_str)} validation subjects and {len(test_subjs_str)} test subjects.')

        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
    
    def init_dataloader(self, batch_size=32, num_workers=0):
        train_loader_subj = DataLoader(TensorDataset(self.X_train, self.y_train), num_workers=num_workers, batch_size=batch_size, shuffle=True)#, persistent_workers=True)
        val_loader_subj = DataLoader(TensorDataset(self.X_val, self.y_val), num_workers=num_workers, batch_size=batch_size, shuffle=False)#, persistent_workers=True)
        
        return train_loader_subj, val_loader_subj
