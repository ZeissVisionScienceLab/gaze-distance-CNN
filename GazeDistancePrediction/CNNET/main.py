import torch
from torch.utils.data import DataLoader, Subset
from training import Trainer
from data.data_cnn import DataCNN, ETData
from data.data_cnn_diopters import DataCNNDiopters
from data.data_cnn_etdata_diopters import DataCNNETDiopters
from data.data_cnn_etdata import DataCNNET
from evaluation import EvaluateModel as eval

import os
import numpy as np


#---------------------------------#
# Adjust according to your needs  #
#---------------------------------#

# Hyperparameters for training

batch_size = 64
num_workers = 4
num_epochs = 100
lr = 1e-3

# Architecture of the model with the name [model_type]_[model_id]_[model_counter] eg. cnnet_100_0

train = True                           # True if model should be trained, only tested otherwise
model_type = 'cnnetdiopters'           # available model types: cnn, cnnclassifier, cnnet, cnnetcov, cnndiopters, cnnclassifierdiopters, cnnetdiopters, cnnetconvdiopters
model_id = 100                         # save/load model with this number
model_counter = 2


#---------------------------------#


# Option to test the model's performance with only one participant
test_only_one_subj = False              # true, if model should be tested only on one subject, trained on the others
id = 3                                  # ids currently range from 3 to 43

# Learning rate optimization
lr_optimization = False                 # True if learning rate optimization should be performed
lrs = [1e-3, 1e-4, 1e-5, 1e-6]          # learning rates for the learning rate optimization

# feature permutation
feature_permutation = ETData.NONE       # for feature permutation: select from ETData.ET, ECCENTRICITY, VERGENCE, IPD, DEPTH
nr_permutations = 20                    # number of permutations for feature permutation

#---------------------------------#
# Do not change below this line   #
#---------------------------------#
 
seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

if feature_permutation != ETData.NONE:
    train = False

save_to_onnx = True

def __main__():

    for id in range(30, 44):
        for seed in range(nr_permutations):
            if feature_permutation == ETData.NONE:
                seed = 42

            # define dataset depending on model type
            print('Seed:', seed)
            train_loader, val_loader, test_loader = initialize_dataset(model_type, test_only_one_subj=test_only_one_subj, id=id , feature_permutation=feature_permutation, seed=seed)

            # initialize model
            model = Trainer(model_type, train_loader, val_loader)


            if train:
                if lr_optimization:
                    losses, vlosses, best_val_loss, best_lr = model.lr_optimization(lrs, nr_epochs=num_epochs, batch_size=batch_size)
                    print('Learning rate optimization done.')
                    print('Best learning rate: ', best_lr)

                else:
                    losses, vlosses, best_val_loss = model.train(lr=lr, nr_epochs=num_epochs, batch_size=batch_size)
                    print('Training done.')
                
                print('Best validation loss: ', best_val_loss)
                
                filepath_results, filepath_model, filepath_runs = initialize_filenames(model_type, feature_permutation=feature_permutation, train=train, model_id=model_id, test_only_one_subj=test_only_one_subj, id=id, model_counter=model_counter)

                model.save_model(filepath_model)
                np.save(filepath_results + '/losses.npy', np.array(losses))
                np.save(filepath_results + '/vlosses.npy', np.array(vlosses))

            else:
                filepath_results, filepath_model, filepath_runs = initialize_filenames(model_type, feature_permutation=feature_permutation, train=train, model_id=model_id, test_only_one_subj=test_only_one_subj, id=id, model_counter=model_counter)
                model.load_model(filepath_model)

                if model is None:
                    print('Model not found.')
                    return
                else:
                    print('Model loaded successfully.')

            if save_to_onnx:
                model.save_to_onnx(filepath_model)

            
            y_test, y_pred = eval.test_model(model, test_loader)
            np.save(filepath_results + '/y_test.npy', y_test)
            np.save(filepath_results + '/y_pred.npy', y_pred)

            print('Results saved to: ', filepath_results)
        
            if feature_permutation == ETData.NONE:
                break
            
        if not test_only_one_subj:
            break
    


def initialize_dataset(model_type, test_only_one_subj=False, id=3 , feature_permutation=ETData.NONE, seed=42):
        
        if model_type == 'cnn' or model_type == 'cnnclassifier':
            print('Model initialized with CNN model. Training on meters.')
            data = DataCNN(test_only_one_subj=test_only_one_subj, subj_id=id, permutation=feature_permutation, seed=seed)
            
        elif model_type == 'cnndiopters' or model_type == 'cnnclassifierdiopters':
            print('Model initialized with CNN model. Training on diopters.')
            data = DataCNNDiopters(test_only_one_subj=test_only_one_subj, subj_id=id, permutation=feature_permutation, seed=seed)
        
        elif model_type == 'cnnet' or model_type == 'cnnetconv':
            print('Model initialized with CNNET model. Training on meters.')
            data = DataCNNET(test_only_one_subj=test_only_one_subj, subj_id=id, permutation=feature_permutation, seed=seed)
        
        elif model_type == 'cnnetdiopters' or model_type == 'cnnetconvdiopters':
            print('Model initialized with CNNET model. Training on diopters.')
            data = DataCNNETDiopters(test_only_one_subj=test_only_one_subj, subj_id=id, permutation=feature_permutation, seed=seed)
        
        else:
            raise ValueError('Model type not supported.')
        
        ds = data.ds
        train_indices = data.train_indices
        val_indices = data.val_indices
        test_indices = data.test_indices
        
        train_loader = DataLoader(Subset(ds, train_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(Subset(ds, val_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(Subset(ds, test_indices), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        return train_loader, val_loader, test_loader

def initialize_filenames(model_type, feature_permutation=ETData.NONE, train=True, model_id=100, test_only_one_subj=test_only_one_subj, id=id, model_counter=0):

    perm_counter = 0

    filename = model_type + '_' + str(model_id) + '_' + str(model_counter)

    filepath_results, filepath_model, filepath_runs = get_filepaths(model_type, filename)

    if train:
        while os.path.exists(filepath_results):

            print('Model already exists. Renaming model.')

            model_counter += 1
            filename = model_type + '_' + str(model_id) + '_' + str(model_counter)
            filepath_results, filepath_model, filepath_runs = get_filepaths(model_type, filename, test_only_one_subj=test_only_one_subj, id=id)

    else:

        if feature_permutation != ETData.NONE:

            if feature_permutation == ETData.ECCENTRICITY:
                perm_type = 'eccentricity'
            elif feature_permutation == ETData.VERGENCE:
                perm_type = 'vergence'
            elif feature_permutation == ETData.DEPTH:
                perm_type = 'depth'
            elif feature_permutation == ETData.ET:
                perm_type = 'et'
            elif feature_permutation == ETData.IPD:
                perm_type = 'ipd'
        
            name_fp = model_type + '_' + perm_type + '_' + str(perm_counter)
            filename_fp = name_fp + '_' + str(model_id) + '_' + str(model_counter)

            filepath_results, _, filepath_runs = get_filepaths(model_type, filename_fp, test_only_one_subj=test_only_one_subj, id=id)

            while os.path.exists(filepath_results):
                perm_counter += 1

                print('Results already exist. Renaming folders for saving feature permutation results.')
                name_fp = model_type + '_' + perm_type + '_' + str(perm_counter)
                filename_fp = name_fp + '_' + str(model_id) + '_' + str(model_counter)
                    
                filepath_results, _, filepath_runs = get_filepaths(model_type, filename_fp, test_only_one_subj=test_only_one_subj, id=id)
    
    if not os.path.exists(filepath_results):
        os.makedirs(filepath_results)
        #os.makedirs(filepath_model)
        os.makedirs(filepath_runs)

    return filepath_results, filepath_model, filepath_runs
        

def get_filepaths(model_type, filename, test_only_one_subj=False, id=3):

    if test_only_one_subj:
        filename += '_subj_' + str(id)

    filepath_results = 'results/' + model_type + '/' + filename
    filepath_model = 'saved_models/' + model_type + '/' + filename + '.pt'
    filepath_runs = 'runs/' + model_type + '/' + filename

    return filepath_results, filepath_model, filepath_runs


if __name__ == "__main__":
    __main__()