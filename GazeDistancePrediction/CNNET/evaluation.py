import numpy as np
import matplotlib.pyplot as plt
import torch
import scripts.utils as utils
from sklearn.metrics import r2_score

class EvaluateModel():

    def evaluate(y_pred, y_test):
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mae = np.mean(np.abs(y_test - y_pred))
        me = np.mean(y_pred - y_test)
        r2 = r2_score(y_pred, y_test)

        mse_diopters = np.mean((1 / y_test - 1 / y_pred) ** 2)
        rmse_diopters = np.sqrt(np.mean((1 / y_test - 1 / y_pred) ** 2))
        mae_diopters = np.mean(np.abs(1 / y_test - 1 / y_pred))
        me_diopters = np.mean(1 / y_pred - 1 / y_test)

        error_diopters = np.abs(1 / y_pred - 1 / y_test)
        error = np.abs(y_pred - y_test)
        percent_below_01 = np.sum(error_diopters < 0.125) / len(error_diopters) * 100
        percent_below_01_m = np.sum(error < 0.01) / len(error) * 100

        error = np.abs(y_pred - y_test)
        percent_below_01_original_metric = np.sum(error < 0.125) / len(error) * 100

        print('MSE: ', mse)
        print('RMSE: ', rmse)
        print('MAE: ', mae)
        print('ME: ', me)
        print('R2: ', r2)

        print('MSE diopters: ', mse_diopters)
        print('RMSE diopters: ', rmse_diopters)
        print('MAE diopters: ', mae_diopters)
        print('ME diopters: ', me_diopters)

        print('Percent below 0.125 diopters: ', percent_below_01)
        print('Percent below 0.01 m: ', percent_below_01_m)

        return mse, rmse, mae, me, r2, mse_diopters, rmse_diopters, mae_diopters, me_diopters, percent_below_01, percent_below_01_original_metric, percent_below_01_m


    def test_model(model_trainer, dataloader):
        """
        Evaluate the model on the test set, compute basic metrics and return the predictions.

        Parameters:
        ----------
        model: torch.nn.Module or Trainer object
            The trained model.
        data: Data object
            The data object containing the test set.

        Returns:
        -------
        y_test_np: numpy array
            The ground truth test set values.
        y_pred_np: numpy array
            The predicted values.
        """

        model = model_trainer.model

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()  # Set model to evaluation mode

        all_outputs = []  # List to store outputs
        all_labels = []   # Optionally, to store all labels (for comparison, etc.)

        with torch.no_grad():  # Disable gradient computation for inference
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)

                # Append the batch outputs and labels
                all_outputs.append(outputs.cpu())  # Move outputs to CPU before storing
                all_labels.append(labels.cpu())    # Optionally append labels

        # Concatenate all batch outputs into a single tensor
        all_outputs = torch.cat(all_outputs, dim=0)  # Concatenates along batch dimension
        all_labels = torch.cat(all_labels, dim=0)    # If you need all labels as well

        # Now all_outputs contains all the outputs for the entire dataset
        print('Output shape: ', all_outputs.shape)
        print('Label shape: ', all_labels.shape)
    	
        #y_pred = model.predict(data.X_test)
        #y_test_np = data.y_test.numpy().flatten()
        #y_pred_np = y_pred.cpu().detach().numpy().flatten()

        y_pred_np = all_outputs.cpu().numpy().flatten()
        y_test_np = all_labels.cpu().numpy().flatten()

        # make predictions between mlp and cnn comparable, by having all depth values in the initial scale
        if model_trainer.model_type == 'cnn' or model_trainer.model_type == 'tinycnn' or model_trainer.model_type == 'smallcnn' or model_trainer.model_type == 'largecnn' or model_trainer.model_type == 'cnnclassifier' or model_trainer.model_type == 'largecnnclassifier' or model_trainer.model_type == 'cnnclassifierdiopters' or model_trainer.model_type == 'cnnet' or model_trainer.model_type == 'cnnetconv':

            # denormalize targets and predictions
            y_pred_np = utils.denormalize_depth(y_pred_np)
            y_test_np = utils.denormalize_depth(y_test_np)

        elif model_trainer.model_type == 'cnnclassifierdiopters' or model_trainer.model_type == 'cnnetdiopters' or model_trainer.model_type == 'cnnetconvdiopters' or model_trainer.model_type == 'cnndiopters':
            max_diopters = 1 / 0.1 

            # denormalize targets and predictions
            y_pred_np = 1 / (y_pred_np * max_diopters)
            y_test_np = 1 / (y_test_np * max_diopters)


        EvaluateModel.evaluate(y_pred_np, y_test_np)

        return y_test_np, y_pred_np


    def plot_losses(losses, vlosses):
        """
        Plot the training and validation losses.

        Parameters:
        ----------
        losses: numpy array
            The training losses.
        vlosses: numpy array
            The validation losses.
        """
        f = plt.figure(1)
        plt.plot(losses, label='Training loss')
        plt.plot(vlosses, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        f.show()


    def plot_predictions(y_test, y_pred):
        """
        Plot the predictions against the ground truth.

        Parameters:
        ----------
        y_test: numpy array
            The ground truth values.

        y_pred: numpy array
            The predicted values.
        """

        g = plt.figure(2)
        plt.hist2d(y_test, y_pred, bins=100, cmap='viridis', range=[[0, 3], [0, 3]])
        plt.colorbar()
        plt.xlabel('ground truth')
        plt.ylabel('prediction')
        plt.xlim(0, 3)
        plt.ylim(0, 3)

        # plot the line of equality
        plt.plot([0, 20], [0, 20], color='red', linewidth=0.5)

        g.show()


