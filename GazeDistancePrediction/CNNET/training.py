import torch
import torch.nn as nn
from architectures.mlp import MLP
from architectures.cnn import CNN, TinyCNN, SmallCNN, LargeCNN, CNNClassifier, LargeCNNClassifier, CNNClassifierDiopters
from architectures.cnnet import CNNET, CNNETConv
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from scripts import utils, params

class Trainer:
    def __init__(self, model_type, train_loader, val_loader, filename='model.pt', epsilon=10**-5, activation=nn.ReLU()):
        """
        Initializes the Trainer object.

        Parameters:
        ----------

        model_type: str
            The type of model to use. Either 'mlp' or 'cnn'.
        batch_size: int
            The batch size to use for training.
        loss_fn: torch.nn loss function
            The loss function to use.
        patience: int
            The number of epochs to wait for early stopping.
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.epsilon = epsilon

        if model_type == 'cnn':
            print('Trainer initialized with CNN model.')
            self.model = CNN().to(self.device)
        elif model_type == 'cnnclassifier':
            print('Trainer initialized with CNN Classifier model.')
            self.model = CNNClassifier().to(self.device)
        elif model_type == 'cnnclassifierdiopters':
            print('Trainer initialized with CNN Classifier Diopters model.')
            self.model = CNNClassifierDiopters().to(self.device)
        elif model_type == 'cnnet':
            print('Trainer initialized with CNNET model.')
            self.model = CNNET(activation).to(self.device)
        elif model_type == 'cnnetconv':
            print('Trainer initialized with CNNETConv model.')
            self.model = CNNETConv(activation).to(self.device)
        elif model_type == 'cnndiopters':
            print('Trainer initialized with CNN Diopters model.')
            self.model = CNN().to(self.device)
        elif model_type == 'cnnetdiopters':
            print('Trainer initialized with CNNET Diopters model.')
            self.model = CNNET(activation).to(self.device)
        elif model_type == 'cnnetconvdiopters':
            print('Trainer initialized with CNNETConv Diopters model.')
            self.model = CNNETConv(activation).to(self.device)
        else:
            raise ValueError('Model type not supported. Use MLP or CNN.')

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.filename = filename
        self.best_model_state_global = None
        self.best_val_loss_global = float('inf')

        print('Model initialized.')


    def train(self, lr=0.001, nr_epochs=100, batch_size=32, patience=10, tensorboard=False):
        """
        Trains the model.

        Parameters:
        ----------
        train_loader: torch.DataLoader
            Dataloader for training data.
        val_loader: torch DataLoader
            Dataloader for validation data.
        n_epochs: int
            The number of epochs to train for.
        
        Returns:
        -------
        losses: list
            The training losses.
        vlosses: list
            The validation losses.
        """

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        best_model_state = None
        best_val_loss = float('inf')

        if tensorboard:        
            # get only the model name from the filename
            model_name = self.filename.split('.')[0]
            writer = SummaryWriter('runs/' + self.model_type + '/' + model_name)

            hparams = {
                'model_type': self.model_type,
                'batch_size': self.batch_size,
                'loss_fn': str(self.loss_fn),
                'patience': self.patience,
                'optimizer': str(self.optimizer),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'nr_parameters': sum(p.numel() for p in self.model.parameters())
            }
        
            metrics = {}

            writer.add_hparams(hparams, metrics)

        epochs_no_improve = 0
        losses = []
        vlosses = []
        num_batches = len(self.train_loader)

        print('Training model with loss function:', self.loss_fn, 'and optimizer:', self.optimizer)

        for epoch in range(nr_epochs):
            self.model.train()
            times = []
            running_loss = 0.0

            for idx, (X_batch, y_batch) in enumerate(self.train_loader):
                print(f'Epoch {epoch+1}, Batch {idx+1}/{num_batches}', end='\r')
                # send to device
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # measure time for one forward pass
                start = time.time()
                y_pred_batch = self.model(X_batch)
                end = time.time()
                times.append(end - start)

                loss = self.loss_fn(y_pred_batch, y_batch)
                self.optimizer.zero_grad(set_to_none=True) # setting to None zeros the memory, while zero_grad() does not
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # output average time for one forward pass
            print('Average time for forward pass:', sum(times) / len(times) / batch_size)

            avg_loss = running_loss / len(self.train_loader)
            losses.append(avg_loss)

            if tensorboard: 
                writer.add_scalar("Loss/train", avg_loss, epoch)
            
            self.model.eval()
            running_vloss = 0.0

            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device)

                    outputs = self.model(data)
                    loss = self.loss_fn(outputs, target)
                    running_vloss += loss.item()

            
            avg_vloss = running_vloss / len(self.val_loader)

            if tensorboard: 
                writer.add_scalar("Loss/validation", avg_vloss, epoch)
                
            vlosses.append(avg_vloss)

            print(f'Epoch {epoch+1}, Training loss: {avg_loss}, Validation loss: {avg_vloss}')

            # Check if the validation loss improved
            if best_val_loss - avg_vloss < self.epsilon:
                print(f'Validation loss did not improve. Best was {best_val_loss:.4f}')
                epochs_no_improve += 1
            else:
                print(f'Validation loss improved from {best_val_loss:.4f} to {avg_vloss:.4f}')
                best_val_loss = avg_vloss
                best_model_state = self.model.state_dict()  # Save best model state
                epochs_no_improve = 0  # Reset early stopping counter

            # Early stopping based on patience
            if epochs_no_improve >= patience:
                print("Early stopping triggered. Restoring best model.")
                break


        if best_val_loss < self.best_val_loss_global:
            self.best_val_loss_global = best_val_loss
            self.best_model_state_global = best_model_state


        self.model.load_state_dict(self.best_model_state_global)

        if tensorboard: 
            writer.flush()
            writer.close()

        return losses, vlosses, best_val_loss
    
    def predict(self, X):
        """
        Predicts the output for the given input.

        Parameters:
        ----------
        X: torch.Tensor
            The input data.
        
        Returns:
        -------
        y_pred: torch.Tensor
            The predicted output.
        """
        X = X.to(self.device)
        y_pred = self.model.predict(X)
        return y_pred

    def lr_optimization(self, lrs=[10**-1, 10*-2, 10**-3, 10**-4, 10**-5, 10**-6], nr_epochs=100, batch_size=32, patience=10):

        losses_lrs = []
        vlosses_lrs = []
        best_val_losses = []
        self.n_epochs = nr_epochs
        self.batch_size = batch_size
        self.patience = patience

        for lr in lrs:
            self.model = self.model.__class__().to(self.device)
            # initialize model with new weights

            losses, vlosses, best_val_loss = self.train(lr=lr, nr_epochs=nr_epochs, batch_size=batch_size, patience=patience)
            losses_lrs.append([losses])
            vlosses_lrs.append([vlosses])
            best_val_losses.append(best_val_loss)

        # choose the best learning rate
        best_lr = lrs[np.argmin(best_val_losses)]
        best_losses = losses_lrs[np.argmin(best_val_losses)]
        best_vlosses = vlosses_lrs[np.argmin(best_val_losses)]
        best_val_loss = best_val_losses[np.argmin(best_val_losses)]

        return best_losses, best_vlosses, best_val_loss, best_lr


    def init_filenames(self, filename):
        self.filename = filename

    
    def save_model(self, path):
        """
        Saves model to file.

        Parameters:
        ----------
        path: str
            The path to save the model to.
        """

        torch.save(self.model.state_dict(), path)
        print('Model saved to', path)

    def save_to_onnx(self, path):
        """
        Saves model to ONNX format.

        Parameters:
        ----------
        path: str
            The path to save the model to.
        """

        # remove .pt extension and add .onnx
        path = path.split('.')[0] + '.onnx'
        inputs = torch.zeros((1, 6, params.kernel_height, params.kernel_height)).to(self.device)
        torch.onnx.export(self.model, inputs, path, input_names=['input'], output_names=['output'], export_params=True, opset=9)
        print('Model saved as ONNX file to', path)



    def save_logs(self, path):
        """
        Saves logs to file.
        """

        with open(path, 'w') as f:
            f.write('Model: ' + str(self.model) + '\n')
            f.write('Optimizer: ' + str(self.optimizer) + '\n')
            f.write('Loss function: ' + str(self.loss_fn) + '\n')
            f.write('Patience: ' + str(self.patience) + '\n')


        print('Logs saved to', path)

    def load_model(self, path):

        self.model.load_state_dict(torch.load(path, weights_only=False))
        print('Model loaded from', path)
        
        return self.model
    
    