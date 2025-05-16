import torch
import torch.nn as nn
max = 20

class CNN(nn.Module):
    def __init__(self, activation=nn.ReLU()):
        super(CNN, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        self.activation = activation

        if isinstance(activation, nn.ReLU):
            nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')

            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')

            print('Weights initialized with He initialization.')

        elif isinstance(activation, nn.Tanh):
            nn.init.xavier_normal_(self.conv1.weight)
            nn.init.xavier_normal_(self.conv2.weight)
            nn.init.xavier_normal_(self.conv3.weight)
            nn.init.xavier_normal_(self.conv4.weight)

            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
            nn.init.xavier_normal_(self.fc3.weight)
            nn.init.xavier_normal_(self.fc4.weight)

            print('Weights initialized with Xavier initialization.')

    
    def forward(self, x):
        # Convolutional layers with ReLU and MaxPooling
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        
        # Flatten the feature maps
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))  # FC1 + ReLU
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # Output layer (no activation for regression)
        

        return x
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
        

class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=0) # no padding as depth values should not arbitrarily be inserted, neither do participants likely look at the edges of the image
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        
        # Fully connected layers
        self.fc4 = nn.Linear(16928, 1)
        
    
    def forward(self, x):
        # Convolutional layers with ReLU and MaxPooling
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 + ReLU + Pooling
        
        # Flatten the feature maps
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = self.fc4(x)  # Output layer (no activation for regression)
        
        return x
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
        
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=0) # no padding as depth values should not arbitrarily be inserted, neither do participants likely look at the edges of the image
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)

        # Fully connected layers
        self.fc4 = nn.Linear(6400, 1)
        
    
    def forward(self, x):
        # Convolutional layers with ReLU and MaxPooling
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv3(x)))
        
        # Flatten the feature maps
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = self.fc4(x)  # Output layer (no activation for regression)
        
        return x
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
        
class LargeCNN(nn.Module):
    def __init__(self):
        super(LargeCNN, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=0) # no padding as depth values should not arbitrarily be inserted, neither do participants likely look at the edges of the image
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        #self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        
    
    def forward(self, x):
        # Convolutional layers with ReLU and MaxPooling
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 + ReLU + Pooling
        x = torch.relu(self.conv3(x))
        x = self.pool(torch.relu(self.conv4(x)))
        x = torch.relu(self.conv5(x))
        x = self.pool(torch.relu(self.conv6(x)))
        #x = torch.relu(self.conv7(x))
        
        # Flatten the feature maps
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))  # Output layer (no activation for regression)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
        

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        # test:
        # padding 0


    
    def forward(self, x):

        gt_depths = x.clone()

        # Convolutional layers with ReLU and MaxPooling
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        
        # Flatten the feature maps
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))  # FC1 + ReLU
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        #x = self.fc4(x)  # Output layer (no activation for regression)
        
        predictions = self.fc4(x).squeeze()  # Output layer (no activation for regression)

        # Assume x or another input tensor is used for finding the nearest value
        nearest_depths = []
        for idx in range(len(x)):
            possible_depth_vals = torch.flatten(gt_depths[idx][0])  # Choose the feature map or other relevant input
            prediction = predictions[idx]

            # Find the soft nearest depth value using the function below
            diff = (possible_depth_vals - prediction)**2
            weights = softmax(-1*diff)
            weighted_sum = compute_weighted_sum(possible_depth_vals, weights)
            nearest_depths.append(weighted_sum)

        # Return tensor with "soft nearest" values
        nearest_depths_tensor = torch.stack(nearest_depths).to(x.device)
        nearest_depths_tensor = nearest_depths_tensor.unsqueeze(1)
        
        return nearest_depths_tensor

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    

        
class LargeCNNClassifier(nn.Module):
    def __init__(self):
        super(LargeCNNClassifier, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=0) # no padding as depth values should not arbitrarily be inserted, neither do participants likely look at the edges of the image
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        
    
    def forward(self, x):

        gt_depths = x.clone()

        # Convolutional layers with ReLU and MaxPooling
        x = self.pool(torch.relu(self.conv1(x)))    # Conv1 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv2(x)))    # Conv2 + ReLU + Pooling
        x = torch.relu(self.conv3(x))               # Conv3 + ReLU
        x = self.pool(torch.relu(self.conv4(x)))
        x = torch.relu(self.conv5(x))
        x = self.pool(torch.relu(self.conv6(x)))
        
        # Flatten the feature maps
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        predictions = self.fc4(x).squeeze()  # Output layer (no activation for regression)

        nearest_depths = []
        for idx in range(len(x)):
            possible_depth_vals = torch.flatten(gt_depths[idx][0])
            prediction = predictions[idx]

            # Find the soft nearest depth value using the function below
            diff = (possible_depth_vals - prediction)**2
            weights = softmax(-1*diff)
            weighted_sum = compute_weighted_sum(possible_depth_vals, weights)
            nearest_depths.append(weighted_sum)

        # Return tensor with "soft nearest" values
        nearest_depths_tensor = torch.stack(nearest_depths).to(x.device)
        nearest_depths_tensor = nearest_depths_tensor.unsqueeze(1)
        
        return nearest_depths_tensor

def softmax(x):
    return nn.Softmax(dim=0)(x)
    
def compute_weighted_sum(x, weights):
    return torch.matmul(x, weights)
    
def find_nearest(tensor, value):
    idx = (tensor - value).abs().argmin()
    return torch.flatten(tensor)[idx]


class CNNClassifierDiopters(nn.Module):
    def __init__(self, activation=nn.ReLU()):
        super(CNNClassifierDiopters, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        self.activation = activation

        if isinstance(activation, nn.ReLU):
            nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')

            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')

            print('Weights initialized with He initialization.')

        elif isinstance(activation, nn.Tanh):
            nn.init.xavier_normal_(self.conv1.weight)
            nn.init.xavier_normal_(self.conv2.weight)
            nn.init.xavier_normal_(self.conv3.weight)
            nn.init.xavier_normal_(self.conv4.weight)

            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
            nn.init.xavier_normal_(self.fc3.weight)
            nn.init.xavier_normal_(self.fc4.weight)

            print('Weights initialized with Xavier initialization.')
    
    def forward(self, x):

        # denormalize depth values
        gt_depths = x.clone()

        # Convolutional layers with ReLU and MaxPooling
        x = self.pool(torch.relu(self.conv1(x)))  # Conv1 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv2(x)))  # Conv2 + ReLU + Pooling
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        
        # Flatten the feature maps
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))  # FC1 + ReLU
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        #x = self.fc4(x)  # Output layer (no activation for regression)
        
        predictions = self.fc4(x).squeeze()  # Output layer (no activation for regression)

        # Assume x or another input tensor is used for finding the nearest value
        nearest_depths = []
        for idx in range(len(x)):
            possible_depth_vals = torch.flatten(gt_depths[idx][0])  # Choose the feature map or other relevant input
            prediction = predictions[idx]

            # Find the soft nearest depth value using the function below
            diff = (possible_depth_vals - prediction)**2
            weights = softmax(-1*diff)
            weighted_sum = compute_weighted_sum(possible_depth_vals, weights)
            nearest_depths.append(weighted_sum)

        # Return tensor with "soft nearest" values
        nearest_depths_tensor = torch.stack(nearest_depths).to(x.device)
        nearest_depths_tensor = nearest_depths_tensor.unsqueeze(1)
        
        return nearest_depths_tensor

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)