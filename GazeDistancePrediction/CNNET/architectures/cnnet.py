import torch
import torch.nn as nn
max = 20

class CNNET(nn.Module):

    def __init__(self, activation=nn.ReLU()):
        super(CNNET, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(261, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 1)

        self.activation = activation

        if isinstance(activation, nn.ReLU):
            nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu') 
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv5.weight, nonlinearity='relu')

            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc5.weight, nonlinearity='relu')

            print('Weights initialized with He initialization.')

        elif isinstance(activation, nn.Tanh):
            nn.init.xavier_normal_(self.conv1.weight)
            nn.init.xavier_normal_(self.conv2.weight)
            nn.init.xavier_normal_(self.conv3.weight)
            nn.init.xavier_normal_(self.conv4.weight)
            nn.init.xavier_normal_(self.conv5.weight)

            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
            nn.init.xavier_normal_(self.fc3.weight)
            nn.init.xavier_normal_(self.fc4.weight)
            nn.init.xavier_normal_(self.fc5.weight)

            print('Weights initialized with Xavier initialization.')


    
    def forward(self, x):

        t = x.clone()

        # set x to first elements of each tensor in the batch
        x = x[:,0,:,:].unsqueeze(1)

        ipd = t[:,1,0,0].unsqueeze(1)
        l_gaze_x = t[:,2,0,0].unsqueeze(1)
        l_gaze_y = t[:,3,0,0].unsqueeze(1)
        r_gaze_x = t[:,4,0,0].unsqueeze(1)
        r_gaze_y = t[:,5,0,0].unsqueeze(1)

        # Convolutional layers with ReLU and MaxPooling
        x = self.pool(self.activation(self.conv1(x)))  # Conv1 + ReLU + Pooling
        x = self.pool(self.activation(self.conv2(x)))  # Conv2 + ReLU + Pooling
        x = self.pool(self.activation(self.conv3(x)))
        x = self.pool(self.activation(self.conv4(x)))
        x = self.pool(self.activation(self.conv5(x)))
        
        # Flatten the feature maps
        x = torch.flatten(x, 1)

        # append ipd, l_gaze_x, l_gaze_y, r_gaze_x, r_gaze_y to the flattened feature maps
        x = torch.cat((x, ipd, l_gaze_x, l_gaze_y, r_gaze_x, r_gaze_y), 1)

        
        # Fully connected layers
        x = self.activation(self.fc1(x))  # FC1 + ReLU
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)  # Output layer (no activation for regression)
        
        return x
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
        

class CNNETConv(nn.Module):
    def __init__(self, activation=nn.ReLU()):
        super(CNNETConv, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=0)
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

        if activation == nn.ReLU():
            nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv4.weight, nonlinearity='relu')

            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')

        elif activation == nn.Tanh():
            nn.init.xavier_normal_(self.conv1.weight)
            nn.init.xavier_normal_(self.conv2.weight)
            nn.init.xavier_normal_(self.conv3.weight)
            nn.init.xavier_normal_(self.conv4.weight)

            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
            nn.init.xavier_normal_(self.fc3.weight)
            nn.init.xavier_normal_(self.fc4.weight)

        # test:
        # padding 0

    
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
        