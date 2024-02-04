# Bharat Chawla and Himaja R. Ginkala
# This class creates and trains the neural network for recognizing plastics into heavy plastic, no plastic, some plastic, and no plastic. 

# import statements
import matplotlib.pyplot as plt
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

# class to build a neural network for categorizing plastics
class NeuralNetwork(nn.Module):

    # create a neural network 
    def __init__(self):
        super().__init__()

        # convolution layers
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        # dropout layer
        self.dropout = nn.Dropout(0.5)

        # flattening layer
        self.fc1 = nn.Linear(1620, 50)
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, 1)
        return x

# utility method to train the network on train dataset
def train_network(model, device, train_loader, optimizer, epoch, train_losses, train_counter, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))

# utility method to train the network on test dataset
def test_network(model, device, test_loader, test_losses, test_counter):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# main function - it loads the dataset and trains the neural network, and plots the model's performance 
def main(argv):
    # handle any command line arguments in argv

    # configs for repeatable experiments
    torch.backends.cudnn.enabled = False
    torch.manual_seed(42)

    # create dataset object
    train_dataset = torchvision.datasets.ImageFolder(root='Data/train', 
                                                    transform=transforms.Compose([
                                                        transforms.Resize((50, 50)),
                                                        #transforms.Grayscale(num_output_channels=1),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.1307,), (0.3081,))]))

    test_dataset = torchvision.datasets.ImageFolder(root='Data/test', 
                                                    transform=transforms.Compose([
                                                        transforms.Resize((50, 50)),
                                                        #transforms.Grayscale(num_output_channels=1),  
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.1307,), (0.3081,))]))

    # convert dataset to dataloader 
    batch_size_train = 16
    batch_size_test = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train)

    
    #for batch in train_loader:
        #print(batch[0].size())
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test)

    # check if we can train our model on a hardware acceleator like GPU, if it's available 
    #evice = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # create an instance of NeuralNetwork and print its structure
    model = NeuralNetwork().to(device)

    # prints summary of the neural network
    summary(model, input_size=(3, 50, 50))

    # define optimizer to update weights and biases which are internal 
    # parameters of a model to reduce the loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # number of epochs - the number of times to iterate oevr the dataset
    n_epochs = 5

    # number of log interval
    log_interval = 10

    # variable to track the progress when training the model 
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs)]

    # each iteration of this optimization loop is called an epoch
    for epoch in range(1, n_epochs + 1):
        train_network(model, device, train_loader, optimizer, epoch, train_losses, train_counter, log_interval)
        test_network(model, device, test_loader, test_losses, test_counter)

    # evaulate the model's performance 
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig.show()
    plt.show()

    # save the model to a file 
    torch.save(model.state_dict(), 'plastics_cnn.pth')

    return

if __name__ == "__main__":
    main(sys.argv)

