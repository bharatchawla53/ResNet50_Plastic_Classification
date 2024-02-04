# Bharat Chawla and Himaja R. Ginkala
# This class reads the newtork and run its on the handwritten test set to predict greek letters.  

import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torchsummary import summary

# main - Loads the existing network and applies it onto the test handwritten set.
def main(argv):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device" + '\n')

    # create a new neural network model instance 
    model = gl.NeuralNetwork().to(device)

    # load the saved network
    model.load_state_dict(torch.load('plastics_cnn.pth'))

    # set the mode to evaluation mode 
    model.eval()

    # prints summary of the neural network
    summary(model, input_size=(3, 50, 50))

    # freeze the network weights
    for param in model.parameters():
        param.requires_grad = False

    # fetches all layers without last one and adds a new linear layer with three nodes 
    truncated_model = nn.Sequential(*list(model.children())[:-1], nn.Linear(50, 3)) 
    print(truncated_model)

    # create dataset object for greek letters
    plastics_dataset = torchvision.datasets.ImageFolder(root='GreekLetters/test', 
                                                    transform=transforms.Compose([
                                                        transforms.Resize((50, 50)),
                                                        #transforms.Grayscale(num_output_channels=1),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.1307,), (0.3081,))]))

    # convert dataset to dataloader 
    batch_size_test = 8

    test_loader = torch.utils.data.DataLoader(plastics_dataset, batch_size=batch_size_test, shuffle=False)

    with torch.no_grad():

        dataiter = iter(test_loader)
       
        while True:
            try:

                data, target = next(dataiter)
                data, target = data.to(device), target.to(device)

                # apply model
                predicted_outputs = model(data)

                # get predicted labels
                _, predicted = torch.max(predicted_outputs, 1) 

                # print outputs, predictions, and actual labels
                for i in range(0, 9):
                    outputElements = 'List of outputs: ['
                    for j in range(0, 9):
                        outputElements += str(round(predicted_outputs[i,j].item(), 2))
                        if (j != 8):
                            outputElements += ', '

                    outputElements += ']'
                    print(outputElements)

                    if (predicted[i].item() == 0):
                        print('PREDICTION: Heavy-Plastic')

                    elif (predicted[i] == 1):
                        print('PREDICTION: No-Image')

                    elif (predicted[i] == 2):
                        print('PREDICTION: No-Plastic')

                    else:
                        print('PREDICTION: Some-Plastic')

                    if (target[i].item() == 0):
                        print('ACTUAL: Heavy-Plastic \n')

                    elif (target[i].item() == 1):
                        print('ACTUAL: No-Image \n')

                    elif (target[i].item() == 2):
                        print('ACTUAL: No-Plastic \n')

                    else:
                        print('ACTUAL: Some-Plastic \n')


            except StopIteration:
                break

    # obtain one batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    images, labels = images.to(device), labels.to(device)

    # get sample outputs 
    output = model(images)

    # convert output probabilities to predicted class 
    _, predicted = torch.max(output, 1)

    # prep images for display
    images = images.cpu().numpy()

    return

if __name__ == "__main__":
    main(sys.argv)
