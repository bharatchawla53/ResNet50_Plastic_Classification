import numpy as np
import torch 
import torchvision 
import torchvision.models as models
import cv2
import os
import torchvision.transforms.functional as tf
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torchsummary import summary
from PIL import Image
import sys

def main(argv):

    # epochs cycle
    epochs = 5

    # Applying Transforms to the Data
    image_transforms = { 
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])
    }

    # batch size 
    bs = 64

    # number of classes 
    classes = 4

    # load train data
    data = {'train': torchvision.datasets.ImageFolder(root='FilteredData/train', transform = image_transforms['train'])
            # we can add test/valid later
            }

    # Size of Data, to be used for calculating Average Loss and Accuracy
    train_data_size = len(data['train'])

    # iterators for the Data loaded using DataLoader module
    train_data = DataLoader(data['train'], batch_size=bs, shuffle=True)

    # Load pretrained ResNet50 Model
    resnet50 = models.resnet50(pretrained=True)

    # Freeze model parameters
    for param in resnet50.parameters():
        param.requires_grad = False

    # Change the final layer of ResNet50 Model for Transfer Learning
    fc_inputs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 10), 
        nn.LogSoftmax(dim=1) # For using NLLLoss()
    )

    # Convert model to be used on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    resnet50 = resnet50.to(device)

    # Define Optimizer and Loss Function
    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(resnet50.parameters())

    criterion = nn.CrossEntropyLoss()


    for epoch in range(epochs):
        #epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        # Set to training mode
        resnet50.train()
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Clean existing gradients
            optimizer.zero_grad()
            # Forward pass - compute outputs on input data using the model
            outputs = resnet50(inputs)
            # Compute loss
            loss = criterion(outputs, labels)
            # Backpropagate the gradients
            loss.backward()
            # Update the parameters
            optimizer.step()
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)

            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))


            ## Validation - No gradient tracking needed
            #with torch.no_grad():
            ## Set to evaluation mode
            #    model.eval()
            #    # Validation loop
            #    for j, (inputs, labels) in enumerate(valid_data_loader):
            #        inputs = inputs.to(device)
            #        labels = labels.to(device)
            #        # Forward pass - compute outputs on input data using the model
            #        outputs = model(inputs)
            #        # Compute loss
            #        loss = loss_criterion(outputs, labels)
            #        # Compute the total loss for the batch and add it to valid_loss
            #        valid_loss += loss.item() * inputs.size(0)
            #        # Calculate validation accuracy
            #        ret, predictions = torch.max(outputs.data, 1)
            #        correct_counts = predictions.eq(labels.data.view_as(predictions))
            #        # Convert correct_counts to float and then compute the mean
            #        acc = torch.mean(correct_counts.type(torch.FloatTensor))
            #        # Compute total accuracy in the whole batch and add to valid_acc
            #        valid_acc += acc.item() * inputs.size(0)
            #        print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            ## Find average training loss and training accuracy
            #avg_train_loss = train_loss/train_data_size 
            #avg_train_acc = train_acc/float(train_data_size)
            ## Find average training loss and training accuracy
            #avg_valid_loss = valid_loss/valid_data_size 
            #avg_valid_acc = valid_acc/float(valid_data_size)
            #history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
            #epoch_end = time.time()
            #print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, nttValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))


            
    return

if __name__ == "__main__":
    main(sys.argv)

