from src.models.resnet import resnet18
from random import randint
import random
import torchvision
import torchvision.transforms as transforms
from src.utils.helpers import device
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from bisect import bisect_left
from torch.utils.data import Subset,TensorDataset
from torch.utils.data import Dataset, DataLoader, Subset

      

def offline_profiling(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    count = 0        
    misclassified_inputs = []
    misclassified_targets = []
    correct_inputs = []
    correct_targets = []
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            count += 1
            
            if predicted.item() != targets.item():
            # Misclassified sample
                misclassified_inputs.append(inputs.cpu())
                misclassified_targets.append(targets.cpu())
            else:
                # Correctly classified sample
                correct_inputs.append(inputs.cpu())
                correct_targets.append(targets.cpu())

        print(f'\nTest set: Average loss: {test_loss/len(testloader):.4f}, Accuracy: {correct}/{total} ({100.*correct/total:.2f}%)\n')    
       
    if correct_inputs:
        # Concatenate tensors
        correct_inputs = torch.cat(correct_inputs, dim=0)
        correct_targets = torch.cat(correct_targets, dim=0)

        # Create a new dataset and DataLoader
        new_test_dataset = TensorDataset(correct_inputs, correct_targets)
        # Save the new dataset
        torch.save({
            'inputs': correct_inputs,
            'targets': correct_targets
        }, 'correct_data.pt')

        new_testloader = DataLoader(
            new_test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=testloader.num_workers
        )
    else:
        print("No correctly classified samples found.")


    correct =0
    total =0
    count =0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(new_testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            count += 1
        print(f'\nTest set: Average loss: {test_loss/len(testloader):.4f}, Accuracy: {correct}/{total} ({100.*correct/total:.2f}%)\n')    
       

   
if __name__ == '__main__':

    model = resnet18(pretrained=True, progress=True,device=device)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])


    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)



    #net.fc = nn.Linear(num_ftrs, 10)  # Modify the last layer to output 10 classes
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    num_ftrs = model.fc.in_features
  

    offline_profiling(0)

 
    


       