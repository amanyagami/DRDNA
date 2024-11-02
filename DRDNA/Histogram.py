from src.models.resnet import resnet18
from src.utils.helpers import device
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
activations= {}
def get_activation(name):
    def hook(model,input,output):
        activations[name] = output.detach()
    return hook


# Testing function
def test_without_fi(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            single_input = inputs[0]  # Extract the first image in the batch
            outputs = model(single_input.unsqueeze(0))  
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            break
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
    # Register hooks to all layers, including convolutional, batch normalization, and fully connected layers
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.bn1.register_forward_hook(get_activation('bn1'))
    model.layer1[0].conv1.register_forward_hook(get_activation('layer1.0.conv1'))
    model.layer1[0].bn1.register_forward_hook(get_activation('layer1.0.bn1'))
    model.layer1[0].conv2.register_forward_hook(get_activation('layer1.0.conv2'))
    model.layer1[0].bn2.register_forward_hook(get_activation('layer1.0.bn2'))
    model.layer2[0].conv1.register_forward_hook(get_activation('layer2.0.conv1'))
    model.layer2[0].bn1.register_forward_hook(get_activation('layer2.0.bn1'))
    model.layer2[0].conv2.register_forward_hook(get_activation('layer2.0.conv2'))
    model.layer2[0].bn2.register_forward_hook(get_activation('layer2.0.bn2'))
    model.layer3[0].conv1.register_forward_hook(get_activation('layer3.0.conv1'))
    model.layer3[0].bn1.register_forward_hook(get_activation('layer3.0.bn1'))
    model.layer3[0].conv2.register_forward_hook(get_activation('layer3.0.conv2'))
    model.layer3[0].bn2.register_forward_hook(get_activation('layer3.0.bn2'))
    model.layer4[0].conv1.register_forward_hook(get_activation('layer4.0.conv1'))
    model.layer4[0].bn1.register_forward_hook(get_activation('layer4.0.bn1'))
    model.layer4[0].conv2.register_forward_hook(get_activation('layer4.0.conv2'))
    model.layer4[0].bn2.register_forward_hook(get_activation('layer4.0.bn2'))
    model.avgpool.register_forward_hook(get_activation('avgpool'))
    model.fc.register_forward_hook(get_activation('fc'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


