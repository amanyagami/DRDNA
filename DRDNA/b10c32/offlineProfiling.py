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
import torch
from torch.utils.data import TensorDataset, DataLoader
from config import Lambda2,Lambda1,Lambda3,bins_num,cohortSize
activations= {}
saare_activations= {}
def get_activation(name):
    def hook(model,input,output):
        activations[name] = (output.detach())
    return hook

def listtohistogram(data,flag,bins= 10):
        # Calculate bin edges for 10 bins
    min_data = min(data)
    max_data = max(data)
    range_data = max_data - min_data
    bin_width = range_data / bins
    bin_edges = [min_data + i * bin_width for i in range(bins+1)]

    # Function to determine bin index
    def find_bin(value, edges):
        return bisect_left(edges, value) - 1

    # Count frequencies per bin
    bin_count = [0] * 10
    for value in data:
        bin_index = find_bin(value, bin_edges)
        bin_count[bin_index] += 1

    # Store in dictionary with bin ranges as keys
    histogram_dict = {
        (bin_edges[i] , bin_edges[i+1]): bin_count[i]
        for i in range(len(bin_edges) - 1)
    }
    if flag == 1 :
        return normalize_histogram(histogram_dict)
    return histogram_dict

def normalize_histogram(histogram):
    # Calculate the total count of all bins
    total_count = sum(histogram.values())
    
    # Normalize each bin frequency
    if total_count > 0:  # To avoid division by zero
        return {key: count / total_count for key, count in histogram.items()}
    else:
        return histogram  # If the total count is zero, return the histogram unchanged



'''
tau processing here

'''
def tau1processing(tau1):
    tau1histtodict = {} 
    for neuron in tau1:
        # print(len(tau1[neuron])," num of neuron activation in tau1")
        tau1histtodict[neuron] = listtohistogram(tau1[neuron],1,bins_num)
    return tau1histtodict
def tau2processing(tau2):
    tau2histtodict = {}
    for layer_name in tau2:
        # print(len(tau2[layer_name])," num of neuron activation in tau1")
        tau2histtodict[layer_name] = listtohistogram(tau2[layer_name],2,bins_num)
    return tau2histtodict
def tau3processing(tau3, count):
    tau3_activation_extremes = {}
    for layer_name in tau3:
        tau3[layer_name] /= count
        tensor= tau3[layer_name]
        flat_tensor = tensor.flatten()
        max_value, max_index = flat_tensor.max(0)
        min_value, min_index = flat_tensor.min(0)
        
        # Convert indices to tensor before using unravel_index
        max_index_tensor = torch.tensor(max_index.item())
        min_index_tensor = torch.tensor(min_index.item())
        
        # Get the index in the original tensor dimensions
        max_location = torch.unravel_index(max_index_tensor, tensor.shape)
        min_location = torch.unravel_index(min_index_tensor, tensor.shape)
        
        # Store the results in the output dictionary
        tau3_activation_extremes[layer_name] = {
            'max_value': max_value.item(),
            'max_location': max_location,
            'min_value': min_value.item(),
            'min_location': min_location
        }
    return tau3_activation_extremes


layer_names = [
    'conv1',
    'bn1',
    'layer1.0.conv1',
    'layer1.0.bn1',
    'layer1.0.conv2',
    'layer1.0.bn2',
    'layer2.0.conv1',
    'layer2.0.bn1',
    'layer2.0.conv2',
    'layer2.0.bn2',
    'layer3.0.conv1',
    'layer3.0.bn1',
    'layer3.0.conv2',
    'layer3.0.bn2',
    'layer4.0.conv1',
    'layer4.0.bn1',
    'layer4.0.conv2',
    'layer4.0.bn2',
    'avgpool',
    'fc'
]

# output dimension [ [] , []] -> out_dim (location of detection sites)
#cohort size = number of detection sites

layer_output_dims = {
    'conv1': [1, 64, 32, 32],
    'bn1': [1, 64, 32, 32],
    'layer1.0.conv1': [1, 64, 16, 16],
    'layer1.0.bn1': [1, 64, 16, 16],
    'layer1.0.conv2': [1, 64, 16, 16],
    'layer1.0.bn2': [1, 64, 16, 16],
    'layer2.0.conv1': [1, 128, 8, 8],
    'layer2.0.bn1': [1, 128, 8, 8],
    'layer2.0.conv2': [1, 128, 8, 8],
    'layer2.0.bn2': [1, 128, 8, 8],
    'layer3.0.conv1': [1, 256, 4, 4],
    'layer3.0.bn1': [1, 256, 4, 4],
    'layer3.0.conv2': [1, 256, 4, 4],
    'layer3.0.bn2': [1, 256, 4, 4],
    'layer4.0.conv1': [1, 512, 2, 2],
    'layer4.0.bn1': [1, 512, 2, 2],
    'layer4.0.conv2': [1, 512, 2, 2],
    'layer4.0.bn2': [1, 512, 2, 2],
    'avgpool': [1, 512, 1, 1],
    'fc': [1, 10]  # Assuming a final fully connected layer with 1000 outputs
}
def chooseRandomNeurons(layer_output_dims, cohort_size):
    final_list = []
    layer_names = layer_output_dims.keys()
    layer_names = list(layer_names)    

    # Function to generate random neuron within a layer
    def generate_random_neuron(layer_name, dimensions):
        if layer_name == 'fc':
            height,width = dimensions
            h = random.randint(0, height - 1)
            w = random.randint(0, width - 1)
            return (layer_name, (h, w))
        else:
            _, channels, height, width = dimensions
            channel = random.randint(0, channels - 1)
            h = random.randint(0, height - 1)
            w = random.randint(0, width - 1)
            return (layer_name, (0, channel, h, w))
    
    # Step: Choose k (cohort Size) neuron from each layer
    for layer_name in layer_names:
        # print(layer_name)
        dimensions = layer_output_dims[layer_name]
        layer_count=0
        counti = 0
        # print(len(final_list)," number of neurons " )
        if layer_name == 'fc': continue # cant have mpre than 10 neuron in last layer  for Resnet for cifar 10.
        while counti < cohort_size :
            neuron = generate_random_neuron(layer_name, dimensions)
            if neuron not in final_list:
                final_list.append(neuron)
                counti += 1
        # print("Done finding neurons")
    # print(final_list)
    return final_list

      

def offline_profiling(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    count = 0
    # it = 0
    tau1 = {}
    tau2= {}
    tau3 = {}
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            for neuron in selected_neurons:
                layer_name,pos = neuron
                #some layers dont have 4 dimenions so this check (eg Fc ->last layer resnet18)
                if len(pos) > 2:       
                    b,c,w,h=pos
                    if neuron in tau1:
                        tau1[neuron].append(activations[layer_name][b,c,w,h].item())
                    else:
                        tau1[neuron]= []
                        tau1[neuron].append(activations[layer_name][b,c,w,h].item())
                else :
                    w,h=pos
                    if neuron in tau1:
                        tau1[neuron].append(activations[layer_name][w,h].item())
                    else:
                        tau1[neuron]= []
                        tau1[neuron].append(activations[layer_name][w,h].item())  
            # print( " tau1 fillng done ")
            # Tau 2
            
         
            for neuron in selected_neurons:
                layer_name,pos = neuron
            #some layers dont have 4 dimenions so this check (eg Fc ->last layer resnet18)
                if len(pos) > 2:       
                    b,c,w,h=pos
                    if layer_name in tau2:
                        tau2[layer_name].append(activations[layer_name][b,c,w,h].item())
                    else:
                        tau2[layer_name]= []
                        tau2[layer_name].append(activations[layer_name][b,c,w,h].item())
                else :
                    w,h=pos
                    if layer_name in tau2:
                        tau2[layer_name].append(activations[layer_name][w,h].item())
                    else:
                        tau2[layer_name]= []
                        tau2[layer_name].append(activations[layer_name][w,h].item()) 
            count += 1
            # print( " tau2 fillng done ")
            #tau3
            for layer_name, tensor in activations.items():
                
                if layer_name in tau3:
                    tau3[layer_name] += activations[layer_name]
                else:
                    tau3[layer_name] = activations[layer_name]
            # print( " tau3 fillng done ")
            if count % 50 ==0:
                print(count)
            if count == 1500:
                break
                 
        tau1 = tau1processing(tau1)
        print("Tau 1 complete")
        tau2 = tau2processing(tau2)   
        print("Tau 2 complete")
        tau3 = tau3processing(tau3,count)        
        print("Tau 3 complete")       
        print(f'\nTest set: Average loss: {test_loss/len(testloader):.4f}, Accuracy: {correct}/{total} ({100.*correct/total:.2f}%)\n')    
        return tau1, tau2, tau3
    




   
if __name__ == '__main__':

    model = resnet18(pretrained=True, progress=True,device=device)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])


    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    # Load the saved tensors
    data = torch.load('correct_data.pt')
    correct_inputs = data['inputs']
    correct_targets = data['targets']

    # Recreate the TensorDataset
    new_test_dataset = TensorDataset(correct_inputs, correct_targets)

    # Create a DataLoader
    testloader = DataLoader(
        new_test_dataset,
        batch_size=1,  # Use the batch size you need
        shuffle=False,
        num_workers=0  # Adjust based on your environment
    )


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
    cohort_size = cohortSize

    selected_neurons = chooseRandomNeurons(layer_output_dims, cohort_size)
    with open('/home/local/ASUAD/asing651/ResnetCifar10pytorchFI/DRDNA/b10c32/DetectionSites.txt', 'w') as f:
        for neuron in selected_neurons:
            layer_name, location = neuron
            f.write(f"{layer_name},{location}\n")  
    tau1 , tau2, tau3 =  offline_profiling(0)

    with open('../ResnetCifar10pytorchFI/DRDNA/b10c32/tau1.pkl', 'wb') as f:
        pickle.dump(tau1 ,f)
    with open('../ResnetCifar10pytorchFI/DRDNA/b10c32/tau2.pkl', 'wb') as f:
        pickle.dump(tau2 ,f)
    with open('../ResnetCifar10pytorchFI/DRDNA/b10c32/tau3.pkl', 'wb') as f:
        pickle.dump(tau3 ,f)
    with open('../activations.pkl', 'wb') as f:
        print("hello")
        pickle.dump(saare_activations ,f)
    
    

    

   