
from pytorchfi.core import fault_injection 
from pytorchfi.neuron_error_models import random_neuron_inj
import torch.nn as nn
from collections import namedtuple
from pytorchfi import core
import torch
import os
import copy
import logging
import pandas as pd
from bisect import bisect_left
import pickle
from torch.utils.data import TensorDataset, DataLoader
import csv
from src.utils.helpers import device
from src.models.resnet import resnet18
from src.utils.customFI_methods import random_neuron_single_bit_inj_Aman
from src.utils.customFI_methods import single_bit_flip_func
from scipy.stats import wasserstein_distance
# from offlineProfiling import cohort_size
from pytorchfi.neuron_error_models import (
    random_inj_per_layer,
    random_inj_per_layer_batched,
    random_neuron_inj,
    random_neuron_inj_batched,
    random_neuron_single_bit_inj,
    random_neuron_single_bit_inj_batched,
    random_batch_element,
    random_neuron_location,
    #declare_neuron_fault_injection
)
# os.environ['LOGLEVEL'] = 'DEBUG'  # Adjust logging level to capture DEBUG messages
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)-15s %(levelname)s %(message)s',
#                     handlers=[
#                         logging.StreamHandler(sys.stdout),  # Log to console
#                         logging.FileHandler('logfile.txt')  # Log to file
#                     ])

all_data = []
df = pd.DataFrame()
cohort_size =32
final_dict = {}
def update_histogram(value, histogram):
    sorted_keys = list(histogram.keys())
    first_bin_range = sorted_keys[0]
    last_bin_range = sorted_keys[-1]

    # Handle values less than the first bin
    if value < first_bin_range[0]:
        histogram[first_bin_range] += 1
        return histogram
    # Handle values greater than or equal to the last bin
    elif value >= last_bin_range[1]:
        histogram[last_bin_range] += 1
        return histogram
    # Handle values within the bin ranges
    else:
        for bin_range in sorted_keys:
            if bin_range[0] <= value < bin_range[1]:
                histogram[bin_range] += 1
                return histogram



def reset_histogram(histogram):
    # Set all frequencies to zero, keeping the bin keys
    return {key: 0 for key in histogram}

def normalize_histogram(histogram):
    # Calculate the total count of all bins
    total_count = sum(histogram.values())
    
    # Normalize each bin frequency
    if total_count > 0:  # To avoid division by zero
        return {key: count / total_count for key, count in histogram.items()}
    else:
        return histogram  # If the total count is zero, return the histogram unchanged


def listtohistogram(data,bins= 10):
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
    return histogram_dict

def TAU3processing(tau3, count):
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
def TAU2processing(tau2):
    tau2histtodict = {}
    for layer_name in tau2:
        # print(len(tau2[layer_name])," num of neuron activation in tau1")
        tau2histtodict[layer_name] = listtohistogram(tau2[layer_name])
    return tau2histtodict
def  abnormility_score1(x,histogram_data):
      # Iterate through the bins (key ranges) and their frequencies
    for (min_val, max_val), frequency in histogram_data.items():
        if min_val <= x < max_val:  # Check if x falls within this bin's range
            return (1-frequency)
    # If no bin is found, abnormality score 1
    return 1

def abnormality_score2(histo1 , histo2):
    bins = [(a + b) / 2 for a, b in histo1.keys()]
    weights1 = list(histo1.values())
    weights2 = list(histo2.values())
    total = sum(weights1) * sum(weights2)
    # Calculate Earth Mover's Distance
    emd_value = wasserstein_distance(bins, bins, weights1, weights2)
    # normalize

    return emd_value/total


def abnormality_score3(a,b):
    if a['max_location'] == b['max_location'] or a['min_location'] == b['max_location']:
        if a['max_location'] == b['max_location'] and a['min_location'] == b['max_location']:
            return 1
        return 0.5
    return 0


activations= {}
def get_activation(name):
    def hook(model,input,output):
        activations[name] = output.detach()
    return hook

def test_with_fault(model, tau1, tau2, tau3):
    '''
    tau1 -> profiled DNA of neurons  
    TAU1 -> Layer wise (abnormality score of neurons added )
    tau2 -> layer DNA from profiling 
    TAU2 -> layer DNA during inference -> then abnormality score layer wise 
    tau3 -> profiled min max of layer  (need to take from profiling)
    TAU3 -> min max of a layer  -> the layer wise abnormality score

    '''
    TAU1 = {}
    Temp2 = copy.deepcopy(tau2)
    for layer_name in Temp2:
        Temp2[layer_name] =reset_histogram(Temp2[layer_name])
    TAU3 = {}
    model.eval()
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

    # b, layer, C, H, W, err_val = [0], [0], [0], [0], [0], [1000]
    test_loss = 0
    correct = 0
    total = 0
    count= 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            pfi.reset_current_layer()
            #single_input = inputs[0]  # Extract the first image in the batch
            #outputs = corrupt_model(single_input.unsqueeze(0))

            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            flag = 0
            if predicted.eq(targets).sum().item() == 0:
                flag = 1
            correct += predicted.eq(targets).sum().item()
            count += 1

            TAU2 = copy.deepcopy(Temp2)

            for layer_name in LayerNames:
                TAU1[layer_name] = 0
            
            # TAU1
            for neuron in selected_neurons:
                layer_name,pos = neuron
                #some layers dont have 4 dimenions so this check (eg Fc ->last layer resnet18)
                if len(pos) > 2:       
                    b,c,w,h=pos
                    TAU1[layer_name] += abnormility_score1(activations[layer_name][b,c,w,h].item(),tau1[neuron])
                else :
                    w,h=pos
                    TAU1[layer_name] += abnormility_score1(activations[layer_name][b,c,w,h].item(),tau1[neuron])

            #TAU2
        
            
            # print(Temp2['avgpool'])
            for neuron in selected_neurons:
                layer_name,pos = neuron
            #some layers dont have 4 dimenions so this check (eg Fc ->last layer resnet18)
                if len(pos) > 2:       
                    b,c,w,h=pos
                    TAU2[layer_name] = update_histogram(activations[layer_name][b,c,w,h].item(),TAU2[layer_name])
                else :
                    w,h=pos
                    TAU2[layer_name] = update_histogram(activations[layer_name][w,h].item(),TAU2[layer_name])

            for layer_name in TAU2 :
                TAU2[layer_name] = abnormality_score2(TAU2[layer_name],tau2[layer_name])


            #TAU3
            for layer_name, tensor in activations.items():
                    TAU3[layer_name] = activations[layer_name]
            TAU3 = TAU3processing(TAU3,1)

            for layer_name in TAU3:
                TAU3[layer_name] = abnormality_score3(TAU3[layer_name],tau3[layer_name])

            #total abnormality score
            lambda1 = 1
            lambda2 = 1
            lambda3 = 1
            Total_score= {}
            prev = 0 
            for layer in LayerNames:
                Total_score[layer] = prev + lambda1 * TAU1[layer] +  lambda2 * TAU2[layer] + lambda3 * TAU3[layer]
                prev= Total_score[layer]
            
            # flag denotes only those inputs which are not masked by the model resulting in SDC
            if flag == 1:
                with open('SDCoutput.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([prev])
            
            if count %100 == 0 :
                print(count)
                if count == 10000:
                    break
            # print(Total_score[-1])
            # print("===================================")
            


    print(f'\nTest set: Average loss: {test_loss/len(testloader):.4f}, Accuracy: {correct}/{total} ({100.*correct/total:.2f}%)\n')

if __name__ == '__main__':

    net = resnet18(pretrained=True, progress=True)
    num_ftrs = net.fc.in_features
    #net.fc = nn.Linear(num_ftrs, 10)  # Modify the last layer to output 10 classes
    net = net.to(device)


    batch_size = 1
    H = 32
    W = 32
    C = 3
    bit_pos = 1
    ranges = [9999,9999,9999,9999,9999,9999,999999999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999,9999]
    pfi = single_bit_flip_func(
                net,
                batch_size=batch_size,
                input_shape=[C,H,W],
                use_cuda=True,
                bits=8,
                random_batch=False,
                bit_pos=bit_pos,
            )

    fi_layer=5
    fi_c = 6
    fi_h = 7
    fi_w = 2
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
    net = random_neuron_single_bit_inj_Aman(pfi, ranges, fi_layer, fi_c, fi_h, fi_w,bit_pos = bit_pos)
    selected_neurons = []
    LayerNames = []
    with open('DetectionSites.txt', 'r') as f:
        for line in f:
            layer_name, location = line.strip().split(',',1)
            location = eval(location.strip()) 
            selected_neurons.append((layer_name, location))
            if layer_name not in LayerNames:
                LayerNames += [layer_name]

        with open('tau1.pkl', 'rb') as f:
            tau1 = pickle.load(f)
        with open('tau2.pkl', 'rb') as f:
            tau2 = pickle.load(f)
        with open('tau3.pkl', 'rb') as f:
            tau3  =pickle.load(f)
        T2 = copy.deepcopy(tau2)
        T1 = copy.deepcopy(tau1)
        T3 = copy.deepcopy(tau3)
        for layer_name in T2:
            reset_histogram(T2[layer_name])

        test_with_fault(net,T1,T2,T3)
