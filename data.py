import collections

import numpy as np
import torch
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset(path, max_vocab_size=2048):
    with open(path, 'r') as f:
        lines = [line for line in f]
    np.random.seed(42)
    np.random.shuffle(lines)

    counts = collections.Counter(char for line in lines for char in line if char != "\n")

    charmap = {'unk':0}
    inv_charmap = ['unk']

    for char, count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(charmap[char])
            else:
                filtered_line.append('unk')
        filtered_lines.append(filtered_line[:-1])

    print("loaded {} lines in dataset".format(len(lines)))
    return filtered_lines, charmap, inv_charmap

def split(lines, test_size = 0.1): #data is already shuffled
    split = int(len(lines) * test_size)
    train_lines = lines[split:len(lines)]
    test_lines = lines[0:split]
    return train_lines, test_lines

def dataloader(lines, batch_size):
    while True:
        np.random.shuffle(lines)
        for i in range(len(lines) // batch_size):
            yield torch.tensor(lines[i*batch_size:(i+1)*batch_size]).to(device=device)
    
def translate(passwords, inv_charmap):
    return ["".join([inv_charmap[c] for c in password]) for password in passwords]

def dump_txt_to_pickle(path, dataset_name, test_size=0.1):
    filtered_lines, charmap, inv_charmap = load_dataset(path) #processed path
    train_lines, test_lines = split(filtered_lines, test_size=test_size)
    pickle.dump(train_lines, open(f"Data/{dataset_name}_train.pickle", 'wb'))
    pickle.dump(test_lines, open(f"Data/{dataset_name}_test.pickle", 'wb'))
    pickle.dump(charmap, open(f"Data/{dataset_name}_charmap.pickle", 'wb'))
    pickle.dump(inv_charmap, open(f"Data/{dataset_name}_inv_charmap.pickle", 'wb'))

def load_data_from_pickle(dataset_name, train_data=True, test_data=False):
    """
    Return order: train_data, test_data, charmap, inv_charmap
    """
    return_values = []
    if train_data:
        train_lines = pickle.load(open(f"Data/{dataset_name}_train.pickle", 'rb'))
        return_values.append(train_lines)
    if test_data:
        test_lines = pickle.load(open(f"Data/{dataset_name}_test.pickle", 'rb'))
        return_values.append(test_lines)
    
    charmap = pickle.load(open(f"Data/{dataset_name}_charmap.pickle", "rb"))
    inv_charmap = pickle.load(open(f"Data/{dataset_name}_inv_charmap.pickle", "rb"))
    
    return_values.append(charmap)
    return_values.append(inv_charmap)

    return return_values