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

def dataloader(lines, batch_size): #refactor into subclass of DataLoader and Dataset?
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

def read_write_lines_hashes(file_name, out_file_name, max_length=10):
    """
    For use with password files from Hashes.org. This function pads passwords with the '|' character to a specified length and removes the passwords greater than the specified length.
    """
    passwords = []
    with open(file_name, 'r', errors='ignore') as f:
        for line in f:
            password = line.strip().split(':')[1]
            if len(password) <= max_length:
                passwords.append(password.ljust(max_length, '|'))

    with open(out_file_name, "w") as f:
        for password in passwords:
            f.write(password + "\n")


def read_write_lines_other(file_name: str, out_file_name: str, max_length=10):
    """
    Pads passwords with the '|' character to a specified length. Removes the passwords greater than the specified length.
    """
    passwords = []
    with open(file_name, 'r', errors='ignore') as f:
        for line in f:
            password = line.strip()
            if len(password) <= max_length:
                passwords.append(password.ljust(max_length, '|'))

    with open(out_file_name, "w") as f:
        for password in passwords:
            f.write(password + "\n")
         
        
# command line arguments from previous version
# if __name__ == "__main__":
#     if len(sys.argv) != 4 or sys.argv[3] not in ["hashes", "other"]:
#         print("usage: python3 data.py <input_file> <output_file> hashes")

#     input_file = sys.argv[1]
#     output_file = sys.argv[2]
#     source = sys.argv[3]

#     t = time()
#     if source == "hashes":
#         read_write_lines_hashes(input_file, output_file, max_length=10)
#     else:
#         read_write_lines_other(input_file, output_file, max_length=10)
#     print(f"{time() - t:.2f} seconds")
