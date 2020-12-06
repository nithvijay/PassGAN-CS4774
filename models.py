import torch
import torch.nn as nn
import torch.nn.functional as F

import datetime

from data import translate


device = "cuda" if torch.cuda.is_available() else "cpu"

class ResidualBlock(nn.Module):
    """
    Residual blocks take BATCH_SIZE x CHANNELS x LENGTH -> BATCH_SIZE x CHANNELS x LENGTH
    """
    def __init__(self, n_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        
    def forward(self, inputs):
        x = F.relu(inputs)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        outputs = inputs + x * 0.3
        return outputs
    
class Generator(nn.Module):
    def __init__(self, charmap, kernel_size=3):
        super(Generator, self).__init__()
        self.lin = nn.Linear(in_features=128, out_features=128*10) #Channels x Length
        self.block1 = ResidualBlock(128)
        self.block2 = ResidualBlock(128)
        self.block3 = ResidualBlock(128)
        self.block4 = ResidualBlock(128)
        self.block5 = ResidualBlock(128)
        self.conv = nn.Conv1d(in_channels=128, out_channels=len(charmap), kernel_size=kernel_size, padding=kernel_size//2)
        
    
    def forward(self, inputs):
        x = self.lin(inputs).reshape(-1, 128, 10) # for residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.conv(x).permute(0, 2, 1)
        x = F.softmax(x, dim=2)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, charmap, kernel_size=3):
        super(Discriminator, self).__init__()
        self.length_charmap = len(charmap)
        self.conv1 = nn.Conv1d(self.length_charmap, 128, kernel_size=kernel_size, padding=kernel_size // 2)
        self.block1 = ResidualBlock(128)
        self.block2 = ResidualBlock(128)
        self.block3 = ResidualBlock(128)
        self.block4 = ResidualBlock(128)
        self.block5 = ResidualBlock(128)
        self.flatten = nn.Flatten()
        self.lin = nn.Linear(in_features=128 * 10, out_features=1)
    
    def forward(self, inputs): #one-hot is input to the discriminator
#        x = F.one_hot(inputs, num_classes=self.length_charmap).permute(0, 2, 1).float()
        x = inputs.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)
        outputs = self.lin(x)
        return outputs

def predict_many(netGs, inv_charmap, num_samples=5):
    latent_noise = torch.randn(num_samples, 128).to(device=device)
    return [translate(netG(latent_noise).argmax(dim=2), inv_charmap) for netG in netGs]

def predict_one(netG, inv_charmap, num_samples):
    latent_noise = torch.randn(num_samples, 128).to(device=device)
    return translate(netG(latent_noise).argmax(dim=2), inv_charmap)

def write_predictions(netG, inv_charmap, batches_per_file, num_files, batch_size=100000):
    print(f"Generating {batch_size * batches_per_file * num_files} passwords...")
    t_total = datetime.datetime.now()
    for i in range(1, num_files+1):
        t = datetime.datetime.now()
        print(f"File {i}")
        print("\tGenerating output...")
        preds = []
        for _ in range(batches_per_file):
            preds.append(netG(torch.randn(batch_size, 128).to(device=device)).argmax(dim=2)) #max 100k fits in memory
        list_of_preds = torch.stack(preds).reshape((-1, 10)).cpu().tolist()
        del preds
        print("\tTranslating output...")
        translated_preds = translate(list_of_preds, inv_charmap)

        del list_of_preds
        print("\tWriting output...")
        with open(f"Predictions/predfile_{i}_{batch_size*batches_per_file}.txt", 'w+') as f:
            for pred in translated_preds:
                f.write(pred + "\n")
        print(f"\t{datetime.datetime.now() - t}")
        del translated_preds
    print(f"\tTotal: {datetime.datetime.now() - t_total}")