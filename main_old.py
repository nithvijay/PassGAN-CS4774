import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.get_device_name(), "|", torch.cuda.is_available())

def load_dataset(max_vocab_size=2048):
    path = "Data/rockyou_processed.txt"
    with open(path, 'r') as f:
        lines = [line for line in f]

    np.random.shuffle(lines)

    import collections
    counts = collections.Counter(char for line in lines for char in line if char != "\n")

    charmap = {'unk':0}
    inv_charmap = ['unk']

    for char,count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    print("loaded {} lines in dataset".format(len(lines)))
    return filtered_lines, charmap, inv_charmap

filtered_lines, charmap, inv_charmap = load_dataset()

def dataloader(lines, batch_size):
    for i in range(len(filtered_lines) // batch_size):
        yield torch.tensor([[charmap[c] for c in l[:-1]] for l in lines[i:i+batch_size]]).to(device=device)

def translate(passwords):
    return ["".join([inv_charmap[c] for c in password]) for password in passwords]

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

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand(real_data.size()).to(device=device)

    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).to(device=device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    # TODO: Make ConvBackward diffentiable
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

lambda_ = 10
LAMBDA = 10
n_critic_iters_per_generator_iter = 10
batch_size = 128
lr = 1e-4
adam_beta1 = 0.5
adam_beta2 = 0.9
iterations = 90000

one = one = torch.tensor(1, dtype=torch.float).to(device=device)
mone = -1 * one

netG = Generator(charmap).to(device=device)
netD = Discriminator(charmap).to(device=device)

optimG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.9))
optimD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.9))
    
train = dataloader(filtered_lines, batch_size)

for iteration in range(1, iterations):
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
        
        
    for i in range(n_critic_iters_per_generator_iter):
        real_inputs_discrete = next(train)
        real_data = F.one_hot(real_inputs_discrete, num_classes=len(charmap)).float()
        real_data_v = autograd.Variable(real_data)
        
        netD.zero_grad()
        
        D_real = netD(real_data_v)
        D_real = D_real.mean()
        # print D_real
        # TODO: Waiting for the bug fix from pytorch
        D_real.backward(mone)
        
        noise = torch.randn(batch_size, 128).to(device=device)
        with torch.no_grad():
            noisev = autograd.Variable(noise)  
        fake = autograd.Variable(netG(noisev).data)
        inputv = fake
        D_fake = netD(inputv)
        D_fake = D_fake.mean()
        # TODO: Waiting for the bug fix from pytorch
        D_fake.backward(one)

        
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        gradient_penalty.backward()
        
        optimD.step()
        netD.zero_grad()
    
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()

    noise = torch.randn(batch_size, 128).to(device=device)
    noisev = autograd.Variable(noise)
    fake = netG(noisev)
    G = netD(fake)
    G = G.mean()
    G.backward(mone)
    G_cost = -G
    optimG.step()

    if iteration % 1000 == 0 or iteration == 1:
        print(f"iterations {iteration}")
        real_translation = translate(real_inputs_discrete[:5].cpu().numpy())
        fake_translation = translate(fake[:5].detach().cpu().numpy().argmax(axis=2))
        print(f"\tFake: {fake_translation}\n\tReal: {real_translation}")
        time = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=-5))).strftime("%I:%M:%S%p_%m-%d-%y")
        torch.save(netG.state_dict(), f"/home/nvijayakumar/gcp-gan/Checkpoints/netG{iteration}{time}")
        torch.save(netD.state_dict(), f"/home/nvijayakumar/gcp-gan/Checkpoints/netD{iteration}{time}")