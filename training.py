import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import datetime
from models import Generator, Discriminator, device
from data import translate

def training_loop(lines, charmap, inv_charmap, dataloader, args):
    # must be better way to do this
    lambda_ = args['lambda_']
    n_critic_iters_per_generator_iter = args['n_critic_iters_per_generator_iter']
    batch_size = args['batch_size']
    lr = args['lr']
    adam_beta1 = args['adam_beta1']
    adam_beta2 = args['adam_beta2']
    iterations = args['iterations']
    continue_training = args['continue_training']
    netG_checkpoint = args['netG_checkpoint']
    netD_checkpoint = args['netD_checkpoint']

    # Start actual training loop

    netG = Generator(charmap).to(device=device)
    netD = Discriminator(charmap).to(device=device)
        
    train = dataloader(lines, batch_size)

    if continue_training: #if continuing from checkpoint
        netG.load_state_dict(torch.load(netG_checkpoint))
        netD.load_state_dict(torch.load(netD_checkpoint))
        start_iter = int(netG_checkpoint.split(":")[0].split("-")[-1][:-2])
        for _ in range(start_iter): #look up better way to do this
            next(train)
            pass
        print(f"Model loaded, starting at {start_iter}...")
    else: 
        start_iter = 1
        
    optimG = optim.Adam(netG.parameters(), lr=lr, betas=(adam_beta1, adam_beta2))
    optimD = optim.Adam(netD.parameters(), lr=lr, betas=(adam_beta1, adam_beta2))


    # start actual training loop
    for iteration in range(start_iter, iterations + 1):
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
            
        for i in range(n_critic_iters_per_generator_iter):
            real_inputs_discrete = next(train)
            real_data = F.one_hot(real_inputs_discrete, num_classes=len(charmap)).float() #x
            latent_variable = torch.randn(batch_size, 128).to(device=device) #z
            alpha = torch.rand(batch_size, 1, 1).to(device=device) #epsilon
            
            fake_data = netG(latent_variable) #x_tilde
            
            interpolates = alpha * real_data + ((1 - alpha) * fake_data) #x_hat
            interpolates = interpolates.clone().detach().requires_grad_(True) #x_hat
            disc_interpolates = netD(interpolates) #D_w(x_hat)
            gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, #grad D_w(x_hat)
                            grad_outputs=torch.ones(disc_interpolates.size()).to(device=device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0] #doesn't populate grad attributes
            
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
            
            disc_real = netD(real_data).mean() #D_w(x)
            disc_fake = netD(fake_data).mean() #D_w(x_tilde)
            
            loss = disc_fake - disc_real + gradient_penalty #L
            loss.backward()
            optimD.step()
            netD.zero_grad()
        
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()
        
        latent_variable = torch.randn(batch_size, 128).to(device=device) #z
        fake_data = netG(latent_variable)    
        G = -netD(fake_data).mean()
        G.backward()
        optimG.step()

        if iteration % 1000 == 0 or iteration == 1:
            print(f"iterations {iteration}")
            real_translation = translate(real_inputs_discrete[:10].cpu().numpy(), inv_charmap)
            fake_translation = translate(fake_data[:10].detach().cpu().numpy().argmax(axis=2), inv_charmap)
            print(f"\tFake: {fake_translation}\n\tReal: {real_translation}")
            time = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=-5))).strftime("%I:%M:%S%p_%m-%d-%y")
            torch.save(netG.state_dict(), f"/home/nvijayakumar/gcp-gan/Checkpoints/netG-{iteration}{time}")
            torch.save(netD.state_dict(), f"/home/nvijayakumar/gcp-gan/Checkpoints/netD-{iteration}{time}")