import datetime
from data import dataloader, dump_txt_to_pickle, load_data_from_pickle
from training import training_loop
from models import device

def main():
    print("Loading data from pickle...")
    t = datetime.datetime.now()
    train_lines, charmap, inv_charmap = load_data_from_pickle("dubsmash")
    print("Finished in:", datetime.datetime.now() - t)


    args = {}
    args['lambda_'] = 10
    args['n_critic_iters_per_generator_iter'] = 10
    args['batch_size'] = 128
    args['lr'] = 1e-4
    args['adam_beta1'] = 0.5
    args['adam_beta2'] = 0.9
    args['iterations'] = 199000
    args['continue_training'] = True
    args['netG_checkpoint'] = "Checkpoints/netG-10300012:39:32AM_12-05-20"
    args['netD_checkpoint'] = "Checkpoints/netD-10300012:39:32AM_12-05-20"

    training_loop(train_lines, charmap, inv_charmap, dataloader, args)

if __name__ == "__main__":
    # print("Loading data for the first time...")
    # t = datetime.datetime.now()
    # dump_txt_to_pickle("Data/dubsmash_processed.txt", "dubsmash", test_size=0.05)
    # print("Finished in:", datetime.datetime.now() - t)
    
    print(f"Using {device}")
    main()
    pass