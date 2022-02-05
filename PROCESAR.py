import torch
from dataset import HorseZebraDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
import os
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

        if not os.path.exists("saved_images/"+str(idx)+"/"):
            os.makedirs("saved_images/"+str(idx)+"/")
        save_image(horse*0.5+0.5, f"saved_images/{idx}/horse.png")
        save_image(zebra*0.5+0.5, f"saved_images/{idx}/zebra.png")
        save_image(fake_horse*0.5+0.5, f"saved_images/{idx}/fakehorse.png")
        save_image(fake_zebra*0.5+0.5, f"saved_images/{idx}/fakezebra.png")
        save_image(fake_zebra*0.5+0.5, f"saved_images/{idx}fakezebra.png")
        save_image(cycle_horse*0.5+0.5, f"saved_images/{idx}/cyclehorse.png")
        save_image(cycle_zebra*0.5+0.5, f"saved_images/{idx}/cyclezebra.png")
        #print (D_H_real)
        #print (D_H_fake)
        #print (D_Z_real)
        #print (D_Z_fake)
        #print (cycle_zebra_loss)
        #print (cycle_horse_loss)


        loop.set_postfix(H_real=H_reals/(idx+1), H_fake=H_fakes/(idx+1))



def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE,
        )

    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR+"/NORMALS", root_zebra=config.TRAIN_DIR+"/STREETVIEW", transform=config.transforms
    )
    val_dataset = HorseZebraDataset(
       root_horse="DATA/VAL/NORMALS", root_zebra="DATA/VAL/STREETVIEW", transform=config.transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)

if __name__ == "__main__":
    main()