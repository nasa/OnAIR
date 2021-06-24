import os
from datetime import datetime 
import torch
from src.util.viz import isNotebook
from torch.utils.tensorboard import SummaryWriter

if isNotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def train(vae, loaders, epochs=20, lr=1e-1, checkpoint=False, logging=True, phases=["train", "val"]):
    """
    Training loop util
    :param loaders: {train: train_loader, val: val_loader} data loaders in dictionary
    :param epochs: (optional int) number of epochs to train for, defaults to 20
    :param lr: (optional float) learning rate, defaults to 1e-1
    :param checkpoint: (optional bool) save model to directory, defaults to False
    :param logging: (optional bool) whether to log run accuracy, defaults to True
    :param phases: (string list) phases in training, defaults to ["train", "val"],
                each phase should have a corresponding data loader
    """
    checkpoint_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),"runs")

    if logging:
        e = datetime.now()
        run_dir = os.path.join(checkpoint_dir, "{}-{}-{}_{}:{}:{}".format(e.day, e.month, e.year, e.hour, e.minute, e.second))

        writer = SummaryWriter(run_dir)
        print("Starting training, see run at", run_dir)

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    for epoch_counter in tqdm(range(epochs)):
        for phase in phases:
            if phase == "train":
                vae.train(True)
            else:
                vae.train(False)

            running_loss = 0.0

            for x in loaders[phase]:
                if phase == "train":
                    vae(x)
                    loss = vae.loss()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        vae(x)
                        loss = vae.loss()

                running_loss += loss

            avg_loss = running_loss / len(loaders[phase])
            if logging:
                writer.add_scalar('Loss/' + phase, avg_loss, epoch_counter)

        if checkpoint:
            checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
            torch.save({
                'epoch': epoch_counter,
                'state_dict': vae.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss
            }, os.path.join(run_dir, checkpoint_name))