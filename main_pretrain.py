import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import random as rd

import torch
import numpy as np
import argparse

from datetime import datetime
from models import *
from utils import *
from data import *
from torch.utils.data import BatchSampler, RandomSampler

isSave = True
torch.use_deterministic_algorithms(True, warn_only=True)
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
parser.add_argument("--epoch", type=int, default=1000, help="Number of training epochs")
parser.add_argument("--lr", type=int, default=0.0001, help="Learning rate")
parser.add_argument("--lr_decay", type=float, default=0.95, help="Learning rate decay factor, lr decays to lr*lr_decay")
parser.add_argument("--lr_decay_steps", type=int, default=100, help="Learning rate decay steps")
parser.add_argument("--seed", type=int, default=0, help="Random seed")

parser.add_argument("--embed_channel", type=int, default=16, help="Embedding size")
parser.add_argument("--embed_size", type=int, default=32, help="Embedding size")
parser.add_argument("--unet_channels", type=str, default="[16, 32, 64, 128]", help="UNet channels")
parser.add_argument("--conv1d_kernel_size", type=int, default=3, help="Conv1d kernel size")
parser.add_argument("--num_sample_steps", type=int, default=1000, help="Number of sample steps")
parser.add_argument("--sampling_steps", type=int, default=10, help="Number of sampling steps")
parser.add_argument("--lambda_rec", type=float, default=1, help="Reconstruction loss weight")

parser.add_argument("--dataset", type=str, default="baby", help="Dataset name")
parser.add_argument("--MR", type=float, default=0.4, help="MR")
parser.add_argument("--complete", type=str, default="zero", help="Complete strategy; Options: mean, zero, mean, random, none, nn")
parser.add_argument("--normalize", type=bool, default=True, help="Normalize data")
parser.add_argument("--reduce_dim", type=bool, default=True, help="Reduce dimensionality")
parser.add_argument("--dim", type=int, default=128, help="Dimensionality")

args = parser.parse_args()

def seed_everything(seed=0):
    rd.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(self, ori_data, incomplete_data, indicator, lr, lr_decay, batch_size,
                 epoch, model, MR, complete, lambda_rec, device=torch.device("cpu"),
                 min_data=None, max_data=None, save_path=None):
        self.ori_data = ori_data
        self.incomplete_data = incomplete_data
        self.indicator = indicator
        self.n_modalities = len(ori_data)
        self.n_sample = ori_data[0].shape[0]
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_steps = args.lr_decay_steps
        self.batch_size = batch_size
        self.epoch = epoch
        self.device = device
        self.model = model.to(device)
        self.lambda_rec = lambda_rec
        if type(min_data) is int:
            self.min_data = [min_data] * self.n_modalities
            self.max_data = [max_data] * self.n_modalities
        else:
            self.min_data = min_data
            self.max_data = max_data
        self.save_path = save_path

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.lr_decay > 0:
            lambda_lr = lambda epoch: self.lr_decay ** (epoch // self.lr_decay_steps)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)
        else:
            self.scheduler = None

        print("Missing rate = {:2f}, complete strategy: {}".format(MR, complete))

    def train(self):
        loss_list = []
        rec_eval_loss_list = []
        diff_loss_list = []
        rec_loss_list = []
        best_eval_loss = np.inf
        next_data = [torch.clone(self.incomplete_data[i]) for i in range(self.n_modalities)]
        for epoch in range(self.epoch):
            batch_loss = []
            batch_diff_loss = []
            batch_rec_loss = []
            batch_idxs = list(BatchSampler(RandomSampler(
                range(self.n_sample)), batch_size=self.batch_size, drop_last=False))
            for i, batch_idx in enumerate(batch_idxs):
                self.model.train()
                self.optimizer.zero_grad()

                batch = [b[batch_idx, :] for b in next_data]
                batch_ind = self.indicator[batch_idx, :]

                diff_loss, rec_loss = self.model(data=batch, indicator=batch_ind, print_progress=False,
                                               min_data=self.min_data, max_data=self.max_data)
                loss = self.lambda_rec * rec_loss + diff_loss
                batch_loss.append(loss.item())
                batch_diff_loss.append(diff_loss.item())
                batch_rec_loss.append(rec_loss.item())

                loss.backward()
                self.optimizer.step()
                print("Epoch: {}, Iter: {}/{}, Loss: {:.4f}, Diffusion Loss: {:.4f}, Reconstruction Loss: {:.4f}"
                      .format(epoch, i+1, len(batch_idxs), loss.item(), diff_loss.item(), rec_loss.item()))

            next_data, rec_train_loss, rec_eval_loss = self.evaluate(next_data=next_data)

            rec_train_loss = np.mean(rec_train_loss)
            rec_eval_loss_list.append(rec_eval_loss)

            mean_loss = np.mean(batch_loss)
            mean_diff_loss = np.mean(batch_diff_loss)
            mean_rec_loss = np.mean(batch_rec_loss)
            loss_list.append(mean_loss)
            diff_loss_list.append(mean_diff_loss)
            rec_loss_list.append(mean_rec_loss)
            print("Epoch: {}, Loss: {:.4f}, Diffusion Loss: {:.4f}, Reconstruction Loss: {:.4f}, Rec Train Loss: {:.4f}, Rec Eval Loss: {:.4f}"
                  .format(epoch, mean_loss, mean_diff_loss, mean_rec_loss, rec_train_loss, rec_eval_loss))
            if self.scheduler is not None:
                self.scheduler.step()

            if rec_eval_loss < best_eval_loss:
                best_eval_loss = rec_eval_loss
                if isSave:
                    torch.save(self.model.state_dict(), self.save_path + 'best_model.pth')
                    print("Best model saved - Epoch: {}, Eval Loss: {:.4f}".format(epoch, best_eval_loss))

        if isSave:
            torch.save(self.model.state_dict(), self.save_path + 'last_model.pth')
            print("Last model saved - Epoch: {}, Eval Loss: {:.4f}".format(epoch, rec_eval_loss))
        pass

    def evaluate(self, next_data):
        self.model.eval()
        with torch.no_grad():
            next_data, rec_train_loss, rec_eval_loss = self.model.complete_train(data=next_data,
                                                                               indicator=self.indicator,
                                                                               target_data=self.ori_data,
                                                                               min_data=self.min_data,
                                                                               max_data=self.max_data)

        return next_data, rec_train_loss.item(), rec_eval_loss.item()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset
    MR = args.MR
    complete = args.complete
    seed = args.seed
    isNormalize = args.normalize
    isReduceDim = args.reduce_dim
    dim = args.dim
    ori_data, incomplete_data, modalities, n_sample, input_size, missing_items, indicator, svd, min_data, max_data = (
        load_multimodal_dataset(dataset=dataset, MR=MR, complete=complete, seed=seed, device=device,
                                normalize=isNormalize, reduced=isReduceDim, dim=dim))
    seed_everything(seed)

    date_s = datetime.now().strftime('%Y%m%d%H%M%S')
    save_path = './checkpoint/' + args.dataset + '/seed-' + str(seed) + '-' + date_s + '/'
    if isSave:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("Save path: " + save_path)

    embed_channel = args.embed_channel
    embed_size = args.embed_size
    conv1d_kernel_size = args.conv1d_kernel_size
    num_sample_steps = args.num_sample_steps
    lambda_rec = args.lambda_rec
    sampling_steps = args.sampling_steps
    Unet_channels = eval(args.unet_channels)

    model = MSDiffusion(modalities=modalities, input_channel=1, input_size=input_size, embed_channel=embed_channel,
                        embed_size=embed_size, conv1d_kernel_size=conv1d_kernel_size, num_sample_steps=num_sample_steps,
                        sampling_steps=sampling_steps, Unet_channels=Unet_channels)

    trainer = Trainer(ori_data=ori_data, incomplete_data=incomplete_data, indicator=indicator, lr=args.lr,
                      lr_decay=args.lr_decay, batch_size=args.batch_size, epoch=args.epoch,
                      model=model, MR=args.MR, complete=args.complete, lambda_rec=lambda_rec, device=device,
                      min_data=0, max_data=1, save_path=save_path)

    trainer.train()



if __name__ == '__main__':
    main(args)