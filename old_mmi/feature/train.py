import config
import torch
import torch.nn as nn
from model import MMI
import math
from tqdm import tqdm
import sys
import os
from optim import Optim
from data import FeatureDataset
from torch.utils.data import DataLoader
from vocab import Vocab
from utils import get_batch, padding

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def evaluate(model, dev_data, dev_loader):
    flag = model.training
    model.eval()
    sum_loss = 0
    with torch.no_grad():
        max_iter = int(math.ceil(len(dev_data)/config.dev_batch_size))
        with tqdm(total=max_iter, desc="validation") as pbar:
            for batch_text, text_len, batch_image in dev_loader:
                batch_size = len(batch_text)
                batch_text = padding(batch_text, model.vocab.word2id['<pad>'])
                loss = model(batch_text, text_len, batch_image)
                target = torch.ones(batch_size, dtype=torch.float, device=model.device)
                loss = nn.functional.mse_loss(loss, target, reduction='mean')
                pbar.set_postfix({"avg_loss": '{%.3f}' % (loss.item())})
                pbar.update(1)
                sum_loss += loss
    if (flag):
        model.train()
    return sum_loss.item()

def train():
    torch.manual_seed(1)
    if (config.cuda):
        torch.cuda.manual_seed(1)
    
    vocab = Vocab(config.dict_path)
    train_data = FeatureDataset(config.data_dir, split='train')
    dev_data = FeatureDataset(config.data_dir, split='valid')
    train_loader = DataLoader(dataset=train_data, batch_size=config.train_batch_size, shuffle=True, collate_fn=get_batch)
    dev_loader = DataLoader(dataset=dev_data, batch_size=config.dev_batch_size, shuffle=True, collate_fn=get_batch)
    device = torch.device("cuda:0" if config.cuda else "cpu")
    
    model = MMI(vocab, device)
    model = model.to(device)
    model.train()

    optimizer = Optim(torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9), config.feature_dim, config.wram_up)

    epoch = 0
    history_vlaid = []
    print("begin training!", file=sys.stderr)
    while (True):
        epoch += 1
        max_iter = int(math.ceil(len(train_data)/config.train_batch_size))
        with tqdm(total=max_iter, desc="train") as pbar:
            for batch_text, text_len, batch_image in train_loader:
                batch_text = padding(batch_text, vocab.word2id['<pad>'])
                optimizer.zero_grad()
                batch_size = len(batch_text)
                loss = model(batch_text, text_len, batch_image)
                target = torch.ones(batch_size, dtype=torch.float, device=device)
                loss = nn.functional.mse_loss(loss, target, reduction='mean')
                loss.backward()
                optimizer.step_and_updata_lr()
                pbar.set_postfix({"epoch": epoch, "avg_loss": '{%.3f}' % (loss.item())})
                pbar.update(1)
        if (epoch % config.valid_iter == 0):
            print("now begin validation ...", file=sys.stderr)
            eval_loss = evaluate(model, dev_data, dev_loader)
            print(eval_loss)
            flag = len(history_vlaid) == 0 or eval_loss < min(history_vlaid)
            if (flag):
                print(f"current model is the best! save to [{config.save_path}]", file=sys.stderr)
                history_vlaid.append(eval_loss)
                model.save(os.path.join(config.save_path, f"{epoch}_{eval_loss}_checkpoint.pth"))
                torch.save(optimizer.optimizer.state_dict(), os.path.join(config.save_path, f"{epoch}_{eval_loss}_optimizer.optim"))
            if (epoch == config.max_epoch):
                print("reach the maximum number of epochs!", file=sys.stderr)
                return

def main():
    train()

if __name__ == '__main__':
    main()