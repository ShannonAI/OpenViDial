import config
import torch
from tqdm import tqdm
import sys
import os
from torch.utils.data import DataLoader
from data import FeatureDataset
from utils import get_batch, padding
from model import MMI
import toch.nn as nn
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def test():
    print(f"load test data from [{config.data_dir}]", file=sys.stderr)
    test_data = FeatureDataset(config.data_dir, split='test')
    test_data_loader = DataLoader(dataset=test_data, batch_size=config.test_batch_size, shuffle=False, collate_fn=get_batch)
    model = MMI.load(config.model_path)
    if (config.cuda):
        model = model.to(torch.device("cuda:0"))
    sum_loss = 0
    with torch.no_grad():
        max_iter = int(math.ceil(len(test_data)/config.test_batch_size))
        with tqdm(total=max_iter, desc="test") as pbar:
            for batch_text, text_len, batch_image, batch_image_mask in test_data_loader:
                batch_size = len(batch_text)
                batch_text = padding(batch_text, model.vocab.word2id['<pad>'])
                loss = model(batch_text, text_len, batch_image, batch_image_mask)
                target = torch.ones(batch_size, dtype=torch.float, device=model.device)
                loss = nn.functional.mse_loss(loss, target, reduction='mean')
                pbar.set_postfix({"avg_loss": '{%.3f}' % (loss.item())})
                pbar.update(1)
                sum_loss += loss
        print(f"loss of test : {sum_loss.item()}")

def main():
    test()

if __name__ == '__main__':
    main()