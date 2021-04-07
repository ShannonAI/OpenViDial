import torch
import torch.nn as nn
import math
import config

class MMI(nn.Module):

    def __init__(self, vocab, device):
        super(MMI, self).__init__()
        self.device = device
        self.vocab = vocab
        self.embeddings = nn.Embedding(num_embeddings=len(vocab), embedding_dim=config.d_model, padding_idx=self.vocab['<pad>'])
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.nhead, dim_feedforward=config.dim_feedforward, dropout=config.dropout)
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=config.layer, norm=self.encoder_norm)
        self.final = nn.Linear(in_features=config.d_model, out_features=config.feature_dim, bias=True)
        self.Dropout = nn.Dropout(config.dropout)
    
    def forward(self, src_text, text_len, src_image):
        text_tensor = torch.tensor(src_text, dtype=torch.long, device=self.device).t()
        image_tensor = torch.tensor(src_image, dtype=torch.float, device=self.device)
        text_feature, text_mask = self.encode(text_tensor)
        text_feature = self.final(text_feature)
        text_feature = text_feature.permute(1, 0, 2) # sen_len * batch_size * feature -> batch_size * sen_len * feature
        image_tensor = torch.unsqueeze(image_tensor, dim=-1)
        output = torch.nn.functional.sigmoid(torch.matmul(text_feature, image_tensor).squeeze(dim=-1)) * text_mask # batch_size * sen_len
        len_tensor = torch.tensor(text_len, dtype=torch.float, device=self.device)
        return output.sum(dim=-1)/len_tensor
    
    def encode(self, src_tensor):
        S = src_tensor.shape[0]
        N = src_tensor.shape[1]
        padding_mask = (src_tensor == self.vocab['<pad>']).bool().t().to(self.device)
        embed_tensor = self.Dropout(self.embeddings(src_tensor).to(self.device)+self.get_position(S, N))
        output = self.encoder(embed_tensor, src_key_padding_mask=padding_mask)
        return output, padding_mask  # sen_len * batch_size * feature_size, batch_size * sen_len

    def get_position(self, sen_len, batch_size):
        pre_PE = []
        for i in range(sen_len):
            shuhe = []
            for j in range(config.d_model):
                if (j % 2 == 0):
                    shuhe.append(math.sin(i/math.pow(10000, j/config.d_model)))
                else:
                    shuhe.append(math.cos(i/math.pow(10000, (j-1)/config.d_model)))
            pre_PE.append(shuhe)
        pre_PE = torch.tensor(pre_PE, dtype=torch.float, device=self.device)
        pre_PE = pre_PE.reshape(pre_PE.shape[0], 1, pre_PE.shape[1])
        pre_PE = pre_PE.expand(pre_PE.shape[0], batch_size, pre_PE.shape[2])
        return pre_PE
    
    def save(self, model_path):
        params = {
            'vocab': self.vocab,
            'device': self.device,
            'state_dict': self.state_dict()
        }
        torch.save(params, model_path)
    
    @staticmethod
    def load(model_path):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = MMI(params['vocab'], params['device'])
        model.load_state_dict(params['state_dict'])
        return model