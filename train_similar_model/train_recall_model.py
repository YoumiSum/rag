from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os
import glob

from torch.utils.tensorboard import SummaryWriter


class DSSM(nn.Module):
    def __init__(self, bert_model, t):
        super(DSSM, self).__init__()
        self.bert_model = bert_model
        self.t = t

    def forward(self, x1, x2):
        v1 = self.bert_model(**x1)
        v2 = self.bert_model(**x2)

        similar = torch.cosine_similarity(v1.pooler_output, v2.pooler_output, dim=1)

        y = nn.Sigmoid()(similar/self.t)
        return y


device = torch.device("cuda:0") 
tokenizer = BertTokenizer.from_pretrained(r'C:\Users\Youmi\Desktop\youmi\rag\train_similar_model\models\google-bert\bert-base-chinese')
model = BertModel.from_pretrained(r'C:\Users\Youmi\Desktop\youmi\rag\train_similar_model\models\google-bert\bert-base-chinese')

dssm = DSSM(model, 0.05)
dssm = dssm.to(device)

with open("train_data", encoding="utf-8") as f:
    lines = [eval(s.strip()) for s in f.readlines()]

# 冻结模型参数
print(model)

train_layers = ["pooler.dense"]
# train_layers = ["encoder.layer.11", "pooler.dense"]
# train_layers = ["encoder.layer.10", "encoder.layer.11", "pooler.dense"]

for name, param in model.named_parameters():
    if any([layer in name for layer in train_layers]):
        param.requires_grad = True
    else:
        param.requires_grad = False


batch_size = 160

log_name = len(glob.glob('logs/*'))
writer = SummaryWriter(f'logs/{log_name}')

optimizer = optim.Adam(dssm.parameters(), lr=0.001)
dssm.train()

cnt_s = 0
for epoch in range(0, 100):
    num = len(lines) // batch_size
    random.shuffle(lines)
    for step in range(0, num):
        print(f"{epoch}: {step} / {num}")
        sub_lines = lines[step*batch_size:(step+1)*batch_size]
        X1, X2, Y = zip(*sub_lines)
        X1 = list(X1)
        X2 = list(X2)

        X1 = tokenizer(X1, padding=True, truncation=True, max_length=512, return_tensors='pt')
        X2 = tokenizer(X2, padding=True, truncation=True, max_length=512, return_tensors='pt')
        Y = torch.tensor(Y, dtype=torch.float32)
        X1, X2, Y = X1.to(device), X2.to(device), Y.to(device)
        output = dssm(X1, X2)
        #loss = F.cross_entropy(output, Y)
        loss = nn.BCELoss()(output.view(-1, 1), Y.float().view(-1, 1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 10 == 0:
            print(loss)
            cnt_s += 1
            writer.add_scalar('loss', loss, cnt_s)

            for name, param in model.named_parameters():
                if any([layer in name for layer in train_layers]):
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), cnt_s)

            writer.flush()

            torch.save(model.state_dict(), os.path.join("my_model01", "pytorch_model.bin"))

writer.close()
