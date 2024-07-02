import glob

from tensorboardX import SummaryWriter
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os

device = torch.device("cuda:0") 
tokenizer = BertTokenizer.from_pretrained(r'C:\Users\Youmi\Desktop\youmi\rag\train_similar_model\models\google-bert\bert-base-chinese')
model = BertForSequenceClassification.from_pretrained(r'C:\Users\Youmi\Desktop\youmi\rag\train_similar_model\models\google-bert\bert-base-chinese')
model = model.to(device)

with open("train_data", encoding="utf-8") as f:
    lines = [eval(s.strip()) for s in f.readlines()]

train_layers = ["pooler.dense"]
# train_layers = ["encoder.layer.11", "pooler.dense"]
# train_layers = ["encoder.layer.10", "encoder.layer.11", "pooler.dense"]

for name, param in model.named_parameters():
    if any([layer in name for layer in train_layers]):
        param.requires_grad = True
    else:
        param.requires_grad = False


def convert_Y(y):
    s = [0, 0]
    s[y] = 1
    return s


optimizer = optim.Adam(model.parameters(), lr=0.001)
batch_size = 2
model.train()

log_name = len(glob.glob('logs/*'))
writer = SummaryWriter(f'logs/{log_name}')

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
        X = [[s1[0:200], s2[0:200]] for s1, s2 in zip(X1, X2)]
        Y = [convert_Y(s) for s in Y]
        X = tokenizer(X, padding=True, truncation=True, max_length=512,return_tensors='pt')
        Y = torch.tensor(Y, dtype=torch.float32)
        X, Y = X.to(device), Y.to(device)
        output = model(**X).logits
        loss = nn.BCEWithLogitsLoss()(output, Y)
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

            torch.save(model.state_dict(), os.path.join("rank_model", "pytorch_model.bin"))
 
writer.close()
