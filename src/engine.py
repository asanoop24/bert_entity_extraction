import torch
from tqdm import tqdm

def train_fn(dataloader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0

    for data in tqdm(dataloader, total=len(dataloader)):
        for k,v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        _, _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss/len(dataloader)

def eval_fn(dataloader, model, device):
    model.eval()
    final_loss = 0

    for data in tqdm(dataloader, total=len(dataloader)):
        for k,v in data.items():
            print(k,v)
            data[k] = v.to(device)
        _, _, loss = model(**data)
        final_loss += loss.item()
    return final_loss/len(dataloader)