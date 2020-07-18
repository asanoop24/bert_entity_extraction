import pandas
import numpy
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
from dataset import EntityDataset
import engine
from model import EntityModel

import joblib


def process_data(data_path):
    # './../input/ner_dataset.csv'
    df = pandas.read_csv(data_path, encoding='latin-1')
    df.loc[:, 'Sentence #'] = df['Sentence #'].fillna(method='ffill')

    enc_pos = LabelEncoder()
    enc_tag = LabelEncoder()

    df.loc[:, 'POS'] = enc_pos.fit_transform(df['POS'])
    df.loc[:, 'Tag'] = enc_tag.fit_transform(df['Tag'])

    text_list = df.groupby('Sentence #')['Word'].apply(list).values
    pos_list = df.groupby('Sentence #')['POS'].apply(list).values
    tag_list = df.groupby('Sentence #')['Tag'].apply(list).values

    return text_list, pos_list, tag_list, enc_pos, enc_tag


if __name__ == '__main__':
    
    data_path = './../input/ner_dataset.csv'
    text_list, pos_list, tag_list, enc_pos, enc_tag = process_data(config.TRAINING_FILE)

    metadata = {
        "enc_pos": enc_pos,
        "enc_tag": enc_tag
    }

    joblib.dump(metadata, 'metadata.bin')

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    (
        train_text,
        valid_text,
        train_pos,
        valid_pos,
        train_tag,
        valid_tag
    ) = train_test_split(text_list, pos_list, tag_list, random_state=42, test_size=0.1)

    train_dataset = EntityDataset(train_text, train_pos, train_tag)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.TRAIN_BATCH_SIZE)
    
    valid_dataset = EntityDataset(valid_text, valid_pos, valid_tag)
    valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=config.VALID_BATCH_SIZE)
    
    device = torch.device('cpu')
    model = EntityModel(num_pos, num_tag)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_parameters = [
        {
            "params": [p for n,p in param_optimizer if not any([n in nd for nd in no_decay])],
            "weight_decay": 0.001
        },
        {
            "params": [p for n,p in param_optimizer if any([n in nd for nd in no_decay])],
            "weight_decay": 0.0
        }
    ]

    num_training_steps = int(len(train_text)/config.TRAIN_BATCH_SIZE*config.NUM_EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    best_loss = numpy.inf
    for epoch in range(config.NUM_EPOCHS):
        train_loss = engine.train_fn(train_dataloader, model, optimizer, device, scheduler)
        valid_loss = engine.eval_fn(valid_dataloader, model, device)
        print(f'Train Loss: {train_loss}; Validation Loss: {valid_loss}')
        if valid_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = valid_loss
