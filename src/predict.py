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
    df.loc[:, 'Tag'] = enc_pos.fit_transform(df['Tag'])

    text_list = df.groupby('Sentence #')['Words'].apply(list).values
    pos_list = df.groupby('Sentence #')['POS'].apply(list).values
    tag_list = df.groupby('Sentence #')['Tag'].apply(list).values

    return text_list, pos_list, tag_list, enc_pos, enc_tag


if __name__ == '__main__':
    
    data_path = './../input/ner_dataset.csv'
    text_list, pos_list, tag_list, enc_pos, enc_tag = process_data(config.TRAINING_FILE)

    metadata = joblib.load('metadata.bin')
    enc_pos = metadata['enc_pos']
    enc_tag = metadata['enc_tag']

    sentence = """
    anoop lives in bangalore
    """

    tokenized_sentence = config.TOKENIZER(sentence)
    sentence = sentence.split()
    print(sentence)
    print(tokenized_sentence)



    test_dataset = EntityDataset(test_text, train_pos, train_tag)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.TRAIN_BATCH_SIZE)
    
    valid_dataset = EntityDataset(valid_text, valid_pos, valid_tag)
    valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=config.VALID_BATCH_SIZE)
    
    device = torch.device('cpu')
    model = EntityModel(num_pos, num_tag)
    model.to(device)

