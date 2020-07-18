import torch
import config

class EntityDataset:

    def __init__(self, text_list, pos_list, tag_list):
        # text_list = [['Hi',',','my','name','is','Anoop'], ['What','is','up','?']........]
        # pos_list = [[1,5,2,8,19,24], [6,8,2,4]........]
        # tag_list = [[12,7,65,87,12,1], [78,7,21,5]........]
        self.text_list = text_list
        self.pos_list = pos_list
        self.tag_list = tag_list

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, item):
        text = self.text_list[item]
        pos = self.pos_list[item]
        tag = self.tag_list[item]

        ids = []
        target_pos = []
        target_tag = []

        for i,s in enumerate(text):
            inputs = config.TOKENIZER.encode(s, add_special_tokens=False)
            
            input_len = len(inputs)
            ids.extend(inputs)
            target_pos.extend([pos[i]] * input_len)
            target_tag.extend([tag[i]] * input_len)

        ids = ids[:config.MAX_LEN - 2]
        target_pos = target_pos[:config.MAX_LEN - 2]
        target_tag = target_tag[:config.MAX_LEN - 2]

        ids = [101] + ids + [102]
        target_pos = [0] + target_pos + [0]
        target_tag = [0] + target_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_pos = target_pos + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "text": text,
            "random":'YO',
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_pos": torch.tensor(target_pos, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long)
        }