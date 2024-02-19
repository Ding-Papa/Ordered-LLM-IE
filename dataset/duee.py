from symbol import argument
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from itertools import permutations
import copy
import random

class DuEEDataset(Dataset):
    def __init__(self, data_path,data_type='extraction', data_split=1):
        self.data_split = data_split
        self._load_schema()
        self._load_dataset(data_path)
        self._process_data()
        self.data_type = data_type
        self._gen_data_for_rl_model()

    def _load_schema(self):
        self.schema = {}
        with open('data1/duee_event_schema.json', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                res = json.loads(line)
                self.schema[res['event_type']] = [item['role'] for item in res['role_list']]

    def _process_data(self):
        self.datas = []
        for data in self.label_datas:
            eve = {}
            for event in data['event_list']:
                if event['event_type'] not in eve.keys():
                    eve[event['event_type']] = []

                tmp_event = {
                    'event_type': event['event_type']
                }
                tmp_event['arguments'] = {}
                for argument in event['arguments']:
                    if argument['role'] not in tmp_event['arguments'].keys():
                        tmp_event['arguments'][argument['role']] = []
                    tmp_event['arguments'][argument['role']].append(argument['argument'])

                event_list = [{}]
                for argument in tmp_event['arguments'].keys():
                    new_event_list = []
                    for el in event_list:
                        for entity in tmp_event['arguments'][argument]:
                            tmp_el = copy.deepcopy(el)
                            tmp_el[argument] = entity
                            new_event_list.append(tmp_el)
                    event_list = new_event_list
                eve[event['event_type']].extend(event_list)
            self.datas.append({
                    'text': data['text'],
                    'event_list': eve
                })

    def _gen_data_for_rl_model(self):
        new_data = []
        for data in tqdm(self.datas, desc='Process data for rl model'):
            '''构造数据：
            1. 把待抽取的句子和Ground Truth的一个关系放在一起，针对关系已知的句子进行要素抽取
            2. 一个Example就是一个抽取的环境，简单而言就是一个（Text, Predicate）对。
            '''
            for event in data['event_list'].keys():
                new_data.append((data['text'], event, data['event_list'][event]))
        self.datas = new_data

    def _load_dataset(self, data_path):
        self.label_datas = []
        # if 'train' in data_path:
        #     with open(data_path[:-5]+f'{self.data_split}.json', 'r', encoding='utf-8') as f:
        #         for line in f.readlines():
        #             self.label_datas.append(json.loads(line))
        # else:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                self.label_datas.append(json.loads(line))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        examples = self.datas[index]
        return examples

    def collate_fn_cuda(self, batch):
        bz = len(batch)
        maxlen = max([len(item[0]) for item in batch])
        batch_input_ids = []
        batch_token_type_ids = []
        batch_labels = torch.zeros(bz,1,maxlen,maxlen)

        for index, (input_ids, token_type_ids, labels) in enumerate(batch):
            batch_input_ids.append(input_ids)
            batch_token_type_ids.append(token_type_ids)
            seqlen = len(input_ids)
            batch_labels[index][0][:seqlen,:seqlen] = labels

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True)
        batch_token_type_ids = torch.nn.utils.rnn.pad_sequence(batch_token_type_ids, batch_first=True)

        return [
            batch_input_ids.cuda(),
            batch_token_type_ids.cuda(),
            batch_labels.cuda()
        ]