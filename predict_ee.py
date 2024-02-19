import torch
import json
from tqdm import tqdm
import argparse
import random
from transformers import BertTokenizerFast
from Environment import ExtractionEnv
from RL_utils.dqn import DQN
from collections import Counter
from client_mixtral import deepseek_33B_api, chatgpt, qwen_14B_api
from utils import text_f1


parser = argparse.ArgumentParser(description='Construtor')
parser.add_argument("--seed", type=int, default=777, help="seed")
parser.add_argument("--dataset", type=str, choices=['DuEE1.0'])
parser.add_argument('--action_mode', type=str, default='RL', choices=['RL','random','sequence'])
parser.add_argument("--rl_model", type=str, default='rl_DuEE', help='choice of rl model file')
parser.add_argument("--write_dir", type=str, default='../predicted/', help='directory to save results')
parser.add_argument("--llm_extractor", type=str, default='chatgpt', choices=['chatgpt','deepseek_33B','qwen_14B'],help='choice of llm_extractor')
args = parser.parse_args()

#Params
dataset = args.dataset
write_dir = args.write_dir
if args.action_mode == 'RL':
    model_name = args.rl_model
else:
    model_name = args.action_mode
plm = '/plm-model/models--hfl--chinese-roberta-wwm-ext'
llm_name = args.llm_extractor
if llm_name == 'chatgpt':
    llm_ext_func = chatgpt
else:
    llm_ext_func = eval(llm_name + '_api')

random.seed(777)

with open(f'../{dataset}/new_test.json', 'r', encoding='utf-8') as f:
    datas = []
    for line in f.readlines():
        datas.append(json.loads(line))

# load tokenizer
tokenizer = BertTokenizerFast.from_pretrained(plm)
if model_name == 'random' or model_name == 'sequence':
    agent1 = None
else:
    agent1 = DQN(plm=plm,epsilon=0, tokenizer=tokenizer, gamma=0.5,buf_sz=10000,batch_sz=32, lr=0, explore_update = 1e10)
    agent1.load_weight(f'weight/{model_name}.pt')
env = ExtractionEnv(llm_ext_func=llm_ext_func,
                data_path=f'../{dataset}/new_test.json',
                dataset=dataset,
                mode='test',
                lang='zh')

class MetricF1:
    def __init__(self):
        self.correct = self.output = self.golden = 0
    
    def append_relax(self, out, ans):  # 松弛化的f1计算
        out, ans = list(set(out)), list(set(ans))  # 去重
        self.output += len(out)
        self.golden += len(ans)
        out = [itms.split('|') for itms in out]
        ans = [itms.split('|') for itms in ans]
        matches = 0
        for outitm in out:
            for ansitm in ans:
                if outitm[0] != ansitm[0]:
                    continue   # 同类型再进行比较
                all_in = True
                try:
                    total_f1 = 0
                    for i in range(1, len(outitm)):
                        s_match = text_f1(outitm[i],ansitm[i])
                        if ansitm[i] not in outitm[i]:
                            all_in = False
                        total_f1 += s_match
                    total_f1 /= (len(outitm) - 1)   # 平均的textf1
                except:
                    total_f1 = 0
                    all_in = False
                if total_f1 > 0.6:  # 做松弛匹配，平均相似度超过0.6即认为抽对了
                    matches += 1
                elif all_in:   # 兼容cot一类的llm
                    matches += 1
        self.correct += matches


    def compute(self, show=True):
        correct, output, golden = self.correct, self.output, self.golden
        prec = correct / max(output, 1);  reca = correct / max(golden, 1);
        f1 = 2 * prec * reca / max(1e-9, prec + reca)
        pstr = 'Prec: %.4f %d/%d, Reca: %.4f %d/%d, F1: %.4f' % (prec, correct, output, reca, correct, golden, f1)
        if show: print(pstr)
        return f1

    # 为了绘制PR curve打印到文件上
    def compute_and_record(self, fout):
        correct, output, golden = self.correct, self.output, self.golden
        prec = correct / max(output, 1);  reca = correct / max(golden, 1);
        f1 = 2 * prec * reca / max(1e-9, prec + reca)
        pstr = 'Prec: %.4f %d/%d, Reca: %.4f %d/%d, F1: %.4f' % (prec, correct, output, reca, correct, golden, f1)
        fout.write(pstr+'\n')
        return (prec, reca, f1)


def ext_with_env(text, ori_cond, choices, model_name):
    state_list, _, _ = env.reset_with_input(text, ori_cond, choices)   # TODO: 也可以不用这样，直接reset()就行？
    slot_list = state_list[0][2]
    slot_num = len(slot_list)
    ep_reward = 0
    for i_step in range(20): 
        new_state_list = []
        for state in state_list:
            cond, text, choices = state
            if model_name == 'sequence':
                action = 0
            elif model_name == 'random':
                action = random.randint(0, len(choices) - 1)
            else:
                action = agent1.select_action(cond, text, choices)
                action = torch.argmax(action)
            next_state_list, reward, done = env.step(cond, action, choices) 
            new_state_list.extend(next_state_list)
        state_list = new_state_list
        if done:
            break
    pre_list = []
    predict_list = env.return_cond()
    for k in predict_list.keys():
        # if '[None]' in k:
        #     continue
        c = Counter(k)
        if c[';'] == 0:
            gt = predict_list[k]   # ground truth
        if c[';'] == slot_num:
            pre = {'event_type': ori_cond}
            predict_word_offset = []
            for slot in slot_list:
                predict_word_offset.append((k.index('; ' + slot+':'), len(slot) + 1, slot))
            predict_word_offset.sort()
            for index, offset in enumerate(predict_word_offset):
                s, l, slot = offset
                vs = s + 2 + l
                if index != len(slot_list) - 1:
                    ve = predict_word_offset[index+1][0]
                else:
                    ve = len(k)
                #if k[vs:ve] != '[None]':
                pre[slot] = k[vs:ve]
            pre_list.append(pre)
    return pre_list

def eve2text(argus, event_type):
    res = argus['event_type'] + '|'
    for role in schemas[event_type]:
        if role in argus:
            res += argus[role] + '|'
        else:
            res += '[None]|'
    return res[:-1]   # 去掉最后一个'|'

# 统一化schema：
schemas = {}
with open('/data1/dingzepeng/Ordered_LLM/DuEE1.0/duee_event_schema.json', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        res = json.loads(line)
        schemas[res['event_type']] = [item['role'] for item in res['role_list']]

# Predict Process
f1 = MetricF1()
wlist = []
ind = 0
for data in tqdm(datas):
    ind += 1
    wdic = {}
    wdic['text'] = data['text']  # 下面都要针对ee进行改动
    wdic['std_ans'] = []
    gold = set()
    for itm in data['event_list']:
        now_dic = {}
        now_dic['event_type'] = itm['event_type']
        for pairs in itm['arguments']:
            now_dic[pairs['role']] = pairs['argument']
        for role in schemas[itm['event_type']]:
            if role not in now_dic:
                now_dic[role] = '[None]'    # 补齐槽位
        gold.add(eve2text(now_dic, itm['event_type']))
        wdic['std_ans'].append(now_dic)
    all_types = [q['event_type'] for q in data['event_list']]
    all_types = list(set(all_types))
    wdic['preds'] = []
    pred = set()
    for event_type in all_types:
        predict = ext_with_env(data['text'], event_type, schemas[event_type], model_name)
        for argus in predict:
                pred.add(eve2text(argus, event_type))
        wdic['preds'] += predict
    print(wdic)
    # print(gold)
    # print(pred)
    wlist.append(wdic)
    f1.append_relax(pred, gold)
    if ind % 50 == 0:
        f1.compute()
        with open(f'{write_dir}{dataset}/{llm_name}_{model_name}.json', 'a', encoding='utf-8') as f:
            json.dump(wlist,f,ensure_ascii=False,indent=4)
        wlist = []

f1.compute()