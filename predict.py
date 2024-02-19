import torch
import json
from tqdm import tqdm
import argparse
import random
from transformers import BertTokenizerFast
from Environment import ExtractionEnv
from RL_utils.dqn import DQN
from collections import Counter
from client_mixtral import mixtral_87_api, mistral_7B_api, deepseek_33B_api, chatgpt, qwen_14B_api
from utils import text_f1


parser = argparse.ArgumentParser(description='Construtor')
parser.add_argument("--seed", type=int, default=777, help="seed")
parser.add_argument("--lang", type=str, default='zh', choices=['zh','en'], help='language for predict')  # 设置语言
parser.add_argument("--dataset", type=str, choices=['WebNLG','DuEE1.0','DuEE-fin','DuIE2.0','HacRED','NYT10','SKE','NYT11-HRL'])
parser.add_argument('--action_mode', type=str, default='RL', choices=['RL','random','sequence'])
parser.add_argument("--rl_model", type=str, default='rl_HacRED', help='choice of rl model file')
parser.add_argument("--write_dir", type=str, default='../predicted/', help='directory to save results')
parser.add_argument("--llm_extractor", type=str, default='chatgpt', choices=['chatgpt','deepseek_33B','qwen_14B','mistral_7B','mixtral_87'],help='choice of llm_extractor')
args = parser.parse_args()

#Params
dataset = args.dataset
lang = args.lang
write_dir = args.write_dir
if args.action_mode == 'RL':
    model_name = args.rl_model
else:
    model_name = args.action_mode
if lang == 'zh':
    plm = '/plm-model/models--hfl--chinese-roberta-wwm-ext'
else:
    plm = '/plm-model/models--bert-base-cased'
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
                lang=lang)

class MetricF1:
    def __init__(self):
        self.correct = self.output = self.golden = 0
    def append(self, out, ans):
        out, ans = set(out), set(ans)
        mid = out & ans
        self.correct += len(mid)
        self.output += len(out)
        self.golden += len(ans)
    
    def append_relax(self, out, ans):  # 松弛化的f1计算
        out, ans = list(set(out)), list(set(ans))  # 去重
        self.output += len(out)
        self.golden += len(ans)
        out = [itms.split('|') for itms in out]
        ans = [itms.split('|') for itms in ans]
        matches = 0
        for outitm in out:
            for ansitm in ans:
                try:
                    s_match = text_f1(outitm[1],ansitm[1])
                    o_match = text_f1(outitm[2],ansitm[2])
                except:
                    s_match = o_match = 0
                if s_match > 0.7 and o_match > 0.7:  # 做松弛匹配，相似度都超过0.7即认为抽对了
                    matches += 1
                elif (ansitm[1] in outitm[1]) and (ansitm[2] in outitm[2]):
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
        if '[None]' in k:
            continue
        c = Counter(k)
        if c[';'] == 0:
            gt = predict_list[k]   # ground truth
        if c[';'] == slot_num:
            pre = {'relation': ori_cond}
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

def spo2text_zh(spo): return spo['relation'] + '|' + spo['主语'] + '|' + spo['宾语']
def spo2text_en(spo): return spo['relation'] + '|' + spo['subject'] + '|' + spo['object']
def spo2text_gt(spo): return spo['label'] + '|' + spo['em1Text'] + '|' + spo['em2Text']

# Predict Process
f1 = MetricF1()
wlist = []
ind = 0
for data in tqdm(datas):
    ind += 1
    wdic = {}
    wdic['text'] = data['sentText']
    wdic['std_ans'] = data['relationMentions']
    all_relations = [q['label'] for q in data['relationMentions']]
    all_relations = list(set(all_relations))
    wdic['preds'] = []
    pred = set()
    for rel_cond in all_relations:
        if lang == 'zh':
            predict = ext_with_env(data['sentText'], rel_cond, ['主语','宾语'], model_name)
        else:
            predict = ext_with_env(data['sentText'], rel_cond, ['subject','object'], model_name)
        for spo in predict:
            if lang == 'zh':
                pred.add(spo2text_zh(spo))
            else:
                pred.add(spo2text_en(spo))
        wdic['preds'] += predict
    print(wdic)
    wlist.append(wdic)
    gold = set([spo2text_gt(spo) for spo in data['relationMentions']])
    f1.append_relax(pred, gold)
    if ind % 50 == 0:
        f1.compute()
        with open(f'{write_dir}{dataset}/{llm_name}_{model_name}.json', 'a', encoding='utf-8') as f:
            json.dump(wlist,f,ensure_ascii=False,indent=4)
        wlist = []

f1.compute()