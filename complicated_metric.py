import json
from tqdm import tqdm
from utils import text_f1

class MetricF1_RE:
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

class MetricF1_EE:
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

file_name = 'qwen_14B_sequence.json'
path = '/data1/DuEE1.0/' + file_name

schemas = {}
with open('/data1/DuEE1.0/duee_event_schema.json', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        res = json.loads(line)
        schemas[res['event_type']] = [item['role'] for item in res['role_list']]
def eve2text(argus, event_type):
    res = argus['event_type'] + '|'
    for role in schemas[event_type]:
        if role in argus:
            res += argus[role] + '|'
        else:
            res += '[None]|'
    return res[:-1]   # 去掉最后一个'|'

f1 = MetricF1_EE()
with open(path,'r',encoding='utf-8') as f:
    outs = json.load(f)

for itm in tqdm(outs):
    nowroles = itm['std_ans'][0].keys()
    if len(nowroles) >=5:
        gold = set()
        pred = set()
        for dic in itm['std_ans']:
            gold.add(eve2text(dic, dic['event_type']))
        for dic in itm['preds']:
            pred.add(eve2text(dic, dic['event_type']))
        f1.append_relax(pred,gold)
f1.compute()