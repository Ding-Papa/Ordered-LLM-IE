import torch
import json
from tqdm import tqdm
import argparse
import random
from client_mixtral import mixtral_87_api, baichuan2_13B_api, mistral_7B_api, deepseek_33B_api, chatgpt, qwen_14B_api

parser = argparse.ArgumentParser(description='Construtor')
parser.add_argument("--seed", type=int, default=777, help="seed")
parser.add_argument("--lang", type=str, default='zh', choices=['zh','en'], help='language for predict')  # 设置语言
parser.add_argument("--dataset", type=str, choices=['DuEE1.0','NYT10','SKE','NYT11-HRL'])
parser.add_argument("--write_dir", type=str, default='../type_classify/', help='directory to save results')
parser.add_argument("--llm_extractor", type=str, default='chatgpt', choices=['chatgpt','deepseek_33B','qwen_14B','mistral_7B','mixtral_87'],help='choice of llm_extractor')
args = parser.parse_args()

dataset = args.dataset
lang = args.lang
write_dir = args.write_dir
llm_name = args.llm_extractor
if llm_name == 'chatgpt':
    llm_ext_func = chatgpt
else:
    llm_ext_func = eval(llm_name + '_api')

class MetricF1:
    def __init__(self):
        self.correct = self.output = self.golden = 0
    def append(self, out, ans):
        out, ans = set(out), set(ans)
        mid = out & ans
        self.correct += len(mid)
        self.output += len(out)
        self.golden += len(ans)


    def compute(self, show=True):
        correct, output, golden = self.correct, self.output, self.golden
        prec = correct / max(output, 1);  reca = correct / max(golden, 1);
        f1 = 2 * prec * reca / max(1e-9, prec + reca)
        pstr = 'Prec: %.4f %d/%d, Reca: %.4f %d/%d, F1: %.4f' % (prec, correct, output, reca, correct, golden, f1)
        if show: print(pstr)
        return f1

random.seed(777)

with open(f'../{dataset}/new_test.json', 'r', encoding='utf-8') as f:  # DuEE要改为new_test
    datas = []
    for line in f.readlines():
        datas.append(json.loads(line))
random.shuffle(datas)

if dataset == 'DuEE1.0':
    with open(f'../{dataset}/duee_event_schema.json', 'r', encoding='utf-8') as f:
        types_list = []
        for line in f.readlines():
            dt = json.loads(line)
            types_list.append(dt['event_type'])
else:
    with open(f'../{dataset}/rel2id.json', 'r', encoding='utf-8') as f:
        types = json.load(f)
    types_list = list(types.keys())

if dataset == 'DuEE1.0':
    inst1 = '''给定事件类型列表，以及原句，请从事件类型列表中选择该句子包含的事件类型并输出；如果包含多种事件类型请用英文逗号隔开。
    示例如下：
    事件类型列表：['财经/交易-上市','财经/交易-降价','产品行为-上映','产品行为-召回','交往-道歉','竞赛行为-晋级','竞赛行为-胜负','司法行为-开庭']
    示例1：输入：'在刚刚结束的一场西部决赛的强强对决中，火箭在主场再次赢球，最终火箭以112:108战胜了勇士队，将总比分扳成2:2！'
    输出：'竞赛行为-胜负'
    示例2：输入：'据悉，此次上市的新股共有10只，其中有7只是科创板新股，另外3只是主板新股。'
    输出：'财经/交易-上市'
    示例3：输入：'英伟达公司在年初将新的一批产品价格调低，并重新在港股上市'
    输出：'财经/交易-降价,财经/交易-上市'
    示例4：输入：'据悉，米哈游有限公司法人代表已经在微博上发表了道歉声明，与腾讯的官司即将开打'
    输出：'交往-道歉,司法行为-开庭'
    '''
elif lang == 'en':
    inst1 = '''Given the relation type list, and a origin sentence, Please select the relation contained in this sentence from the list. 
    If more than one relation in list is included in this sentence, please separate them with commas(',') and output.
    Here are some examples:
    relation list: ["location contains","place lived","born in","neighborhood of","country capital"]
    example1: INPUT: 'Donations may be made to Special Olympics , Massachusetts , Attn : Donations , 450 Maple Street , Building 1 , Danvers , MA 01923 .' 
    OUTPUT: 'location contains'
    example2: INPUT: 'Earlier in the day , Senator Jim Bunning , Republican of Kentucky and a Hall of Fame pitcher , testified that he thought that players who used steroids should have their records wiped from the book .'
    OUTPUT: 'place lived'
    example3: INPUT: 'The Soviet-era grenade was discovered in the crowd roughly 100 to 120 feet from where Mr. Bush and Mikheil Saakashvili , the president of Georgia , appeared together at a huge outdoor rally in Tbilisi 's government center , the officials said . '
    OUTPUT: 'place lived, born in'
    '''
else:
    inst1 = '''给定关系类型列表，以及原句，请从关系类型列表中选择该句子包含的关系类型并输出；如果包含多种关系类型请用英文逗号隔开。
    示例如下：
    关系类型列表：["朝代","妻子","丈夫","民族","毕业院校","编剧","出品公司","父亲","出版社","作词","作曲","母亲","成立日期"]
    示例1：输入：'上海万安医院投资管理有限公司于2009年9月28日在闵行区市场监督管理局登记成立'
    输出：'成立日期'
    示例2：输入：'刘弗陵（前94年－前74年），即汉昭帝，西汉第八位皇帝，汉武帝刘彻少子，赵婕妤（钩弋夫人）所生'
    输出：'朝代，出生日期，父亲，母亲'
    示例3：输入：'《讲不出再见》是谭咏麟演唱的一首歌曲，由华夏传媒出品，作词：林夕，作曲：罗大佑'
    输出：'出品公司，作词，作曲'
    '''

# Predict Process
f1 = MetricF1()
wlist = []
ind = 0
for data in tqdm(datas[:200]):
    ind += 1
    if dataset == 'DuEE1.0':
        wdic = {}
        text = data['text']
        wdic['text'] = text
        inst = inst1 + f'现在给定如下的事件类型列表：{types_list}，以及原句{text}\n。请从事件类型列表中选择该句包含的事件类型并输出，如果包含多种事件类型请用英文逗号分隔。注意：仅考虑给定列表里面的事件类型；输出内容仅包含该句所含的事件类型，不要有任何额外的输出。'
    elif lang == 'en':
        wdic = {}
        text = data['sentText']
        wdic['text'] = text
        inst = inst1 + f'Now given the relation type list{types_list}, and a sentence {text}, Please select the relation(s) contained in this sentence from the list. If more than one relation in list is included in this sentence, please separate them with commas(",") and output. Note: Consider only the types provided in the list, and do not output anything redundant.'
    else:
        wdic = {}
        text = data['sentText']
        wdic['text'] = text
        inst = inst1 + f'现在给定如下的关系类型列表：{types_list}，以及原句{text}\n。请从关系类型列表中选择该句包含的关系类型并输出，如果包含多种关系类型请用英文逗号分隔。注意：仅考虑给定列表里面的关系类型；输出内容仅包含该句所含的关系类型，不要有任何额外的输出。'
    res = llm_ext_func(inst)
    print(res)
    if dataset == 'DuEE1.0':
        ans = [a['event_type'] for a in data['event_list']]
        ans = list(set(ans))
        wdic['std_ans'] = ans
    else:
        ans = [a['label'] for a in data['relationMentions']]
        ans = list(set(ans))
        wdic['std_ans'] = ans
    tmp = res.split(',')
    tmp = [t.strip() for t in tmp]
    wdic['preds'] = []   # 松弛匹配，并筛掉不在候选列表里面的
    for ts in tmp:
        if ts not in types_list:
            for t in types_list:
                if t in ts:
                    wdic['preds'].append(t)
        else:
            wdic['preds'].append(ts)
    wdic['preds'] = list(set(wdic['preds']))
    print(wdic)
    f1.append(wdic['preds'], wdic['std_ans'])
    wlist.append(wdic)
    if ind % 50 == 0:
        f1.compute()
        with open(f'{write_dir}{dataset}/{llm_name}.json', 'a', encoding='utf-8') as f:
            for w in wlist:
                f.write(json.dumps(w, ensure_ascii=False) + '\n')
        wlist = []
f1.compute()