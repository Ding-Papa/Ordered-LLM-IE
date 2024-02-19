from html import entities
import json
import torch
from dataset.nyt import NYTDataset
from dataset.duee import DuEEDataset
from dataset.duie import DuIEDataset
from dataset.hacred import HacREDDataset
from dataset.ske import SKEDataset
# from model import GlobalPointerModel
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers.generation import GenerationConfig
import random, math
import copy
import numpy as np
from utils import text_f1  # 这个可以沿用

class ExtractionEnv:
    def __init__(self, llm_ext_func, data_path, dataset='WebNLG', lang='en', mode='train', reward_type='v1', data_split=1):
        self.data = None
        self.state = None
        self.data_split = data_split
        self.llm_ext = llm_ext_func
        self.mode = mode
        self.dname = dataset
        self.index = 0
        self.lang = lang
        self.reward_type = reward_type

        if dataset == 'HacRED':
            self.dataset = HacREDDataset(data_path=data_path, data_type='rl', data_split=self.data_split)
            self.dataset_len = len(self.dataset)
        elif dataset == 'NYT':
            self.dataset = NYTDataset(data_path=data_path, data_type='rl', data_split=self.data_split)
            self.dataset_len = len(self.dataset)
        elif dataset == 'SKE':
            self.dataset = SKEDataset(data_path=data_path, data_type='rl', data_split=self.data_split)
            self.dataset_len = len(self.dataset)
        elif dataset == 'DuIE2.0':
            self.dataset = DuIEDataset(data_path=data_path, data_type='rl', data_split=self.data_split)
            self.dataset_len = len(self.dataset)
        elif dataset == 'DuEE1.0':
            self.dataset = DuEEDataset(data_path=data_path, data_type='rl', data_split=self.data_split)
            self._load_schema()
            self.dataset_len = len(self.dataset)

    def _load_schema(self):
        self.schema = {}
        if self.dname == 'DuEE1.0':
            with open('/data1/DuEE1.0/duee_event_schema.json', 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    self.schema[res['event_type']] = [item['role'] for item in res['role_list']]
    
    def _example_generation(self, input_texts, cond, hist_ext, slot_name):
        if self.dname == 'DuEE1.0':
            inst = '''现在在进行事件抽取任务，给定原句、事件类型、已经抽取出的内容，请你根据已有的内容按要求识别我想要的实体。
            我提供的输入会包含原句信息、事件类型、已经抽取的结果和我需要的元素，请你按照指定格式返回我想要的实体。原句中没有的信息不要考虑。
            要求输出内容仅包含我想要的实体，并且需要与输入中提供的事件类型和已抽取内容相匹配，不要有任何额外的输出。如果对应的实体有多个，请用英文逗号分隔，如果没有相应的实体，返回'None'。示例如下：
            示例1：输入：{'原句':'雀巢裁员4000人：时代抛弃你时，连招呼都不会打！','事件类型':'组织关系-裁员','已经抽取的内容':{'裁员人数':'4000人'},'我想要的元素':'裁员方'}
            输出：'雀巢'
            示例2：输入：{'原句':'火车上醉酒散德行，男子辱骂殴打乘警被刑拘！这样的人还有不少！','事件类型':'司法行为-拘捕','已经抽取的内容':{},'我想要的元素':'被拘捕者'}
            输出：'男子'
            示例3：输入：{'原句':'千盼万唤始出来，美联储降息25个基点，10年来首次；中国央行也降息30个基点','事件类型':'财经/交易-降息','已经抽取的内容':{'降息机构':'美联储'},'我想要的元素':'降息幅度'}
            输出：'25个基点'
            示例4：输入：{'原句':'传音控股如约挂牌 被华为起诉未阻断上市进程','事件类型':'财经/交易-上市','已经抽取的内容':{'上市企业':'传音控股'},'我想要的元素':'融资金额'}
            输出：'None'
            示例5：输入：{'原句':'千盼万唤始出来，美联储降息25个基点，10年来首次；中国央行也降息30个基点','事件类型':'财经/交易-降息','已经抽取的内容':{},'我想要的元素':'降息机构'}
            输出：'美联储,中国央行'
            示例6：输入：{'原句':'8月23号英雄联盟LPL赛区IG战队输给了LNG，与此同时LCK赛区的SKT战胜DK战队离冠军更进一步','事件类型':'竞赛行为-胜负','已经抽取的内容':{'败者':'IG战队'},'我想要的元素':'胜者'}
            输出：'LNG'
            示例7：输入：{'原句':'8月23号英雄联盟LPL赛区IG战队输给了LNG，与此同时LCK赛区的SKT战胜DK战队离冠军更进一步','事件类型':'竞赛行为-胜负','已经抽取的内容':{},'我想要的元素':'败者'}
            输出：'IG战队,DK战队'
            现在，请你仿照上面的例子，按要求完成下面的任务。再次强调，输出内容仅包含我想要的实体，并且需要与输入中提供的关系类型和已抽取内容相匹配，不要有任何额外的输出。原句中没有的信息不要考虑。如果对应的实体有多个，请用英文逗号分隔。
            输入：'''
            dic = {'原句':input_texts,'事件类型':cond,'已经抽取的内容':hist_ext,'我想要的元素':slot_name}
            inst = inst + str(dic) + '\n' + '输出：'
        elif self.lang == 'zh':
            inst = '''现在在进行关系三元组抽取任务，给定原句、感兴趣的关系类型，也可能会给定已经抽取出的内容，请你根据已有的内容按要求识别我想要的实体。
            我提供的输入会包含原句信息、关系类型、已经抽取的结果和我需要的实体类型，请你按照指定格式返回我想要的实体。关系类型字段如果有括号“（）”，括号内表示的是主语或宾语应当符合的类别要求。
            要求输出内容仅包含我想要的实体，并且需要与输入中提供的关系类型和已抽取内容相匹配，不要有任何额外的输出。如果对应的实体有多个，请用英文逗号分隔，如果没有相应的实体，返回'None'。示例如下：
            示例1：输入：{'原句':'《女学报》是中国最早的妇女报纸，1898年（光绪二十四年）7月24日创刊于上海，这是中国近代创办的第一份女报。由中国女学会主办，主笔为康有为的女儿康同薇、梁启超的夫人李蕙仙等。','关系类型':'（人）妻子是（人）','已经抽取的内容':{'主语':'梁启超'},'我想要的实体类型':'宾语'}
            输出：'李蕙仙'
            示例2：输入：{'原句':'刘凯年，男，重庆师范大学数学与计算机科学学院教授，硕导，中国数学奥林匹克高级教练，重庆市数学学会理事，重庆市数学竞赛委员会副主任。学历：研究生，学位：硕士，最后学位所学专业：数学与应用数学。','关系类型':'（人）所属机构（机构名称）','已经抽取的内容':{'主语':'刘凯年'},'我想要的实体类型':'宾语'}
            输出：'重庆市数学学会, 重庆师范大学数学与计算机科学学院, 重庆市数学竞赛委员会'
            示例3：输入：{'原句':'乔布斯和比尔盖茨分别创立了苹果公司和微软公司','关系类型':'创立','已经抽取的内容':{},'我想要的实体类型':'宾语'}
            输出：'苹果公司,微软公司'
            示例4：输入：{'原句':'乔布斯和比尔盖茨分别创立了苹果公司和微软公司','关系类型':'创立','已经抽取的内容':{'宾语':'微软公司'},'我想要的实体类型':'主语'}
            输出：'乔布斯'
            示例5：输入：{'原句':'丁薛祥毕业于复旦大学政治系','关系类型':'（人）毕业院校（学校）','已经抽取的内容':{'宾语':'清华大学'},'我想要的实体类型':'主语'}
            输出：'None'
            现在，请你仿照上面的例子，按要求完成下面的任务。再次强调，输出内容仅包含我想要的实体，不要包含任何多余内容，并且需要与输入中提供的关系类型和已抽取内容相匹配，不要有任何额外的输出。如果对应的实体有多个，请用英文逗号分隔。
            输入：'''
            dic = {'原句':input_texts,'关系类型':cond,'已经抽取的内容':hist_ext,'我想要的实体类型':slot_name}
            inst = inst + str(dic) + '\n' + '输出：'
        elif self.lang == 'en':
            inst = '''Now we are doing the task of relation triple extraction. Given the text and the interested relation, we may also give the extracted content. Please identify the entities I want according to the given content.
            The input I provide will contain the text information, the relation, the extracted results and the entity type I need. Please return the entities I want in the specified format. If there are parentheses in the 'Relation', the contents of the parentheses indicate the requirements of the category to be met by the subject or object.
            The output content only contains the entities I want, and needs to match the relation and extracted content provided in the input. If there are multiple corresponding entities, please separate them with English commas. If there is no corresponding entity, return 'None'. The examples are as follows:
            example1: INPUT: {'text':'Donations may be made to Special Olympics , Massachusetts , Attn : Donations , 450 Maple Street , Building 1 , Danvers , MA 01923 .','relation':'(place) location contains (place)','extracted content':{'subject':'Massachusetts'},'entity type I need':'object'}
            OUTPUT: 'Danvers'
            example2: INPUT: {'text':'Earlier in the day , Senator Jim Bunning , Republican of Kentucky and a Hall of Fame pitcher , testified that he thought that players who used steroids should have their records wiped from the book .','relation':'(penson) place lived (place)','extracted content':{'object':'Kentucky'},'entity type I need':'subject'}
            OUTPUT: 'Jim Bunning'
            example3: INPUT: {'text':'Steve Jobs and Bill Gates founded Apple and Microsoft respectively.','relation':'(people) founded (company)','extracted content':{},'entity type I need':'subject'}
            OUTPUT: 'Steve Jobs, Bill Gates'
            example4: INPUT: {'text':'Steve Jobs and Bill Gates founded Apple and Microsoft respectively.','relation':'(people) founded (company)','extracted content':{'subject':'Steve Jobs'},'entity type I need':'object'}
            OUTPUT: 'Apple'
            example5: INPUT: {'text':'Ding Xuexiang graduated from the Department of Politics of Fudan University.','relation':'(people) graduated from (school)','extracted content':{'object':'Tsinghua University'},'entity type I need':'subject'}
            OUTPUT: 'None'
            Now, please complete the following task according to the requirements. Again, the output content only contains the entities I want, and needs to match the relation and extracted content provided in the input. If there are multiple corresponding entities, please separate them with English commas.
            INPUT:'''
            dic = {'text':input_texts,'relation':cond,'extracted content':hist_ext,'entity type I need':slot_name}
            inst = inst + str(dic) + '\n' + 'OUTPUT:'
        new_resp = self.llm_ext(inst)
        new_resp = new_resp.split('\n')[0].strip()  # mixtral识别的结果有时候会附上换行的explanation，需要去掉
        new_resp = new_resp.strip("'")
        return new_resp

    def sigmoid(self, i):
        return 1 / (math.exp(-i) + 1)

    def score2prob(self, entities):
        '''
        Input: Entity without filtering
        Output: Unduplicated list: [(entity, prob, score)]
        '''
        entities_mention = list(set([e[0] for e in entities]))
        logsum = sum([math.exp(e[1]) for e in entities])
        entities = [(e[0],math.exp(e[1])/logsum, e[1]) for e in entities]
        entities_score = [(name, sum([i[1] for i in entities if i[0] == name]), max([i[2] for i in entities if i[0] == name])) for name in entities_mention]
        return entities_score

    def choice_decision(self, cond, choices, action):   # action是整数（choices列表的索引）; cond里面包含了关系+已经抽取的结果
        """ print('='* 50)
        print(choices[action]) """
        cond_list = cond.split(';')
        rel_type = cond_list[0].strip()
        hist = {}
        try:
            for item in cond_list[1:]:
                kv = item.split(':')
                hist[kv[0].strip()] = kv[1].strip()    # 不用担心槽位重复，step中会针对每个entity分别往后探索
        except:
            hist = {}
        ori_resp = self._example_generation(self.text, rel_type, hist, choices[action])
        std_ans = ''
        for spo in self.spo_list[cond]:
            if self.dname == 'DuEE1.0':
                if spo[choices[action]] == '[None]':
                    std_ans += ''
                else:
                    std_ans += ',' + spo[choices[action]].strip()
            else:
                if spo[choices[action]] == '[None]':
                    std_ans += ''
                else:
                    std_ans += ','+spo[choices[action]].strip()  # choices[action]即槽位名
        std_ans = std_ans.strip(',')   # 去掉两端的逗号
        std_ans = list(set(std_ans.split(',')))   # 去重
        std_ans = ','.join(std_ans)
        # print('标准答案:'+std_ans)
        entities_1step = self._recognize(ori_resp, threshold=0.19, std_ans=std_ans)
        # print('识别后结果:'+str(entities_1step))
        entities_1step = self.score2prob(entities_1step)
        if self.dname == 'DuEE1.0':
            if entities_1step == []:
                if std_ans.strip():
                    entities_1step.append(('[None]',0.9, 0.0))   # 有标准答案但没抽出来
                else:
                    entities_1step.append(('[None]',0.9, 9.99))    # 本来这个槽位就应该是空的，那么奖励就是最高档
        elif entities_1step == []:
            entities_1step.append(('[None]',0.9, 0.0))   # reward给一个很小的值
        return entities_1step

    def _recognize(self, llm_resp, threshold = 0.39, std_ans=''):  # 删掉textf1小于0.4的结果
        '''
        根据GlobalPointer模型的输出进行抽取结果的识别，返回形式：
            [(entity, scores),...]
        std_ans用于计算text_f1，仅在训练时提供，测试时不提供
        '''
        # TODO: 改成对LLM输出进行规范化的版本，不要score——直接对输出逗号分隔（多个实体情况）然后规范化处理，返回实体列表即可
        # TODO: score改成textf1，作为reward
        if llm_resp == 'None':
            return []
        entity_list = llm_resp.split(',')
        entities = []
        std_ans = std_ans.strip()
        if not std_ans:
            entities = [(en.strip(), 0.0) for en in entity_list]
            return entities
        for ent in entity_list:
            score = text_f1(ent.strip(), std_ans)
            if (score > threshold) or (std_ans in ent):   # 防止cot类输出被误判
                entities.append((ent.strip(), score*10))   # 乘以10做个放缩，与dqn量级匹配
        if entities:
            entities.sort(key=lambda x:x[-1], reverse=True)
        return entities

    def step(self, cond, action, choices):
        '''
        Action就是预测下一个待抽取的槽位,
        一个action需要返回多个下个状态（如果抽取多个候选要素，进行状态的分裂）
        return:
        Reward, []
        '''
        ##########################################################
        slot_name = choices[action]
        entities = self.choice_decision(cond, choices, action)
        reward = sum([entity[2] for entity in entities]) / len(entities)

        entities = list(set([e[0] for e in entities]))
        valid_conds = []
        for entity in entities:
            new_cond = f'{cond}; {slot_name}:{entity}'  # new_cond是下一个状态的cond，包含了关系以及拼接好的已经抽取的内容
            # if self.lang == 'en':
            #     new_cond = f'{cond}; {slot_name}:{entity}'
            # elif self.lang == 'zh':
            #     new_cond = f'{cond}； {slot_name}：{entity}'
            if new_cond not in self.spo_list.keys():
                self.spo_list[new_cond] = []
            valid_conds.append(new_cond)
            for spo in self.spo_list[cond]:
                if self.dname == 'DuEE1.0':
                    if (text_f1(spo[slot_name], entity) > 0.6) or (spo[slot_name] in entity): # 这里要改  不然好多“std_ans”算不出来，没法分配reward
                        self.spo_list[new_cond].append(spo)
                else:
                    if (text_f1(spo[slot_name], entity) > 0.6) or (spo[slot_name] in entity):
                        self.spo_list[new_cond].append(spo)

        new_choices = copy.deepcopy(choices)
        del new_choices[action]

        if new_choices:
            done = False
        else:
            done = True
        #return
        return [(_cond , self.text, new_choices) for _cond in valid_conds], reward, done

    def return_cond(self):  # 用于test时返回最终抽取的结果
        return self.spo_list

    def slot_fill_(self, slot_list, cond):
        if self.dname == 'DuEE1.0':
            for slot_name in slot_list:
                for rel in self.spo_list[cond]:
                    if slot_name not in rel.keys():
                        # rel[slot_name] = ('[None]',-1)
                        rel[slot_name] = '[None]'
        else:
            for slot_name in slot_list:
                for rel in self.spo_list[cond]:
                    if slot_name not in rel.keys():
                        rel[slot_name] = '[None]'

    def reset_with_input(self, text, cond, choices):
        self.spo_list = {}
        self.spo_list[cond] = {}
        self.text = text
        self.gt_num = 1e12
        return [(cond, text, choices)], 0, False

    def reset(self):   # reset单条数据（一条即一个抽取环境，参见各数据脚本里面的_gen_data_for_rl_model）
        '''
        data的形式: [
            Text, 
            predicate, 
            (s,o) list for the corresponding relation
        ]
        '''
        self.spo_list = {}
        if self.mode == 'train':
            index = self.index % self.dataset_len
            self.index += 1
            #index = random.randint(0,self.dataset_len - 1)
            #index = 24
        else:
            index = self.index
            self.index += 1

        self.data = self.dataset[index]
        cond = self.data[1]    # relation，或事件类型
        text = self.data[0]
        self.gt_num = len(self.data[2])    # 该句子中关于该关系的三元组个数（ground_truth）
        self.spo_list[cond] = self.data[2]   # 理解成“ground truth”
        #print(self.spo_list)
        if self.dname == 'WebNLG' or self.dname == 'NYT':
            choices = ['subject', 'object']
        elif self.dname in ['HacRED','SKE','DuIE2.0']:
            choices = ['主语','宾语']
        elif self.dname == 'DuEE1.0':
            choices = self.schema[cond]   # 问一下这个触发词是什么意思，为什么要加到action里面
            # slot 补全
            self.slot_fill_(choices, cond)
        self.text = text
        return [(cond, self.text, choices)], 0, False