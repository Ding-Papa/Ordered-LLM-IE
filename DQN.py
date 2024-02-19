from model import ActorModel
import torch
import torch.nn as nn
import random
from transformers import BertTokenizerFast, RobertaTokenizerFast, set_seed
from Environment import ExtractionEnv
import argparse
import numpy as np
from tqdm import tqdm
from utils import *
from collections import Counter
from RL_utils.dqn import DQN
import math
import time
# import wandb
from client_mixtral import mistral_7B_api, qwen_14B_api

parser = argparse.ArgumentParser(description='Construtor')
parser.add_argument("--seed", type=int, default=888, help="seed")
# parser.add_argument("--plm", type=str, default='/plm-model/models--bert-base-cased', help='pretrain language model for bert')  # 英文数据
parser.add_argument("--plm", type=str, default='/plm-model/models--hfl--chinese-roberta-wwm-ext', help='pretrain language model for bert')   # 中文数据
parser.add_argument("--exploration_update", type=int, default=100, help='how many step to update exploration ratio')
parser.add_argument("--max_step", type=int, default=15, help='max step on a single episode')
parser.add_argument("--num_episode", type=int, default=10, help='sample episode number')  # DuIE是15
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate for the DQN model')
parser.add_argument("--do_train", type=bool, default=False, help='whether train')
parser.add_argument("--weight_file", type=str, default='rl_NYT.pt', help='path to save rl model weight')
parser.add_argument('--dname', type=str, choices=['WebNLG','DuEE1.0','DuEE-fin','DuIE2.0','HacRED','NYT','SKE'])
parser.add_argument('--buf_size', type=int, default=5000, help='the size of the memory size')   # 适当减小
parser.add_argument('--action_strategy', type=str, default='RL', choices=['RL','Random','Sequence'])
parser.add_argument('--reward_type', type=str, choices=['v1', 'v2', 'v3'], default='v1')
parser.add_argument('--data_split', type=int, choices=[1,2], default=1)
args = parser.parse_args()

"""
Parameters
"""
seed = args.seed
plm = args.plm
lr = args.learning_rate
target_update = 20
iters_save = 500

if args.do_train:
    """
    Training
    """
    
    np.random.seed(seed)
    set_seed(seed)
    config = {
        "learning_rate": args.learning_rate,
        "epochs": args.num_episode,
        "batch_size": 32,
        "seed": args.seed,
    }
    tokenizer = BertTokenizerFast.from_pretrained(plm)
    if args.dname == 'HacRED':
        env = ExtractionEnv(llm_ext_func=qwen_14B_api,
                        data_path='/data1/HacRED/new_train.json',
                        dataset='HacRED',
                        reward_type=args.reward_type,
                        lang='zh',
                        data_split=args.data_split)
    elif args.dname == 'NYT':
        env = ExtractionEnv(llm_ext_func=mistral_7B_api,
                        data_path='/data1/NYT11-HRL/new_train.json',
                        dataset='NYT',
                        reward_type=args.reward_type,
                        lang='en',
                        data_split=args.data_split)
    elif args.dname == 'DuIE2.0':
        env = ExtractionEnv(llm_ext_func=qwen_14B_api,
                        data_path='/data1/DuIE2.0/new_train.json',
                        dataset='DuIE2.0',
                        reward_type=args.reward_type,
                        lang='zh',
                        data_split=args.data_split)
    elif args.dname == 'SKE':
        env = ExtractionEnv(llm_ext_func=qwen_14B_api,
                        data_path='/data1/SKE/new_train.json',
                        dataset='SKE',
                        reward_type=args.reward_type,
                        lang='zh',
                        data_split=args.data_split)
    elif args.dname == 'DuEE1.0':
        env = ExtractionEnv(llm_ext_func=qwen_14B_api,
                        data_path='/data1/DuEE1.0/new_train.json', 
                        dataset='DuEE1.0',
                        reward_type=args.reward_type,
                        lang='zh',
                        data_split=args.data_split)

    tot_step = args.num_episode * env.dataset_len
    n = math.log(0.05 / 0.9) / math.log(0.95)
    explore_update = int(tot_step // n)
    print('Explore update:', explore_update)
    agent = DQN(plm=plm,epsilon=0.9, tokenizer=tokenizer, gamma=0.5,buf_sz=args.buf_size,batch_sz=32, lr=lr, explore_update=explore_update)    # epsilon用来控制随机选择action还是继续策略，考虑降低该值
    
    print('Begin RL training!')

    rewards = [] # 记录总的rewards
    moving_average_rewards = [] # 记录总的经滑动平均处理后的rewards
    ep_steps = []
    for i_episode in tqdm(range(tot_step), desc='Training RL agent'): # cfg.max_episodes为最大训练的episode数
        state_list, _, _ = env.reset() # reset环境状态
        ep_reward = 0
        for i_step in range(args.max_step): # cfg.max_steps为每个episode的步长
            new_state_list = []
            for state in state_list:
                cond, text, choices = state
                action = agent.select_action(cond, text, choices) # 根据当前环境state选择action
                action = torch.argmax(action)
                #print(action)
                next_state_list, reward, done = env.step(cond, action, choices) # 更新环境参数
                ep_reward += reward / len(state_list)    # TODO: 这里reward如何对齐

                agent.store_transition(state, action, reward, next_state_list, done)
                """ for next_state in next_state_list:
                    agent.store_transition(state, action[1], reward, next_state, done) """
                new_state_list.extend(next_state_list) # 跳转到下一个状态
                print('reward:'+str(reward))
                # wandb.log({'reward': reward})
            state_list = new_state_list
            if done:
                break
        agent.update() # 每步更新网络
        print('ep_reward:'+str(ep_reward))
        # wandb.log({'ep_reward': ep_reward})
        # 更新target network，复制DQN中的所有weights and biases
        if i_episode % target_update == 0: #  cfg.target_update为target_net的更新频率
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        if i_episode % iters_save == 0 and i_episode != 0:
            agent.save_weight(f'weight/{args.weight_file}')

        #print('Episode:', i_episode, ' Reward: %.2f' %
        #    ep_reward, 'n_steps:', i_step, 'done: ', done,' Explore: %.2f' % agent.epsilon)
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if i_episode == 0:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1] + 0.1*ep_reward)