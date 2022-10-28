import datetime
import importlib
import os
import random

import numpy as np
import torch

from recbole.utils.enum_type import ModelType


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur


def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_model(model_name):
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        Recommender: model class
    """
    model_submodule = [
        'sequential_recommender'
    ]

    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = '.'.join(['recbole.model', submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        raise ValueError('`model_name` [{}] is not the name of an existing model.'.format(model_name))
    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer(model_type, model_name):
    r"""Automatically select trainer class based on model type and model name

    Args:
        model_type (ModelType): model type
        model_name (str): model name

    Returns:
        Trainer: trainer class
    """
    try:
        return getattr(importlib.import_module('recbole.trainer'), model_name + 'Trainer')
    except AttributeError:
        if model_type == ModelType.KNOWLEDGE:
            return getattr(importlib.import_module('recbole.trainer'), 'KGTrainer')
        elif model_type == ModelType.TRADITIONAL:
            return getattr(importlib.import_module('recbole.trainer'), 'TraditionalTrainer')
        else:
            return getattr(importlib.import_module('recbole.trainer'), 'Trainer')


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def calculate_valid_score(valid_result, valid_metric=None):
    r""" return valid score from valid result

    Args:
        valid_result (dict): valid result
        valid_metric (str, optional): the selected metric in valid result for valid score

    Returns:
        float: valid score
    """
    if valid_metric:
        return valid_result[valid_metric]
    else:
        return valid_result['Recall@10']


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ' : ' + str(value) + '    '
    return result_str


def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'



# 绘制图像
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from pylab import *
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# 获取正负样本的相似度
def get_sim(sequence_output_0,aug_0,aug_1):
    import torch

    # 点积
    aug_0_1=torch.concat((aug_0,aug_1),dim=0)
    sim_0_1 = torch.matmul(sequence_output_0, aug_0_1.t())
    sim_o = torch.matmul(sequence_output_0, sequence_output_0.t())
    sim_o = torch.diag(sim_o)
    sim_0 = []
    sim_1 = []
    for i in range(256):
        # print(sim_0_1[i, i], sim_0_1[i, i + 256])
        temp_0 = sim_0_1[i, i].detach().cpu().tolist()
        temp_1 = sim_0_1[i, i + 256].detach().cpu().tolist()
        sim_0.append(temp_0)
        sim_1.append(temp_1)
    sim_o = sim_o.detach().cpu().tolist()
    # 余弦相似度
    import torch.nn as nn
    sim_o=nn.CosineSimilarity()(sequence_output_0,sequence_output_0).detach().cpu().tolist()
    sim_0=nn.CosineSimilarity()(sequence_output_0,aug_0).detach().cpu().tolist()
    sim_1=nn.CosineSimilarity()(sequence_output_0,aug_1).detach().cpu().tolist()



    # print(sim_0, sim_1, len(sim_0), sim_o)
    # 最大最小化
    ma = max(max(sim_o), max(sim_0), max(sim_1))
    mi = min(min(sim_o), min(sim_0), min(sim_1))
    for i in range(256):
        sim_0[i] = min_max(sim_0[i], ma, mi)
        sim_1[i] = min_max(sim_1[i], ma, mi)


    # 存放字典
    dic_data = {0: 0, 1: 0, 2: 0, 3: 0}
    dic_data = get_dict(dic_data, sim_0)
    dic_data = get_dict(dic_data, sim_1)
    return dic_data



def min_max(a,ma,mi):
    return (a-mi)/(ma-mi)

def get_dict(dic,a):
    for d in a:
        if d<=0.25:
            dic[0]+=1
        elif d<=0.5 and d>0.25:
            dic[1]+=1
        elif d<=0.75 and d>0.5:
            dic[2]+=1
        else:
            dic[3]+=1
    return dic












