import torch
import random
import numpy as np
from torch.utils.data.sampler import Sampler
import datetime
import torch.nn as nn
import os


def set_random_seed(seed=0): 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
#每个随机种子对应的是要应用的库上
#random.seed(seed)：设置 Python 内置 random 模块的随机种子。这通常用于 Python 标准库中的随机数生成函数
#np.random.seed(seed)：设置 NumPy 库的随机种子。NumPy 用于生成随机数的函数（如 numpy.random.rand、numpy.random.randint 等）会受到这个设置的影响。
#torch.manual_seed(seed)：设置 PyTorch 库中的随机种子。这确保了 PyTorch 在 CPU 上生成的随机数是确定性的。PyTorch 的许多操作（如初始化模型参数、数据采样等）依赖于随机数生成
#torch.cuda.manual_seed(seed)：设置 PyTorch 在 GPU 上生成的随机数的种子（如果可用）。这对于使用 GPU 的计算来说也非常重要，因为 GPU 的随机数生成和 CPU 是分开的。
#如果不设置 GPU 随机种子，即使在相同的代码和输入下，可能也会得到不同的结果

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # # 设置 Python 哈希种子，防止哈希值的随机性影响实验结果
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  ## 确保卷积操作的确定性

#设置环境变量 PYTHONHASHSEED。Python 中的哈希值在某些情况下会受到随机性影响（特别是在 Python 3.3 及以后版本）。通过设置这个环境变量，你可以确保哈希值的生成是确定性的，从而避免由于哈希随机性带来的不确定性。
#设置 PyTorch 的 cuDNN 后端为确定性模式。cuDNN 是 NVIDIA 提供的深度学习库，它包含许多优化的卷积操作。在确定性模式下，cuDNN 将使用可重复的算法，以确保在相同输入下，卷积操作的结果是确定性的。注意，这可能会影响性能，因为它可能会禁用一些优化。

# 在每个epoch开始的时候, 自定义从数据集中取样本的策略
class DTISampler(Sampler):  # 采样器
    def __init__(self, weights, num_samples, replacement=True):
        weights = np.array(weights) / np.sum(weights)  # 返回一个矩阵，np.sum(weights) = 2,  weights之和等于1 #将 weights 除以其总和，以使得归一化后的权重总和为 1。这样，每个权重可以被视为样本被选中的概率。
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement #存储是否允许重复采样的标志。

    def __iter__(self):
        # return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())
        retval = np.random.choice(len(self.weights), self.num_samples, replace=self.replacement, p=self.weights)
        #在len(self.weights)的样本数量中，依照概率p,采样self.num_samples 个数量）
        return iter(retval.tolist())

    def __len__(self):
        return self.num_samples


class EarlyStopping(object):#提前停止
    def __init__(self,  mode='higher', patience=15, filename=None, tolerance=0.0):
        if filename is None:
            dt = datetime.datetime.now()
            filename = './model_save/early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.tolerance = tolerance
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        # return (score > prev_best_score)
        return score / prev_best_score > 1 + self.tolerance #放回的是布尔值，判断当前评分是否优于之前的最佳评分。

    def _check_lower(self, score, prev_best_score):
        # return (score < prev_best_score)
        return prev_best_score / score > 1 + self.tolerance  #检查当前评分是否显著低于之前的最佳评分。

    #这个 step 方法是典型的早停（Early Stopping）策略的一部分。早停是一种用于防止模型过拟合的技术，在训练过程中监控模型的表现，避免其在训练集上过拟合。
    #当模型的表现没有显著提升时，训练可以提前停止。这个方法的作用是检查当前模型的表现，并决定是否更新最佳模型或停止训练。
#即这个函数的早停只发生在允许的步数内没有显著改进 时
    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: #self.patience 允许的最大步长，超过最大步长，模型还是没有改善，直接停止
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)  # 保存网络中的参数, 速度快，占空间少, 以字典格式存储
        #state_dict() 方法返回模型的状态字典，这个字典包含了模型所有的参数（权重和偏置）及其对应的值。它是保存和加载模型时最重要的部分，因为它包含了模型的学习到的所有信息。

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])
        #model.load_state_dict 是 PyTorch 中的一个方法，用于将状态字典（模型参数）加载到模型实例中。
        #这个方法会更新模型的参数，使其与保存的状态字典中的参数一致。（微调的时候类似）


class MyLoss(nn.Module):
    def __init__(self, alph):
        super(MyLoss, self).__init__()  
        self.alph = alph
# input 通常代表模型的预测值，而 target 代表真实的标签值
        sum_xy = torch.sum(torch.sum(input * target))
        sum_x = torch.sum(torch.sum(input))
        sum_y = torch.sum(torch.sum(target))
        sum_x2 = torch.sum(torch.sum(input * input))
        sum_y2 = torch.sum(torch.sum(target * target))
        n = input.size()[0]
        pcc = (n * sum_xy - sum_x * sum_y) / torch.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        return self.alph*(1-torch.abs(pcc)) + (1-self.alph)*torch.nn.functional.mse_loss(input, target)

#1 - torch.abs(pcc): 皮尔逊相关系数的绝对值被从 1 中减去，以使得皮尔逊相关系数的绝对值越大，损失越小。目的是鼓励 input 和 target 之间有更高的相关性。
#torch.nn.functional.mse_loss(input, target): 计算 input 和 target 之间的均方误差损失。
#self.alph*(1-torch.abs(pcc)) + (1-self.alph)*torch.nn.functional.mse_loss(input, target):
#self.alph 控制皮尔逊相关系数部分和均方误差部分的权重。
#损失函数综合了皮尔逊相关系数和均方误差两者，以此来优化模型的相关性和预测精度。
