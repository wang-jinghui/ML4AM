# -*- coding: utf-8 -*-
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import learning_curve,GridSearchCV
from sklearn.model_selection import LeaveOneOut

'''
通过交叉验证选择带宽
在核密度估计（KDE）中，带宽的选择对于获得合适的密度估计至关重要，
它是控制密度估计中偏差与方差权衡的调节参数：
带宽过窄会导致估计的方差较高（即过拟合），
此时单个数据点的有无就会使估计结果产生很大差异。
带宽过宽会导致估计的偏差较高（即欠拟合），
此时数据中的结构会被过宽的核函数平滑掉。

在统计学中，基于对数据的较严格假设来快速估计最优带宽的方法
已有悠久的历史。例如，如果查看 SciPy 和 StatsModels 库中的
KDE 实现，就会发现其中有些实现
采用了这类规则。

在机器学习中，这类超参数调优通常
通过交叉验证进行经验性选择。考虑到这一点，Scikit-Learn 中的 KernelDensity 估计器
在设计上支持直接配合 Scikit-Learn 的标准
网格搜索工具使用。这里我们将使用 GridSearchCV
为前面的数据集优化带宽。由于数据集很小，
我们将采用留一法交叉验证，以尽可能减少
每次交叉验证时训练集样本数量的损失：

https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
'''

''' 这也是书中的练习 2.7：
“扩展代码片段 2.2 中的 fitKDE 函数，使其能够通过
交叉验证估计 bWidth（带宽）的最优值。”
'''

def findOptimalBWidth(eigenvalues):
    bandwidths = 10 ** np.linspace(-2, 1, 100)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=LeaveOneOut())
    grid.fit(eigenvalues[:, None]);
    
    #Now we can find the choice of bandwidth which maximizes the score (which in this case defaults to the log-likelihood):
    
    return grid.best_params_