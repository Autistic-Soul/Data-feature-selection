# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 13:17:57 2018

@author: User
"""
#通过卡方检验选定数据特征
#卡方检验是统计样本的实际观测值与理论推断值之间的偏离程度，偏离程度决定了卡方值的大小
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#导入数据
a = 'data.csv'
b = ['1','2','3','4','5','6']
data = read_csv(a, names = b)
#将数据分为输入数据和输出结果
array = data.values
X = array[:, 0:5]
Y = array[:, 5]
#特征选定
test = SelectKBest(score_func = chi2,k = 4)
fit = test.fit(X,Y)
set_printoptions(precision = 3)
print(fit.scores_)
features = fit.transform(X)
print(features)