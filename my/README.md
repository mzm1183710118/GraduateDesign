# 仅介绍各个文件/文件夹的作用
本毕设在linux环境下运行，具体实验环境见毕设论文
## data介绍
使用的数据集为公开数据集FI-2010，具体描述见论文Benchmark dataset for mid-price prediction of limit order book data with machine learning methods
### origin子文件夹
源数据
### processed子文件夹
将5支股票的数据进行切分，分别存在各自文件夹中
## attentionWeights文件夹
存储C3L-AED模型训练得到的tensorflow模型参数以及Attention分数（最后补充的）
## savedModels_attention文件夹
存储C3L-AED模型训练得到的tensorflow模型参数
## savedModels_rightOrder文件夹
存储C3L模型以及4个baseline在各种情况下训练得到的pytorch模型参数
存储的参数命名规则：stock1_MLP_40_10指使用stock1的数据，MLP模型，$x_t$使用40个特征，标签使用$y_{10}$
## utils文件夹
存储各种经过测试好的工具类
## 剩下的各种ipynb文件
建议使用vscode打开，里面都保存了注释、代码、运行结果、得到的各种图片