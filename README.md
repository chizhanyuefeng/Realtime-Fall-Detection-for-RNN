# Fall Detection for RNN(AFD-RNN)

<p align="left">
<img src="https://github.com/chizhanyuefeng/Fall_Detection_for_RNN/blob/master/result/rnn.gif", width="720">
</p>

## 使用RNN来完成跌倒数据的分析和检测识别
由于传感器是按照时间序列进行获取的数据，所以本项目采用RNN来进行网络模型设计。
其中数据采集频率为50HZ（大于50HZ的数据，通过截取降为50HZ）。

## 训练和测试的数据集

暂时使用MobileFall的数据集合进行网络的训练和测试，来检测网络模型的优劣。
后期使用自己的数据进行完善网络模型。

## 检测识别种类
坐下、起立、站立、慢跑、走路、上楼梯、下楼梯、跌倒、跳、躺下等10种动作。

## 项目进展
构建完成RNN模型，正在训练调试模型中。