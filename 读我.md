# Real-time Fall Detection for RNN(AFD-RNN)

<p align="left">
<img src="https://github.com/chizhanyuefeng/Fall_Detection_for_RNN/blob/master/result/rnn.gif", width="720">
</p>

结果图说明：

- 图中红、绿、蓝线，分别代表加速度传感器的x、y、z轴数据。
- correct为label，predict为rnn预测值
- Fall1、Fall2、Fall3、Fall4分别代表4种跌倒（前向跌倒、侧向跌倒、后向跌倒、膝盖着地的前向跌倒）


## 环境配置
- TensorFlow >= 1.4
- python3
- matplotlib

## 使用RNN来完成跌倒数据的分析和检测识别
由于传感器是按照时间序列进行获取的数据，所以本项目采用RNN来进行网络模型设计。
其中数据采集频率为50HZ（大于50HZ的数据，通过截取降为50HZ）。

## 训练和测试的数据集

使用[MobileFall](http://www.bmi.teicrete.gr/index.php/research/mobiact)的数据集合进行网络的训练和测试，来检测网络模型的优劣。
准确率达到98.78%

## 检测识别种类
坐下、起立、站立、慢跑、走路、上楼梯、下楼梯、跌倒、跳、躺下等10种动作。

## 网络训练

### 1.训练数据要求
- 传感器采集频率50Hz
- 包含加速度传感器、陀螺仪传感器

### 2.训练前准备
将数据放到./dataset/train/中，进新kalman滤波


    python utils.py

### 3.网络训练和测试
    
    python train_rnn.py
    
## 4.测试数据
将测试数据同样进行kalman滤波后，放入./dataset/test/中


    python run_rnn.py