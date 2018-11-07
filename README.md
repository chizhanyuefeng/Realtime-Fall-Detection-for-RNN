# Real-time Fall Detection for RNN(AFD-RNN)

<p align="left">
<img src="https://github.com/chizhanyuefeng/Fall_Detection_for_RNN/blob/master/result/rnn.gif", width="720">
</p>

result picture illustrate：

- The red,green,blue lines is acceleration sensor's x,y,z data。
- In the picture ,"correct" is the ground truth,"predict" is AFD-RNN network predict data
- Fall1、Fall2、Fall3 and Fall4 are represent Forward-lying,Front-knees-lying,Back-sitting-chair,Sideward-lying 

## AFD-RNN using RNN 
The sensors(acceleration and gyroscope sensor) is realtime to collect data,so we using rnn to detect the people movement.

## Requirenment
- TensorFlow >= 1.4
- python3
- matplotlib

## Class
Sitting,standing,stand to sit,sit to stand,upstairs,downstairs,lying,jumping,joging,walking and fall.

## Train and test

### 1.Train data
- The data collect frequence is 50Hz
- Need acceleration and gyroscope sensor

### 2.Before training
Put the train data to ./dataset/train/,and use kalman filter  to handle the data.


    python utils.py

### 3.Training
    
    python train_rnn.py
    
## 4.Testing
Put the test data to ./dataset/test/,and use kalman filter  to handle the data.


    python run_rnn.py
    
## Dataset

We using public dataset [MobileFall](http://www.bmi.teicrete.gr/index.php/research/mobiact) to train and test our net.

I upload the dataset at [Baidu网盘](https://pan.baidu.com/s/1arZMNPs1GzWrQf4beJFCSQ),if you cant download from [MobileFall](http://www.bmi.teicrete.gr/index.php/research/mobiact),you can try this

The final accuracy is 98.78%
