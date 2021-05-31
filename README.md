# Real-Time Digital Video Stabilization of Bioinspired Robotic Fish Using Estimation-and-Prediction Framework
## Prerequisites
- Linux
- Python 3.6
- NVIDIA GPU (12G or 24G memory) + CUDA cuDNN
- pytorch-gpu==1.4.0
- numpy
- ...

## Getting Started
### Testing
The proposed IOLK, IGLK, and ITLK methods can be implemented with following three commands. Note that the motion parameters of the robot should be adjusted according to the input video. 
```bash
python3 Method_IOLK.py;
python3 Method_IGLK.py;
python3 Method_ITLK.py;
```

### Training

DeepStab dataset (7.9GB) http://cg.cs.tsinghua.edu.cn/download/DeepStab.zip

It is tough to directly capture videos from abundant underwater scenes. Therefore, we modify the ground dataset 'DeepStab'  to train our TENet. 

The training data of LSTM-based predictor are provided in 'expCamera.csv'.

```bash
python3 TENet_train;
python3 LSTM_train;
```
