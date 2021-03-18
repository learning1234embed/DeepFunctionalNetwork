# [IPSN 2021] Deep Functional Network (DFN): Functional Interpretation of Deep Neural Network for Embedded Systems


## Introduction
This Git repository provides the ***Deep Functional Network (DFN) Framework***, which is an open-source code of [IPSN 2021](https://ipsn.acm.org/2021/) submission titled "***Deep Functional Network (DFN): Semantic Interpretation of Deep Neural Network for Intelligent Sensing Systems***". The DFN translates a black-box deep neural network (DNN) into a human-understandable computer program consisting of functions written in high-level programming languages (Python here), which can be optimized to run on resource-constrained embedded systems.

This repository provides the proposed DFN framework (Python/TensorFlow) and shows an example of DNN interpretation for the real system that is implemented in the paper, i.e., obstacle detection (avoidance) on a mobile robot. The target DNN to be interpreted is [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) trained with the [obstacle-avoidance dataset](https://github.com/varunverlencar/Dynamic-Obstacle-Avoidance-DL). The following steps demonstrate the procedure for the interpretation of AlexNet into an explainable DFN, which consists of 1) function estimation, 2) DFN network formation, and 3) interpretability optimization. For the reviewers' convenience, we provide three Python scripts (**function_estimator.py**, **network_formation.py**, **optimization.py**) that automatically performs each step with simple command-line options. This repository showcases the first two steps of DFN: 1) function estimation and 2) DFN network formation.


## Software Install and Code Cloning
The DFN framework is implemented based on Python and TensorFlow with a GPU. We used Tensorflow 1.13.1 and Python 2.7.

**Step 1.** Install [Python (>= 2.7)](https://www.python.org/downloads/).

**Step 2.** Install [Tensorflow >= 1.13.1)](https://www.tensorflow.org/).

**Step 3.** Clone this DeepFunctionalNetwork repository.
```sh
> git clone https://github.com/dfn-sensys2020/DeepFunctionalNetwork.git
Cloning into 'DeepFunctionalNetwork'...
remote: Enumerating objects: 3, done.
remote: Counting objects: 100% (3/3), done.
remote: Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
Unpacking objects: 100% (3/3), done.
```

## Step 0) Download Dataset and AlexNet Model (Preliminary)

As a preliminary step before getting into the interpretation of AlexNet, we first download the obstacle dataset and AlexNet model. Then, the output of AlexNet is obtained by running it with the train data, which will be used at the network formation step. The dataset (.npy files) and AlexNet model (a .pb file) will be downloaded in the 'obstacle/' folder.

1) Download the obstacle dataset by running the script (**download_dataset.sh**) as follows.
```sh
DeepFunctionalNetwork> ./download_dataset.sh
Downloading the obstacle dataset...
Downloading the test data... obstacle/obstacle_test_data.npy
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   408    0   408    0     0    218      0 --:--:--  0:00:01 --:--:--   217
  0     0    0     0    0     0      0      0 --:--:--  0:00:03 --:--:--     0
  0     0    0     0    0     0      0      0 --:--:--  0:00:03 --:--:--     0
100 78.4M    0 78.4M    0     0  17.3M      0 --:--:--  0:00:04 --:--:-- 92.6M
Downloading the test label... obstacle/obstacle_test_label.npy
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   408    0   408    0     0   2503      0 --:--:-- --:--:-- --:--:--  2503
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100 91488  100 91488    0     0   182k      0 --:--:-- --:--:-- --:--:--  182k
Downloading the train data... obstacle/obstacle_train_data.npy
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   408    0   408    0     0   3517      0 --:--:-- --:--:-- --:--:--  3487
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100  156M    0  156M    0     0  75.6M      0 --:--:--  0:00:02 --:--:--  111M
Downloading the train label... obstacle/obstacle_train_label.npy
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   408    0   408    0     0   2428      0 --:--:-- --:--:-- --:--:--  2428
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100  178k  100  178k    0     0   333k      0 --:--:-- --:--:-- --:--:--  333k

```

2) Download AlexNet model by running the script (**download_alexnet.sh**) as follows.
```sh
DeepFunctionalNetwork> ./download_alexnet.sh
Downloading AlexNet...
Downloading the model... obstacle/alexnet.pb
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   408    0   408    0     0   2873      0 --:--:-- --:--:-- --:--:--  2873
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100  130M    0  130M    0     0  70.9M      0 --:--:--  0:00:01 --:--:--  111M
Downloading the execution code... obstacle/execute_alexnet.py
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   408    0   408    0     0   2684      0 --:--:-- --:--:-- --:--:--  2684
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100  2317  100  2317    0     0   5092      0 --:--:-- --:--:-- --:--:--  5092
```

3) Run AlexNet with the train data to obtain its output by 1) moving to the obstacle folder, and 2) running the Python script (**execute_alexnet.py**) as follows. The output file will be genearted as 'alexnet.npy'.
```sh
DeepFunctionalNetwork> cd obstacle/
DeepFunctionalNetwork/obstacle> python execute_alexnet.py -frozen_model_filename=alexnet.pb -input_data=obstacle_train_data.npy
input_feed_data.shape: (5716, 45, 80, 1)
2020-06-22 15:10:56.566681: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-22 15:10:56.680565: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:998] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-06-22 15:10:56.681055: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x55fdc79d4c70 executing computations on platform CUDA. Devices:
2020-06-22 15:10:56.681068: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): GeForce RTX 2080 Ti, Compute Capability 7.5
2020-06-22 15:10:56.701485: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600000000 Hz
2020-06-22 15:10:56.703232: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x55fdc7a29620 executing computations on platform Host. Devices:
2020-06-22 15:10:56.703276: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
2020-06-22 15:10:56.703491: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties:
name: GeForce RTX 2080 Ti major: 7 minor: 5 memoryClockRate(GHz): 1.545
pciBusID: 0000:01:00.0
totalMemory: 10.76GiB freeMemory: 10.56GiB
2020-06-22 15:10:56.703522: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2020-06-22 15:10:56.704649: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-06-22 15:10:56.704671: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0
2020-06-22 15:10:56.704681: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N
2020-06-22 15:10:56.704785: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10271 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2080 Ti, pci bus id: 0000:01:00.0, compute capability: 7.5)
2020-06-22 15:10:58.024368: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library libcublas.so.10.0 locally
2020-06-22 15:10:58.152994: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library libcupti.so.10.0 locally
output.shape: (5716, 4)
[[7.2145015e-01 5.4784544e-04 2.7639109e-01 1.6109460e-03]
 [9.1262382e-01 3.2675141e-04 8.6310193e-02 7.3924597e-04]
 [9.8687279e-01 8.4636667e-05 1.2785754e-02 2.5684928e-04]
 ...
 [5.5088266e-03 2.2196768e-05 6.9510228e-05 9.9439949e-01]
 [1.0210629e-02 4.0019957e-05 1.0221004e-04 9.8964721e-01]
 [1.0210629e-02 4.0019957e-05 1.0221004e-04 9.8964721e-01]]
```

## Step 1) Function Estimation (function_estimator.py)

The first step of generating a DFN from the target DNN is to estimate the function distribution in the expected solution DFN, which will be used at the network formation step (Step2). The function distribution is estimated by Function Distribution Estimator (FDE)---a special-designed independent DNN that infers the distribution of functions from the end-to-end input/output pair of the target DNN.

1) Create a new function estimator by running the script (**function_estimator.py**) as follows.
```sh
DeepFunctionalNetwork/obstacle> cd ..
DeepFunctionalNetwork> python function_estimator.py -dnn_name=obstacle -mode=c -input_size=3600 -output_size=4 -placeholder_size=6
neuron: 1 functions: ['neuron']
signal_proc: 12 functions: ['irfft', 'max_pool', 'irfft2d', 'conv1d', 'idct', 'rfft', 'stft', 'depthwise_conv2d', 'avg_pool', 'dct', 'rfft2d', 'conv2d']
image_proc: 14 functions: ['flip_h', 'g_to_rgb', 'rot180', 'rot90', 'rot270', 'img_hgrad', 'flip_v', 'erosion2d', 'rgb_to_g', 'sobel_edges', 'img_transpose', 'total_variation', 'dilation2d', 'img_vgrad']
math: 22 functions: ['reciproc', 'divide', 'square', 'cumsum', 'min', 'neg', 'sum', 'add', 'sqrt', 'lgamma', 'bessel_i1e', 'abs', 'max', 'ceil', 'multiply', 'subtract', 'floor', 'cumprod', 'bessel_i0e', 'lbeta', 'digamma', 'round']
statistics: 6 functions: ['std', 'moments', 'erf', 'erfc', 'variance', 'mean']
linalg: 6 functions: ['diag_part', 'trace', 'transpose', 'cross', 'norm', 'dot']
activation: 6 functions: ['tanh', 'sigmoid', 'softsign', 'relu', 'softplus', 'leaky_relu']
trigonometric: 3 functions: ['cos', 'sin', 'tan']
[c] creating a function estimator
total 69 functions in function pool
input_emb Tensor("Add:0", shape=(?, 512), dtype=float32)
output_emb Tensor("Add_1:0", shape=(?, 512), dtype=float32)
Tensor("neuron_0:0", shape=(?, 1024), dtype=float32)
self.num_of_neuron_per_layer[layer_no] [256]
fc_parameter {'weights': <tf.Variable 'weight_0:0' shape=(1024, 256) dtype=float32_ref>, 'biases': <tf.Variable 'bias_0:0' shape=(256,) dtype=float32_ref>}
Tensor("neuron_1:0", shape=(?, 256), dtype=float32)
self.num_of_neuron_per_layer[layer_no] [256]
fc_parameter {'weights': <tf.Variable 'weight_1:0' shape=(256, 256) dtype=float32_ref>, 'biases': <tf.Variable 'bias_1:0' shape=(256,) dtype=float32_ref>}
Tensor("neuron_2:0", shape=(?, 256), dtype=float32)
self.num_of_neuron_per_layer[layer_no] [256]
fc_parameter {'weights': <tf.Variable 'weight_2:0' shape=(256, 256) dtype=float32_ref>, 'biases': <tf.Variable 'bias_2:0' shape=(256,) dtype=float32_ref>}
Tensor("neuron_3:0", shape=(?, 256), dtype=float32)
self.num_of_neuron_per_layer[layer_no] [69]
fc_parameter {'weights': <tf.Variable 'weight_3:0' shape=(256, 69) dtype=float32_ref>, 'biases': <tf.Variable 'bias_3:0' shape=(69,) dtype=float32_ref>}
Tensor("output:0", shape=(?, 69), dtype=float32)
Tensor("neuron_4:0", shape=(?, 69), dtype=float32)
```

2) Generate traing dataset for the function estimator by running the script (**function_estimator.py**) as follows.
```sh
DeepFunctionalNetwork> python function_estimator.py -dnn_name=obstacle -mode=g
neuron: 1 functions: ['neuron']
signal_proc: 12 functions: ['irfft', 'max_pool', 'irfft2d', 'conv1d', 'idct', 'rfft', 'stft', 'depthwise_conv2d', 'avg_pool', 'dct', 'rfft2d', 'conv2d']
image_proc: 14 functions: ['flip_h', 'g_to_rgb', 'rot180', 'rot90', 'rot270', 'img_hgrad', 'flip_v', 'erosion2d', 'rgb_to_g', 'sobel_edges', 'img_transpose', 'total_variation', 'dilation2d', 'img_vgrad']
math: 22 functions: ['reciproc', 'divide', 'square', 'cumsum', 'min', 'neg', 'sum', 'add', 'sqrt', 'lgamma', 'bessel_i1e', 'abs', 'max', 'ceil', 'multiply', 'subtract', 'floor', 'cumprod', 'bessel_i0e', 'lbeta', 'digamma', 'round']
statistics: 6 functions: ['std', 'moments', 'erf', 'erfc', 'variance', 'mean']
linalg: 6 functions: ['diag_part', 'trace', 'transpose', 'cross', 'norm', 'dot']
activation: 6 functions: ['tanh', 'sigmoid', 'softsign', 'relu', 'softplus', 'leaky_relu']
trigonometric: 3 functions: ['cos', 'sin', 'tan']
[g] generating training data for the function estimator
total 69 functions in function pool
generating 100 program with input size 1000 and placholder size 6
0
10
20
30
40
50
60
70
80
90
99
train_data (100000, 3604)
train_label (100000, 69)
generating 10 program with input size 100 and placholder size 6
0
9
test_data (1000, 3604)
test_label (1000, 69)
generating 10 program with input size 100 and placholder size 6
0
9
val_data (1000, 3604)
val_label (1000, 69)
```

3) Train the function estimator with the training dataset by running the script (**function_estimator.py**) as follows.
```sh
DeepFunctionalNetwork> python function_estimator.py -dnn_name=obstacle -mode=t
neuron: 1 functions: ['neuron']
signal_proc: 12 functions: ['irfft', 'max_pool', 'irfft2d', 'conv1d', 'idct', 'rfft', 'stft', 'depthwise_conv2d', 'avg_pool', 'dct', 'rfft2d', 'conv2d']
image_proc: 14 functions: ['flip_h', 'g_to_rgb', 'rot180', 'rot90', 'rot270', 'img_hgrad', 'flip_v', 'erosion2d', 'rgb_to_g', 'sobel_edges', 'img_transpose', 'total_variation', 'dilation2d', 'img_vgrad']
math: 22 functions: ['reciproc', 'divide', 'square', 'cumsum', 'min', 'neg', 'sum', 'add', 'sqrt', 'lgamma', 'bessel_i1e', 'abs', 'max', 'ceil', 'multiply', 'subtract', 'floor', 'cumprod', 'bessel_i0e', 'lbeta', 'digamma', 'round']
statistics: 6 functions: ['std', 'moments', 'erf', 'erfc', 'variance', 'mean']
linalg: 6 functions: ['diag_part', 'trace', 'transpose', 'cross', 'norm', 'dot']
activation: 6 functions: ['tanh', 'sigmoid', 'softsign', 'relu', 'softplus', 'leaky_relu']
trigonometric: 3 functions: ['cos', 'sin', 'tan']
[t] training the function estimator
[t] data: train/val.shape: (100000, 3604) (1000, 3604)
train
doTrain
step 0, training diff: 0.459302 ce: 48.328934 mse: 0.239929
step 0, Validation diff: 0.456995 ce: 48.448208 mse: 0.239013
step 100, training diff: 0.025043 ce: 4.630960 mse: 0.004455
step 100, Validation diff: 0.026652 ce: 5.480234 mse: 0.003574
step 200, training diff: 0.025779 ce: 4.834342 mse: 0.005429
step 200, Validation diff: 0.026692 ce: 5.524333 mse: 0.003586
step 300, training diff: 0.025469 ce: 4.728193 mse: 0.004979
step 300, Validation diff: 0.026701 ce: 5.563686 mse: 0.003597
...
...
step 3800, training diff: 0.024972 ce: 4.645852 mse: 0.004203
step 3800, Validation diff: 0.026655 ce: 5.705174 mse: 0.003597
step 3900, training diff: 0.024580 ce: 4.585587 mse: 0.004442
step 3900, Validation diff: 0.026461 ce: 5.742161 mse: 0.003608
step 3999, training diff: 0.025387 ce: 4.779102 mse: 0.005105
step 3999, Validation diff: 0.026408 ce: 5.697950 mse: 0.003592
```

4) Estimate the function distirubtion of AlexNet by feeding the output of AlexNetthe to the trained function estimator by running the script (**function_estimator.py**) as follows.
```sh
DeepFunctionalNetwork> python function_estimator.py -dnn_name=obstacle -mode=e -input_data=obstacle/obstacle_train_data.npy -output_data=obstacle/alexnet.npy
neuron: 1 functions: ['neuron']
signal_proc: 12 functions: ['irfft', 'max_pool', 'irfft2d', 'conv1d', 'idct', 'rfft', 'stft', 'depthwise_conv2d', 'avg_pool', 'dct', 'rfft2d', 'conv2d']
image_proc: 14 functions: ['flip_h', 'g_to_rgb', 'rot180', 'rot90', 'rot270', 'img_hgrad', 'flip_v', 'erosion2d', 'rgb_to_g', 'sobel_edges', 'img_transpose', 'total_variation', 'dilation2d', 'img_vgrad']
math: 22 functions: ['reciproc', 'divide', 'square', 'cumsum', 'min', 'neg', 'sum', 'add', 'sqrt', 'lgamma', 'bessel_i1e', 'abs', 'max', 'ceil', 'multiply', 'subtract', 'floor', 'cumprod', 'bessel_i0e', 'lbeta', 'digamma', 'round']
statistics: 6 functions: ['std', 'moments', 'erf', 'erfc', 'variance', 'mean']
linalg: 6 functions: ['diag_part', 'trace', 'transpose', 'cross', 'norm', 'dot']
activation: 6 functions: ['tanh', 'sigmoid', 'softsign', 'relu', 'softplus', 'leaky_relu']
trigonometric: 3 functions: ['cos', 'sin', 'tan']
[e] executing the function estimator
input_data (5716, 45, 80)
output_data (5716, 4)
infer
function estimation completed
```

## Step 2) Network Formation (network_formation.py)
The second step of generating a DFN is to form a network structure of the solution DFN, given the function distribution obtained at the function estimation step. The network formation consists of two steps: 1) network structure search, and 2) edge weight search.

1) Create a new DFN that will interpret AlexNet by running the script (**network_formation.py**) as follows.
```sh
DeepFunctionalNetwork> python network_formation.py -dnn_name=obstacle -dfn_name=dfn_1 -mode=c -placeholder_size=6 -population_size=32 -use_distribution=1 -input_size=3600 -output_size=4
neuron: 1 functions: ['neuron']
signal_proc: 12 functions: ['irfft', 'max_pool', 'irfft2d', 'conv1d', 'idct', 'rfft', 'stft', 'depthwise_conv2d', 'avg_pool', 'dct', 'rfft2d', 'conv2d']
image_proc: 14 functions: ['flip_h', 'g_to_rgb', 'rot180', 'rot90', 'rot270', 'img_hgrad', 'flip_v', 'erosion2d', 'rgb_to_g', 'sobel_edges', 'img_transpose', 'total_variation', 'dilation2d', 'img_vgrad']
math: 22 functions: ['reciproc', 'divide', 'square', 'cumsum', 'min', 'neg', 'sum', 'add', 'sqrt', 'lgamma', 'bessel_i1e', 'abs', 'max', 'ceil', 'multiply', 'subtract', 'floor', 'cumprod', 'bessel_i0e', 'lbeta', 'digamma', 'round']
statistics: 6 functions: ['std', 'moments', 'erf', 'erfc', 'variance', 'mean']
linalg: 6 functions: ['diag_part', 'trace', 'transpose', 'cross', 'norm', 'dot']
activation: 6 functions: ['tanh', 'sigmoid', 'softsign', 'relu', 'softplus', 'leaky_relu']
trigonometric: 3 functions: ['cos', 'sin', 'tan']
dfn_file_path /.../.../DeepFunctionalNetwork/obstacle/dfn_1/dfn_1
[c] creating a deep functional network
total 69 functions in function pool
fe_file_path /.../.../DeepFunctionalNetwork/obstacle/function_estimator/function_estimator.obj
generating 70 primitive functions
generating a dfn
generating population of size 32
```

2) Perform the network structure search by running the script (**network_formation.py**) as follows. This step would take one or two days.
```sh
DeepFunctionalNetwork> python network_formation.py -dnn_name=obstacle -dfn_name=dfn_1 -mode=s -input_data=obstacle/obstacle_train_data.npy -output_data=obstacle/alexnet.npy
neuron: 1 functions: ['neuron']
signal_proc: 12 functions: ['irfft', 'max_pool', 'irfft2d', 'conv1d', 'idct', 'rfft', 'stft', 'depthwise_conv2d', 'avg_pool', 'dct', 'rfft2d', 'conv2d']
image_proc: 14 functions: ['flip_h', 'g_to_rgb', 'rot180', 'rot90', 'rot270', 'img_hgrad', 'flip_v', 'erosion2d', 'rgb_to_g', 'sobel_edges', 'img_transpose', 'total_variation', 'dilation2d', 'img_vgrad']
math: 22 functions: ['reciproc', 'divide', 'square', 'cumsum', 'min', 'neg', 'sum', 'add', 'sqrt', 'lgamma', 'bessel_i1e', 'abs', 'max', 'ceil', 'multiply', 'subtract', 'floor', 'cumprod', 'bessel_i0e', 'lbeta', 'digamma', 'round']
statistics: 6 functions: ['std', 'moments', 'erf', 'erfc', 'variance', 'mean']
linalg: 6 functions: ['diag_part', 'trace', 'transpose', 'cross', 'norm', 'dot']
activation: 6 functions: ['tanh', 'sigmoid', 'softsign', 'relu', 'softplus', 'leaky_relu']
trigonometric: 3 functions: ['cos', 'sin', 'tan']
dfn_file_path /.../.../DeepFunctionalNetwork/obstacle/dfn_1/dfn_1
[s] searching the dfn architecture
input_data (5716, 45, 80)
output_data (5716, 4)
[generation] 0
[0][0] best_fitnes: [0.16726619005203247, 0.16726619, -1]
[0][1] best_fitnes: [0.1988099068403244, 0.1988099, -1]
[0][2] best_fitnes: [0.2587803602218628, 0.25878036, -1]
[0][3] best_fitnes: [0.23742617666721344, 0.23742618, -1]
...
...
```

3) Perform the edge weight search by running the script (**network_formation.py**) as follows.
```sh
DeepFunctionalNetwork> python network_formation.py -dnn_name=obstacle -dfn_name=dfn_1 -mode=u -input_data=obstacle/obstacle_train_data.npy -output_data=obstacle/alexnet.npy -save=1
neuron: 1 functions: ['neuron']
signal_proc: 12 functions: ['irfft', 'max_pool', 'irfft2d', 'conv1d', 'idct', 'rfft', 'stft', 'depthwise_conv2d', 'avg_pool', 'dct', 'rfft2d', 'conv2d']
image_proc: 14 functions: ['flip_h', 'g_to_rgb', 'rot180', 'rot90', 'rot270', 'img_hgrad', 'flip_v', 'erosion2d', 'rgb_to_g', 'sobel_edges', 'img_transpose', 'total_variation', 'dilation2d', 'img_vgrad']
math: 22 functions: ['reciproc', 'divide', 'square', 'cumsum', 'min', 'neg', 'sum', 'add', 'sqrt', 'lgamma', 'bessel_i1e', 'abs', 'max', 'ceil', 'multiply', 'subtract', 'floor', 'cumprod', 'bessel_i0e', 'lbeta', 'digamma', 'round']
statistics: 6 functions: ['std', 'moments', 'erf', 'erfc', 'variance', 'mean']
linalg: 6 functions: ['diag_part', 'trace', 'transpose', 'cross', 'norm', 'dot']
activation: 6 functions: ['tanh', 'sigmoid', 'softsign', 'relu', 'softplus', 'leaky_relu']
trigonometric: 3 functions: ['cos', 'sin', 'tan']
dfn_file_path /.../.../DeepFunctionalNetwork/obstacle/dfn_1/dfn_1
[u] update weight of the population[0]
input_data (5716, 45, 80)
output_data (5716, 4)
step 0, validation fitness: 0.043049
2020-06-25 21:31:42.367130: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:704] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-06-25 21:31:42.375430: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:704] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-06-25 21:31:42.433947: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:704] Iteration = 0, topological sort failed with message: The graph couldn't be sorted in topological order.
2020-06-25 21:31:42.439563: E tensorflow/core/grappler/optimizers/dependency_optimizer.cc:704] Iteration = 1, topological sort failed with message: The graph couldn't be sorted in topological order.
step 100, validation fitness: 0.034216
save weight for 0.034215790770317056
step 200, validation fitness: 0.029897
save weight for 0.02989715289981466
step 300, validation fitness: 0.027461
save weight for 0.0274610956296575
...
...
```

4) Evaluate the inference accuracy of the interpreted DFN by running the script (**network_formation.py**) as follows, which is **0.9800350262697023**.

```sh
DeepFunctionalNetwork> python network_formation.py -dnn_name=obstacle -dfn_name=dfn_1 -mode=e input_data=obstacle/obstacle_test_data.npy -output_data=obstacle/obstacle_test_label.npy
neuron: 1 functions: ['neuron']
signal_proc: 12 functions: ['irfft', 'max_pool', 'irfft2d', 'conv1d', 'idct', 'rfft', 'stft', 'depthwise_conv2d', 'avg_pool', 'dct', 'rfft2d', 'conv2d']
image_proc: 14 functions: ['flip_h', 'g_to_rgb', 'rot180', 'rot90', 'rot270', 'img_hgrad', 'flip_v', 'erosion2d', 'rgb_to_g', 'sobel_edges', 'img_transpose', 'total_variation', 'dilation2d', 'img_vgrad']
math: 22 functions: ['reciproc', 'divide', 'square', 'cumsum', 'min', 'neg', 'sum', 'add', 'sqrt', 'lgamma', 'bessel_i1e', 'abs', 'max', 'ceil', 'multiply', 'subtract', 'floor', 'cumprod', 'bessel_i0e', 'lbeta', 'digamma', 'round']
statistics: 6 functions: ['std', 'moments', 'erf', 'erfc', 'variance', 'mean']
linalg: 6 functions: ['diag_part', 'trace', 'transpose', 'cross', 'norm', 'dot']
activation: 6 functions: ['tanh', 'sigmoid', 'softsign', 'relu', 'softplus', 'leaky_relu']
trigonometric: 3 functions: ['cos', 'sin', 'tan']
dfn_file_path /.../.../DeepFunctionalNetwork/obstacle/dfn_1/dfn_1
[e] executing the population[0]
input_data (2855, 45, 80)
output_data (2855, 4)
(2855,)
(2855,)
test accuracy 0.9800350262697023
```

5) Draw the DFN graph by running the script (**network_formation.py**) as follows. 
```sh
DeepFunctionalNetwork> python draw_graph.py -mode=d -dnn_name=obstacle -dfn_name=dfn_1
neuron: 1 functions: ['neuron']
signal_proc: 12 functions: ['irfft', 'max_pool', 'irfft2d', 'conv1d', 'idct', 'rfft', 'stft', 'depthwise_conv2d', 'avg_pool', 'dct', 'rfft2d', 'conv2d']
image_proc: 14 functions: ['flip_h', 'g_to_rgb', 'rot180', 'rot90', 'rot270', 'img_hgrad', 'flip_v', 'erosion2d', 'rgb_to_g', 'sobel_edges', 'img_transpose', 'total_variation', 'dilation2d', 'img_vgrad']
math: 22 functions: ['reciproc', 'divide', 'square', 'cumsum', 'min', 'neg', 'sum', 'add', 'sqrt', 'lgamma', 'bessel_i1e', 'abs', 'max', 'ceil', 'multiply', 'subtract', 'floor', 'cumprod', 'bessel_i0e', 'lbeta', 'digamma', 'round']
statistics: 6 functions: ['std', 'moments', 'erf', 'erfc', 'variance', 'mean']
linalg: 6 functions: ['diag_part', 'trace', 'transpose', 'cross', 'norm', 'dot']
activation: 6 functions: ['tanh', 'sigmoid', 'softsign', 'relu', 'softplus', 'leaky_relu']
trigonometric: 3 functions: ['cos', 'sin', 'tan']
dfn_file_path /.../.../DeepFunctionalNetwork/obstacle/dfn_1/dfn_1
[d] drawing a deep functional network
['max_pool:0', 'dct:1', 'erosion2d:0', 'stft:0', 'g_to_rgb:1', 'rgb_to_g:1', 'output', 'input']
```

* The DFN graph file is saved as 'obstacle_dfn_1_0.png', which is shown in the below.
![](/image/obstacle_dfn_1_0.png)
