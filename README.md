# CIFAR-10 Deep Learning

> Through Deep learning to achieve images classification

### Essential matters

```
MaxPooling2D, Dropout, Flatten, BatchNormalization
```

### Shape

```
train_data = np.zeros((50000, 32, 32, 3))
train_labels = np.zeros((50000,))

test_data = np.zeros((10000, 32, 32, 3))
test_labels = np.zeros((10000,))
```

### Accuracy

scale: 0.7008 - 0.7233 

![TrainingAndValidationAccuracy](https://github.com/ZhangYizhe/CIFAR-10-DeepLearning/blob/main/TrainingAndValidationAccuracy.png)

![TrainingAndValidationLoss](https://github.com/ZhangYizhe/CIFAR-10-DeepLearning/blob/main/TrainingAndValidationLoss.png)

### Process

First, you need to use CIFAR10-Py-ProcessData.py file to convert data from cifar-10-batches-py.

Then you can use CIFAR10-Py.py file to start learning.

### Dataset

https://www.cs.toronto.edu/~kriz/cifar.html

### Summary:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 30, 30, 32)        896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496     
_________________________________________________________________
dropout (Dropout)            (None, 13, 13, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 11, 11, 128)       73856     
_________________________________________________________________
dropout_1 (Dropout)          (None, 11, 11, 128)       0         
_________________________________________________________________
flatten (Flatten)            (None, 15488)             0         
_________________________________________________________________
batch_normalization (BatchNo (None, 15488)             61952     
_________________________________________________________________
dense (Dense)                (None, 256)               3965184   
_________________________________________________________________
dense_1 (Dense)              (None, 10)                2570      
=================================================================
Total params: 4,122,954
Trainable params: 4,091,978
Non-trainable params: 30,976
```

```
_________________________________________________________________
2021-03-01 16:07:52.769187: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
Epoch 1/50
196/196 [==============================] - 47s 236ms/step - loss: 1.7473 - acc: 0.3829 - val_loss: 2.1307 - val_acc: 0.3157
Epoch 2/50
196/196 [==============================] - 46s 234ms/step - loss: 1.1363 - acc: 0.5954 - val_loss: 1.8643 - val_acc: 0.5376
Epoch 3/50
196/196 [==============================] - 46s 233ms/step - loss: 0.9510 - acc: 0.6638 - val_loss: 1.1390 - val_acc: 0.6612
Epoch 4/50
196/196 [==============================] - 46s 233ms/step - loss: 0.8210 - acc: 0.7094 - val_loss: 0.9285 - val_acc: 0.6727
Epoch 5/50
196/196 [==============================] - 46s 233ms/step - loss: 0.7071 - acc: 0.7489 - val_loss: 0.8407 - val_acc: 0.7072
Epoch 6/50
196/196 [==============================] - 46s 233ms/step - loss: 0.6145 - acc: 0.7849 - val_loss: 0.9462 - val_acc: 0.6854
Epoch 7/50
196/196 [==============================] - 46s 232ms/step - loss: 0.5201 - acc: 0.8157 - val_loss: 0.8710 - val_acc: 0.7244
Epoch 8/50
196/196 [==============================] - 46s 233ms/step - loss: 0.4565 - acc: 0.8388 - val_loss: 0.8585 - val_acc: 0.7298
Epoch 9/50
196/196 [==============================] - 46s 233ms/step - loss: 0.3917 - acc: 0.8605 - val_loss: 0.9335 - val_acc: 0.7155
Epoch 10/50
196/196 [==============================] - 46s 233ms/step - loss: 0.3336 - acc: 0.8823 - val_loss: 1.0797 - val_acc: 0.6935
Epoch 11/50
196/196 [==============================] - 46s 233ms/step - loss: 0.3112 - acc: 0.8893 - val_loss: 1.1387 - val_acc: 0.6877
Epoch 12/50
196/196 [==============================] - 46s 232ms/step - loss: 0.2746 - acc: 0.9024 - val_loss: 1.0856 - val_acc: 0.7193
Epoch 13/50
196/196 [==============================] - 46s 233ms/step - loss: 0.2411 - acc: 0.9153 - val_loss: 1.0715 - val_acc: 0.7206
Epoch 14/50
196/196 [==============================] - 46s 233ms/step - loss: 0.2163 - acc: 0.9230 - val_loss: 1.1008 - val_acc: 0.7203
Epoch 15/50
196/196 [==============================] - 46s 233ms/step - loss: 0.2029 - acc: 0.9286 - val_loss: 1.1364 - val_acc: 0.7255
Epoch 16/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1855 - acc: 0.9344 - val_loss: 1.2187 - val_acc: 0.7120
Epoch 17/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1848 - acc: 0.9340 - val_loss: 1.2466 - val_acc: 0.7154
Epoch 18/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1650 - acc: 0.9413 - val_loss: 1.2132 - val_acc: 0.7262
Epoch 19/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1592 - acc: 0.9453 - val_loss: 1.4363 - val_acc: 0.7065
Epoch 20/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1510 - acc: 0.9484 - val_loss: 1.3741 - val_acc: 0.7070
Epoch 21/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1503 - acc: 0.9476 - val_loss: 1.4155 - val_acc: 0.7234
Epoch 22/50
196/196 [==============================] - 46s 232ms/step - loss: 0.1456 - acc: 0.9497 - val_loss: 1.3704 - val_acc: 0.7107
Epoch 23/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1346 - acc: 0.9537 - val_loss: 1.4357 - val_acc: 0.7208
Epoch 24/50
196/196 [==============================] - 46s 232ms/step - loss: 0.1487 - acc: 0.9492 - val_loss: 1.3794 - val_acc: 0.7183
Epoch 25/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1320 - acc: 0.9537 - val_loss: 1.4372 - val_acc: 0.7098
Epoch 26/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1272 - acc: 0.9572 - val_loss: 1.6112 - val_acc: 0.7085
Epoch 27/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1246 - acc: 0.9570 - val_loss: 1.4745 - val_acc: 0.7128
Epoch 28/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1232 - acc: 0.9590 - val_loss: 1.6015 - val_acc: 0.7046
Epoch 29/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1169 - acc: 0.9595 - val_loss: 1.5039 - val_acc: 0.7178
Epoch 30/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1127 - acc: 0.9619 - val_loss: 1.5798 - val_acc: 0.7141
Epoch 31/50
196/196 [==============================] - 46s 235ms/step - loss: 0.1184 - acc: 0.9603 - val_loss: 1.6177 - val_acc: 0.7182
Epoch 32/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1204 - acc: 0.9586 - val_loss: 1.5789 - val_acc: 0.7213
Epoch 33/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1020 - acc: 0.9645 - val_loss: 1.5500 - val_acc: 0.7121
Epoch 34/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1019 - acc: 0.9654 - val_loss: 1.7174 - val_acc: 0.7153
Epoch 35/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1075 - acc: 0.9645 - val_loss: 1.5246 - val_acc: 0.7178
Epoch 36/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1126 - acc: 0.9630 - val_loss: 1.5152 - val_acc: 0.7237
Epoch 37/50
196/196 [==============================] - 46s 233ms/step - loss: 0.0979 - acc: 0.9665 - val_loss: 1.7075 - val_acc: 0.7186
Epoch 38/50
196/196 [==============================] - 46s 233ms/step - loss: 0.0986 - acc: 0.9677 - val_loss: 1.5515 - val_acc: 0.7265
Epoch 39/50
196/196 [==============================] - 46s 233ms/step - loss: 0.0993 - acc: 0.9671 - val_loss: 1.6099 - val_acc: 0.7146
Epoch 40/50
196/196 [==============================] - 46s 233ms/step - loss: 0.0998 - acc: 0.9671 - val_loss: 1.6778 - val_acc: 0.7136
Epoch 41/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1066 - acc: 0.9667 - val_loss: 1.5866 - val_acc: 0.7157
Epoch 42/50
196/196 [==============================] - 46s 233ms/step - loss: 0.0932 - acc: 0.9690 - val_loss: 1.7256 - val_acc: 0.7192
Epoch 43/50
196/196 [==============================] - 46s 233ms/step - loss: 0.0964 - acc: 0.9683 - val_loss: 1.6212 - val_acc: 0.7215
Epoch 44/50
196/196 [==============================] - 46s 233ms/step - loss: 0.1018 - acc: 0.9669 - val_loss: 1.7039 - val_acc: 0.7113
Epoch 45/50
196/196 [==============================] - 46s 233ms/step - loss: 0.0945 - acc: 0.9702 - val_loss: 1.6666 - val_acc: 0.7233
Epoch 46/50
196/196 [==============================] - 46s 233ms/step - loss: 0.0872 - acc: 0.9707 - val_loss: 1.6739 - val_acc: 0.7184
Epoch 47/50
196/196 [==============================] - 46s 233ms/step - loss: 0.0886 - acc: 0.9715 - val_loss: 1.7019 - val_acc: 0.7250
Epoch 48/50
196/196 [==============================] - 46s 233ms/step - loss: 0.0950 - acc: 0.9696 - val_loss: 1.7507 - val_acc: 0.7209
Epoch 49/50
196/196 [==============================] - 47s 242ms/step - loss: 0.0916 - acc: 0.9689 - val_loss: 1.7249 - val_acc: 0.7121
Epoch 50/50
196/196 [==============================] - 47s 239ms/step - loss: 0.0810 - acc: 0.9735 - val_loss: 2.0224 - val_acc: 0.7008

```