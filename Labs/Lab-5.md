The following is the code for lab-5:
```
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNN(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        ks = 3
        pd = 1

        self.cv1 = nn.Conv2d(3, 32, kernel_size=ks, padding=pd)
        self.cv2 = nn.Conv2d(32, 64, kernel_size=ks, padding=pd)
        self.cv3 = nn.Conv2d(64, 128, kernel_size=ks, padding=pd)
        self.cv4 = nn.Conv2d(128, 256, kernel_size=ks, padding=pd)
        self.fc1 = nn.Linear(256*2*2, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.cv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.cv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.cv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.cv4(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x
```

Running through the model script, we have a [[Kernel Size]] of 3. One for each RGB channel, and [[Padding]] of 1 or same to maintain the same [[Spatial Dimension]]s.

The first [[Convolution Layer]] takes the input channels and maps them to 32 [[Feature Map]]s. That is, if you think of the spatial dimensions as length and width and the RGB channels as depth, each channel R, G and B can be seen as slices of the overall image each describing a particular set of patterns. We take those features and combine them in different ways to further expand on them to create more feature maps. In consequent layers, we can extract more features out of the feature maps we produce. This can be used to learn patterns like colours, contrasts, edges, shapes and other abstract concepts.

RELU activation is applied to introduce non-linearity to the model. It can help to classify more complex interweavings of data.

Max pooling reduces the spatial dimensions of the [[Feature Map]]s by passing a filter over them and taking the max value over the area of the filter. Looking at the 3D analogy, this is akin to reducing the area after increasing the depth. This reduces the amount of pixels carried across while maintaining the important features, making the later layers more efficient. We start with an image that is 32 x 32 pixels in size and apply [[Max Pooling]] 4 times with a filter size that is 2 x 2 and a stride of 2. (Stride must be at least the same size as the filter dimensions to prevent overlap).

* $32 \times 32$
* $16 \times 16$
* $8 \times 8$
* $4 \times 4$
* $2 \times 2$

This leaves the spatial dimensions as 2 x 2.

So, the [[Convolution Layer]]s increased the depth from 3 to 256 and the[[ Max Pooling ]] reduced the length and width down to 2 x 2 meaning our [[Dense Layer]] needs 2 x 2 x 256 inputs.

```
x = x.view(x.size(0), -1)
```

This function before the [[Dense Layer]] is just another way to apply the [[Flatten]] function. View reshapes the tensor without affecting the data it contains and x.size(0) takes the [[Batch Size]] we defined in the [[Hyperparameter]]s. the -1 dynamically takes the dimension but we have already calculated it to be 256 x 2 x 2. This just makes it easier if we change the parameters.

[[Dropout]] is added to randomly drop a percentage of the neurons to stop the model from [[Overfitting]]. It is a form of added noise. We had a dropout of 0.5 or 50%.

Moving on to the [[Hyperparameter]]s, I set the training to last for 70 [[Epoch]]s.  Any more than that and the model would just plateau and there would be no point.

The [[Batch Size]] was 64, not too big, not too small. Larger batch sizes would train the data faster but would fail to generalise as well. I didn't mind waiting a bit longer.

[[Learning Rate]] was set at 0.05. Again, a larger learning rate would train the model faster, but too large and you start to get oscillations in the training. This is small enough to be bearable.

[[Momentum]] adds a bit of memory to the [[Gradient]] descent so that it can more easily navigate out of local minima. I had no issues with this so I kept it as the standard 0.9 value. A low momentum leads to slow training while a high momentum can cause oscillations like the [[Learning Rate]].

The [[Loss Function]] used was [[Cross Entropy]] as this is suited for multiclass regression and was combined with [[Softmax]].

The optimiser was stochastic gradient descent or [[SGD]] which gives the best combination of parameters to the support vector machine [[SVM]] algorithm.

[[Weight Decay]] is a form of [[Regularisation]]  also known as L2 Regularisation which adds a penalty to large weights to prevent relying on one too heavily. It is a method to prevent [[Overfitting]].

I didn't use any [[Batch Normalisation]] in this because I was having issues with it, but normally it can be used to reduce exploding / vanishing gradients and can allow you to use a higher learning rate.