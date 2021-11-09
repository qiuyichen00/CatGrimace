import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
""" 
Information about architecture config:
Tuple is structured by and signifies a convolutional block (filters, kernel_size, stride) 
Every convolutional layer is a same convolution with kernal_size 3. 
List: ["B",#of Repeats]
"S" is for a scale prediction block and computing the yolo loss
"U" is for upsampling the feature map
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1], #two conv layers, one is # of prev filters/2, the other is # of prev filter, then residual
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    # first route from the end of the previous block
    (512, 3, 2),
    ["B", 8],
    # second route from the end of the previous block
    (1024, 3, 2),
    ["B", 4],
    # until here is YOLO-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

# in darknet, there is batch normalization followed by leaky activation
# in prediction, only conv layer, so add an extra bn_act as a new parameter in class
class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels,bn_act = True, **kwargs): #kwargs to store other parameters
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.bn_act = bn_act

    def forward(self, x):
        if self.bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

#sometimes, skip instead of residual
class ResidualBlock(nn.Module):
    def __init__(self, channels, num_repeats=1, residual=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.channels = channels
        self.num_repeats = num_repeats
        self.residual = residual
        for i in range(num_repeats):
            self.layers+=[(nn.Sequential(
                CNNBlock(self.channels, self.channels //2, kernel_size = 1),
                CNNBlock(self.channels//2,self.channels,kernel_size = 3,padding = 1)
            ))]
    def forward(self,x):

        for layer in self.layers:
            x = layer(x) + self.residual * x
        return x

#prediction
#one conv layer with two times output filter, plus one that leads to the results
#result shape (batch size, anchors per scale, grid size, grid size, 5 + number of classes)

class ScalePrediction(nn.Module):
    def __init__(self,channels,num_classes,anchors=3):
        super().__init__()
        self.channels = channels
        self.anchors = anchors
        self.num_classes = num_classes

        self.pred = nn.Sequential(
            CNNBlock(in_channels=channels,out_channels=channels*2,kernel_size =3 ,padding = 1),
            #bn_act = false because it's the output layer
            CNNBlock(in_channels=channels*2,out_channels=3*(5+self.num_classes),bn_act=False,kernel_size = 1)
        )

    def forward(self,x):
        return(self.pred(x). #need to print the shape here
               reshape(x.shape[0],self.anchors,self.num_classes+5,x.shape[2],x.shape[3]).
               permute(0,1,3,4,2)) #put bounding box vector at end
        # current x is of shape: N * 3 * 13 * 13 * 5+num_classes


class YOLOv3(nn.Module):
    #put everything together
    def __init__(self,in_channels=3,num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self,x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer,ScalePrediction):
                outputs.append(layer(x))
                continue #multiple output layers

            x = layer(x)

            if isinstance(layer,ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                #print(x.shape,route_connections[-1].shape)
                x = torch.cat([x,route_connections[-1]],dim = 1)
                route_connections.pop()
        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module,tuple):#conv block
                out_channels,kernel_size,stride = module
                layers.append(
                    CNNBlock(in_channels = in_channels,
                                       out_channels=out_channels,
                                       kernel_size = kernel_size,
                                       stride=stride,
                                       padding = 1 if kernel_size == 3 else 0,
                                       )
                              )
                in_channels = out_channels #update the in_channels

            #residual
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(in_channels,num_repeats=num_repeats)
                )

            elif isinstance(module,str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels,residual=False,num_repeats=1),
                        CNNBlock(in_channels,in_channels // 2, kernel_size = 1),
                        ScalePrediction(in_channels // 2,num_classes = self.num_classes)
                    ]
                    in_channels = in_channels // 2
                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3 #concatenate prev layer that has 2 times of filters
        return layers

if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2,3,IMAGE_SIZE,IMAGE_SIZE))
    out = model(x)
    print("results' shapes:")
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)