import math
import torch
import torch.nn as nn

cfg = {
       'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       }

def make_layers(input_ch, cfg, batch_norm=False):
    layers  = []
    
    in_channels = input_ch
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            relu = nn.ReLU(inplace=False)
            if batch_norm:
                bn = nn.BatchNorm2d(v)
                layers += [conv2d, bn, relu]
            else:
                layers += [conv2d, relu]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, im_size, input_ch, num_classes, net_type, batch_norm):
        super(VGG, self).__init__()
        
        self.features = make_layers(input_ch, cfg[net_type], batch_norm=batch_norm)
        
        num_strides = sum([layer == 'M' for layer in cfg[net_type]])
        kernel_numel = int((im_size / (2**num_strides))**2)

        relu1 = nn.ReLU(inplace=False)
        relu2 = nn.ReLU(inplace=False)
        
        lin1 = nn.Linear(512 * kernel_numel, 4096, bias=False)
        lin2 = nn.Linear(4096, 4096, bias=False)
        lin3 = nn.Linear(4096, 1000)
        
        bn1 = nn.BatchNorm1d(4096)
        bn2 = nn.BatchNorm1d(4096)
        
        self.classifier = nn.Sequential(
            lin1,
            bn1,
            relu1,
            lin2,
            bn2,
            relu2,
            lin3
        )
        
        self._initialize_weights()
        
        mod = list(self.classifier.children())
        mod.pop()
        
        lin4 = torch.nn.Linear(4096, num_classes)
        
        mod.append(lin4)
        self.classifier = torch.nn.Sequential(*mod)
        self.classifier[-1].weight.data.normal_(0, 0.01)
        self.classifier[-1].bias.data.zero_()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                try:
                    m.bias.data.zero_()
                except:
                    pass
    
class VGG11_bn(VGG):
    def __init__(self, im_size, input_ch, num_classes):
        super(VGG11_bn, self).__init__(im_size, input_ch, num_classes, 'A', True)
