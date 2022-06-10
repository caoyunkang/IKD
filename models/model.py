from models.hrnet.hrnet import hrnet18, hrnet32, hrnet48, _hrnet
import torch
from torch.nn import functional as F
from torch import nn


class School(nn.Module):
    def __init__(self, model='hrnet18'):
        super(School, self).__init__()

        if model == 'hrnet18':
            self.teacher = hrnet18(pretrained=True)
            self.student = hrnet18(pretrained=False)
        elif model == 'hrnet32':
            self.teacher = hrnet32(pretrained=True)
            self.student = hrnet32(pretrained=False)
        elif model == 'hrnet48':
            self.teacher = hrnet48(pretrained=True)
            self.student = hrnet48(pretrained=False)
        else:
            raise NotImplementedError

    def forward_one(self, model: _hrnet, x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.conv2(x)
        x = model.bn2(x)
        x = model.relu(x)
        x = model.layer1(x)

        x_list = []
        for i in range(model.stage2_cfg['NUM_BRANCHES']):
            if model.transition1[i] is not None:
                x_list.append(model.transition1[i](x))
            else:
                x_list.append(x)
        y_list = model.stage2(x_list)

        x_list = []
        for i in range(model.stage3_cfg['NUM_BRANCHES']):
            if model.transition2[i] is not None:
                if i < model.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(model.transition2[i](y_list[i]))
                else:
                    x_list.append(model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = model.stage3(x_list)

        x_list = []
        for i in range(model.stage4_cfg['NUM_BRANCHES']):
            if model.transition3[i] is not None:
                if i < model.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(model.transition3[i](y_list[i]))
                else:
                    x_list.append(model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = model.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        return {'x0': x[0], 'x1': x1, 'x2': x2, 'x3': x3}

    def forward(self, x):
        with torch.no_grad():
            x_t = self.forward_one(self.teacher, x)
        x_s = self.forward_one(self.student, x)
        return x_t, x_s