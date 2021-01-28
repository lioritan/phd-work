import torch
import torch.nn.functional as F


class SimpleNet(torch.nn.Module):

    def __init__(self, net_params, in_size, out_size):
        super(SimpleNet, self).__init__()
        self.fc_in = torch.nn.Linear(in_size, net_params[0])
        self.fc_out = torch.nn.Linear(net_params[-1], out_size)
        self.fc_layers = []
        for i, layer_size in enumerate(net_params):
            if i == len(net_params) - 1:
                pass
            else:
                layer = torch.nn.Linear(net_params[i], net_params[i + 1])
                self.fc_layers.append(layer)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        for mid_layer in self.fc_layers:
            x = F.relu(mid_layer(x))
        x = self.fc_out(x)
        return x
