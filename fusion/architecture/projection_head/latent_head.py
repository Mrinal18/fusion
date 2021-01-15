from fusion.architecture import ABaseArchitecture
import torch.nn as nn


class LatentHead(ABaseArchitecture):
    def __init__(
        self,
        dim_in,
        dim_l,
        dim_h=0,
        num_h_layers=0,
        use_linear=False,
        use_bias=False,
        use_bn=True
    ):
        super(LatentHead, self).__init__()
        self._num_h_layers = num_h_layers
        self._use_linear = use_linear
        self._head = nn.ModuleList([])
        if self._use_linear:
            if self._num_h_layers == 0:
                self._head.append(
                    nn.Linear(dim_in, dim_l, bias=use_bias)
                )
            else:
                assert dim_h != 0
                self._head.append(nn.Linear(dim_in, dim_h, bias=use_bias))
                if use_bn:
                    self._head.append(nn.BatchNorm1d(dim_h))
                self._head.append(nn.ReLU(inplace=True))
                for i in range(1, self._num_h_layers):
                    self._head.append(nn.Linear(dim_h, dim_h, bias=use_bias))
                    if use_bn:
                        self._head.append(nn.BatchNorm1d(dim_h))
                    self._head.append(nn.ReLU(inplace=True))
                self._head.append(nn.Linear(dim_h, dim_l, bias=use_bias))
        self._head = nn.Sequential(*self._head)
        self.init_weights()

    def forward(self, x):
        if self._use_linear:
            x = self._head(x)
        return x

    def init_weights(self):
        for layer in self._head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(
                    layer.weight, gain=nn.init.calculate_gain('relu'))
                if not isinstance(layer.bias, type(None)):
                    nn.init.constant_(layer.bias, 0)
