from fusion.architecture.base_block import Flatten
import torch
import torch.nn as nn


class LinearEvaluator(nn.Module):
    def __init__(self, **kwargs):
        """
        Args:
            :param kwargs:

        Return:

        """
        super(LinearEvaluator, self).__init__()
        self.flatten = Flatten()
        self.linear = nn.Linear(kwargs['dim_rkhs'], kwargs['num_classes'])

    def forward(self, x):
        """

        Args:

            :param x:
        Return:

        """
        x = self.flatten(x)
        x = self.linear(x)
        return x


class LinearEvaluatorWithEncoder(nn.Module):
    def __init__(self, encoder, num_classes, view, **kwargs):
        """

        Args:
            :param encoder:
            :param num_classes:
            :param view:
            :param kwargs:

        Return:

        """
        super(LinearEvaluatorWithEncoder, self).__init__()
        self.encoder = encoder
        self.encoder.eval()
        self.flatten = Flatten()
        self.linear = nn.Linear(kwargs['dim_rkhs'], num_classes)
        self.view = view

    def forward(self, x):
        """
        Args:
            :param x: input tensor
        Return

        """
        x = x[self.view]
        x = self.encoder(x)[0]
        x = x.detach()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class FusedLinearEvaluator(nn.Module):
    def __init__(self, encoder_list, num_classes, num_views, **kwargs):
        """
        Args:
            :param encoder_list:
            :param num_classes:
            :param num_views:
            :param kwargs:
        Return:

        """
        super(FusedLinearEvaluator, self).__init__()
        self.encoder_list = nn.ModuleList(encoder_list)
        for encoder in self.encoder_list:
            encoder.eval()
            encoder.cuda()
        self.flatten = Flatten()
        self.linear = nn.Linear(num_views * kwargs['dim_rkhs'], num_classes)
        self.num_views = num_views

    def forward(self, x):
        """
        Args:
            :param x: input tensor
        Return:

        """
        z = None
        for view, encoder in enumerate(self.encoder_list):
            z_temp = encoder(x[view])[0].detach()
            if z is None:
                z = z_temp
            else:
                z = torch.cat([z, z_temp], dim=1)
        z = self.flatten(z)
        z = self.linear(z)
        return z


class EvalEncoder(nn.Module):
    def __init__(self, encoder, view):
        """
        Args:
            :param encoder:
            :param view:
        Return:

        """
        super(EvalEncoder, self).__init__()
        self.encoder = encoder
        self.encoder.eval()
        self.view = view

    def forward(self, x):
        """
        Args:
            :param x: input tensor
        Return:

        """
        x = x[self.view]
        x = self.encoder(x)
        if isinstance(x, tuple):
            x = x[0]
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = x.detach()
        x = x.reshape(x.size(0), -1)
        return x