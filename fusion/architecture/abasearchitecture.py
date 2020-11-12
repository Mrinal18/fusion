from fusion.architecture.ibasearchitecture import IBaseArchitecture
import torch.nn as nn
import zope.interface


@zope.interface.implementer(IBaseArchitecture)
class ABaseArchitecture(nn.Module):
    @zope.interface.interfacemethod
    def forward(self, x):
        pass
