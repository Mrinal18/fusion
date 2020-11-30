from fusion.utils import ObjectProvider
from torch.optim.lr_scheduler import OneCycleLR


scheduler_provider = ObjectProvider()
scheduler_provider.register_object('OneCycleLR', OneCycleLR)
