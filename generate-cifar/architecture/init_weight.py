from torch.nn import init

def init_func(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init.constant_(m.weight.data, 1.0)
        if hasattr(m, 'bias') and m.bias is not None:
            init.normal_(m.bias.data, mean=0.0, std=0.01)
    elif hasattr(m, 'weight') and (
        classname.find('Conv') != -1 or
        classname.find('Linear') != -1 or
        classname.find('Embedding') != -1
    ):
        init.orthogonal_(m.weight.data, gain=1.0)
        if hasattr(m, 'bias') and m.bias is not None:
            init.normal_(m.bias.data, mean=0.0, std=0.01)
