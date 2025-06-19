from .M3D import M3DFEL

__all__ = ['M3DFEL']

def create_model(args):
    """create model according to args

    Args:
        args
    """
    model = M3DFEL(args)

    return model
