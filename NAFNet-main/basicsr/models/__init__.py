# This code is adapted from the original BasicSR and NAFNet repositories.
# It allows creating the correct model type based on the configuration.

import importlib
from os import path as osp

# The original NAFNet/BasicSR code dynamically scans for models,
# but for this application, we can import the required model directly.
from .image_restoration_model import ImageRestorationModel


def create_model(opt):
    """
    Factory function for creating a model.
    This function is called by app.py and train.py.
    """
    model_type = opt.get('model_type')

    if model_type == 'ImageRestorationModel':
        model = ImageRestorationModel(opt)
    else:
        raise ValueError(f"Model type '{model_type}' is not recognized.")

    return model