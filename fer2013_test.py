import json
import os
import random
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import imgaug
import numpy as np
import torch

from main_fer2013 import get_dataset, get_model
from trainers.tta_trainer import FER2013Trainer
from utils.datasets.fer2013dataset import fer2013

seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# call this module like this: python fer2013_test.py <config_path> <checkpoint_filename>
try:
    config_path = sys.argv[1]
    checkpoint_filename = sys.argv[2]
except:
    config_path = "./configs/fer2013_ResMaskingNet_74.14_test_acc_config.json"
    # this checkpoint must be located in ./saved/checkpoints
    checkpoint_filename = "resmasking_dropout1_rot30_2019Nov17_14.33"


def main(config_path, checkpoint):
    """
    This is the main function for testing
    a model on fer2013 dataset

    Parameters:
    -----------
    config_path : str
        path to config file
    checkpoint : str
        checkpoint filename

    Note:
    -----------
    Parameters in config file must be
    compatible with the checkpoint file
    (arch, in_channels, num_classes...)
    """
    # load configs and set random seed
    configs = json.load(open(config_path))
    configs["cwd"] = os.getcwd()

    # load model and data_loader
    model = get_model(configs)
    train_set, val_set, test_set = get_dataset(configs)

    # initialize trainer, in trainer there is test method
    trainer = FER2013Trainer(model, train_set, val_set, test_set, configs)

    # load weights (checkpoint) in model
    state = torch.load(os.path.join("saved/checkpoints", checkpoint))
    trainer._model.load_state_dict(state["net"])

    # calculate acc on test dataset with tta
    trainer._calc_acc_on_private_test_with_tta()

    # to calculate acc without tta, first we need
    # test set without tta
    test_set_without_tta = fer2013("test", configs, tta=False)

    # initialize trainer with test set without tta
    trainer = FER2013Trainer(model, train_set, val_set, test_set_without_tta, configs)

    trainer._model.load_state_dict(state["net"])

    # calculate acc on test dataset without tta
    trainer._calc_acc_on_private_test()


if __name__ == "__main__":
    main(config_path=config_path, checkpoint=checkpoint_filename)
