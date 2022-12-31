import json
import os
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import imgaug
import numpy as np
import torch
import torch.multiprocessing as mp
from main_fer2013 import get_dataset, get_model

seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import models
from models import segmentation


def main(config_path):
    """
    This is the main function for testing a model

    Parameters:
    -----------
    config_path : str
        path to config file
    """
    # load configs and set random seed
    configs = json.load(open(config_path))
    configs["cwd"] = os.getcwd()

    model_dict = {
        "model_name": "resmasking_dropout1",
        "checkpoint_path": "resmasking_dropout1_rot30_2019Nov17_14.33",
    }

    # load model and data_loader

    model = getattr(models, model_dict.get("model_name"))
    model = model(in_channels=3, num_classes=7)

    state = torch.load(
        os.path.join("saved/checkpoints", model_dict.get("checkpoint_path"))
    )
    model.load_state_dict(state["net"])

    # model.cuda()
    # model.eval()

    train_set, val_set, test_set = get_dataset(configs)

    from trainers.tta_trainer import FER2013Trainer

    trainer = FER2013Trainer(model, train_set, val_set, test_set, configs)
    trainer._calc_acc_on_private_test_with_tta()


if __name__ == "__main__":
    main("./configs/fer2013_config.json")
