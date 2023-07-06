import json
import os
import random

import imgaug
import numpy as np
import torch

import time

seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn.functional as F
from tqdm import tqdm

import models
from utils.datasets.fer2013dataset import fer2013
from utils.generals import make_batch

model_dict = [
    ("resnet18", "resnet18_rot30_2019Nov05_17.44"),
    ("resnet50_pretrained_vgg", "resnet50_pretrained_vgg_rot30_2019Nov13_08.20"),
    ("resnet101", "resnet101_rot30_2019Nov14_18.12"),
    ("cbam_resnet50", "cbam_resnet50_rot30_2019Nov15_12.40"),
    ("efficientnet_b2b", "efficientnet_b2b_rot30_2019Nov15_20.02"),
    ("resmasking_dropout1", "resmasking_dropout1_rot30_2019Nov17_14.33"),
    ("resmasking", "resmasking_rot30_2019Nov14_04.38"),
]


def main():
    with open("./configs/fer2013_config.json") as f:
        configs = json.load(f)

    test_set = fer2013("test", configs, tta=True, tta_size=8)

    prediction_time = 0

    for model_name, checkpoint_path in model_dict:
        prediction_list = []  # each item is 7-ele array

        print("Processing", checkpoint_path)
        if os.path.exists("./saved/results/{}.npy".format(checkpoint_path)):
            continue

        test_targets = []
        if os.path.exists("./saved/test_targets.npy"):
            targets_saved = True
        else:
            targets_saved = False

        model = getattr(models, model_name)
        model = model(in_channels=3, num_classes=7)

        state = torch.load(os.path.join("saved/checkpoints", checkpoint_path))
        model.load_state_dict(state["net"])

        model.cuda()
        model.eval()

        with torch.no_grad():
            for idx in tqdm(range(len(test_set)), total=len(test_set), leave=False):
                images, targets = test_set[idx]
                if not targets_saved:
                    test_targets.append(targets)

                images = make_batch(images)
                images = images.cuda(non_blocking=True)

                start_time = time.perf_counter()
                outputs = model(images).cpu()
                outputs = F.softmax(outputs, 1)
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                prediction_time += execution_time

                outputs = torch.sum(outputs, 0)  # outputs.shape [tta_size, 7]

                outputs = [round(o, 4) for o in outputs.numpy()]
                prediction_list.append(outputs)

        if not targets_saved:
            test_targets = np.array(test_targets)
            np.save("./saved/{}.npy".format("test_targets"), test_targets)
            targets_saved = True

        np.save("./saved/results/{}.npy".format(checkpoint_path), prediction_list)

        print("Prediction time: " + str(prediction_time))


if __name__ == "__main__":
    main()
