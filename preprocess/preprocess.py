import os
from os.path import join as PJ
import random

random.seed(1024)

DATASET = "CIFAR10"
DATA_ROOT = PJ(os.path.dirname(os.getcwd()), "datasets", DATASET)

for data_file in ["test", "train"]:

    dir = os.path.expanduser(PJ(DATA_ROOT, data_file))
    class_names = os.listdir(dir)

    if data_file == 'train':
        items = []
        val_items = []
        for c in class_names:
            pathes = [PJ(data_file, c, p) for p in os.listdir(PJ(dir, c))]
            val_pathes = random.sample(pathes, len(pathes) // 10)
            train_pathes = list(set(pathes) - set(val_pathes))

            items += [" ".join([tp, c]) for tp in train_pathes]

            val_items += [" ".join([vp, c]) for vp in val_pathes]
            with open(PJ(DATA_ROOT, "val.txt"), "w") as f:
                f.write('\n'.join(val_items))
    else:
        items = [" ".join([PJ(data_file, c, p), c]) for c in sorted(class_names) for p in os.listdir(PJ(dir, c))]


    with open(PJ(DATA_ROOT, data_file + ".txt"), "w") as f:
        f.write('\n'.join(items))