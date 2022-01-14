from data_loader import cnn_data_loader_cv
from cross_validation import CV

import argparse
import os


def make_savedir(config):
    """logディレクトリを作成する"""
    if config.parent_path is None:
        os.makedirs(
            f"../logs/{config.name}/{config.model_name}/{config.batchsize}_{config.adam_lr}",
            exist_ok=True,
        )
        save_dir = f"../logs/{config.name}/{config.model_name}/{config.batchsize}_{config.adam_lr}"
    else:
        os.makedirs(
            f"{config.parent_path}/logs/{config.name}/{config.model_name}/{config.batchsize}_{config.adam_lr}",
            exist_ok=True,
        )
        save_dir = f"../{config.parent_path}/logs/{config.name}/{config.model_name}/{config.batchsize}_{config.adam_lr}"

    os.makedirs(save_dir + "/train_log", exist_ok=True)
    os.makedirs(save_dir + "/model_weight", exist_ok=True)

    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # log
    parser.add_argument("--parent_path", type=str, default=None)
    # data
    parser.add_argument("--resize", type=int, default=100)
    parser.add_argument("--name", type=str, default="snapshot_2D")
    # model setting
    parser.add_argument("--model_name", type=str, default="CNN")
    # train params
    parser.add_argument("--adam_lr", type=float, default=0.001)
    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--batchsize", type=int, default=32)

    parser.add_argument("--path", type=str, default=None)

    config = parser.parse_args()
    # load dataset
    dataset, num_class = cnn_data_loader_cv(config.name, config.resize, config.path)
    # make save dir
    save_dir = make_savedir(config)
    # train and validation
    CV(num_class, dataset, save_dir, config)
