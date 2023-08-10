from TrainCondition import train, eval


def main(model_config=None):
    modelConfig = {
        "state": "eval", # or eval
        "epoch": 70,
        "batch_size": 20,
        "T": 200,
        "channel": 32,
        "channel_mult": [1, 2, 3, 4],
        "num_res_blocks": 2,
        "dropout": 0,
        "lr": 1e-4,
        "multiplier": 2,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "signal_length": 1000,
        "grad_clip": 1.,
        "device": "cpu",
        "w": 1.8,
        "save_dir": "CheckpointsCondition/",
        "training_load_weight": None,
        "test_load_weight": "ckpt_0_.pt"
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()
