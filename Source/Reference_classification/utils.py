from Utils import paths
import torch
import os


def save_config(path, **kwargs):

    folder = paths.pretrained_classifiers_folder

    with open(
            os.path.join(
                path,
                f"config_{kwargs['MODEL_NAME'].split('/')[1]}_{str(kwargs['VERSION'])}"),
            "w"
    ) as config_file:

        for k, v in kwargs.items():
            config_file.write(f"{k}: {v}\n")

    config_file.close()


def save_model(CONFIG, model, loss_records, checkpoint=None):

    out_path = os.path.join(CONFIG["OUTPUT_DIR"],
                            f"{CONFIG['MODEL_NAME'].split('/')[1]}_{CONFIG['VERSION']}"
                            )

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    model_name = f"{CONFIG['MODEL_NAME'].split('/')[1]}_{CONFIG['VERSION']}"

    checkpoint_suffix = f"_checkpoint_{checkpoint}" if checkpoint is not None else ""
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_labels': CONFIG["NUM_CLASSES"],
        'model_name': model_name + checkpoint_suffix},
        os.path.join(out_path, model_name + checkpoint_suffix)
    )

    CONFIG["Losses"] = loss_records
    save_config(out_path, **CONFIG)