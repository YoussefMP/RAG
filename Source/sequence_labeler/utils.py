from Source import paths
import os


def save_config(**kwargs):

    folder = paths.pretrained_classifiers_folder

    with open(os.path.join(folder, f"config_{kwargs['MODEL_NAME'].split('/')[1]}_v{str(kwargs['VERSION'])}"), "w") as config_file:

        for k, v in kwargs.items():
            config_file.write(f"{k}: {v}\n")

    config_file.close()
