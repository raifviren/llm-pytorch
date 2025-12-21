# This is a sample Python script.
from pathlib import Path

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import random

from model_builder import TinyVGG
from predictions import pred_and_plot_image

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"
HIDDEN_UNITS = 10
class_names=  ['pizza', 'steak', 'sushi']
PATH = "models/05_going_modular_script_mode_tinyvgg_model.pth"

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Use PyTorch 2.9 or newer for stable mps results
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 1):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device

def predict():
    device = get_device()
    num_images_to_plot = 3
    test_image_path_list = list(Path(test_dir).glob("*/*.jpg"))  # get list all image paths from test data
    test_image_path_sample = random.sample(population=test_image_path_list,  # go through all of the test image paths
                                           k=num_images_to_plot)  # randomly select 'k' image paths to pred and plot

    #load model
    model = TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)
    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()

    # Make predictions on and plot the images
    for image_path in test_image_path_sample:
        pred_and_plot_image(model=model,
                            image_path=image_path,
                            class_names=class_names,
                            # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
                            image_size=(64, 64),
                            device=device)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    predict()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
