# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import logging
# import data_setup
# import tiktoken

from model_builders.qwen3_model import Qwen3Model

from config import get_config
from utils import download_weights, load_weights_into_qwen,get_tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 2


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


def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)[:, -1]
            next_token = torch.argmax(out, dim=-1, keepdim=True)

            if (eos_token_id is not None
                    and torch.all(next_token == eos_token_id)):
                break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)

def main():
    device = get_device()
    logger.info(f"Running on device : {device}")

    # Create DataLoaders with help from data_setup.py
    # train_dataloader, test_dataloader = data_setup.create_dataloaders(
    #     batch_size=BATCH_SIZE,
    #     drop_last=True,
    #     shuffle=True
    # )
    #
    # model = Qwen3Model(QWEN3_CONFIG).to(device)
    model_version = "0.6B"
    USE_REASONING_MODEL = False
    USE_INSTRUCT_MODEL = False
    QWEN3_CONFIG= get_config(model_version)



    model = Qwen3Model(QWEN3_CONFIG)
    print(model)

    weights_dict = download_weights(model_version,USE_REASONING_MODEL,USE_INSTRUCT_MODEL)
    load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
    model.to(device)

    tokenizer = get_tokenizer(model_version, USE_REASONING_MODEL, USE_INSTRUCT_MODEL)


    prompt = "Give me a short introduction to large language models."

    input_token_ids = tokenizer.encode(prompt)
    input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

    for token in generate_text_basic_stream(
            model=model,
            token_ids=input_token_ids_tensor,
            max_new_tokens=500,
            eos_token_id=tokenizer.eos_token_id
    ):
        # generated_tokens += 1
        token_id = token.squeeze(0).tolist()
        print(
            tokenizer.decode(token_id),
            end="",
            flush=True
        )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
