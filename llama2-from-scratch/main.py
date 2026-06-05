# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import logging
from huggingface_hub import hf_hub_download

from model_builders.llama2_model import Llama2Model
from config import LLAMA2_CONFIG_7B

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

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def main():
    device = get_device()
    logger.info(f"Running on device : {device}")

    # Create DataLoaders with help from data_setup.py
    # train_dataloader, test_dataloader = data_setup.create_dataloaders(
    #     batch_size=BATCH_SIZE,
    #     drop_last=True,
    #     shuffle=True
    # )

    model = Llama2Model(LLAMA2_CONFIG_7B)
    print(model)
    start_context = "Hello, I am"



    # tokenizer = tiktoken.get_encoding("gpt2")
    # encoded = tokenizer.encode(start_context)
    # encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    # print(f"\n{50 * '='}\n{22 * ' '}IN\n{50 * '='}")
    # print("\nInput text:", start_context)
    # print("Encoded input text:", encoded)
    # print("encoded_tensor.shape:", encoded_tensor.shape)

    # out = generate_text_simple(
    #     model=model,
    #     idx=encoded_tensor,
    #     max_new_tokens=10,
    #     context_size=GPT_CONFIG_124M["context_length"]
    # )
    # decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    #
    # print(f"\n\n{50 * '='}\n{22 * ' '}OUT\n{50 * '='}")
    # print("\nOutput:", out)
    # print("Output length:", len(out[0]))
    # print("Output text:", decoded_text)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
