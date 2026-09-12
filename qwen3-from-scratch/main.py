# This is a sample Python script.
from pathlib import Path

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import logging
import time

from model_builders.qwen3_model import Qwen3Model
from qwen3_tokenizer import Qwen3Tokenizer
from utils import download_from_huggingface
from config import QWEN_CONFIG_06_B

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()

USE_REASONING_MODEL = True
# Uses the base model if USE_REASONING_MODEL = False

USE_INSTRUCT_MODEL = False
# Uses the instruct mode (without reasoning) if
# USE_REASONING_MODEL = True
# USE_INSTRUCT_MODEL = True
# This setting does have no effect if USE_REASONING_MODEL = False

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1

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

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

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
    repo_id = "rasbt/qwen3-from-scratch"
    if USE_REASONING_MODEL:
        filename = "qwen3-0.6B.pth"
        local_dir = "models/Qwen3-0.6B"
        tokenizer_file_path = "tokenizer.json"
    else:
        filename = "qwen3-0.6B-base.pth"
        local_dir = "models/Qwen3-0.6B-Base"
        tokenizer_file_path = "tokenizer-base.json"

    download_from_huggingface(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir
    )
    model_file = Path(local_dir) / filename
    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_file, weights_only=True, map_location="cpu"))
    # model = model.to(torch.float16)
    model.float()
    model.to(device)

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        repo_id=repo_id,
        apply_chat_template=USE_REASONING_MODEL,
        add_generation_prompt=USE_REASONING_MODEL,
        add_thinking=not USE_INSTRUCT_MODEL
    )

    prompt = "Give me a short introduction to large language models."
    input_token_ids = tokenizer.encode(prompt)

    torch.manual_seed(123)

    start = time.time()

    output_token_ids = generate(
        model=model,
        idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
        max_new_tokens=150,
        context_size=QWEN_CONFIG_06_B["context_length"],
        top_k=1,
        temperature=0.
    )

    total_time = time.time() - start
    print(f"Time: {total_time:.2f} sec")
    print(f"{int(len(output_token_ids[0]) / total_time)} tokens/sec")

    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")

    output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())

    print("\n\nOutput text:\n\n", output_text + "...")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
