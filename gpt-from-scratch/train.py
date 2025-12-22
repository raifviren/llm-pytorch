"""
Created at 22/12/25
@author: raif.viren@gmail.com
"""
import torch
import logging
import data_setup, engine, utils
import tiktoken

from model_builders.gpt_model import GPTModel

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

def train(gpt_config, settings):
    device = get_device()
    logger.info(f"Running on device : {device}")

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader = data_setup.create_dataloaders(
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    model = GPTModel(gpt_config).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    # Start training with help from engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=NUM_EPOCHS,
                 device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                     target_dir="models",
                     model_name="gpt_from_scratch_model.pth")

    # start_context = "Hello, I am"
    #
    # tokenizer = tiktoken.get_encoding("gpt2")
    # encoded = tokenizer.encode(start_context)
    # encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    #
    # print(f"\n{50 * '='}\n{22 * ' '}IN\n{50 * '='}")
    # print("\nInput text:", start_context)
    # print("Encoded input text:", encoded)
    # print("encoded_tensor.shape:", encoded_tensor.shape)
    #
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
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 256,  # Shortened context length (orig: 1024)
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False  # Query-key-value bias
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1
    }
    train(GPT_CONFIG_124M,OTHER_SETTINGS)