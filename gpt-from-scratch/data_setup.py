"""
Created at 21/12/25
@author: raif.viren@gmail.com
"""
import os
import tiktoken
import torch
import requests
import logging

from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger()

NUM_WORKERS = os.cpu_count()


def download_data():
    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
    text_data = ""
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    return text_data


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1 :  i + 1 + max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


def create_dataloaders(
        batch_size: int = 2,
        max_length=256,
        stride=128,
        drop_last: bool = True,
        shuffle: bool = True,
        num_workers: int = NUM_WORKERS
):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
      batch_size: Number of samples per batch in each of the DataLoaders.
      drop_last : drop last batch if it does not fit batch_size
      shuffle : shuffle the data
      num_workers: An integer for number of workers per DataLoader.

    Returns:
      A tuple of (train_dataloader, test_dataloader).
      Example usage:
        train_dataloader, test_dataloader = \
          = create_dataloaders(train_dir=path/to/train_dir,
                               test_dir=path/to/test_dir,
                               batch_size=32,
                               num_workers=4)
    """
    text_data = download_data()
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    test_data = text_data[split_idx:]

    logger.info(f"Train Data Size : {len(train_data)}")
    logger.info(f"Test Data Size : {len(test_data)}")

    train_dataloader = create_dataloader_v1(
        train_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=drop_last,
        shuffle=shuffle,
        num_workers=num_workers
    )

    test_dataloader = create_dataloader_v1(
        test_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers
    )

    return train_dataloader, test_dataloader
