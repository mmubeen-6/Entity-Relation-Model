from typing import List

import torch
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def get_dataset(dataset_name: str) -> DatasetDict:
    """Load a dataset from the HuggingFace datasets library.
    Currently supported datasets:
    - kbp37_formatted

    Args:
        dataset_name (str): Name of the dataset to be loaded.

    Returns:
        DatasetDict: Dataset loaded from the HuggingFace datasets library.
    """
    assert dataset_name in [
        "kbp37_formatted",
    ], "Invalid dataset name"

    dataset = load_dataset("DFKI-SLT/kbp37", name="kbp37_formatted")
    return dataset


def get_tokenizer(tokenizer_name: str):
    """Create a tokenizer from a given tokenizer name.
    Currently supported tokenizers:
    - bert-base-uncased
    - bert-large-uncased

    Args:
        tokenizer_name (str): Name of the tokenizer to be used.

    Returns:
        Tokenizer.
    """
    assert tokenizer_name in [
        "bert-base-uncased",
        "bert-large-uncased",
    ], "Invalid tokenizer name"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer


class EntityRelationDataset(Dataset):
    """Dataset class for entity relation extraction.

    Args:
        Dataset: HuggingFace dataset object.
        tokenizer: HuggingFace tokenizer object.
        max_length (int, optional): Maximum length of the input sequence.
                Defaults to 512.
    """

    def __init__(self, dataset, tokenizer, max_length: int = 512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx) -> dict:
        """Return an item from the dataset.

        Args:
            idx (int): Index of the item to be returned.

        Returns:
            dict: Dictionary containing the input_ids, attention_mask and label
                    for the given index.
        """
        idx_item = self.dataset[idx]
        marked_tokens = self.mark_entities_in_sentence(
            idx_item["token"],
            idx_item["e1_start"],
            idx_item["e1_end"],
            idx_item["e2_start"],
            idx_item["e2_end"],
        )
        sentence = " ".join(marked_tokens)
        encoded_input = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded_input["input_ids"].flatten(),
            "attention_mask": encoded_input["attention_mask"].flatten(),
            "label": torch.tensor(
                int(idx_item["relation"]), dtype=torch.int64
            ),
        }

    def mark_entities_in_sentence(
        self,
        tokens: List[str],
        e1_start: int,
        e1_end: int,
        e2_start: int,
        e2_end: int,
    ) -> List[str]:
        """Mark the entities in the sentence with special tokens.
        Cuurently, the special tokens are:
        - [E1] for the first entity
        - [/E1] for the end of the first entity
        - [E2] for the second entity
        - [/E2] for the end of the second entity

        Args:
            tokens (List[str]): List of tokens in the sentence.
            e1_start (int): Start index of the first entity.
            e1_end (int): End index of the first entity.
            e2_start (int): Start index of the second entity.
            e2_end (int): End index of the second entity.

        Returns:
            List[str]: List of tokens with special tokens inserted.
        """

        # Insert special tokens for the second entity
        tokens.insert(e2_end, "[/E2]")
        tokens.insert(e2_start, "[E2]")

        # Insert special tokens for the first entity
        tokens.insert(e1_end, "[/E1]")
        tokens.insert(e1_start, "[E1]")

        return tokens
