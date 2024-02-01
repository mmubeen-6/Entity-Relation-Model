import torch.nn as nn
from transformers import AutoModel


def get_base_model(model_name: str):
    """
    Create a base model from a given model name.

    Args:
        model_name (str): Name of the model to be used.
                should be one of the following:
                - bert-base-uncased
                - bert-large-uncased

    Returns:
        Base model.
    """
    assert model_name in [
        "bert-base-uncased",
        "bert-large-uncased",
    ], "Invalid model name"

    base_model = AutoModel.from_pretrained(model_name)
    return base_model


def update_model_tokenizer(model, tokenizer):
    model.resize_token_embeddings(len(tokenizer))
    return model


class EntityRelationModel(nn.Module):
    """Entity Relation Extraction Model.
    It is a simple feed forward neural network with 2 dense layers.
    Uses BERT as the base model and a linear layer as the output layer.
    """

    def __init__(self, base_model, num_classes: int):
        """Initialize the model.

        Args:
            base_model: Base model to be used.
            num_classes (int): Number of classes.
        """
        super(EntityRelationModel, self).__init__()

        self.bert = base_model
        out_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(out_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)  # softmax

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

