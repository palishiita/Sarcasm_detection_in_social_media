import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import get_linear_schedule_with_warmup, BertTokenizer, BertForSequenceClassification, BertModel

def get_bert_model(num_labels=2):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels
    )
    return model, tokenizer


def tokenize_data(texts, tokenizer, max_length=128):
    encodings = tokenizer(
        list(texts),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return encodings


def make_dataloader(encodings, labels, batch_size=16, shuffle=True):
    dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        encodings["token_type_ids"],
        torch.tensor(labels.values, dtype=torch.long)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_bert(model, train_dataloader, epochs=3, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids, attention_mask, token_type_ids, labels = [
                b.to(device) for b in batch
            ]
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

    return model


def predict_bert(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, token_type_ids, _ = [
                b.to(device) for b in batch
            ]
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    return all_preds



class DualEncoderBert(nn.Module):
    """
    Reference: Conneau et al. (2017), Hazarika et al. (2018)
    Encodes context and reply separately through two BERT encoders,
    then classifies based on the relationship between their CLS vectors.

    Classifier input = [CLS_context | CLS_reply | CLS_context * CLS_reply]
    The element-wise product explicitly captures semantic incongruity
    between context and reply — the core signal in sarcasm detection.
    """
    def __init__(self, num_labels=2, dropout=0.1):
        super(DualEncoderBert, self).__init__()

        self.context_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.reply_encoder = BertModel.from_pretrained("bert-base-uncased")

        hidden_size = self.context_encoder.config.hidden_size  # 768

        self.dropout = nn.Dropout(dropout)
        # * 3 because: CLS_context + CLS_reply + CLS_context * CLS_reply
        self.classifier = nn.Linear(hidden_size * 3, num_labels)

    def forward(self, context_input_ids, context_attention_mask,
                reply_input_ids, reply_attention_mask, labels=None):

        # Encode context — CLS token is index 0
        cls_context = self.context_encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask
        ).last_hidden_state[:, 0, :]  # (batch, 768)

        # Encode reply — CLS token is index 0
        cls_reply = self.reply_encoder(
            input_ids=reply_input_ids,
            attention_mask=reply_attention_mask
        ).last_hidden_state[:, 0, :]  # (batch, 768)

        # Combine: concatenate + element-wise product for interaction signal
        combined = torch.cat(
            [cls_context, cls_reply, cls_context * cls_reply],
            dim=-1
        )  # (batch, 768 * 3)

        logits = self.classifier(self.dropout(combined))  # (batch, 2)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return loss, logits


class DualEncoderDataset(Dataset):
    """
    Tokenises context and reply separately.
    Used to build dataloaders for DualEncoderBert.
    """
    def __init__(self, contexts, replies, labels, tokenizer, max_length=128):
        self.context_encodings = tokenizer(
            list(contexts),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        self.reply_encodings = tokenizer(
            list(replies),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "context_input_ids": self.context_encodings["input_ids"][idx],
            "context_attention_mask": self.context_encodings["attention_mask"][idx],
            "reply_input_ids": self.reply_encodings["input_ids"][idx],
            "reply_attention_mask": self.reply_encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }


def train_dual_encoder(model, train_dataloader, epochs=3, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            context_input_ids = batch["context_input_ids"].to(device)
            context_attention_mask = batch["context_attention_mask"].to(device)
            reply_input_ids = batch["reply_input_ids"].to(device)
            reply_attention_mask = batch["reply_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss, _ = model(
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                reply_input_ids=reply_input_ids,
                reply_attention_mask=reply_attention_mask,
                labels=labels
            )
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

    return model


def predict_dual_encoder(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            context_input_ids = batch["context_input_ids"].to(device)
            context_attention_mask = batch["context_attention_mask"].to(device)
            reply_input_ids = batch["reply_input_ids"].to(device)
            reply_attention_mask = batch["reply_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            _, logits = model(
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                reply_input_ids=reply_input_ids,
                reply_attention_mask=reply_attention_mask,
                labels=labels
            )
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    return all_preds