import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report
from datetime import datetime
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from shutil import copyfile
import numpy as np
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, RandomSampler
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
import torch.optim
from google.colab import drive
import pandas as pd
import json
import math
from transformers import BertConfig, BertModel, BertForSequenceClassification, BertTokenizer
import torch.nn as nn
import torch


class AdaptiveBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear layers for transformations
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout layer
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Add a learnable scaling factor for each attention head
        self.scaling_factors = nn.Parameter(
            torch.ones(self.num_attention_heads))

    # Transpose the input tensor to calculate attention scores
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False,
                encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, use_cache=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # Attention scores calculation
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Calculate the attention scores for each head (Scaled Dot-Product)
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        # Apply the learnable scaling factors to the attention scores of each head
        attention_scores = attention_scores * \
            self.scaling_factors.view(1, -1, 1, 1)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (
            context_layer,)
        return outputs


class CustomBertModel(BertModel):
    def __init__(self, config):
        super().__init__(config)

        # Replace the original self-attention mechanism with the custom one
        for i in range(config.num_hidden_layers):
            self.encoder.layer[i].attention.self = AdaptiveBertSelfAttention(
                config)


class CustomBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = CustomBertModel(config)


# Load the custom BERT model for sequence classification with the modified multi-head attention mechanism
num_labels = 4
config = BertConfig.from_pretrained('bert-base-uncased', num_labels=num_labels)

model = CustomBertForSequenceClassification.from_pretrained(
    'bert-base-uncased', config=config)

# Mount Google Drive
drive.mount('/content/drive')

# Load json data from the file
data_file = "/content/drive/MyDrive/Final Year Prototype/Datasets/ER-Data-03.json"

with open(data_file, 'r') as file:
    data = json.load(file)

# Initialize Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Extract entities and realtion labels
# Mask the text-description for model traning


def extract_relations(data):
    processed_data = []

    for doc in data:
        document = doc["document"]
        relations = doc["relations"]
        entities = {entity["token_start"]: entity for entity in doc["tokens"]}

        for relation in relations:
            # Entity 1 -> Relation Label -> Entity 2
            entity1_start = entities[relation["head"]]["token_start"]
            entity1_end = entities[relation["head"]]["token_end"]
            entity1_text = document[entities[relation["head"]]
                                    ["start"]:entities[relation["head"]]["end"]]

            entity2_start = entities[relation["child"]]["token_start"]
            entity2_end = entities[relation["child"]]["token_end"]
            entity2_text = document[entities[relation["child"]]
                                    ["start"]:entities[relation["child"]]["end"]]

            relation_label = relation["relationLabel"]

            # Tokenize using BERT tokenizer
            entity1_tokens = tokenizer.tokenize(entity1_text)
            entity2_tokens = tokenizer.tokenize(entity2_text)
            document_tokens = tokenizer.tokenize(document)

            marked_document = document[:entities[relation["head"]]["start"]] + \
                '[E1]' + entity1_text + '[/E1]' + \
                document[entities[relation["head"]]["end"]:entities[relation["child"]]["start"]] + \
                '[E2]' + entity2_text + '[/E2]' + \
                document[entities[relation["child"]]["end"]:]

            marked_document_tokens = tokenizer.tokenize(marked_document)
            marked_document_token_ids = tokenizer.convert_tokens_to_ids(
                marked_document_tokens)

            # Find the start and end positions of the entities in the tokenized document
            entity1_start_token, entity1_end_token = None, None
            entity2_start_token, entity2_end_token = None, None

            for i, token in enumerate(document_tokens):
                if token == entity1_tokens[0] and document_tokens[i:i+len(entity1_tokens)] == entity1_tokens:
                    entity1_start_token = i
                    entity1_end_token = i + len(entity1_tokens) - 1
                if token == entity2_tokens[0] and document_tokens[i:i+len(entity2_tokens)] == entity2_tokens:
                    entity2_start_token = i
                    entity2_end_token = i + len(entity2_tokens) - 1

            processed_data.append({
                "document": document,
                "marked_document": marked_document,
                "marked_document_tokens": marked_document_tokens,
                "marked_document_token_ids": marked_document_token_ids,
                "entity1": entity1_text,
                "entity1_start": entity1_start_token,
                "entity1_end": entity1_end_token,
                "entity2": entity2_text,
                "entity2_start": entity2_start_token,
                "entity2_end": entity2_end_token,
                "relation_label": relation_label
            })

    return processed_data


tokenized_data = extract_relations(data)
tokenized_df = pd.DataFrame(tokenized_data)

print("Sample masked doc: ", tokenized_df.loc[0]['marked_document'])
tokenized_df


# Relationship labels
relation_labels = [
    "INCLUDE_SERVICE",
    "HAS_FUNCTION",
    "ACCESS",
    "USES"
]

label2idx = {label: idx for idx, label in enumerate(relation_labels)}
label2idx


class RelationshipDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        marked_document = item["marked_document"]
        relation_label = label2idx[item["relation_label"]]

        # Tokenize and encode the marked document
        encoding = self.tokenizer(marked_document,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_length,
                                  return_tensors='pt')

        # Prepare the input features
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        token_type_ids = encoding['token_type_ids'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': torch.tensor(relation_label, dtype=torch.long)
        }


# Dataset split
train_data, val_data = train_test_split(
    tokenized_data, test_size=0.2, random_state=32)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 512

train_dataset = RelationshipDataset(train_data, tokenizer, max_length)
val_dataset = RelationshipDataset(val_data, tokenizer, max_length)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(
    train_dataset, sampler=train_sampler, batch_size=16)
val_dataloader = DataLoader(val_dataset, batch_size=16)

num_labels = len(relation_labels)
print("No of labels: ", num_labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 10
total_steps = len(train_dataloader) * num_epochs
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


log_dir = '/content/logs/'
writer = SummaryWriter(log_dir)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir /content/logs/ --host=0.0.0.0 --port=6006

current_time = datetime.now().strftime("%Y-%m-%d_%H")

save_directory = f"/content/drive/MyDrive/Final Year Prototype/Models/Test/Models-{current_time}"
os.makedirs(save_directory, exist_ok=True)

best_val_loss = float('inf')
eval_every_n_epochs = 5

model.train()

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    total_train_loss = 0
    total_train_accuracy = 0

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        model.zero_grad()

        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels)

        loss = outputs[0]
        logits = outputs[1]
        total_train_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        # Update training accuracy
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_train_accuracy += flat_accuracy(logits, label_ids)

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)

    print(f"Average training loss: {avg_train_loss}")
    print(f"Average training accuracy: {avg_train_accuracy}")

    writer.add_scalar("Training Loss", avg_train_loss, epoch)

    if (epoch + 1) % eval_every_n_epochs == 0:
        # Validation
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0

        y_true = []
        y_pred = []

        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            with torch.no_grad():
                outputs = model(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=labels)

            loss = outputs[0]
            logits = outputs[1]

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

            y_true.extend(label_ids.tolist())
            y_pred.extend(np.argmax(logits, axis=1).tolist())

        avg_val_loss = total_eval_loss / len(val_dataloader)
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        # Log the validation loss and accuracy to TensorBoard
        writer.add_scalar("Validation Loss", avg_val_loss, epoch)
        writer.add_scalar("Validation Accuracy", avg_val_accuracy, epoch)

        print(f"Validation loss: {avg_val_loss}")
        print(f"Validation accuracy: {avg_val_accuracy}")

        if avg_val_loss < best_val_loss:
            print("\nBest validation loss found. Saving model...")
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(
                save_directory, f"best_model_{avg_val_loss}.pt")
            torch.save(model.state_dict(), model_save_path)

        # Calculate evaluation metrics
            # Calculate evaluation metrics
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(
            y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(
            y_true, y_pred, average='weighted', zero_division=1)

        print(f"Weighted F1-score: {f1}")
        print(f"Weighted Precision: {precision}")
        print(f"Weighted Recall: {recall}")

        # Log the evaluation metrics to TensorBoard
        writer.add_scalar("F1-score", f1, epoch)
        writer.add_scalar("Precision", precision, epoch)
        writer.add_scalar("Recall", recall, epoch)

        # Print classification report
        class_report = classification_report(y_true, y_pred, zero_division=1)
        print("\nClassification report:\n", class_report)

# Close the SummaryWriter object to free up system resources
writer.close()

plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

# Accuracy curve
plt.figure()
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.show()

# Save the fine-tuned BERT model
model_save_path = 'fine_tuned_bert_relationship_model'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Save the model to drive
save_directory = f"/content/drive/MyDrive/Final Year Prototype/F-Models/Fine-Tuned-Models-{current_time}"
os.makedirs(save_directory, exist_ok=True)

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)


def mark_entities(sentence, entity1, entity2):
    marked_sentence = sentence.replace(entity1, f"[E1]{entity1}[/E1]")
    marked_sentence = marked_sentence.replace(entity2, f"[E2]{entity2}[/E2]")
    return marked_sentence


def prepare_input(sentence, entity1, entity2, tokenizer, max_length=512):
    marked_sentence = mark_entities(sentence, entity1, entity2)
    encoding = tokenizer(marked_sentence,
                         padding='max_length',
                         truncation=True,
                         max_length=max_length,
                         return_tensors='pt')
    return encoding


def predict_relationship(model, tokenizer, sentence, entity1, entity2, idx2label, device='cuda'):
    model.eval()
    model.to(device)

    # Prepare the input
    encoding = prepare_input(sentence, entity1, entity2, tokenizer)

    # Move tensors to the device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)

    # Make the prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

    logits = outputs[0]
    predicted_index = torch.argmax(logits, dim=1).item()
    predicted_label = idx2label[predicted_index]

    return predicted_label


# Example input
sentence = "EG is an energy provider who allows their customers to access the service using web browsers and also from mobile devices. The online service will allow users to login to the system and submit the electricity and gas meter readings. EG will notify users on successful submission of meter readings also sends the bill amount based on the calculation."
entity1 = "online service"
entity2 = "authentication"

# Create idx2label dictionary
idx2label = {idx: label for label, idx in label2idx.items()}

# Make a prediction
predicted_label = predict_relationship(
    model, tokenizer, sentence, entity1, entity2, idx2label)
print(f"Predicted relationship: {predicted_label}")


def get_true_and_predicted_labels(model, tokenizer, dataset, idx2label, device='cuda'):
    true_labels = []
    predicted_labels = []

    print(dataset[0])

    for item in dataset:
        sentence = item['document']
        entity1 = item['entity1']
        entity2 = item['entity2']
        true_label = item['relation_label']

        # Make a prediction
        predicted_label = predict_relationship(
            model, tokenizer, sentence, entity1, entity2, idx2label, device)

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    return true_labels, predicted_labels


true_labels, predicted_labels = get_true_and_predicted_labels(
    model, tokenizer, tokenized_data, idx2label)


report = classification_report(true_labels, predicted_labels, zero_division=1)
print(report)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy}")

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("\nConfusion Matrix:")
print(conf_matrix)


def plot_confusion_matrix(conf_matrix, labels, title='Confusion Matrix', cmap='Blues', normalize=False):
    if normalize:
        conf_matrix = conf_matrix.astype(
            'float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        title = 'Normalized ' + title

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f' if normalize else 'd',
                cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()


plot_confusion_matrix(conf_matrix, labels=list(
    label2idx.keys()), normalize=True)
