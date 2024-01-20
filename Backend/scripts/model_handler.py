import torch
import spacy
from transformers import BertForSequenceClassification, BertTokenizer
from scripts.data_preprocessing import DataPreProcesser


class ModelHandler:
    def __init__(self):
        self.preprocessor = DataPreProcesser()

    def load_transformer_model(self):
        # Load the fine-tuned model and tokenizer
        model_path = './models/RE/relationship_model_v2'
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        return model, tokenizer

    def mark_entities(self, sentence, entity1, entity2):
        marked_sentence = sentence.replace(entity1, f"[E1]{entity1}[/E1]")
        marked_sentence = marked_sentence.replace(
            entity2, f"[E2]{entity2}[/E2]")
        
        # Get the wording between the two entities
        start_index = sentence.find(entity1) + len(entity1)
        end_index = sentence.find(entity2)
        wording_between_entities = sentence[start_index:end_index].strip()
        verbs = self.preprocessor.extract_verbs_between_entities(wording_between_entities)

        return marked_sentence, verbs

    def prepare_input(self, sentence, entity1, entity2, tokenizer, max_length=512):
        marked_sentence, verbs = self.mark_entities(sentence, entity1, entity2)
        encoding = tokenizer(marked_sentence,
                             padding='max_length',
                             truncation=True,
                             max_length=max_length,
                             return_tensors='pt')
        return encoding, verbs

    def predict_relationship(self, model, tokenizer, sentence, entity1, entity2, label2idx, idx2label, device='cpu'):
        model.eval()
        model.to(device)

        # Prepare the input
        encoding, verbs = self.prepare_input(sentence, entity1, entity2, tokenizer)

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

        return predicted_label, verbs

    def use_transformer(self, sentence, entity1, entity2):

        # Create idx2label dictionary
        labels = ['INCLUDE_SERVICE', 'HAS_FUNCTION', 'ACCESS', 'USES']
        label2idx = {label: idx for idx, label in enumerate(labels)}

        idx2label = {idx: label for label, idx in label2idx.items()}

        # Load the fine-tuned model and tokenizer
        model, tokenizer = self.load_transformer_model()

        # Make a prediction
        predicted_label, verbs = self.predict_relationship(
            model, tokenizer, sentence, entity1, entity2, label2idx, idx2label)

        print(f"Predicted relationship: {predicted_label} verbs: {verbs}")
        if len(verbs) > 0:
            return f"{predicted_label}-{verbs}"
        else:
            return predicted_label
