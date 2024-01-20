import spacy

class DataPreProcesser:

    def __init__(self):
        self.lan_model = spacy.load('en_core_web_trf')

    # Sentence Tokenization
    def sentence_tokenization(self, data):
        document = self.lan_model(data)
        return [sent.text for sent in document.sents]

    # Word Tokenization
    def word_tokenization(self, data):
        document = self.lan_model(data)
        return [token.text for token in document]

    # POS Tagging
    def pos_tagging(self, data):
        document = self.lan_model(data)
        return [(token.text, token.pos_) for token in document]

    # Dependency Parsing
    def dependency_parsing(self, data):
        document = self.lan_model(data)
        return [(token.text, token.dep_) for token in document]

    # Standard Named Entity Recognition
    def default_ner(self, data):
        document = self.lan_model(data)
        return [(ent.text, ent.label_) for ent in document.ents]
    
    # Custom Named Entity Recognition
    def custom_ner(self, data):
        ner_model = spacy.load("./models/NER/model-best_v2")

        for doc in ner_model.pipe([data], disable=["tagger", "parser"]):
            return [(ent.text, ent.label_) for ent in doc.ents]
    
    # Dependecy based Verb Extraction
    def extract_verbs_between_entities(self, sentence):
        doc = self.lan_model(sentence)
        verbs = [token for token in doc if token.pos_ == "VERB"]
        return verbs
        
    def extract_triplets_from_sentence(self, sentence):
        triplets = []

        doc = self.lan_model(sentence)

        for token in doc:
            if token.dep_ in ("attr", "dobj"):
                subj = [t for t in token.head.lefts if t.dep_ == "nsubj"]
                if subj:
                    triplets.append((subj[0], token.head, token))
            elif token.dep_ == "prep":
                subj = [t for t in token.head.lefts if t.dep_ == "nsubj"]
                obj = [t for t in token.rights if t.dep_ in ("pobj", "attr", "dobj")]
                if subj and obj:
                    triplets.append((subj[0], token.head, obj[0]))

        return triplets
