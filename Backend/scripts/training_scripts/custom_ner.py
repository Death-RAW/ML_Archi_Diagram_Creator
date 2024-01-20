import json
import spacy
from tqdm import tqdm
from spacy.tokens import DocBin

# Import annotated labels
with open('data/annotations/sample.json', 'r') as file:
    annotated_data = json.load(file)

# Create a blank spacy model
spacy_lan_model = spacy.load('en_core_web_trf')
ner_model = spacy.blank("en")
docBin = DocBin()

def create_traning_dataset(data, docBin):
  for text, annot in tqdm(data['annotations']): 
    doc = spacy_lan_model.make_doc(text) 
    entities = []
    for start, end, label in annot["entities"]:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            entities.append(span)
    doc.ents = entities 
    docBin.add(doc)

  docBin.to_disk("./data/training_data.spacy")


# create_traning_dataset(annotated_data, docBin)

def display_entities(text, model):
  doc = model(text)

  colors = {"SYSTEM": "#AAE3E2", "COMPONENT": "#D9ACF5", "HARDWARE":"#FFCEFE", "APPLICATION": "#FDEBED", "FEATURE": "#FFE7CC","TECH REQ" : "#FFD369","BUSINESS REQ": "#E26241","USER ROLE": "#7286D3","FUNCTION": "#645CBB","MISC": "#FFEA20","LOC": "#08D9D6","DATA" : "#FCD1D1","DEPARTMENT": "#FCD8D4"}
  options = {"colors": colors} 
  spacy.displacy.render(doc, style="ent", options= options, jupyter=True)

sample_text = """EG is an energy provider who allows their customers to access the service using web browsers and also from mobile devices.
The online service will allow users to login to the system and submit the electricity and gas meter readings.
EG will notify users on successful submission of meter readings also sends the bill amount based on the calculation."""

ner_model = spacy.load("model-best") 
display_entities(sample_text, ner_model)