# Handle the processing of the data
import re
import json
import logging
from inflect import engine
from scripts.data_preprocessing import DataPreProcesser
from scripts.persistence_service import PersistenceService
from scripts.model_handler import ModelHandler
from scripts.diagram_generator import DiagramGenerator


class Processor:
    def __init__(self, input_data):
        self.data = input_data
        self.modleHandeler = ModelHandler()
        self.preprocessor = DataPreProcesser()
        self.persistence_service = PersistenceService()

        self.words = []
        self.sentences = []
        self.custom_ne = []
        self.default_ne = []

        self.architecture_data = {
            "diagram_name": "Context Diagram",
            "organization": "",
            "actors": [],
            "systems": [],
            "devices": [],
            "databases": [],
            "containers": [],

            "payments": [],
            "notification": [],
            "authentication": [],

            "processes": [],
            "relationships": [],
            "intergrations": []
        }

        self.sentence_level_relationships = []

    def process(self):
        logging.info("Pre-Processing data...")

        self.sentences = self.preprocessor.sentence_tokenization(self.data)
        extracted_ner = self.preprocessor.default_ner(self.data)

        self.default_ne = list(set(extracted_ner))

        # Identify Diagram Name based on Default NER
        self.prefill_data()
        other_relationships_identified = []

        for sentence in self.sentences:
            self.sentence_level_data_creation(sentence)
            other_relationships_identified.append(self.preprocessor.extract_triplets_from_sentence(sentence))

        print(other_relationships_identified)
        diagram_name = DiagramGenerator.generate_diagram_json(self.architecture_data)
        diagram_code = DiagramGenerator.generate_text_representation(self.architecture_data)

        # Generate container diagram
        container_architecture = self.generate_container_diagram(self.architecture_data);
        container_diagram_name = DiagramGenerator.generate_diagram_json(container_architecture)
        container_code = DiagramGenerator.generate_text_representation(container_architecture)

        return {
            "diagram_name": diagram_name,
            "diagram_code": diagram_code,
            "container_diagram_name": container_diagram_name,
            "container_code": container_code
        }

    def prefill_data(self):
        """
            Prefill the diagram name based on the default NER
        """
        for entity in self.default_ne:
            if 'ORG' in entity[1]:
                self.architecture_data["diagram_name"] += " for " + entity[0]
                self.architecture_data["organization"] = entity[0]
                if 'System' not in entity[0]:
                    self.architecture_data["diagram_name"] += " Service"
                break;

            elif 'PRODUCT' in entity[1]:
                self.architecture_data["diagram_name"] = entity[0] + \
                    " " + self.architecture_data["diagram_name"]
                self.architecture_data["organization"] = entity[0]
                break;

        print(self.architecture_data["diagram_name"])

    def identify_components(self, entity_pairs):
        """
            Identify the components of the architecture

            Args:
                sentence (str): Sentence
                entity_pairs (list): List of entity pairs
        """
        singularized_entity_pairs = [(engine().singular_noun(
            entity) or entity, entity_type) for entity, entity_type in entity_pairs]

        for entity, entity_type in singularized_entity_pairs:
            if entity_type == "DATABASE":
                if not any(d.get("name", "").lower() == entity.lower() for d in self.architecture_data["databases"]):
                    self.architecture_data["databases"].append({
                        "name": entity,
                        "technology": "",
                        "description": ""
                    })
            elif entity_type in ["SYSTEM", "APPLICATION"]:
                if not any(d.get("name", "").lower() == entity.lower() for d in self.architecture_data["systems"]):
                    self.architecture_data["systems"].append({
                        "name": entity,
                        "description": "",
                        "external": entity_type == "APPLICATION"
                    })
            elif entity_type == "CONTAINER":
                if not any(d.get("name", "").lower() == entity.lower() for d in self.architecture_data["containers"]):
                    self.architecture_data["containers"].append({
                        "name": entity,
                        "description": "",
                        "technology": ""
                    })
            elif entity_type == "PERSON":
                if not any(d.get("name", "").lower() == entity.lower() for d in self.architecture_data["actors"]):
                    self.architecture_data["actors"].append({
                        "name": entity,
                        "description": ""
                    })
            elif entity_type == "AUTHENTICATION":
                if not any(d.get("name", "").lower() == entity.lower() for d in self.architecture_data["authentication"]):
                    self.architecture_data["authentication"].append({
                        "name": entity,
                        "description": ""
                    })
            elif entity_type == "DEVICE":
                if not any(d.get("name", "").lower() == entity.lower() for d in self.architecture_data["devices"]):
                    self.architecture_data["devices"].append({
                        "name": entity,
                        "description": ""
                    })
            elif entity_type == "NOTIFICATION":
                if not any(d.get("name", "").lower() == entity.lower() for d in self.architecture_data["notification"]):
                    self.architecture_data["notification"].append({
                        "name": entity,
                        "description": ""
                    })
            elif entity_type == "INTEGRATION":
                if not any(d.get("name", "").lower() == entity.lower() for d in self.architecture_data["integrations"]):
                    self.architecture_data["integrations"].append({
                        "name": entity,
                        "description": ""
                    })
            elif entity_type == "PAYMENT":
                if not any(d.get("name", "").lower() == entity.lower() for d in self.architecture_data["payments"]):
                    self.architecture_data["payments"].append({
                        "name": entity,
                        "description": ""
                    })
            elif entity_type in ["PROCESS", "FUNCTION"]:
                if not any(d.get("name", "").lower() == entity.lower() for d in self.architecture_data["processes"]):
                    self.architecture_data["processes"].append({
                        "name": entity,
                        "description": entity_type
                    })

    def sentence_level_relationship_extraction(self, sentence, entity_pairs):
        """
            Extract the relationships between the entities in a sentence

            Args:
                sentence (str): Sentence
                entity_pairs (list): List of entity pairs
        """
        singularized_entity_pairs = [(engine().singular_noun(
            entity) or entity, entity_type) for entity, entity_type in entity_pairs]
        exclude_pairs = [("FUNCTION", "FUNCTION"), ("FUNCTION", "PROCESS"),
                         ("PROCESS", "FUNCTION"), ("DEVICE", "DEVICE")]

        # Each entity should be checked with all the other entities for possible relationships
        for i in range(len(singularized_entity_pairs)):
            for j in range(i+1, len(entity_pairs)):
                entity_one = singularized_entity_pairs[i]
                entity_two = singularized_entity_pairs[j]

                # Skip relationship creation if entity pair has excluded types
                if (entity_one[1], entity_two[1]) in exclude_pairs or (entity_two[1], entity_one[1]) in exclude_pairs:
                    continue

                relation = self.modleHandeler.use_transformer(
                    sentence, entity_one[0], entity_two[0])

                relationship_data = {
                    "entityOne": {
                        "name": entity_one[0],
                        "type": entity_one[1]
                    },
                    "entityTwo": {
                        "name": entity_two[0],
                        "type": entity_two[1]
                    },
                    "relationship": relation,
                    "sentence": sentence
                }

                target_data = {
                    "source": entity_one[0],
                    "relationship": relation,
                    "target": entity_two[0]
                }

                self.sentence_level_relationships.append(relationship_data)
                self.architecture_data["relationships"].append(target_data)

    def structure_data(self, is_container=False):
        # Add the entities to the architecture data
        if self.architecture_data["systems"]:
            for system in self.architecture_data["systems"]:
                if re.search(r"online service|online application|system", system["name"], re.IGNORECASE):
                    self.architecture_data["systems"].remove(system)
                    self.architecture_data["systems"].append(
                        {
                            "name": "Backend Service",
                            "description": system["name"] or "",
                            "external": False
                        }
                    )

                    # iterate over the relationships and update the source and target
                    for relationship in self.architecture_data["relationships"]:
                        if relationship["source"] == system["name"]:
                            relationship["source"] = "Backend Service"
                        if relationship["target"] == system["name"]:
                            relationship["target"] = "Backend Service"

        if self.architecture_data["notification"]:
            for notification in self.architecture_data["notification"]:
                if re.search(r"notify|notification|alert|alarm", notification["name"], re.IGNORECASE):
                    self.architecture_data["systems"].append(
                        {
                            "name": "Notification",
                            "description": notification["name"] or "",
                            "external": False
                        }
                    )

                    for relationship in self.architecture_data["relationships"]:
                        if relationship["source"] == notification["name"]:
                            relationship["source"] = "Notification"
                        if relationship["target"] == notification["name"]:
                            relationship["target"] = "Notification"

                elif re.search(r"message|messaging|email|sms", notification["name"], re.IGNORECASE):
                    self.architecture_data["systems"].append(
                        {
                            "name": "Messaging",
                            "description": notification["name"] or "",
                            "external": False
                        }
                    )

                    for relationship in self.architecture_data["relationships"]:
                        if relationship["source"] == notification["name"]:
                            relationship["source"] = "Messaging"
                        if relationship["target"] == notification["name"]:
                            relationship["target"] = "Messaging"

        if self.architecture_data["authentication"]:
            for authentication in self.architecture_data["authentication"]:
                if re.search(r"authentication|auth|login|sign in", authentication["name"], re.IGNORECASE):
                    self.architecture_data["systems"].append(
                        {
                            "name": "Authentication",
                            "description": authentication["name"] or "",
                            "external": False
                        }
                    )

                    for relationship in self.architecture_data["relationships"]:
                        if relationship["source"] == authentication["name"]:
                            relationship["source"] = "Authentication"
                        if relationship["target"] == authentication["name"]:
                            relationship["target"] = "Authentication"

        if self.architecture_data["payments"]:
            for payment in self.architecture_data["payments"]:
                if re.search(r"payment|pay|checkout|bill", payment["name"], re.IGNORECASE):
                    self.architecture_data["systems"].append(
                        {
                            "name": "Payments",
                            "description": payment["name"] or "",
                            "external": False
                        }
                    )

                    for relationship in self.architecture_data["relationships"]:
                        if relationship["source"] == payment["name"]:
                            relationship["source"] = "Payments"
                        if relationship["target"] == payment["name"]:
                            relationship["target"] = "Payments"

        if is_container and len(self.architecture_data["devices"]) > 0:
            for device in self.architecture_data["devices"]:
                self.architecture_data["containers"].append(
                    {
                        "name": device["name"],
                        "description": "",
                        "technology": ""
                    }
                )

    def sentence_level_data_creation(self, sentence):
        entity_pairs = self.preprocessor.custom_ner(sentence)
        # System, Container, Database, Function
        self.identify_components(entity_pairs)

        # Extract relationships between entities
        self.sentence_level_relationship_extraction(sentence, entity_pairs)

        # Rule based stucturing of data
        self.structure_data(is_container=True)

        # Remove duplicates from the data
        self.remove_duplicates()

    def remove_duplicates(self):
        # Remove duplicate systems
        self.architecture_data["systems"] = list(
            {system["name"]: system for system in self.architecture_data["systems"]}.values())

        # Remove duplicate containers
        self.architecture_data["containers"] = list(
            {container["name"]: container for container in self.architecture_data["containers"]}.values())

        # Remove duplicate databases
        self.architecture_data["databases"] = list(
            {database["name"]: database for database in self.architecture_data["databases"]}.values())

        # Remove duplicate relationships
        self.architecture_data["relationships"] = list(
            {relationship["source"] + relationship["relationship"] + relationship["target"]: relationship for relationship in self.architecture_data["relationships"]}.values())

    def generate_container_diagram(self, architecture_data):
        # Generate the container diagram
        for container in architecture_data["containers"]:
            possible_relationships = self.persistence_service.find_possible_relationships_by_entity_and_relation(container['name'].lower(), "include_service")
            for relation in possible_relationships:
                architecture_data["systems"].append(
                    {
                        "name": relation['entity2'],
                        "description": "",
                        "external": False
                    }
                )

                architecture_data["relationships"].append(
                    {
                        "source": container['name'],
                        "relationship": "include_service".upper(),
                        "target": relation['entity2']
                    }
                )

        for system in architecture_data["systems"]:
            possible_relationships = self.persistence_service.find_possible_relationships_by_entity_and_relation(system['name'].lower(), "include_service")
            for relation in possible_relationships:
                architecture_data["systems"].append(
                    {
                        "name": relation['entity2'],
                        "description": "",
                        "external": False
                    }
                )

                architecture_data["relationships"].append(
                    {
                        "source": system['name'],
                        "relationship": "include_service".upper(),
                        "target": relation['entity2']
                    }
                )
        
        for database in architecture_data["databases"]:
            possible_relationships = self.persistence_service.find_possible_relationships_by_entity_and_relation(database['name'].lower(), "include_service")
            for relation in possible_relationships:
                architecture_data["systems"].append(
                    {
                        "name": relation['entity2'],
                        "description": "",
                        "external": False
                    }
                )

                architecture_data["relationships"].append(
                    {
                        "source": database['name'],
                        "relationship": "include_service".upper(),
                        "target": relation['entity2']
                    }
                )

        return architecture_data

        

