# Import necessary libraries
import os
import re
import json
import uuid

from typing import List
from diagrams import Diagram
from diagrams.c4 import Person, Container, Database, System, SystemBoundary, Relationship

class DiagramGenerator:
    def __init__(self):
        pass

    # Json to object convertion
    @staticmethod
    def json_to_architecture_data(json_data):
        actors = [ActorDTO(
            name=actor["name"],
            description=actor["description"]
        ) for actor in json_data["actors"]]

        containers = [ContainerDTO(
            name=container["name"],
            technology=container["technology"],
            description=container["description"]
        ) for container in json_data["containers"]]

        databases = [DatabaseDTO(
            name=db["name"],
            technology=db["technology"],
            description=db["description"]
        ) for db in json_data["databases"]]

        systems = [SystemDTO(
            name=system["name"],
            description=system["description"],
            external=system.get("external", False)
        ) for system in json_data["systems"]]

        relationships = [RelationshipDTO(
            source=relationship["source"],
            relationship=relationship["relationship"],
            target=relationship["target"]
        ) for relationship in json_data["relationships"]]

        return ArchitectureData(
            diagram_name=json_data["diagram_name"],
            actors=actors,
            containers=containers,
            databases=databases,
            systems=systems,
            relationships=relationships)

    # Function to generate diagrams using diagrams api
    @staticmethod
    def generate_diagram_default():
        graph_attr = {
            "splines": "spline",
        }

        # Create the 'outputs' folder if it doesn't exist
        if not os.path.exists('outputs'):
            os.makedirs('outputs')

        # Set the output path to the 'outputs' folder
        out_path = os.path.join('outputs', 'internet_banking_2.png')

        with Diagram("Container diagram for Internet Banking System", direction="TB", graph_attr=graph_attr, outformat='png', filename=out_path, show=False):
            customer = Person(
                name="Customer"
            )

            with SystemBoundary("[ORG] System"):
                webapp = Container(
                    name="[ORG] User Account"
                )

                spa = Container(
                    name="[ORG] Backend"
                )

                mobileapp = Container(
                    name="Payments"
                )

                api = Container(
                    name="Notification"
                )

            customer >> Relationship("Uses") >> webapp
            webapp >> Relationship("Uses") >> spa
            spa >> Relationship("Raises") >> mobileapp
            webapp >> Relationship("Sends") >> api
            api >> Relationship("sends") >> webapp

    # Function to generate diagrams
    @staticmethod
    def generate_diagram_json(architecture_data):
        splines_type = "spline"
        output_filetype = "png"
        filename = str(uuid.uuid4())

        # Convert json to objects
        data = DiagramGenerator.json_to_architecture_data(architecture_data)

        if not os.path.exists('outputs'):
            os.makedirs('outputs')

        out_path = os.path.join('outputs', filename)

        # Diagram rendering
        graph_attr = {"splines": splines_type, }
        with Diagram(data.diagram_name, direction="TB", graph_attr=graph_attr, outformat=output_filetype, filename=out_path, show=False):
            actors = {actor.name: Person(
                name=actor.name, description=actor.description) for actor in data.actors}

            with SystemBoundary("Main System"):
                containers = {container.name: Container(
                    name=container.name, description=container.description) for container in data.containers}
                databases = {database.name: Database(
                    name=database.name, description=database.description) for database in data.databases}
                systems = {system.name: System(
                    name=system.name, description=system.description) for system in data.systems}

                for relationship in data.relationships:
                    source_name = relationship.source
                    target_name = relationship.target
                    relationship_type = relationship.relationship

                    source = actors.get(source_name, None) or containers.get(
                        source_name, None) or databases.get(source_name, None) or systems.get(source_name, None)
                    target = actors.get(target_name, None) or containers.get(
                        target_name, None) or databases.get(target_name, None) or systems.get(target_name, None)

                    if source and target:
                        source >> Relationship(relationship_type) >> target
        return filename + "." + output_filetype

    def generate_text_representation(data):
        text = f"Diagram: {data['diagram_name']}\nOrganization: {data['organization']}\n\nActors:"
        for i, actor in enumerate(data['actors'], start=1):
            text += f"\nActors {i:02}: {{{actor['name']}: {actor['description']}}}"
            
        text += "\n\nSystems:"
        for i, system in enumerate(data['systems'], start=1):
            identifier = f"{i:02}"
            text += f"\nSystems {identifier}: {{{system['name']}: {system['description']} (External: {system['external']})}}"
            
        text += "\n\nContainers:"
        for i, device in enumerate(data['containers'], start=1):
            identifier = f"{i:02}"
            text += f"\nContainers {identifier}: {{{device['name']}: {device['description']}}}"

        text += "\n\nDatabases:"
        for i, database in enumerate(data['databases'], start=1):
            identifier = f"{i:02}"
            text += f"\nDatabases {identifier}: {{{database['name']}: {database['description']}}}"

        text += "\n\nRelationships:"
        for relationship in data['relationships']:
            text += f"\n- {relationship['source']} >> {relationship['relationship']} >> {relationship['target']}"

        return text

    @staticmethod
    def parse_text_to_code(text):
        data = {
            'diagram_name': '',
            'organization': '',
            'actors': [],
            'systems': [],
            'containers': [],
            'databases': [],
            'relationships': []
        }

        for line in text.split('\n'):
            if "Diagram:" in line:
                data['diagram_name'] = line.split(":", 1)[1].strip()
            elif "Organization:" in line:
                data['organization'] = line.split(":", 1)[1].strip()
            elif "Actors:" in line:
                continue
            elif "Systems:" in line:
                continue
            elif "Containers:" in line:
                continue
            elif "Databases:" in line:
                continue
            elif "Relationships:" in line:
                continue
            elif "Actors" in line:
                actor = re.match(r"Actors \d+:\s*\{(.+?):\s*(.*)\}", line)
                data['actors'].append({'name': actor.group(1), 'description': actor.group(2)})
            elif "Systems" in line:
                system = re.match(r"Systems \d+:\s*\{(.+?):\s*(.*?)\s*(\(External:\s*(True|False)\))?\}", line)
                data['systems'].append({'name': system.group(1), 'description': system.group(2), 'external': system.group(4) == 'True'})
            elif "Containers" in line:
                container = re.match(r"Containers \d+:\s*\{(.+?):\s*(.*)\}", line)
                data['containers'].append({'name': container.group(1), 'description': container.group(2), 'technology': ''})
            elif "Databases" in line:
                database = re.match(r"Databases \d+:\s*\{(.+?):\s*(.*)\}", line)
                data['databases'].append({'name': database.group(1), 'description': database.group(2), 'technology': ''})
            elif ">>" in line:
                line = line.lstrip('- ') 
                relationship = re.match(r"(.+?)\s*>>\s*(.+?)\s*>>\s*(.+)", line)
                data['relationships'].append({'source': relationship.group(1), 'relationship': relationship.group(2), 'target': relationship.group(3)})

        return data

# Actor, Container, Database, System, Relationship DTOs
class ActorDTO:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def __repr__(self):
        return f"Actor(name={self.name!r}, description={self.description!r})"


class ContainerDTO:
    def __init__(self, name: str, technology: str, description: str):
        self.name = name
        self.technology = technology
        self.description = description

    def __repr__(self):
        return f"Container(name={self.name!r}, technology={self.technology!r}, description={self.description!r})"


class DatabaseDTO:
    def __init__(self, name: str, technology: str, description: str):
        self.name = name
        self.technology = technology
        self.description = description

    def __repr__(self):
        return f"Database(name={self.name!r}, technology={self.technology!r}, description={self.description!r})"


class SystemDTO:
    def __init__(self, name: str, description: str, external: bool = False):
        self.name = name
        self.description = description
        self.external = external

    def __repr__(self):
        return f"System(name={self.name!r}, description={self.description!r}, external={self.external!r})"


class RelationshipDTO:
    def __init__(self, source: str, relationship: str, target: str):
        self.source = source
        self.relationship = relationship
        self.target = target

    def __repr__(self):
        return f"Relationship(source={self.source!r}, relationship={self.relationship!r}, target={self.target!r})"


# ArchitectureData DTO
class ArchitectureData:
    def __init__(self, diagram_name: str, actors: List[ActorDTO], containers: List[ContainerDTO], databases: List[DatabaseDTO], systems: List[SystemDTO], relationships: List[RelationshipDTO]):
        self.diagram_name = diagram_name
        self.actors = actors
        self.containers = containers
        self.databases = databases
        self.systems = systems
        self.relationships = relationships

    def __repr__(self):
        return f"ArchitectureData(diagram_name={self.diagram_name!r}, actors={self.actors!r}, containers={self.containers!r}, databases={self.databases!r}, systems={self.systems!r}, relationships={self.relationships!r})"
