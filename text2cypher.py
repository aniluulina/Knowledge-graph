import fitz  # PyMuPDF
import openai
import csv
import re
from typing import List, Dict
import os

class PDFTextExtractor:
    def extract_text(self, pdf_path: str) -> str:
        document = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text

class LLMProcessor:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def generate_prompt(self, text: str) -> str:
        # return f"""
        # You are a data scientist working for a company that is building a graph database. Your task is to extract information from data and convert it into a graph database. 
        # Provide a set of Nodes in the form [ENTITY, TYPE, PROPERTIES] and a set of relationships in the form [ENTITY1, RELATIONSHIP, ENTITY2, PROPERTIES]. 
        # Pay attention to the type of the properties, if you can't find data for a property set it to null. Don't make anything up and don't add any extra data. If you can't find any data for a node or relationship don't add it.
        # Only add nodes where the TYPE is 'Company'. Relationships should still include any valid ENTITY1 and ENTITY2 pairs even if one or both are not 'Company' nodes.
        # data: {text}
        # """
    
        return f"""
        You are a data scientist working for a company that builds a knowledge graph database. Your task is to extract information from text data and convert it into a graph database.
        Text data is especially related with company's business reports. Therefore, the knowledge graph would look like company business knowledge graph.
        Provide a set of Nodes in the form [ENTITY, TYPE, PROPERTIES] and a set of relationships in the form [ENTITY1, RELATIONSHIP, ENTITY2, PROPERTIES]. 
        Pay attention to the type of the nodes and properties, if you can't find data for property set it to blank. Don't make anything up and don't add any extra data. If you can't find any data for a node or relationship don't add it.
        Add nodes whose type is 'COMPANY'.
        data: {text}
        """

    def extract_nodes_and_relationships(self, text: str) -> Dict:
        prompt = self.generate_prompt(text)
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0
        )
        result = response['choices'][0]['message']['content'].strip()

        print("LLM Response:\n", result)

        return self.get_nodes_and_relationships_from_result(result)

    def get_nodes_and_relationships_from_result(self, result: str) -> Dict:
        regex = r"Nodes:\s+(.*?)\s?Relationships:\s?(.*)"
        internal_regex = r"\[(.*?)\]"
        nodes = []
        relationships = []
        
        parsing = re.match(regex, result, flags=re.S)
        if parsing:
            raw_nodes = str(parsing.group(1))
            raw_relationships = parsing.group(2)
            nodes.extend(re.findall(internal_regex, raw_nodes))
            relationships.extend(re.findall(internal_regex, raw_relationships))

        return {
            "nodes": [self.convert_to_dict(node, is_relationship=False) for node in nodes],
            "relationships": [self.convert_to_dict(relationship, is_relationship=True) for relationship in relationships if len(relationship.split(',')) > 2]
        }

    def convert_to_dict(self, data: str, is_relationship: bool) -> Dict:
        elements = [e.strip() for e in data.split(",")]
        if is_relationship:
            return {
                "entity1": elements[0],
                "type": elements[1] if len(elements) > 1 else None,
                "entity2": elements[2] if len(elements) > 2 else None,
                "properties": eval(elements[3]) if len(elements) > 3 and elements[3] != "null" else {}
            }
        else:
            return {
                "entity": elements[0],
                "type": elements[1] if len(elements) > 1 else None,
                "properties": eval(elements[2]) if len(elements) > 2 and elements[2] != "null" else {}
            }

class CSVExporter:
    def export_to_csv(self, nodes: List[Dict], relationships: List[Dict], output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        nodes_file = os.path.join(output_dir, "nodes.csv")
        relationships_file = os.path.join(output_dir, "relationships.csv")
        
        with open(nodes_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar='\\')
            writer.writerow(["entity", "type", "properties"])
            for node in nodes:
                cleaned_node = [field.replace('"', '').replace('\\', '') for field in [node["entity"], node["type"], str(node["properties"])]]
                writer.writerow(cleaned_node)

        with open(relationships_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_NONE, escapechar='\\')
            writer.writerow(["entity1", "relationship", "entity2", "properties"])
            for relationship in relationships:
                cleaned_relationship = [field.replace('"', '').replace('\\', '') for field in [relationship["entity1"], relationship["type"], relationship["entity2"], str(relationship["properties"])]]
                writer.writerow(cleaned_relationship)

        return nodes_file, relationships_file

    def import_csv_to_neo4j(self, graph_uri: str, username: str, password: str, nodes_file: str, relationships_file: str):
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(graph_uri, auth=(username, password))

        with driver.session() as session:
            session.run("""
            LOAD CSV WITH HEADERS FROM 'file:///nodes_for_apple.csv' AS row
            CREATE (n:NodeType {name: row.entity})
            SET n += row.properties
            """)

            session.run("""
            LOAD CSV WITH HEADERS FROM 'file:///relationships_for_apple.csv' AS row
            MATCH (a {name: row.entity1}), (b {name: row.entity2})
            CREATE (a)-[r:RELATIONSHIP_TYPE]->(b)
            SET r += row.properties
            """)

class PDFToGraphPipeline:
    def __init__(self, api_key: str, graph_uri: str, user: str, password: str):
        self.text_extractor = PDFTextExtractor()
        self.llm_processor = LLMProcessor(api_key)
        self.csv_exporter = CSVExporter()
        self.graph_uri = graph_uri
        self.user = user
        self.password = password

    def process_pdf(self, pdf_path: str, output_dir: str):
        text = self.text_extractor.extract_text(pdf_path)
        
        result = self.llm_processor.extract_nodes_and_relationships(text)
        
        nodes_file, relationships_file = self.csv_exporter.export_to_csv(result["nodes"], result["relationships"], output_dir)
        
        self.csv_exporter.import_csv_to_neo4j(self.graph_uri, self.user, self.password, nodes_file, relationships_file)





if __name__ == "__main__":
    api_key = 
    neo4j_uri = 
    neo4j_user = 
    neo4j_password = 
    output_dir = 

    pipeline = PDFToGraphPipeline(api_key, neo4j_uri, neo4j_user, neo4j_password)
    
    pipeline.process_pdf("apple_2023_1-20.pdf", output_dir) 
