import json
from langchain.prompts import PromptTemplate

class PromptTemplateManager:
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.map_prompt = PromptTemplate.from_template(data['map_template'])
        self.reduce_prompt = PromptTemplate.from_template(data['reduce_template'])

    def get_map_prompt(self):
        return self.map_prompt

    def get_reduce_prompt(self):
        return self.reduce_prompt

    def save_to_json(self, json_file):
        data = {
            'map_template': self.map_prompt.template,
            'reduce_template': self.reduce_prompt.template
        }
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)


# Example usage:
json_file = "prompt_templates.json"

# Create an instance of PromptTemplateManager by reading templates from JSON file
prompt_manager = PromptTemplateManager(json_file)

# Retrieve map and reduce prompts
map_prompt = prompt_manager.get_map_prompt()
reduce_prompt = prompt_manager.get_reduce_prompt()


