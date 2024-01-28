from datasets import load_dataset
from langchain.text_splitter import Document

class DatasetProcessor:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_documents(self):
        # Load the specified dataset
        dataset = load_dataset(self.dataset_name)

        # Extract dialogue from the dataset
        dialogues = dataset['train']['dialogue']

        # Prepare documents as instances of the Document class
        documents = [Document(page_content=dialogue) for dialogue in dialogues]

        # Return the documents
        return documents
