from datasets import load_dataset
from langchain.text_splitter import Document

class DatasetProcessor:
    """
    Class for processing datasets and preparing documents.

    Attributes:
        dataset_name (str): Name of the dataset to be processed.
    """

    def __init__(self, dataset_name):
        """
        Initialize DatasetProcessor with the dataset name.

        Args:
            dataset_name (str): Name of the dataset to be processed.
        """
        self.dataset_name = dataset_name

    def get_documents(self):
        """
        Load the dataset, extract dialogues, and prepare documents.

        Returns:
            List[Document]: List of documents created from dialogues.
        """
        # Load the specified dataset
        dataset = load_dataset(self.dataset_name)

        # Extract dialogues from the dataset
        dialogues = dataset['train']['dialogue']

        # Prepare documents as instances of the Document class
        documents = [Document(page_content=dialogue) for dialogue in dialogues]

        # Return the documents
        return documents
