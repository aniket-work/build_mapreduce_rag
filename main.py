import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import HuggingFacePipeline
from PromptTemplateManager import PromptTemplateManager
from ChainManager import ChainManager

def main():
    """
    Main function to execute the processing pipeline.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize tokenizer and model for text summarization
    tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization", max_length=512)
    model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization")

    # Initialize HuggingFace pipeline for summarization
    pipe = pipeline(
        "summarization", 
        model=model, 
        tokenizer=tokenizer, 
        min_length=28,
        max_length=60,    
        temperature=0.9,
        do_sample=True
    )

    # Initialize HuggingFace pipeline for Language Model operations
    llm = HuggingFacePipeline(pipeline=pipe)

    # Load text using Langchain text loader
    text_loader = TextLoader("story.txt")  # Adjust path as needed

    # Initialize PromptTemplateManager to manage prompts
    prompt_manager = PromptTemplateManager("prompt_templates.json")

    # Retrieve map and reduce prompts
    map_prompt = prompt_manager.get_map_prompt()
    reduce_prompt = prompt_manager.get_reduce_prompt()
    logging.info("Map Prompt: %s", map_prompt)
    logging.info("Reduce Prompt: %s", reduce_prompt)

    # Initialize RecursiveCharacterTextSplitter for splitting text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=30)

    # Load text documents using Langchain text loader
    split_docs = text_loader.load()
    logging.info("Loaded %d documents", len(split_docs))

    # Split documents if necessary
    split_docs = text_splitter.split_documents(split_docs)
    logging.info("Split %d documents", len(split_docs))

    # Initialize ChainManager for managing processing chains
    chain_manager = ChainManager(llm, map_prompt, reduce_prompt)

    # Execute processing pipeline on the split documents
    result = chain_manager.execute_processing_pipeline(split_docs)
    logging.info("Execution result: %s", result)

if __name__ == "__main__":
    main()
