import logging
import transformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import HuggingFacePipeline
from PromptTemplateManager import PromptTemplateManager
from ChainManager import ChainManager

# Configure logging
logging.basicConfig(level=logging.INFO)

tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization", max_length=512)

model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization")
prompt_manager = PromptTemplateManager("prompt_templates.json")

pipe = pipeline(
    "summarization", 
    model=model, 
    tokenizer=tokenizer, 
    min_length=28,
    max_length=60,    
    temperature=0.9,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=pipe)

# Load text using Langchain text loader
text_loader = TextLoader("story.txt")  # Adjust path as needed

# Retrieve map and reduce prompts
map_prompt = prompt_manager.get_map_prompt()
reduce_prompt = prompt_manager.get_reduce_prompt()
logging.info("Map Prompt: %s", map_prompt)
logging.info("Reduce Prompt: %s", reduce_prompt)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20)

split_docs = text_loader.load()  # Load text using Langchain text loader
logging.info("Loaded %d documents", len(split_docs))

# Splitting documents if necessary
split_docs = text_splitter.split_documents(split_docs)
logging.info("Split %d documents", len(split_docs))

chainManager = ChainManager(llm, map_prompt, reduce_prompt)
result = chainManager.execute_processing_pipeline(split_docs)
logging.info("Execution result: %s", result)

