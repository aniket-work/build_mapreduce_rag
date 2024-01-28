import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")

model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization")

dataset = load_dataset("samsum")