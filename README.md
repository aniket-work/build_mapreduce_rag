# 📝 build_mapreduce_rag

This project demonstrates building a **MapReduce-style pipeline** using **Langchain** for text summarization. 📚

## 🛠 Setup

To set up the environment, run these commands in your terminal:

```bash
python -m venv build_mapreduce_rag
cd build_mapreduce_rag
Scripts\activate
pip install -r requirements.txt
```

This will create a **virtual environment** and install the required packages from `requirements.txt`:

- langchain
- tiktoken 
- chromadb
- pypdf
- transformers
- InstructorEmbedding
- torch
- datasets
- langchain-community
- tqdm

## 🚀 Usage

The main driver script is `main.py`. This loads a text document, splits it into chunks, summarizes each chunk, and aggregates the summaries. 🙌

Key components:

- `TextLoader` to load document text 📄
- `TextSplitter` to split text into chunks 🗂
- `PromptTemplateManager` to manage prompt templates 📝
- `ChainManager` to define and execute the MapReduce pipeline 🔗
- `HuggingFacePipeline` to call the summarization model 🤗

The pipeline works as follows:

1. Load text document 📥
2. Split text into chunks 🔪
3. Apply "map" operation to summarize each chunk 🗺
4. Apply "reduce" operation to aggregate chunk summaries 🔎
5. Output final summarized text 📤

The map and reduce operations are defined using prompt templates. 🎨

## ⚙ Configuration

The summarization model, prompt templates, and other parameters can be configured by modifying:

- `main.py` - parameters for pipeline components 🧩
- `prompt_templates.json` - defines prompt templates 🎨
- `story.txt` - sample text document 📄

## 🏃 Running

To execute the summarization pipeline, run this command in your terminal:

```bash
python main.py
```

Output will be logged to the console showing the original text, split chunks, chunk summaries, and final aggregated summary. 📊

## 🌟 Extending

To use this with a different model or task, modify the `HuggingFacePipeline` instantiation and `prompt_templates.json`. The high-level pipeline architecture can remain the same. 🏗
