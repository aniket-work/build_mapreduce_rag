from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain

class ChainManager:
    def __init__(self, llm, map_template, reduce_template):
        self.llm = llm
        
        self.map_prompt = map_template
        self.reduce_prompt = reduce_template

        self.map_chain = LLMChain(llm=self.llm, prompt=self.map_prompt)
        self.reduce_chain = LLMChain(llm=self.llm, prompt=self.reduce_prompt)
        
        self.combine_documents_chain = StuffDocumentsChain(
            llm_chain=self.reduce_chain, document_variable_name="doc_summaries"
        )
        self.reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=self.combine_documents_chain,
            collapse_documents_chain=self.combine_documents_chain,
            token_max=4000,
        )
        self.map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=self.map_chain,
            reduce_documents_chain=self.reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
        )

    def execute_processing_pipeline(self, documents):
        # Execute the processing pipeline on the provided documents
        return self.map_reduce_chain.run(documents)
