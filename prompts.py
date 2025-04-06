from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.string import PromptTemplateFormat

prompt_template = """You are a helpful assistant. Use the following context 

Context:
{context}

and then structure the response to be cohesive and nice to read.
you may use the following query to structure or refine your response 
as needed

{query}

Answer:""".strip()

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "query"],
)
