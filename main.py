# pip install llama-index-embeddings-ollama

import ast
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    PromptTemplate,
    Settings
)
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse

from code_reader import code_reader
from prompts import context, code_parser_template
from pydantic import BaseModel


class CodeOutput(BaseModel):
    code: str
    description: str
    filename: str


load_dotenv()
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

default_llm = Ollama(model="llama3", request_timeout=300.0)
code_llm = Ollama(model="codellama", request_timeout=30.0)


pdf_parser = LlamaParse(result_type="markdown")
code_parser = PydanticOutputParser(CodeOutput)
file_extractor = {".pdf": pdf_parser}

documents = SimpleDirectoryReader(
    input_dir="data", file_extractor=file_extractor,
).load_data()

vector_index = VectorStoreIndex.from_documents(documents=documents)
query_engine = vector_index.as_query_engine(llm=default_llm)
qet = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="api_documentation",
        description="this gives documentation about code for an API. Use this for reading docs for the API",
    ), 
)


tools = [
    qet, code_reader
]


agent = ReActAgent.from_tools(
    tools=tools, llm=code_llm, verbose=True, context=context
)

json_prompt_str = code_parser.format(code_parser_template)
json_prompt_tmpl = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, default_llm])


while (prompt := input("Ask me a question (q to quit): ")) != "q":
    retries = 0

    while retries < 3:
        try:
            result = agent.query(prompt)
            next_result = output_pipeline.run(response=result)
            cleaned_json = ast.literal_eval(str(next_result).replace("assistant:", ""))
            break
        except Exception as e:
            retries += 1
            print(f"Error occurred, retry #{retries}:", e)

    if retries >= 3:
        print("Unable to process request, try again...")
        continue

    print("Code generated")
    print(cleaned_json["code"])
    print("\n\nDescription:", cleaned_json["description"])

# read the contents of test.py and write a python script that calls the post endpoint to make a new item