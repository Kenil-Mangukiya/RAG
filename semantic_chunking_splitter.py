from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_paragraph = """
This line is crafted to contain exactly seventy six characters total. Now!!!
This line is crafted to contain exactly seventy six characters total. Now!!!


This line has exactly forty six characters. OK
it includes seventy character total so far

This longer paragraph line is intentionally written to reach a very specific character count so it can be used to test how recursive and text splitters behave in RAG pipelines.!
"""

split = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70
)

chunks = split.split_text(text_paragraph)

print("Semantic chunks are :--------- ")

print(f"Chunks are : {chunks}")