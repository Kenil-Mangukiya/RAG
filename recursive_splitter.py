from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

text_paragraph = """
This line is crafted to contain exactly seventy six characters total. Now!!!

This line has exactly forty six characters. OK

This longer paragraph line is intentionally written to reach a very specific character count so it can be used to test how recursive and text splitters behave in RAG pipelines.!
"""

# split = CharacterTextSplitter(
#     chunk_size = 100,
#     chunk_overlap = 0,
#     separator = " "
# )

# result = split.split_text(text_paragraph)

# print(f"Result is : ",result)

split = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separators = ["\n\n","\n", ". ", " ", ""]
)

result = split.split_text(text_paragraph)

print(f"Result is : {result}")



