from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# text = "Hi I am using rag system here"

# embedding = embedding_model.embed_query(text)

# print(len(embedding))