from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from embeddingModel import embedding_model
from llmModel import llm
from PromptTemplates.PromptTemplate import prompt
from langchain_core.runnables import RunnableParallel ,RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from urllib.parse import urlparse, parse_qs


load_dotenv()


# Step 1 : Getting the transcript of a particular video from utube
# video_id = "oX7OduG1YmI"


url = input("Enter YouTube URL: ")

video_id = parse_qs(urlparse(url).query).get("v", [None])[0]



# To get the transcript of any youtube video.........
try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id,languages=['en'])
    raw_transcript = transcript_list.to_raw_data()
    #print(type(raw_transcript))
    transcript = " ".join(chunk["text"] for chunk in raw_transcript)
    #print(transcript)
except Exception as e:
    print("No transcript is available....")
    print(f"Error: {e}")


# Step 2 : Performing splitting of the doc. which we get from the transcript
# Splitting the transcript -- Created chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, chunk_overlap = 200
)
chunks = splitter.create_documents([transcript])


# Step 3 : Creating embedding and store it into vector stores
# Embedding and vector stores

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)


# Step 4 : Created a retreiver which gonna help to fetch from the vector space and note that it will get the similiar doc.
# Creation of Retriever

# a - This retreiver help to get the most related 4 docu. from the vector DB

question = input("Ask anything related to a particular video : ")
retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})
res = retriever.invoke(question)


# LLM creation.... done in llmModel file
# Prompt -- Template -- in PromptTemplate.py file


# Step 5 : Similiar doc we get from retreiver we will now fetch the raw text from those docs
# Creating context text 

def format_docs_come_from_store(retrieved_doc):
    context_text = "\n\n".join(docs.page_content for docs in retrieved_doc)
    return context_text



# Step 5 : final prompt that contain both the data from the vectorStore and the query
final_prompt = prompt.invoke({
   "context" : format_docs_come_from_store(res),
   "question" : question
})


# Step 6 : final step
# Generation stage where llm are used and this is done by creating chains


#  Chain formation 

parallelChain = RunnableParallel({
    'context' : retriever | RunnableLambda(format_docs_come_from_store),
    'question' : RunnablePassthrough()
})



parser = StrOutputParser()

mainChain = parallelChain | prompt | llm | parser

res = mainChain.invoke(question)
print(res)