import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from embeddingModel import embedding_model
from llmModel import llm
from PromptTemplates.SummaryPromptTemplate import summary_prompt
from PromptTemplates.PromptTemplate import prompt
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough
)
from langchain_core.output_parsers import StrOutputParser
from urllib.parse import urlparse, parse_qs


load_dotenv()


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = None

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []


def format_docs_come_from_store(retrieved_doc):
    context_text = "\n\n".join(
        docs.page_content for docs in retrieved_doc
    )
    return context_text

def getSummary(url,SummaryPrompt):
    video_id = parse_qs(
        urlparse(url).query
    ).get("v", [None])[0]
    try:
        transcript_list = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en"]
        )

        raw_transcript = transcript_list.to_raw_data()

        transcript = " ".join(
            chunk["text"] for chunk in raw_transcript
        )
        parser = StrOutputParser()

        summary_chain = (
            SummaryPrompt 
            | llm
            | parser
        )

        return summary_chain.invoke({
            "transcript" : transcript
        })

        
    except Exception as e:
        return f"Transcript not available.\n\nError: {e}"
    

def create_vector_store(url):

    video_id = parse_qs(
        urlparse(url).query
    ).get("v", [None])[0]

    transcript_list = YouTubeTranscriptApi().fetch(
        video_id,
        languages=["en"]
    )

    raw_transcript = transcript_list.to_raw_data()

    transcript = " ".join(
        chunk["text"] for chunk in raw_transcript
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vector_store


def get_answer(vector_store, question):

    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    similiar_transcript = retriever.invoke(question)
    #print(similiar_transcript)

    parallelChain = RunnableParallel({
        "context": retriever
                   | RunnableLambda(format_docs_come_from_store),
        "question": RunnablePassthrough()
    })

    parser = StrOutputParser()

    mainChain = (
        parallelChain
        | prompt
        | llm
        | parser
    )

    result = mainChain.invoke(question)

    return {
        "result" : result,
        "src" : similiar_transcript
    }


# ------------------ UI ------------------

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------

st.title("🎥 YouTube Video Assistant")

st.markdown("""
Ask questions about any YouTube video or generate a complete summary using AI.
""")

st.divider()

# -----------------------------
# Inputs
# -----------------------------

youtube_url = st.text_input(
    "🔗 Enter YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

question = st.text_area(
    "❓ Ask a Question About the Video",
    placeholder="What are the main concepts discussed in the video?"
)

# -----------------------------
# Buttons
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    answer_btn = st.button(
        "🔍 Get Answer",
        use_container_width=True
    )

with col2:
    summary_btn = st.button(
        "📝 Generate Summary",
        use_container_width=True
    )

# -----------------------------
# Q&A Section
# -----------------------------

if answer_btn:

    if not youtube_url:
        st.warning("Please enter a YouTube URL.")
        st.stop()

    if not question:
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Analyzing video and generating answer..."):

        try:
            if (
            st.session_state.vector_store is None
            or st.session_state.loaded_url != youtube_url
           ):


              st.session_state.vector_store = create_vector_store(
              youtube_url
              )
              st.session_state.loaded_url = youtube_url
            #   print(st.session_state.loaded_url)
            #   print(youtube_url)
            answer = get_answer(
              st.session_state.vector_store,
              question
          )
            result = answer["result"]
            sources = answer["src"]
            st.session_state.qa_history.append(
            {
                  "question": question,
                 "answer": result
            }
)

            st.success("Answer Generated Successfully")

            st.subheader("📌 Answer")

            st.write(result)

            st.subheader("📚 Source Chunks")

            for i, doc in enumerate(sources, start=1):

                with st.expander(f"Chunk {i}"):

                   st.write(doc.page_content)

        except Exception as e:

            st.error(f"Error: {e}")

# -----------------------------
# Summary Section
# -----------------------------

if summary_btn:

    if not youtube_url:
        st.warning("Please enter a YouTube URL.")
        st.stop()

    with st.spinner("Generating video summary..."):

        try:

            summary = getSummary(
                youtube_url,
                summary_prompt
            )

            st.success("Summary Generated Successfully")

            st.subheader("📄 Video Summary")

            st.write(summary)

        except Exception as e:

            st.error(f"Error: {e}")

    # Display the history..........

if st.session_state.qa_history:


    st.divider()

    st.subheader("📜 Question History")
    delete_btn = st.button(
        "Delete History",
        use_container_width=True
    )


    for item in reversed(st.session_state.qa_history):

        with st.expander(item["question"]):

            st.write(item["answer"])
    
    if delete_btn:
        st.session_state.qa_history = []

# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "Built using LangChain • FAISS • HuggingFace Embeddings • Mistral AI • Streamlit"
)