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
    
    


def get_answer(url, question):

    video_id = parse_qs(
        urlparse(url).query
    ).get("v", [None])[0]

    if not video_id:
        return "Invalid YouTube URL"

    try:
        transcript_list = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en"]
        )

        raw_transcript = transcript_list.to_raw_data()

        transcript = " ".join(
            chunk["text"] for chunk in raw_transcript
        )

    except Exception as e:
        return f"Transcript not available.\n\nError: {e}"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

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
            answer = get_answer(
                youtube_url,
                question
            )
            result = answer["result"]
            sources = answer["src"]

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

# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "Built using LangChain • FAISS • HuggingFace Embeddings • Mistral AI • Streamlit"
)