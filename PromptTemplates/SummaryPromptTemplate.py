from langchain_core.prompts import PromptTemplate


summary_prompt = PromptTemplate(
   template="""
      You are an expert summarizer.

      Summarize the following transcript of a particular youtube video.

      Transcript: {transcript}
      Provide: 
      1. A short and concise summary
      2. Key takeaways in bullet points
   """,
   input_variables=["transcript"]
)