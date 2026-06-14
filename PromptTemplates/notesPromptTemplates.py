from langchain_core.prompts import PromptTemplate


notesPrompt =  PromptTemplate(
    template="""
You are an expert note maker.

Analyze the following YouTube transcript and create well-structured study notes.

Transcript:
{transcript}

Instructions:
- Create clear headings.
- Use bullet points.
- Include important concepts.
- Keep the notes concise and easy to revise.
- Do not add information not present in the transcript.

Output Format:

# Topic

## Section 1
- Point
- Point

## Section 2
- Point
- Point
""",
    input_variables=["transcript"]
)