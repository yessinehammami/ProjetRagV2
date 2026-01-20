import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from typer import prompt
from src.prompt_template import pt
from src.rag import retriever
load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

llm = AzureChatOpenAI(
		azure_endpoint=endpoint,
		azure_deployment=deployment,
		api_version=api_version,
		api_key=api_key,
		temperature=0.1,
	)
import streamlit as st

st.set_page_config(page_title="Infos sur WW1", layout="wide")

st.title("Assistant WW1")
st.caption("Pose une question.")

# Simple text input for user query

user_query = st.text_input("Ecris ta question ici :", "")


if user_query:
	
    docs = retriever.invoke(user_query)
    context_text = "\n".join([doc.page_content for doc in docs])

    filled_prompt = pt.format(query=user_query, context=context_text)
    response = llm.invoke(filled_prompt)

    st.subheader("Réponse de l'assistant :")
    st.write(response.content)

    st.subheader("Prompt :")
    st.write(filled_prompt)