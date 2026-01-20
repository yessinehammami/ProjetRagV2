import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
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


st.title("RAG Premiere Guerre Mondiale")
st.markdown(
    """ 
    Ce site permet d'intéroger un model d'intéligence artificielle sur la Premiere Uerre Mondiale. 
    """
)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
if prompt := st.chat_input("Votre question"):

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    docs = retriever.invoke(prompt)
    context_text = "\n".join([doc.page_content for doc in docs])
    filled_prompt = pt.format(query=prompt, context=context_text)
         
    with st.chat_message("assistant"):
       response = st.write_stream(llm.stream(filled_prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})



	

