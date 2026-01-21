import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from src.prompt_template import pt
from src.rag import preprocess_pdfs, retriever, vectorstore
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

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
if uploaded_file := st.file_uploader("Upload un fichier PDF", type=["pdf"]):
    save_path = f"data/{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    new_docs = preprocess_pdfs(save_path)
    vectorstore.add_documents(new_docs)
    st.success("Fichier PDF ajouté et indexé avec succès !")

    retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs= {"k":5}
    )
    
if prompt := st.chat_input("Posez votre question..."):

    docs = retriever.invoke(prompt)
    context_text = "\n".join([doc.page_content for doc in docs])
    filled_prompt = pt.format(query=prompt, context=context_text, history=st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
         
    with st.chat_message("assistant"):
       response = st.write_stream(llm.stream(filled_prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    resources = [doc.metadata['source'] for doc in docs]

    st.write("Sources utilisées :", "\n")
    unique_resources = set()
    for resource in resources:
        trimmed = resource[5:]
        if trimmed not in unique_resources:
            st.write(trimmed)
            unique_resources.add(trimmed)


with st.sidebar:
    st.title("RAG Première Guerre Mondiale")
    st.markdown(
    """ 
    Ce site permet d'interroger un modèle d'intelligence artificielle sur la Première Guerre Mondiale. 
    """
    )

    st.container(height=500, border=False)

    if st.button("Supprimer l'historique", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()






	

