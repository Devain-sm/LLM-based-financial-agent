from langchain.document_loaders import TextLoader 
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredURLLoader 
from sentence_transformers import SentenceTransformer 
import pandas as pd 
#loading text data to a variable
loader=TextLoader("Documents/LLM/for_ai_project.txt",encoding="utf-8") 
data=loader.load()
#loading csv data to a variable
excel=CSVLoader("Documents/LLM/car_data.csv", encoding="utf-8")
data1=excel.load()
df = pd.DataFrame(data1)
print(df.columns)
#loading url data to a variable
links=UnstructuredURLLoader(urls=["https://www.moneycontrol.com/news/business/earnings/idfc-first-bank-q1-profit-drops-32-to-rs-462-6-crore-asset-quality-weakens-13331675.html",
                                  "https://www.moneycontrol.com/city/they-told-us-to-sit-students-warned-teachers-before-rajasthan-school-roof-collapsed-article-13329906.html"])
data2=links.load()
#printing everything 
print(data2[0].page_content)
print(len(data1))
print(data)
print(loader)
pd.set_option('display.max_colwidth', None)  # Set to None to display full text in columns
df=pd.read_csv("Documents/LLM/car_name.csv", encoding="utf-8")
#importing sentence transformer model to convert text to vectors from huggingface
encoder = SentenceTransformer('all-mpnet-base-v2') 
vectors=encoder.encode(df.description)
print(vectors)  # Print the shape of the vectors 
dim=vectors.shape[1]  # Get the dimension of the vectors
print(f"Vector dimension: {dim}")  # Print the dimension of the vectors
import faiss
index = faiss.IndexFlatL2(dim)  # Create a FAISS index for L2 distance
print(index) 
print(index.add(vectors))  # Add vectors to the index
search_query = encoder.encode(["What is the best car for a family?"])
print(search_query.shape)  # Print the search query vector
import numpy as np 
svec=np.arry(search_query).reshape(1, -1)  # Reshape the search query vector
print(svec.shape)
distances, I=index.search(svec, k=5)
print("Distances:", distances)  # Print the distances of the nearest neighbors
print(df.loc[I[0]])  # Print the nearest neighbors based on the search query

#this part goes after the dependency installation for the actual project 
os.environ['OPENAI.API_KEY'] =  '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' #set your openai api key here
llm=OpenAI(temperature=0.9, max_tokens=500) #initialize the llm model
print(llm)  # Print the llm object to verify initialization
loaders=UnstructuredURLLoader(urls=["https://techcrunch.com/2025/08/19/spotifys-latest-feature-lets-you-add-your-own-transitions-to-playlists/"]) #load the url data
data=loaders.load()
print(len(data))
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #split the data into chunks
docs=text_splitter.split_documents(data)
print(len(docs)) 
embeddings=OpenAIEmbeddings() #initialize the embeddings model
print(embeddings)  # Print the embeddings object to verify initialization
vectorstore=FAISS.from_documents(docs, embeddings) #create the vector store
print(vectorstore)  # Print the vector store object to verify creation
file_path="vector_index.pkl" #file path to save the vector store
with open(file_path, "wb") as f:
    pickle.dump(vectorstore, f)  # Save the vector store to a file
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorIndex = pickle.load(f)  # Load the vector store from the file
 
chain=RetrievalQAWithSourcesChain(llm=llm,retriever=vectorIndex.as_retriever(), chain_type="stuff") #create the chain
print(chain)  # Print the chain object
query="How can I add my own transitions to Spotify playlists?"
langchain.debug = True  # Enable debugging mode
chain({"query": query},return_only_outputs=True)  # Execute the chain with the query

# part 2 of the project starts here THE ACTUAL PROJECT
import pickle
import time
import os
import streamlit as st
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import SeleniumURLLoader
from dotenv import load_dotenv
# Load environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
load_dotenv()  # take environment variables from .env (especially openai api key)
# Streamlit UI
st.title("Research and Analysis System 📈")
st.sidebar.title("News Article URLs 🔗")

# Sidebar inputs for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"Enter URL {i + 1}", key=f"url_{i + 1}")
    if url:
        urls.append(url)

# Sidebar button
process_url_clicked = st.sidebar.button("Process")

# File path for FAISS store
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()

# Initialize LLM
llm = OpenAI(temperature=0.9, max_tokens=500)

# Process URLs
if process_url_clicked and urls:
    loader = SeleniumURLLoader(urls=urls)
    main_placeholder.text("Loading and processing the data...🔁")
    data = loader.load()
    st.write(f"Loaded {len(data)} documents.")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("Splitting the data into chunks...🔁")
    docs = text_splitter.split_documents(data)

    # Embeddings + FAISS vector store
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Building embedded vector store...🔁")
    time.sleep(2)

    # Save vector store
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

# Question input
query = main_placeholder.text_input("Question:")

# Answering
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore_openai = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore_openai.as_retriever()
        )
        result = chain({"question": query}, return_only_outputs=True)

        # Show result
        st.header("Answer")
        st.write(result["answer"])
        if "sources" in result:
            st.subheader("Sources")
            st.write(result["sources"])