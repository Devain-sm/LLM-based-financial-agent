from langchain.document_loaders import TextLoader 
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredURLLoader 
from sentence_transformers import SentenceTransformer 
import pandas as pd 
loader=TextLoader("Documents/LLM/for_ai_project.txt",encoding="utf-8")
data=loader.load()
excel=CSVLoader("Documents/LLM/car_data.csv", encoding="utf-8")
data1=excel.load()
df = pd.DataFrame(data1)
print(df.columns)
links=UnstructuredURLLoader(urls=["https://www.moneycontrol.com/news/business/earnings/idfc-first-bank-q1-profit-drops-32-to-rs-462-6-crore-asset-quality-weakens-13331675.html",
                                  "https://www.moneycontrol.com/city/they-told-us-to-sit-students-warned-teachers-before-rajasthan-school-roof-collapsed-article-13329906.html"])
data2=links.load()
print(data2[0].page_content)
print(len(data1))
print(data)
print(loader) 
pd.set_option('display.max_colwidth', None)  # Set to None to display full text in columns
df=pd.read_csv("Documents/LLM/car_name.csv", encoding="utf-8")

encoder = SentenceTransformer('all-mpnet-base-v2') 
vectors=encoder.encode(df.description)
print(vectors)  # Print the shape of the vectors 