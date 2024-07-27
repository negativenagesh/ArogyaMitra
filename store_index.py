from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_APE_KEY= os.environ.get('PINECONE_APE_KEY')
PINECONE_ENV=os.environ.get('PINECONE_ENV')

print(PINECONE_APE_KEY)
print(PINECONE_ENV)