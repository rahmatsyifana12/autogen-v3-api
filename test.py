import os
import pinecone
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
        # Create an instance of the Pinecone class
pinecone_client = pinecone.Pinecone(api_key=api_key)
index_list = pinecone_client.list_indexes()

# Print the list of indexes
print("Indexes in your Pinecone account:")
for index in index_list.names():
    print(index)
