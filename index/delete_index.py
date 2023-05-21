import pinecone
import env_loader as e

pinecone.init(api_key="a433ce32-e7cb-4684-9280-1d201daccc85", environment="eu-west1-gcp")

print(f"Deleting index..")
pinecone.delete_index(e.index_name)
print(f"DONE")
