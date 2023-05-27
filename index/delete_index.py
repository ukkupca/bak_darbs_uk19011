import pinecone
import env_loader as e

pinecone.init(api_key="", environment="")

print(f"Deleting index..")
pinecone.delete_index(e.index_name)
print(f"DONE")
