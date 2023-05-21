import pinecone
import env_loader as e

# Use to create an index that would not index all metadata
metadata_config = {
    "indexed": ["timestamp"]
}
print(f"Creating index..")
index = pinecone.create_index(
    e.index_name,
    dimension=1536,
    metric="cosine",
    pod_type="p1",
    metadata_config=metadata_config
)
print(f"DONE")
