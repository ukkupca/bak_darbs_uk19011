import pinecone

pinecone.init(api_key="a433ce32-e7cb-4684-9280-1d201daccc85", environment="eu-west1-gcp")

# Use to create an index that would not index all metadata
metadata_config = {
    "indexed": ["timestamp"]
}
print(f"Creating index..")
index = pinecone.create_index(
    "virtual-agent-v0",
    dimension=1536,
    metric="cosine",
    pod_type="p1",
    metadata_config=metadata_config
)
print(f"DONE")
