from dotenv import load_dotenv
import pinecone
import os

load_dotenv()
pinecone.init(api_key=os.getenv('KEY_PINECONE'), environment=os.getenv('ENV_PINECONE'))
index = pinecone.Index("virtual-agent-v0")
openai_api_key = os.getenv('KEY_OPEN_AI')
openai_model = 'gpt-3.5-turbo'  # gpt-3.5-turbo gpt-4
