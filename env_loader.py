from dotenv import load_dotenv
import pinecone
import os

load_dotenv()
openai_api_key = os.getenv('KEY_OPEN_AI')

# Setting model and config and index
openai_model = 'gpt-3.5-turbo'  # gpt-3.5-turbo / gpt-4
single_agent_config = 'prompt-configs/single_agent_gpt_3_config'
index_name = 'virtual-agent-gpt-3-iz'

# Pinecone
pinecone.init(api_key=os.getenv('KEY_PINECONE'), environment=os.getenv('ENV_PINECONE'))
index = pinecone.Index(index_name)
