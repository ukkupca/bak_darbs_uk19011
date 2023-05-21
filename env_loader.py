from dotenv import load_dotenv
import pinecone
import os

load_dotenv()
openai_api_key = os.getenv('KEY_OPEN_AI')

# Setting model and config and index
openai_model = 'gpt-3.5-turbo'  # gpt-3.5-turbo / gpt-4
single_agent_config = 'agent-prompt-configs/gpt-3.5-turbo/agent_config_iza'  # _iza
index_name = 'iza-gpt3'
log_name = index_name + '-origin-chat'

# Pinecone
pinecone.init(api_key=os.getenv('KEY_PINECONE'), environment=os.getenv('ENV_PINECONE'))
index = pinecone.Index(index_name)
