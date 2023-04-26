# Import things that are needed generically
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, Tool, initialize_agent, tool
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
import pinecone

import common
import process_index
import os


class SearchUserChatHistory(BaseTool):
    name = "SearchUserChatHistory"
    description = "pass a query to find relevant messages in chat history"

    def _run(self, query: str) -> str:
        pinecone.init(api_key=os.getenv('KEY_PINECONE'), environment=os.getenv('ENV_PINECONE'))
        index = pinecone.Index("virtual-agent-v0")
        index_user_history = index.query(namespace='USER',
                                         vector=common.gpt_embedding(query),
                                         top_k=100,
                                         include_values=False,
                                         include_metadata=True)
        return process_index.process_history(index_user_history, query, 'USER')

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


agent_tools = [SearchUserChatHistory()]
