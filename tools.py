from langchain.tools import BaseTool
import common
import process_index
import env_loader as e


class SearchUserDatabase(BaseTool):
    name = "SearchUserDatabase"
    description = "Pass query of separate keywords to find information in database about the user. " \
                  "Format: keyword, keyword, ..., keyword" \
                  "Summarized information on what user has said in past conversations will be returned if available."

    def _run(self, query: str) -> str:
        index_user_history = e.index.query(namespace='USER',
                                           vector=common.gpt_embedding(query),
                                           top_k=100,
                                           include_values=False,
                                           include_metadata=True)
        return process_index.process_history(index_user_history, query, 'USER')

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SearchUserDatabase does not support async")


class SearchAgentDatabase(BaseTool):
    name = "SearchAgentDatabase"
    description = "Pass query of important separate keywords to find information on what " \
                  "assistant Eve has said in past messages. " \
                  "Format: keyword, keyword, ..., keyword" \
                  "Summarized information on what assistant Eve has said in past conversations will " \
                  "be returned if available. If returned information is not helpful, try again " \
                  "with different keywords or assume that assistant Eve has not talked about the topic previously"

    def _run(self, query: str) -> str:
        index_user_history = e.index.query(namespace='AGENT',
                                           vector=common.gpt_embedding(query),
                                           top_k=100,
                                           include_values=False,
                                           include_metadata=True)
        return process_index.process_history(index_user_history, query, 'AGENT')

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SearchAgentDatabase does not support async")
