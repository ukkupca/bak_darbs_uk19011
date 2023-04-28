from langchain.tools import BaseTool
import common
import process_index
import env_loader as e


class SearchUserHistory(BaseTool):
    name = "SearchUserHistory"
    description = "Pass query of separate keywords to find information that you know in database about the user. " \
                  "Format: keyword, keyword, ..., keyword" \
                  "Information may not be relevant to what you needed. Try again with more specific keywords or " \
                  "assume that user has not talked about the topic previously"

    def _run(self, query: str) -> str:
        index_user_history = e.index.query(namespace='USER',
                                           vector=common.gpt_embedding(query),
                                           top_k=100,
                                           include_values=False,
                                           include_metadata=True)
        return process_index.process_history(index_user_history, query, 'USER')

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SearchUserHistory does not support async")


class SearchEveHistory(BaseTool):
    name = "SearchEveHistory"
    description = "Pass query of important separate keywords to find information on what Eve has said in past " \
                  "messages. " \
                  "Format: keyword, keyword, ..., keyword" \
                  "Information may not be relevant to needed information. Try again with more specific keywords or " \
                  "assume that you have not said anything about the topic previously"

    def _run(self, query: str) -> str:
        index_user_history = e.index.query(namespace='AGENT',
                                           vector=common.gpt_embedding(query),
                                           top_k=100,
                                           include_values=False,
                                           include_metadata=True)
        return process_index.process_history(index_user_history, query, 'AGENT')

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SearchYourPastMessages does not support async")


