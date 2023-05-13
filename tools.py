from langchain.tools import BaseTool
import common
import process_index
import env_loader as e


class SearchUserDatabase(BaseTool):
    name = "SearchUserPastMessages"
    description = "Pass query of separate keywords to find information in database " \
                  "on what user has shared in the past. " \
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


class SearchChatbotDatabase(BaseTool):
    name = "SearchChatbotPastMessages"
    description = "Pass query of separate keywords to find information on what " \
                  "assistant Eve has said in past messages. " \
                  "Format: keyword, keyword, ..., keyword" \
                  "Summarized information on what chatbot Eve has said in past conversations will " \
                  "be returned if available."

    def _run(self, query: str) -> str:
        index_user_history = e.index.query(namespace='AGENT',
                                           vector=common.gpt_embedding(query),
                                           top_k=100,
                                           include_values=False,
                                           include_metadata=True)
        return process_index.process_history(index_user_history, query, 'AGENT')

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SearchChatbotDatabase does not support async")


class SearchPastConversations(BaseTool):
    name = "SearchPastConversations"
    description = "Pass a question to find relevant information in past conversations. "

    def _run(self, query: str) -> str:
        index_summaries = e.index.query(namespace='SUMMARY',
                                        vector=common.gpt_embedding(query),
                                        top_k=100,
                                        include_values=False,
                                        include_metadata=True)
        return process_index.process_summaries(index_summaries, query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SearchPastConversations does not support async")
