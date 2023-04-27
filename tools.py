import sys
from langchain.tools import BaseTool
import common
import process_index
import env_loader as e


class SearchUserHistory(BaseTool):
    name = "SearchUserHistory"
    description = "Pass query of important keywords to find relevant messages in database from the user. " \
                  "Format: keyword, keyword, ..., keyword" \
                  "Facts from the user messages will be returned." \
                  "Facts may not be relevant to the keywords at all. Try again with more specific keywords or " \
                  "assume that user has not shared information about the topic previously"

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


class SearchAgentHistory(BaseTool):
    name = "SearchAgentHistory"
    description = "Use to look up what you have shared in past conversations about a certain topic." \
                  "Pass query of important keywords to find relevant messages in database from yourself. " \
                  "Facts from your own messages will be returned." \
                  "Facts may not be relevant to the keywords at all. Try again with more specific keywords or " \
                  "assume that you have not written about the topic previously"

    def _run(self, query: str) -> str:
        index_user_history = e.index.query(namespace='AGENT',
                                           vector=common.gpt_embedding(query),
                                           top_k=100,
                                           include_values=False,
                                           include_metadata=True)
        return process_index.process_history(index_user_history, query, 'AGENT')

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SearchAgentHistory does not support async")


class AnswerUser(BaseTool):
    name = "AnswerUser"
    description = "Pass your answer to the user when you have an answer ready, receive users response"

    def _run(self, query: str) -> str:
        sys.stdout.write("\nEve: %s" % query)
        print()  # Newline
        user_input = input("You: ")
        return user_input

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("AnswerUser does not support async")
