from langchain.tools import BaseTool
import common
import process_index
import env_loader as e
from logs import log_service


class SearchUserPastMessages(BaseTool):
    name = "SearchUserPastMessages"
    description = "Pass query of separate keywords to find information in database " \
                  "on what user has shared in the past. " \
                  "Format: keyword, keyword, ..., keyword" \
                  "Summarized information on what user has said in past conversations will be returned if available."

    def _run(self, query: str) -> str:
        log_service.logs.append(['Agent', 'SearchUserPastMessages', query])
        index_user_history = e.index.query(namespace='USER',
                                           vector=common.gpt_embedding(query),
                                           top_k=20,
                                           include_values=False,
                                           include_metadata=True)
        response = process_index.process_user_messages(index_user_history, query)
        log_service.logs.append(['Agent', 'SearchUserPastMessagesResponse', response])
        return response

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SearchUserDatabase does not support async")


class SearchChatbotPastMessages(BaseTool):
    name = "SearchChatbotPastMessages"
    description = "Pass query of separate keywords to find information on what " \
                  "assistant Eve has said in past messages. " \
                  "Format: keyword, keyword, ..., keyword" \
                  "Summarized information on what chatbot Eve has said in past conversations will " \
                  "be returned if available."

    def _run(self, query: str) -> str:
        log_service.logs.append(['Agent', 'SearchChatbotPastMessages', query])
        index_agent_history = e.index.query(namespace='AGENT',
                                            vector=common.gpt_embedding(query),
                                            top_k=20,
                                            include_values=False,
                                            include_metadata=True)
        response = process_index.process_agent_messages(index_agent_history, query)
        log_service.logs.append(['Agent', 'SearchChatbotPastMessagesResponse', response])
        return response

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SearchChatbotDatabase does not support async")


class SearchBatchMemory(BaseTool):
    name = "SearchMemory"
    description = "Pass a question to find out what is saved in memories."

    def _run(self, query: str) -> str:
        log_service.logs.append(['Agent', 'SearchBatchMemory', query])
        index_agent_history = e.index.query(namespace='BATCH',
                                            vector=common.gpt_embedding(query),
                                            top_k=10,
                                            include_values=False,
                                            include_metadata=True)
        response = process_index.process_batch_messages(index_agent_history, query)
        log_service.logs.append(['Agent', 'SearchBatchMemoryResponse', response])
        return response

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SearchBatchMemory does not support async")


class SearchSummaryMemory(BaseTool):
    name = "SearchMemory"
    description = "Pass a question to find out what is saved in memories."

    def _run(self, query: str) -> str:
        log_service.logs.append(['Agent', 'SearchSummaryMemory', query])
        index_summaries = e.index.query(namespace='SUMMARY',
                                        vector=common.gpt_embedding(query),
                                        top_k=2,
                                        include_values=False,
                                        include_metadata=True)
        response = process_index.process_summaries(index_summaries, query)
        log_service.logs.append(['Agent', 'SearchSummaryMemoryResponse', response])
        return response

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SearchSummaryMemory does not support async")


class SearchEntityMemory(BaseTool):
    name = "SearchMemory"
    description = "Pass a question to find out what is saved in memories."

    def _run(self, query: str) -> str:
        log_service.logs.append(['Agent', 'SearchEntityMemory', query])
        index_entities = e.index.query(namespace='ENTITY-2',
                                       vector=common.gpt_embedding(query),
                                       top_k=5,
                                       include_values=False,
                                       include_metadata=True)
        response = process_index.process_entities(index_entities, query)
        log_service.logs.append(['Agent', 'SearchEntityMemoryResponse', response])
        return response

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SearchEntityMemory does not support async")


class SearchGraphMemory(BaseTool):
    name = "SearchMemory"
    description = "Pass a question to find out what is saved in memories."

    def _run(self, query: str) -> str:
        log_service.logs.append(['Agent', 'SearchGraphMemory', query])
        index_graphs = e.index.query(namespace='GRAPH',
                                     vector=common.gpt_embedding(query),
                                     top_k=100,
                                     include_values=False,
                                     include_metadata=True)
        response = process_index.process_graphs(index_graphs, query)
        log_service.logs.append(['Agent', 'SearchGraphMemoryResponse', response])
        return response

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("SearchGraphMemory does not support async")
