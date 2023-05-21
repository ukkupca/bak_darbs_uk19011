from langchain.llms import OpenAI
from langchain.memory import ConversationKGMemory
import env_loader as e
from index import index_service


class GraphMemory:
    llm = OpenAI(
        temperature=0,
        openai_api_key=e.openai_api_key
    )
    memory = ConversationKGMemory(llm=llm)
    last_user_input = ""
    knowledge_triplets = []

    def set_last_user_input(self, chat_input):
        self.last_user_input = chat_input
        self.memory.load_memory_variables({"input": self.last_user_input})
        self.knowledge_triplets.append(self.memory.get_knowledge_triplets(self.last_user_input))

    def set_last_agent_input_and_save(self, agent_input):
        self.memory.save_context({"input": self.last_user_input}, {"output": agent_input})

    def upsert_to_db(self):
        # not implemented
        summary_payload = list()
        # index_service.prepare_and_add(self.memory.load_memory_variables(), summary_payload)
        # e.index.upsert(summary_payload, 'SUMMARY')
