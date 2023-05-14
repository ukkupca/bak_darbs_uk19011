from langchain.llms import OpenAI
from langchain.memory import ConversationEntityMemory
import env_loader as e
from index import index_service


class EntityMemory:
    llm = OpenAI(
        temperature=0,
        openai_api_key=e.openai_api_key
    )
    memory = ConversationEntityMemory(llm=llm)
    last_user_input = ""

    def set_last_user_input(self, chat_input):
        self.last_user_input = chat_input
        self.memory.load_memory_variables({"input": self.last_user_input})

    def set_last_agent_input_and_save(self, agent_input):
        self.memory.save_context({"input": self.last_user_input}, {"output": agent_input})

    def upsert_to_db(self):
        entities = self.memory.entity_store
        # TODO: how to save

