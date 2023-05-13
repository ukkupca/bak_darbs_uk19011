from langchain.memory import ConversationSummaryMemory
from langchain.llms import OpenAI
import env_loader as e


class SummaryMemory:
    llm = OpenAI(
        temperature=0,
        openai_api_key=e.openai_api_key
    )
    memory = ConversationSummaryMemory(
        human_prefix='user',
        ai_prefix='agent',
        llm=llm
    )
    last_user_input = ""

    def set_last_user_input(self, chat_input):
        self.last_user_input = chat_input

    def set_last_agent_input_and_save(self, agent_input):
        self.memory.save_context({"input": self.last_user_input}, {"output": agent_input})
