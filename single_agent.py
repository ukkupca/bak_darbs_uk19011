import sys
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
from langchain.tools import BaseTool
import common
import control_session_service
import env_loader
import tools as t
import env_loader as e
from control_session_service import IS_CONTROL_SESSION
from memory.entity_memory import EntityMemory
from memory.graph_memory import GraphMemory
from memory.single_message_memory import SingleMessageMemory
from memory.summary_memory import SummaryMemory
from logs import log_service

# MEMORY
SAVE_SINGLE_MESSAGE_MEMORY = False
SAVE_SUMMARY_MEMORY = False
SAVE_ENTITY_MEMORY = False
# SAVE_GRAPH_MEMORY = False

last_user_input = None

single_message_memory = SingleMessageMemory()
summary_memory = SummaryMemory()
entity_memory = EntityMemory()


# graph_memory = GraphMemory()

class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]

    def format_messages(self, **kwargs) -> List[HumanMessage]:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


class AnswerUser(BaseTool):
    name = "AnswerUser"
    description = "Pass your answer to the user when you have an answer ready. Receive users response."

    # description = "Pass your answer to the user when you have an answer ready. Receive users response." \ "Remember
    # that you want to be a close friend to the user. Be funny and kind. The conversation must " \ "flow " \
    # "naturally, share your own thoughts and opinions, ask questions and encourage user to talk about " \
    # "themselves. This will make the conversation more engaging and enjoyable for the user." \ "Talk about one topic
    # at a time. Only the user can end a conversation with you. When one " \ "conversation comes to a natural end
    # figure out a new topic you could talk about."

    def _run(self, query: str) -> str:
        sys.stdout.write("\nEve: %s" % query)
        print()  # Newline
        log_service.logs.append(['Agent', 'AnswerUser', query])

        user_input = input("You: ")
        log_service.logs.append(['User', 'Input', user_input])
        global last_user_input
        last_user_input = user_input

        new_messages = 'eve: %s \nuser: %s \n<<end_of_messages>>' % (query, user_input)
        agent.llm_chain.prompt.template = agent.llm_chain.prompt.template \
            .replace('<<end_of_messages>>', new_messages)

        # common.save_json('logs/%s.json' % int(time.time()), agent.llm_chain.prompt.template)

        # Handling memory
        single_message_memory.add_agent_message_to_payload(query)
        single_message_memory.add_user_message_to_payload(user_input)

        summary_memory.set_last_agent_input_and_save(query)
        summary_memory.set_last_user_input(user_input)

        entity_memory.set_last_agent_input_and_save(query)
        entity_memory.set_last_user_input(user_input)

        # graph_memory.set_last_agent_input_and_save(query)
        # graph_memory.set_last_user_input(user_input)

        return user_input

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("AnswerUser does not support async")


# VIRTUAL AGENT
output_parser = CustomOutputParser()
llm = ChatOpenAI(
    model=env_loader.openai_model,
    temperature=0,
    openai_api_key=e.openai_api_key
)
tools = [AnswerUser(), t.SearchUserPastMessages(),
         t.SearchChatbotPastMessages()]  # t.searchSummaryMemory t.SearchEntityMemory()
tool_names = [tool.name for tool in tools]
prompt = CustomPromptTemplate(
    template=common.open_file(e.single_agent_config),
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)
# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
    verbose=False
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# AGENT LOOP
if IS_CONTROL_SESSION is False:

    # INITIAL INPUT
    initial_input = input("You: ")
    log_service.logs.append(['User', 'Input', initial_input])
    last_user_input = initial_input

    # Handling first input in memories
    single_message_memory.add_user_message_to_payload(initial_input)
    summary_memory.set_last_user_input(initial_input)
    entity_memory.set_last_user_input(initial_input)
    # graph_memory.set_last_user_input(initial_input)

    # Adding first input to prompt
    initial_user_message = 'user: %s \n<<end_of_messages>>\n' % initial_input
    agent.llm_chain.prompt.template = agent.llm_chain.prompt.template \
        .replace('<<end_of_messages>>', initial_user_message)

    while True:
        try:
            agent_executor.run(last_user_input)
        except ValueError as error:
            sys.stdout.write(str(error))
            sys.stdout.write("System error, restoring connection...")
        except KeyboardInterrupt:
            if log_service.SAVE_LOGS:
                sys.stdout.write("\nSaving logs..")
                log_service.save()
                sys.stdout.write("\nLogs saved")

            if SAVE_SINGLE_MESSAGE_MEMORY:
                sys.stdout.write("\nSaving single message memory..")
                single_message_memory.upsert_all_messages()
                sys.stdout.write("\nSingle message memory saved")

            if SAVE_SUMMARY_MEMORY:
                sys.stdout.write("\nSaving summary memory..")
                summary_memory.upsert_to_db()
                sys.stdout.write("\nSummary memory saved")

            if SAVE_ENTITY_MEMORY:
                sys.stdout.write("\nSaving entity memory..")
                entity_memory.upsert_to_db()
                sys.stdout.write("\nEntity memory saved")
            break
else:
    control_session_service.import_control_questions()
    for question in control_session_service.control_questions:
        for m in range(1, 4):
            response = agent_executor.run(question['controlQuestion'])
            control_session_service.add_result([question['id'], e.openai_model, m, response])
    control_session_service.save()
