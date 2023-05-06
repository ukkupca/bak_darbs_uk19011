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
import env_loader
import tools
import env_loader as e
from index import index_service


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
            # TODO: Implement retry so it fixes format
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


class AnswerUser(BaseTool):
    name = "AnswerUser"
    # description = "Pass your answer to the user when you have an answer ready. Receive users response."
    description = "Pass your answer to the user when you have an answer ready. Receive users response." \
                  "Remember that you want to be a close friend to the user. Be funny and kind. The conversation must " \
                  "flow " \
                  "naturally, share your own thoughts and opinions, ask questions and encourage user to talk about " \
                  "themselves. This will make the conversation more engaging and enjoyable for the user." \
                  "Talk about one topic at a time. Only the user can end a conversation with you. When one " \
                  "conversation comes to a natural end figure out a new topic you could talk about." \
                  ""

    def _run(self, query: str) -> str:
        user_payload = list()
        agent_payload = list()
        index_service.prepare_and_add(query, agent_payload)
        sys.stdout.write("\nEve: %s" % query)
        print()  # Newline
        new_agent_message = 'eve: %s \n<<end_of_messages>>\n' % query
        agent.llm_chain.prompt.template = agent.llm_chain.prompt.template \
            .replace('<<end_of_messages>>', new_agent_message)
        index_service.prepare_and_add(query, agent_payload)
        e.index.upsert(agent_payload, 'AGENT')
        user_input = input("You: ")
        new_user_message = 'user: %s \n<<end_of_messages>>\n' % user_input
        agent.llm_chain.prompt.template = agent.llm_chain.prompt.template \
            .replace('<<end_of_messages>>', new_user_message)
        index_service.prepare_and_add(user_input, user_payload)
        e.index.upsert(user_payload, 'USER')
        return user_input

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("AnswerUser does not support async")


output_parser = CustomOutputParser()
llm = ChatOpenAI(
    model=env_loader.openai_model,
    temperature=0,
    openai_api_key=e.openai_api_key
)

tools = [AnswerUser(), tools.SearchUserDatabase(), tools.SearchChatbotDatabase()]
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
    allowed_tools=tool_names
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

while True:
    # Getting initial user input, adding to local history and prepping index payload
    user = input("You: ")
    initial_user_message = 'user: %s \n<<end_of_messages>>\n' % user
    agent.llm_chain.prompt.template = agent.llm_chain.prompt.template \
        .replace('<<end_of_messages>>', initial_user_message)
    for n in range(5):
        try:
            agent_executor.run(user)
        except ValueError as error:
            sys.stdout.write(str(error))
            sys.stdout.write("System error, restoring connection...")

