import sys
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
import os
from langchain.tools import BaseTool
import common
from index import index_service
import tools
import env_loader as e


class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]

    def format_messages(self, **kwargs) -> list[HumanMessage]:
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

    def insert_history(self, array):
        chat_history = common.history_to_string(array)
        self.template = self.template.replace('{chat_history}', chat_history)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # TODO: Implement retry
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


tools = [tools.SearchUserHistory(), tools.SearchAgentHistory()]
tool_names = [tool.name for tool in tools]
output_parser = CustomOutputParser()
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=e.openai_api_key
)
current_conversation_history = []
while True:
    user_payload = list()
    agent_payload = list()

    # Getting user input, adding to local history and prepping index payload
    chat_input = input("You: ")
    index_service.prepare_and_add(chat_input, user_payload)

    base_prompt = CustomPromptTemplate(
        template=common.open_file('../agent-prompt-configs/old/base_agent_config'),
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )
    base_prompt.insert_history(current_conversation_history)

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=base_prompt)
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    agent_output = agent_executor.run(chat_input)
    sys.stdout.write("Eve: %s" % agent_output)
    print()  # Newline

    # Adding question and response to local current conversation history
    current_conversation_history.append({"role": "User input", "content": chat_input})
    current_conversation_history.append({"role": "Your answer", "content": agent_output})

    index_service.prepare_and_add(agent_output, agent_payload)

    e.index.upsert(user_payload, "USER")
    e.index.upsert(agent_payload, "AGENT")
