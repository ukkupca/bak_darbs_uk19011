import pinecone
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
import os
from langchain.tools import BaseTool
import common
from dotenv import load_dotenv

import process_index

load_dotenv()
pinecone.init(api_key=os.getenv('KEY_PINECONE'), environment=os.getenv('ENV_PINECONE'))
index = pinecone.Index("virtual-agent-v0")


class SearchUserChatHistory(BaseTool):
    name = "SearchUserChatHistory"
    description = "Pass a query to find relevant messages in chat history from the user. If no messages are found an " \
                  "empty string will be returned, meaning that assistant has no prior knowledge about the query or " \
                  "that the query was not good for searching history"

    def _run(self, query: str) -> str:
        # TODO: needs second parameter - task for LLM
        index_user_history = index.query(namespace='USER',
                                         vector=common.gpt_embedding(query),
                                         top_k=100,
                                         include_values=False,
                                         include_metadata=True)
        return process_index.process_history(index_user_history, query, 'USER')

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


class ReadCurrentChatHistory(BaseTool):
    name = "ReadLocalChatHistory"
    description = "Read current chat history with the user if users last input provides little context for what needs "\
                  "to be answered"

    def _run(self, query: str) -> str:
        index_user_history = index.query(namespace='USER',
                                         vector=common.gpt_embedding(query),
                                         top_k=100,
                                         include_values=False,
                                         include_metadata=True)
        return process_index.process_history(index_user_history, query, 'USER')

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


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


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Answer To The User:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Answer To The User:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


tools = [SearchUserChatHistory()]
prompt = CustomPromptTemplate(
    template=common.open_file('prompt-configs/base_agent_config'),
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)
output_parser = CustomOutputParser()
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.getenv('KEY_OPEN_AI')
)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
agent_output = agent_executor.run("What is my thesis about?")
