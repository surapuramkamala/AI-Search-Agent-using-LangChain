import os
import certifi
import streamlit as st
from dotenv import load_dotenv

#from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
#from langchain import langchainhub
#import langchainhub as hub
#from langchain import hub

from langchain_classic.agents import create_react_agent, AgentExecutor

#=================
#LOAD ENV VARIABLE
#=================

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

#=====================
#STREAMLIT PAGE CONFIG
#=================
st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout='centered'
)
st.title("Agentic AI Assistant")
st.markdown(
    """
    # Agentic AI Assistant
    Ask a question and this assistant can reason over web search results before responding.
    """
)

#================
#SEARCH TOOL
#=================
search_tool = TavilySearch(max_results=1)
search_tool.name = "Search"
search_tool.description = "Use this tool to search the web for current facts."

result = search_tool.invoke("Give me the latest news on AI")

print(result)


# =====
# LLM
# =====
from langchain_ollama import ChatOllama  
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

response = llm.invoke('tell me about a joke?')
print(response)

#======
# PROMPT
#=======
from langchain_core.prompts import PromptTemplate

template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)
print(prompt)

tools =[search_tool]

#===============
#CREATING AGENT
#===============
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt

)

#===========
#AGENT EXECUTOR OBJECT
#===========

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

#=========
#RUN
#=========
#response = agent_executor.invoke({
#    'input' :(
#        'what is india capital city?'
#    )
#})
#print(response["output"])

#===============
#UI INPUT
#===============
user_input = st.text_input(
    "Ask a question:",
    placeholder="Type your question here..."
    )


#=================
#RUN AGENT
#=================
if st.button("Submit"):
    if user_input:
        with st.spinner('Agent is thinking..'):
            try:
                response = agent_executor.invoke({
                   "input": user_input 
                })
                st.success("Response Generated")
                st.markdown("## Final Response")
                st.write(response["output"])
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question before submitting.")