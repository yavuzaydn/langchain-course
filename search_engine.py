from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

load_dotenv()



llm = ChatOllama(temperature=0, model="gemma4:12b")
tools = [TavilySearch()]
agent = create_agent(llm, tools)


def main():
    result = agent.invoke({"messages": [HumanMessage(content="Whats the weather like in Istanbul?")]})
    print(result)


if __name__ == "__main__":
    main()