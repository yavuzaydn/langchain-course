from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    # Initialize Gemini Chat Model
    # Note: Use a model supported by your key, such as "gemini-2.5-flash" or "gemini-2.0-flash"
    #llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    llm = ChatOllama(temperature=0, model="gemma4:12b")

    information = """
    William Henry Gates III (born October 28, 1955) is an American businessman and philanthropist. A pioneer of the microcomputer revolution of the 1970s and 1980s, he co-founded the software company Microsoft in 1975 with his childhood friend Paul Allen. Following Microsoft's initial public offering in 1986 and the subsequent increase in its stock price, Gates became the world's then-youngest billionaire in 1987, at age 31. Forbes magazine ranked him as the world's wealthiest person in their The World's Billionaires list for 18 out of 24 years between 1995 and 2017, including 13 years consecutively from 1995 to 2007. Gates became the first centibillionaire in 1999, when his net worth briefly surpassed US$100 billion. According to Forbes, as of February 2026, his net worth stood at US$107.7 billion, making him the 18th-wealthiest individual in the world. 
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """
    prompt = PromptTemplate(template=summary_template, input_variables=["information"])
    chain = prompt | llm
    response = chain.invoke({"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
