import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# 1. Load environment variables
load_dotenv()

# 2. Configure Gemini LLM (gemini-3.6-flash)
gemini_llm = LLM(
    model="gemini/gemini-3.6-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

# 3. Custom File Reader Tool
@tool("Read Market Data File")
def read_market_data(query: str = "") -> str:
    """Reads the entire contents of the market data knowledge base file."""
    file_path = os.path.join("knowledge_base", "market_data.txt")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Error: Market data file not found."

# 4. Define Agents
retriever_agent = Agent(
    role="Market Data Specialist",
    goal="Retrieve precise financial facts from market data.",
    backstory="An expert researcher focused on extracting accurate facts from financial documents.",
    tools=[read_market_data],
    llm=gemini_llm,
    verbose=True
)

analyst_agent = Agent(
    role="Senior Financial Analyst",
    goal="Synthesize financial facts into clear executive briefs.",
    backstory="A seasoned analyst who turns raw figures into strategic takeaways.",
    llm=gemini_llm,
    verbose=True
)

# 5. Define Tasks
retrieval_task = Task(
    description="Read market_data.txt to find revenue growth, profit margins, growth drivers, and risk factors for Acme Corp.",
    expected_output="A list of raw financial facts retrieved directly from the file.",
    agent=retriever_agent
)

analysis_task = Task(
    description="Summarize the retrieved facts into a 3-bullet-point executive report on financial performance and risk.",
    expected_output="An executive briefing containing exactly 3 strategic takeaways.",
    agent=analyst_agent
)

# 6. Execute Crew
market_crew = Crew(
    agents=[retriever_agent, analyst_agent],
    tasks=[retrieval_task, analysis_task],
    process=Process.sequential
)

if __name__ == "__main__":
    print("--- Starting Agentic RAG Execution ---")
    result = market_crew.kickoff()
    print("\n=== Final Executive Summary ===")
    print(result)