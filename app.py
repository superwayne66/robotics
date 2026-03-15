import os
import streamlit as st
from langchain_openai import OpenAI
from langchain_core.tools import Tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# -----------------------------
# Tools
# -----------------------------
def compare_items(query: str) -> str:
    try:
        parts = [p.strip() for p in query.split(",") if p.strip()]
        if len(parts) < 3:
            return "Error: please provide at least two items and a category."
        items = parts[:-1]
        category = parts[-1]

        comparison_template = """
You are an expert product analyst.

Category: {category}
Items: {items}

Compare the items in a concise but informative way:
- Brief overview of each item (1 line each)
- Key differences (bullets)
- Pros/cons for each
- Best choice for: budget / performance / overall (adapt to category)
Return a clean, readable answer.
"""
        comparison_prompt = PromptTemplate(
            input_variables=["items", "category"],
            template=comparison_template.strip()
        )
        comparison_chain = LLMChain(llm=st.session_state.llm, prompt=comparison_prompt)
        result = comparison_chain.invoke({"items": ", ".join(items), "category": category})["text"]
        return result.strip()
    except Exception as e:
        return f"Error in compare_items: {str(e)}"

def analyze_results(query: str) -> str:
    analysis_template = """
You are a concise analyst.

Analyze the following text and produce:
1) 1-2 sentence summary
2) Key takeaways (3-6 bullets)
3) Any caveats/uncertainties (if applicable)

Text:
{text}
"""
    analysis_prompt = PromptTemplate(
        input_variables=["text"],
        template=analysis_template.strip()
    )
    analysis_chain = LLMChain(llm=st.session_state.llm, prompt=analysis_prompt)
    result = analysis_chain.invoke({"text": query})["text"]
    return result.strip()

@st.cache_resource
def build_agent(openai_key: str, serpapi_key: str):
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["SERPAPI_API_KEY"] = serpapi_key

    llm = OpenAI(temperature=0)

    search_tool = load_tools(["serpapi"], llm=llm)[0]

    compare_tool = Tool(
        name="Compare",
        func=compare_items,
        description=(
            "Compare multiple items in a category. "
            "Input format: 'item1, item2, ..., category' "
            "Example: 'iPhone 15 Pro, Samsung Galaxy S24 Ultra, smartphones'."
        )
    )
    analyze_tool = Tool(
        name="Analyze",
        func=analyze_results,
        description="Summarize and extract key takeaways from any text (search results, comparisons, notes)."
    )

    tools = [search_tool, compare_tool, analyze_tool]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    return llm, agent

def process_query(agent, query: str):
    out = agent.invoke({"input": query})
    return out

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("ReAct Agent")
st.write("Ask a complex question and let the agent reason through it step by step.")

# Keys: Streamlit secrets (preferred) or environment vars
openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
serpapi_key = st.secrets.get("SERPAPI_API_KEY", os.getenv("SERPAPI_API_KEY", ""))

if not openai_key or not serpapi_key:
    st.warning("Missing API keys. Set OPENAI_API_KEY and SERPAPI_API_KEY in st.secrets or environment variables.")
    st.stop()

llm, agent = build_agent(openai_key, serpapi_key)
st.session_state.llm = llm  # used by compare/analyze tools

query = st.text_input("Enter your query:")

if st.button("Submit"):
    if query:
        with st.spinner("Thinking..."):
            output = process_query(agent, query)
            result = output.get("output", "")
            steps = output.get("intermediate_steps", [])

        st.subheader("Answer")
        st.write(result)

        with st.expander("Step-by-step trace (Thought / Action / Observation)", expanded=False):
            if not steps:
                st.write("No intermediate steps returned.")
            else:
                for idx, (action, observation) in enumerate(steps, start=1):
                    st.markdown(f"**Step {idx}**")
                    st.markdown(f"- **Action:** `{action.tool}`")
                    st.markdown(f"- **Action Input:** `{action.tool_input}`")
                    st.markdown(f"- **Observation:** {observation}")
                    st.write("---")
    else:
        st.warning("Please enter a query before submitting.")
