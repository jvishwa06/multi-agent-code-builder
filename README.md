# 🛠️Agentic AI Engineering Team

**Agentic AI Engineering Team** is an AI-powered coding assistant built with [LangGraph](https://github.com/langchain-ai/langgraph).  
It works like a multi-agent development team that can take a natural language request and transform it into a complete, working project — file by file — using real developer workflows.

---

## 🏗️ Architecture

- **SENIOR SOFTWARE ENGINEERING PLANNER AGENT** – Analyzes your request and generates a detailed project plan.
- **SENIOR SOFTWARE ARCHITECT AGENT** – Breaks down the plan into specific engineering tasks with explicit context for each file.
- **SENIOR SOFTWARE ENGINEER AGENT** – Implements each task, writes directly into files, and uses available tools like a real developer.

---

## 🚀 Getting Started
### Prerequisites
- Make sure you have uv installed, follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/) to install it.
- Ensure that you have created a groq account and have your API key ready. Create an API key [here](https://console.groq.com/keys).

### ⚙️ **Instsllstion and Startup**
- Create a virtual environment using: `uv venv` and activate it using `source .venv/bin/activate`
- Install the dependencies using: `uv pip install -r pyproject.toml`
- Create a `.env` file and add the variables and their respective values mentioned in the `.sample_env` file

Now that we are done with all the set-up & installation steps we can start the application using the following command:
  ```bash
    python main.py
  ```