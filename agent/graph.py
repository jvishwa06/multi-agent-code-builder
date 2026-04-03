from dotenv import load_dotenv
from langchain_core.globals import set_verbose, set_debug
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from agent.prompts import *
from agent.states import *
from agent.tools import write_file, read_file, init_project_root_with_name

load_dotenv()

set_debug(True)
set_verbose(True)

llm = ChatGroq(model="openai/gpt-oss-120b")


def planner_agent(state: AgentGraphState) -> dict:
    """Converts user prompt into a structured Plan."""
    user_prompt = state["user_prompt"]
    try:
        resp = llm.with_structured_output(Plan).invoke(planner_prompt(user_prompt))
    except Exception as e:
        print(f"Warning: Structured output failed, using fallback: {e}")
        # Fallback: create a minimal Plan
        resp = Plan(
            name="Generated Project",
            description="Project created from user prompt",
            techstack="python,javascript",
            features=["feature1", "feature2"],
            files=[]
        )
    
    if resp is None:
        raise ValueError("Planner did not return a valid response.")
    return {"plan": resp}


def init_project_agent(state: AgentGraphState) -> dict:
    """Initialize the project root with the project name."""
    plan: Plan = state["plan"]
    project_root_path = init_project_root_with_name(plan.name)
    print(f"📁 Project folder created: {project_root_path}")
    return {"status": "project_initialized"}


def architect_agent(state: AgentGraphState) -> dict:
    """Creates TaskPlan from Plan with proper task breakdown."""
    plan: Plan = state["plan"]
    
    # Generate implementation tasks from the plan's files
    implementation_steps = []
    
    if plan.files:
        # Create a task for each file in the plan
        for file_obj in plan.files:
            task = ImplementationTask(
                filepath=file_obj.path,
                task_description=f"Create {file_obj.path}: {file_obj.purpose}"
            )
            implementation_steps.append(task)
    else:
        # Fallback: try structured output
        try:
            resp = llm.with_structured_output(TaskPlan).invoke(architect_prompt(plan=plan.model_dump_json()))
            implementation_steps = resp.implementation_steps if resp else []
        except Exception as e:
            print(f"Warning: Could not generate tasks: {e}")
            implementation_steps = [
                ImplementationTask(
                    filepath="README.md",
                    task_description="Create project documentation and setup files"
                )
            ]
    
    # Create the final TaskPlan
    task_plan = TaskPlan(implementation_steps=implementation_steps)
    task_plan.plan = plan
    
    print(f"Generated {len(implementation_steps)} implementation tasks:")
    for i, step in enumerate(implementation_steps, 1):
        print(f"  {i}. {step.filepath}: {step.task_description[:60]}...")
    
    return {"task_plan": task_plan}


def coder_agent(state: AgentGraphState) -> dict:
    """LangGraph tool-using coder agent with proper file generation."""
    coder_state: CoderState = state.get("coder_state")
    if coder_state is None:
        # Initialize with first step
        task_plan = state.get("task_plan")
        if not task_plan or not task_plan.implementation_steps:
            # No steps to process - mark as complete
            return {"coder_state": None, "status": "DONE"}
        coder_state = CoderState(task_plan=task_plan, current_step_idx=0)

    steps = coder_state.task_plan.implementation_steps
    current_idx = coder_state.current_step_idx
    
    # Safety check: if index out of bounds, mark as done
    if current_idx >= len(steps):
        print(f"All {len(steps)} steps completed!")
        return {"coder_state": coder_state, "status": "DONE"}

    current_task = steps[current_idx]
    filepath = current_task.filepath
    
    # Try to read existing content
    try:
        existing_content = read_file.invoke({"path": filepath})
    except Exception as e:
        existing_content = ""

    system_prompt = coder_system_prompt()
    user_message = (
        f"Task {current_idx + 1}/{len(steps)}: {current_task.task_description}\n"
        f"File: {filepath}\n"
        f"Existing content (if any):\n{existing_content[:500] if existing_content else '[Empty or new file]'}\n\n"
        f"Please generate the complete content for this file. "
        f"Provide the full code/config, not just a description."
    )

    # Invoke the LLM to generate file content
    print(f"\n📝 Step {current_idx + 1}/{len(steps)}: Creating {filepath}")
    
    try:
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        
        # Extract the response content
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Try to extract code blocks if present
        if "```" in response_text:
            # Extract code between triple backticks
            import re
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', response_text, re.DOTALL)
            if code_blocks:
                file_content = code_blocks[0]
            else:
                file_content = response_text
        else:
            file_content = response_text
        
        # Write the file using invoke
        write_result = write_file.invoke({"path": filepath, "content": file_content})
        print(f"✅ Created: {write_result}")
        
    except Exception as e:
        print(f"⚠️ Warning: Step {current_idx} failed: {e}")
        # Create a minimal placeholder file so we can continue
        try:
            write_file.invoke({"path": filepath, "content": f"// TODO: Complete implementation for {filepath}\n// Error: {str(e)}"})
        except Exception as write_err:
            print(f"⚠️ Could not write fallback file: {write_err}")

    # Create new CoderState with incremented index
    new_coder_state = CoderState(
        task_plan=coder_state.task_plan,
        current_step_idx=current_idx + 1,
        current_file_content=existing_content
    )
    
    # Check if this was the last step
    if new_coder_state.current_step_idx >= len(steps):
        print(f"\n🎉 All {len(steps)} files have been created!")
        return {"coder_state": new_coder_state, "status": "DONE"}
    
    # Continue to next task
    return {"coder_state": new_coder_state}


graph = StateGraph(AgentGraphState)

graph.add_node("planner", planner_agent)
graph.add_node("init_project", init_project_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder", coder_agent)

graph.add_edge(START, "planner")
graph.add_edge("planner", "init_project")
graph.add_edge("init_project", "architect")
graph.add_edge("architect", "coder")
graph.add_conditional_edges(
    "coder",
    lambda s: END if s.get("status") == "DONE" else "coder",
    {END: END, "coder": "coder"}
)
agent = graph.compile()
if __name__ == "__main__":
    result = agent.invoke({"user_prompt": "Build a colourful modern todo app in html css and js"},
                          {"recursion_limit": 100})
    print("Final State:", result)
    