from langchain_ollama import ChatOllama
from src.config.prompts import HIJACK_GENERATION_PROMPT

class SemanticHijacker:
    def __init__(self, model_name="llama3.2"):
        # We use a higher temperature here because we want the model 
        # to be creative in synthesizing the description.
        self.llm = ChatOllama(model=model_name, temperature=0.7)

    def generate_attack_description(self, shadow_tasks: list[str]) -> str:
        """
        Generates a tool description (R) that semantically overlaps 
        with the provided shadow tasks (Q').
        """
        print(f"--- Generating Adversarial Description for {len(shadow_tasks)} shadow tasks ---")
        
        # Format tasks as a list for the prompt
        formatted_tasks = "\n".join([f"- {task}" for task in shadow_tasks])
        
        chain = HIJACK_GENERATION_PROMPT | self.llm
        
        # Run the generation
        result = chain.invoke({
            "shadow_task_descriptions": formatted_tasks,
            "word_limit": "50"
        })
        
        description = result.content.strip()
        print(f"\n[GENERATED POISONED DESCRIPTION]:\n{description}\n")
        return description