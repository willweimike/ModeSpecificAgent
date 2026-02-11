# ModeSpecificAgent

* **Multi-Agent Architecture:** A conversational AI system built using the LangChain and LangGraph frameworks to manage state and workflows between different nodes.
* **Local LLM Integration:** Powered entirely by a local language model (`qwen3:8b`) running through Ollama, ensuring privacy and local execution.
* **Dynamic Intent Classification:** Utilizes an intelligent routing system with structured LLM output (Pydantic) to classify user messages as either "emotional" or "logical" in real-time.
* **Specialized Responders:**
* **Emotional Agent:** Prompted to act as a compassionate therapist, focusing on empathy, feeling validation, and emotional processing rather than immediate problem-solving.
* **Logical Agent:** Prompted to act as a purely factual assistant, delivering direct, evidence-based, and concise answers to practical queries.
