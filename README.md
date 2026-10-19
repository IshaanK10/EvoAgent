# EvoAgent — Local AI Agent with Persistent Memory

EvoAgent is a Python-based AI assistant powered by a locally running large language model (LLM). It combines tool execution, persistent memory, and feedback-based machine learning to create an assistant that can interact with its environment and improve a separate predictive model using user feedback.

## Features

- **Local LLM:** Uses Llama 3.2 (3B) through Ollama for local inference.
- **Persistent Memory:** Stores user-provided facts in SQLite so they can be retrieved across sessions.
- **Tool Calling:** Supports arithmetic calculations, listing notes, and reading text or Markdown files.
- **Bounded Agent Loop:** Allows the model to select tools, inspect their results, and continue for up to five steps.
- **Feedback-Based Learning:** Uses scikit-learn's incremental `SGDClassifier` to learn from helpful and unhelpful response ratings.
- **Model Persistence:** Saves the feedback classifier using Joblib.
- **Tool Safety:** Validates tool names and arguments and restricts calculator expressions and note access.
- **Unit Testing:** Includes tests for calculator behavior, invalid inputs, file access, and unsupported tools.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| LLM | Llama 3.2 (3B) |
| Local inference | Ollama |
| Agent orchestration | Python, Requests |
| Persistent memory | SQLite |
| Feedback learning | scikit-learn |
| Model persistence | Joblib |
| Testing | Python `unittest` |

## Architecture

```text
User
 |
 v
main.py
 |
 +----> memory.py ----> SQLite
 |
 v
agent.py
 |
 +----> Ollama / Llama 3.2
 |
 +----> tools.py
 |       +-- Calculator
 |       +-- List notes
 |       +-- Read note
 |
 v
Assistant response
 |
 v
User feedback
 |
 v
learner.py
 |
 +----> Incremental classifier
         |
         +----> Joblib model persistence
```

## Project Structure

```text
EvoAgent/
├── main.py
├── agent.py
├── memory.py
├── learner.py
├── tools.py
├── tests/
│   └── test_tools.py
├── workspace/
│   └── notes/
├── data/                 # Local SQLite data
├── models/               # Saved learner model
└── .gitignore
```

Local databases, saved model files, and the virtual environment are excluded from version control.

## Prerequisites

- Python 3.11 or later recommended
- Ollama
- Git

The project has been developed and tested in a Python 3.13 virtual environment.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/IshaanK10/EvoAgent.git
cd EvoAgent
```

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install requests scikit-learn joblib
```

### 4. Install Ollama and download the model

Install Ollama from [ollama.com](https://ollama.com/), then run:

```bash
ollama pull llama3.2:3b
```

Ensure Ollama is running locally before starting the assistant.

### 5. Start EvoAgent

```bash
python main.py
```

## Usage

EvoAgent supports interactive questions and the following commands:

| Command | Description |
|---|---|
| `/remember <fact>` | Save a fact to persistent memory |
| `/memories` | Display saved memories |
| `/stats` | Display feedback-learning statistics |
| `/quit` | Exit the assistant |

### Example prompts

**Calculator**

```text
Calculate (125 * 8) + 40.
```

Expected result: `1040`

**Read a note**

```text
Read project.md and tell me which database the project uses.
```

**Save a memory**

```text
/remember I am building an AI agent called EvoAgent.
```

**View saved memories**

```text
/memories
```

## Feedback-Based Learning

After receiving a response, the user can rate whether it was helpful. The feedback is used to update an incremental scikit-learn classifier.

The classifier learns patterns associated with helpful and unhelpful responses from the question-and-answer text. Its state is persisted so learning can continue across sessions.

**Important:** This component trains a separate feedback classifier. It does not fine-tune the Llama 3.2 model or automatically modify the underlying LLM weights.

## Testing

Run the tool safety tests with:

```bash
python -m unittest discover -s tests -v
```

The current tool test suite contains 10 tests covering:

- Arithmetic evaluation
- Parentheses and mathematical expressions
- Rejection of arbitrary code
- Division by zero
- Unsupported operators and extra arguments
- Missing notes and path traversal
- Unknown tools and unsupported file types

All 10 tool tests passed in the development environment.

## Security Considerations

- Calculator input is evaluated using a restricted expression parser rather than arbitrary Python execution.
- Tool execution is restricted to an allowlist.
- Note access is restricted to the designated notes directory.
- The agent loop has a maximum of five tool steps.
- Local databases, learned models, and environment files should remain untracked.

This is a personal development project and has not been independently security-audited.

## Current Limitations

- Tool selection depends on the local LLM's output.
- Feedback learning updates a separate classifier, not the LLM itself.
- The feedback classifier's effectiveness depends on the quantity and quality of user ratings.
- Evaluation currently focuses on tool behavior and basic functional correctness.
- The assistant requires Ollama and the configured local model to be available.

## Future Improvements

- Add a repeatable task-evaluation benchmark.
- Improve tool-selection reliability and error recovery.
- Add conversation summarization and memory retrieval.
- Expand the tool ecosystem with additional permission-controlled capabilities.
- Add structured logs and performance measurements.

## Author

**Ishaan Khatwani**

GitHub: [@IshaanK10](https://github.com/IshaanK10)

## License

No license has been added yet. If you intend to let others use, modify, or distribute this project, consider adding an appropriate open-source license.