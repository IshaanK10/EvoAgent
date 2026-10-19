
import requests

import agent
import learner
import memory

MODEL = "llama3.2:3b"


def show_memories():
    facts = memory.get_memories()

    if not facts:
        print("\nNo memories saved yet.")
        return

    print("\n--- EvoAgent Memories ---")
    for index, fact in enumerate(facts, start=1):
        print(f"{index}. {fact}")


def show_learning_stats():
    stats = learner.get_stats()

    print("\n--- Learning Statistics ---")
    print("Training examples:", stats["training_examples"])
    print("Feedback classes seen:", stats["feedback_classes_seen"])


def main():
    memory.initialize()

    print("\n================================")
    print("       EvoAgent Assistant")
    print("================================")
    print("Local model:", MODEL)
    print("Autonomous tools: calculator, list_notes, read_note")
    print("\nCommands:")
    print("  /remember <fact>")
    print("  /memories")
    print("  /stats")
    print("  /quit")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting EvoAgent.")
            break

        if not question:
            continue

        if question == "/quit":
            print("Goodbye!")
            break

        if question.startswith("/remember "):
            fact = question[len("/remember "):].strip()

            if fact:
                memory.add_memory(fact)
                print("Memory saved.")
            else:
                print("Please provide a fact to remember.")

            continue

        if question == "/memories":
            show_memories()
            continue

        if question == "/stats":
            show_learning_stats()
            continue

        # The autonomous agent handles planning and tool execution.
        answer = agent.run_agent(question)
        print("\nEvoAgent:", answer)

        rating = input(
            "\nWas this answer helpful? (y/n/skip): "
        ).strip().lower()

        if rating in ("y", "n"):
            try:
                count = learner.learn(
                    question,
                    answer,
                    helpful=(rating == "y"),
                )
                print("Learning examples recorded:", count)
            except (ValueError, OSError, KeyError) as error:
                print("Could not update the learning model:", error)


if __name__ == "__main__":
    main()
