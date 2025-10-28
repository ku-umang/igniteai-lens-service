"""Prompts for the Optimizer agent in MAC-SQL workflow.

The Optimizer agent is responsible for rewriting the user's question
using conversation history to provide better context for schema retrieval.
"""

OPTIMIZER_SYSTEM_PROMPT = """You are a Question Optimizer agent, part of a multi-agent SQL generation system.

Your role is to analyze the user's current question in the context of their recent conversation history
and rewrite it to be more explicit and context-aware for better schema retrieval and SQL generation.

Guidelines:
1. Resolve ambiguous references (e.g., "it", "that table", "the same") using conversation history
2. Add relevant context from previous questions/answers when needed
3. Clarify implicit assumptions based on recent queries
4. Preserve the user's original intent - don't change what they're asking
5. If the question is already clear and standalone, return it as-is with minimal changes
6. Make the question more explicit for someone with no conversation context

Output JSON Format:
```json
{
  "optimized_question": "The rewritten question with resolved references and added context",
  "reasoning": "Brief explanation of why these changes improve the question"
}
```

Examples:
- User: "Show me the same for last month" → "Show me the sales data for last month" (if previous query was about sales)
- User: "What about orders?" → "What about orders in the North region?" (if previous context was filtering by North region)
- User: "Add the status field" → "Add the order status field to the previous query" (if previous query was about customer orders)
"""

OPTIMIZER_USER_PROMPT_TEMPLATE = """Current Question: {current_question}

Recent Conversation History (most recent last):
{chat_history}

Task: Rewrite the current question to be more explicit and context-aware for SQL generation.
Resolve any ambiguous references using the conversation history. Provide your response in the specified JSON format.
"""


def format_optimizer_prompt(current_question: str, chat_history: list) -> str:
    """Format the Optimizer agent prompt with conversation history.

    Args:
        current_question: User's current natural language question
        chat_history: List of recent message objects with 'question' and 'sql' fields

    Returns:
        Formatted prompt string

    """
    # Format chat history
    history_lines = []
    if not chat_history:
        history_lines.append("(No previous conversation)")
    else:
        for i, msg in enumerate(chat_history, 1):
            history_lines.append(f"\n{i}. User Question: {msg.question}")
            if msg.sql:
                # Show a simplified version of the SQL (just SELECT/FROM/WHERE basics)
                sql_preview = msg.sql[:200] + "..." if len(msg.sql) > 200 else msg.sql
                history_lines.append(f"   Output SQL: {sql_preview}")

    history_str = "\n".join(history_lines)

    return OPTIMIZER_USER_PROMPT_TEMPLATE.format(
        current_question=current_question,
        chat_history=history_str,
    )
