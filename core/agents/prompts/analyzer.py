ANALYZER_SYSTEM_PROMPT = """You are an Analysis agent in a multi-agent SQL generation and analytics system.

**Role**: Analyze query results and synthesize insights to answer the user's question based on the data.

**Analysis Strategies by Question Type:**

| Type | Approach |
|------|----------|
| Scenario Analysis ("What if") | Compare baseline vs scenario; calculate differences/impacts; highlight key changes/magnitude |
| Time-based Patterns | Identify growth/decline; calculate rate of change; highlight inflection points/anomalies |
| Relationships | Assess strength/direction; identify correlations; note confounding factors; quantify associations |
| Future Prediction | Analyze historical patterns; project trends forward; quantify uncertainty; note prediction limitations |
| Group Comparison | Rank groups by metrics; identify leaders/laggards; calculate relative differences; highlight variations |
| Clustering/Grouping | Identify distinct groups; describe segment characteristics; quantify sizes; label meaningfully |
| Outlier Detection | Identify unusual data points; explain what makes them anomalous; assess severity; suggest causes |
| Descriptive Statistics | Summarize key metrics; provide context; highlight notable findings; answer directly |

**Guidelines:**
- Be concise but comprehensive; use specific numbers from data
- Provide actionable insights, not just data summary
- Acknowledge limitations or caveats; if data insufficient, state what's missing

**Output JSON Format:**
```json
{
  "answer": "Direct, clear answer to the user's question (2-3 sentences)",
  "insights": ["Specific insight #1 with supporting data", "Insight #2", "Insight #3"],
  "supporting_evidence": [{"metric": "metric name", "value": "actual value", "context": "why this matters"}],
  "confidence": 0.9,
  "recommendations": ["Actionable recommendation #1 (if applicable)", "Recommendation #2"],
  "limitations": ["Data limitation or caveat #1", "Caveat #2"],
  "reasoning": "Your analytical reasoning process"
}
```

**Confidence Scoring:**
0.9-1.0 (complete data, clear patterns), 0.7-0.9 (good data, some uncertainty), 0.5-0.7 (limited data/mixed signals),
0.3-0.5 (insufficient data/unclear patterns), 0.0-0.3 (very limited/contradictory data)
"""

ANALYZER_USER_PROMPT_TEMPLATE = """Original Question: {question}

Execution Plan Strategy: {plan_strategy}

Query Result:
{query_result}

Task: Analyze this result and provide a comprehensive answer to the user's question.
Infer question type and adapt analysis accordingly. Be specific with numbers, adapt to question nature, acknowledge limitations.
Provide analysis in the specified JSON format.
"""


def format_analyzer_prompt(
    question: str,
    plan_strategy: str,
    query_result: dict,
) -> str:
    """Format the Analysis agent prompt.

    Args:
        question: User's original question
        plan_strategy: High-level strategy from execution plan
        query_result: Single execution result dict with data

    Returns:
        Formatted prompt string

    """
    # Format query result
    result_lines = []

    if not query_result.get("success", False):
        result_lines.append("Status: FAILED")
        result_lines.append(f"Error: {query_result.get('error_message', 'Unknown error')}")
    else:
        result_lines.append("Status: SUCCESS")
        result_lines.append(f"Rows Returned: {query_result.get('rows_returned', 0)}")

        # Include actual data
        rows = query_result.get("rows", [])
        if rows:
            result_lines.append("\nData:")

            # Show column headers
            if len(rows) > 0:
                columns = list(rows[0].keys())
                result_lines.append(f"  Columns: {', '.join(columns)}")

            # Show sample rows (up to 10)
            sample_size = min(10, len(rows))
            result_lines.append(f"  Sample ({sample_size} of {len(rows)} rows):")

            for row_idx, row in enumerate(rows[:sample_size], 1):
                row_str = ", ".join([f"{k}={v}" for k, v in list(row.items())[:6]])  # First 6 columns
                result_lines.append(f"    {row_idx}. {row_str}")

            # If more than 10 rows, provide summary statistics for numeric columns
            if len(rows) > 10:
                result_lines.append(f"\n  Note: {len(rows)} total rows available for analysis")

        else:
            result_lines.append("\nData: No rows returned")

        # Include execution metadata
        if query_result.get("execution_time_ms"):
            result_lines.append(f"\nExecution Time: {query_result.get('execution_time_ms')}ms")
        if query_result.get("cached"):
            result_lines.append("Cached: Yes")

    query_result_str = "\n".join(result_lines)

    return ANALYZER_USER_PROMPT_TEMPLATE.format(
        question=question,
        plan_strategy=plan_strategy,
        query_result=query_result_str,
    )
