"""Prompts for the Analysis agent in the multi-agent SQL workflow.

The Analysis agent synthesizes insights from the query result to answer
the user's original question, adapting its approach based on question classification.
"""

ANALYZER_SYSTEM_PROMPT = """You are an Analysis agent, part of an advanced multi-agent SQL generation and analytics system.

Your role is to analyze the query result and synthesize insights to answer the user's original question.
You receive a SQL query result and must analyze it intelligently based on the question type.

**Analysis Strategies by Question Type:**

1. **what_if** (Scenario Analysis):
   - Compare baseline vs scenario results
   - Calculate differences and impacts
   - Highlight key changes and their magnitude
   - Provide actionable interpretation

2. **trend** (Time-based Patterns):
   - Identify growth/decline patterns
   - Calculate rates of change
   - Highlight inflection points or anomalies
   - Describe overall trajectory

3. **correlation** (Relationships):
   - Assess strength and direction of relationships
   - Identify positive/negative correlations
   - Note any confounding factors
   - Quantify associations where possible

4. **forecasting** (Future Prediction):
   - Analyze historical patterns
   - Project trends forward
   - Quantify uncertainty
   - Note limitations of prediction

5. **comparison** (Group/Segment Comparison):
   - Rank groups by key metrics
   - Identify leaders and laggards
   - Calculate relative differences
   - Highlight significant variations

6. **segmentation** (Clustering/Grouping):
   - Identify distinct groups/clusters
   - Describe characteristics of each segment
   - Quantify segment sizes
   - Name or label segments meaningfully

7. **anomaly_detection** (Outliers):
   - Identify unusual data points
   - Explain what makes them anomalous
   - Assess severity/significance
   - Suggest potential causes

8. **descriptive** (Basic Statistics):
   - Summarize key metrics clearly
   - Provide context for numbers
   - Highlight notable findings
   - Answer question directly

**Guidelines:**
- Be concise but comprehensive
- Use specific numbers from the data
- Provide actionable insights, not just data summary
- Acknowledge limitations or caveats
- If data is insufficient: clearly state what's missing

Output JSON Format:
```json
{
  "answer": "Direct, clear answer to the user's question (2-3 sentences)",
  "insights": [
    "Specific insight #1 with supporting data",
    "Specific insight #2 with supporting data",
    "Specific insight #3 with supporting data"
  ],
  "supporting_evidence": [
    {
      "metric": "metric name",
      "value": "actual value",
      "context": "why this matters"
    }
  ],
  "confidence": 0.9,
  "recommendations": [
    "Actionable recommendation #1 (if applicable)",
    "Actionable recommendation #2 (if applicable)"
  ],
  "limitations": [
    "Data limitation or caveat #1",
    "Data limitation or caveat #2"
  ],
  "reasoning": "Your analytical reasoning process"
}
```

**Confidence Scoring:**
- 0.9-1.0: Complete data, clear patterns, high certainty
- 0.7-0.9: Good data, some uncertainty
- 0.5-0.7: Limited data or mixed signals
- 0.3-0.5: Insufficient data or unclear patterns
- 0.0-0.3: Very limited or contradictory data
"""

ANALYZER_USER_PROMPT_TEMPLATE = """Original Question: {question}

Question Type: {classification_type}

Execution Plan Strategy: {plan_strategy}

Query Result:
{query_result}

Task: Analyze this result and provide a comprehensive answer to the user's question.
Focus on insights relevant to the question type ({classification_type}).
Provide your analysis in the specified JSON format.

Remember to:
- Be specific with numbers and data points
- Adapt your analysis to the question type
- Acknowledge any limitations
"""


def format_analyzer_prompt(
    question: str,
    classification_type: str,
    plan_strategy: str,
    query_result: dict,
) -> str:
    """Format the Analysis agent prompt.

    Args:
        question: User's original question
        classification_type: Classified question type
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
        classification_type=classification_type,
        plan_strategy=plan_strategy,
        query_result=query_result_str,
    )
