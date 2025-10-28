"""Prompts for the Planner agent in the multi-agent SQL workflow.

The Planner agent creates multi-step execution plans based on question classification,
supporting iterative planning where subsequent queries depend on previous results.
"""

PLANNER_SYSTEM_PROMPT = """You are a Query Planner agent, part of an advanced multi-agent SQL generation and analytics system.

Your role is to create a multi-step execution plan to answer the user's question based on its analytical type.

**Classification-Aware Planning Strategies:**

1. **what_if** (Scenario Analysis):
   - Query baseline/current state
   - Apply hypothetical changes via calculations
   - Compare scenarios
   - May need multiple queries to gather parameters and compute scenarios

2. **trend** (Time-based Patterns):
   - Query time-series data with appropriate granularity
   - Usually single query with time grouping
   - May need additional query for period-over-period comparison

3. **correlation** (Relationships):
   - Query both/all variables of interest
   - Usually single query joining relevant data
   - May need separate queries if variables are in distant tables

4. **forecasting** (Future Prediction):
   - Query historical data for the metric
   - Pattern: historical data query + analysis step
   - Note: Forecasting logic happens in analysis, not SQL

5. **comparison** (Group/Segment Comparison):
   - Single query with grouping by comparison dimension
   - May need multiple queries if comparing complex derived metrics

6. **segmentation** (Clustering/Grouping):
   - Query base data with relevant features
   - May need exploratory query first, then detailed query
   - Segmentation logic may happen in analysis layer

7. **anomaly_detection** (Outliers):
   - Query data with statistical aggregates (avg, stddev, etc.)
   - Usually single query with window functions or aggregations
   - Detection logic may be SQL-based or analysis-based

8. **descriptive** (Basic Statistics):
   - Usually single simple query
   - Straightforward aggregations, counts, or filters

**Multi-Step Planning Guidelines:**
- Simple questions (descriptive, basic comparison): 1 query step
- Moderate complexity (trend, correlation): 1-2 query steps
- High complexity (what_if, forecasting): 2-4 query steps
- Only create multiple steps if truly necessary
- Each step should build on previous results when dependent
- Set `requires_iteration: true` if you might need more queries after seeing first results

**Planning Considerations:**
- Classification type and confidence
- Schema complexity
- Previous query results (if in iterative mode)
- Whether analysis layer can handle certain computations vs needing SQL

Output JSON Format:
```json
{
  "steps": [
    {
      "step_number": 1,
      "description": "Clear description of what this query does",
      "purpose": "Why this step is needed to answer the question",
      "depends_on": [],
      "required_tables": ["table1", "table2"],
      "aggregations": ["SUM(revenue)", "COUNT(*)"],
      "filters": ["date >= '2024-01-01'"]
    },
    {
      "step_number": 2,
      "description": "Description of second query (if needed)",
      "purpose": "Why this step is needed",
      "depends_on": [1],
      "required_tables": ["table3"],
      "aggregations": [],
      "filters": []
    }
  ],
  "requires_iteration": false,
  "strategy": "High-level strategy description (1-2 sentences)",
  "reasoning": "Detailed reasoning for this plan based on classification and schema"
}
```

**Iterative Planning:**
If called with previous query results, you can:
- Add more steps if needed based on what you learned
- Mark the plan as complete if no more queries are needed
- Adjust strategy based on data patterns observed
"""

PLANNER_USER_PROMPT_TEMPLATE = """User Question: {question}

Question Classification:
- Type: {classification_type}
- Confidence: {classification_confidence}
- Characteristics: {classification_characteristics}

{previous_results_section}

{schema_section}

Task: Create a multi-step execution plan to answer this question.
Consider the question type and plan accordingly. Each step should be a distinct SQL query.
Keep it as simple as possible - only add steps if truly necessary.

Provide your plan in the specified JSON format.
"""

PLANNER_ITERATIVE_PROMPT_TEMPLATE = """User Question: {question}

Question Classification:
- Type: {classification_type}
- Confidence: {classification_confidence}

Current Execution Plan Progress:
{plan_progress}

Previous Query Results Summary:
{previous_results}

Task: Based on the results so far, decide if more query steps are needed.

Options:
1. Add more steps to the plan if additional queries are required
2. Mark the plan as complete if you have sufficient data

Provide your updated plan in the specified JSON format.
If adding steps, number them sequentially continuing from the last completed step.
"""


def format_planner_prompt(
    question: str,
    classification_type: str,
    classification_confidence: float,
    classification_characteristics: list,
    schema_context: dict | None = None,
) -> str:
    """Format the Planner agent prompt for initial planning.

    Args:
        question: User's natural language question (optimized if available)
        classification_type: Classified question type
        classification_confidence: Classification confidence score
        classification_characteristics: Key characteristics identified
        schema_context: Optional schema context from previous iteration

    Returns:
        Formatted prompt string

    """
    # Format characteristics
    chars_str = ", ".join(classification_characteristics) if classification_characteristics else "None identified"

    # Format schema section if available
    if schema_context:
        schema_lines = []
        tables = schema_context.get("tables", [])
        columns = schema_context.get("columns", {})

        for table in tables:
            table_name = table if isinstance(table, str) else table.get("metadata", {}).get("table_name", "unknown")
            table_columns = columns.get(table_name, [])
            schema_lines.append(f"Table: {table_name}")
            if table_columns:
                schema_lines.append(f"  Columns: {', '.join(table_columns[:15])}")  # Limit displayed columns

        relationships = schema_context.get("relationships", [])
        if relationships:
            schema_lines.append("\nJoin Paths:")
            for rel in relationships[:5]:  # Limit to top 5
                from_table = rel.get("from_table", "unknown")
                to_table = rel.get("to_table", "unknown")
                schema_lines.append(f"  {from_table} -> {to_table}")

        schema_section = "Selected Schema:\n" + "\n".join(schema_lines)
    else:
        schema_section = "Schema: Not yet retrieved (you may need to trigger schema selection first)"

    return PLANNER_USER_PROMPT_TEMPLATE.format(
        question=question,
        classification_type=classification_type,
        classification_confidence=classification_confidence,
        classification_characteristics=chars_str,
        previous_results_section="",
        schema_section=schema_section,
    )


def format_planner_iterative_prompt(
    question: str,
    classification_type: str,
    classification_confidence: float,
    completed_steps: list,
    previous_results: list,
) -> str:
    """Format the Planner agent prompt for iterative planning.

    Args:
        question: User's natural language question
        classification_type: Classified question type
        classification_confidence: Classification confidence score
        completed_steps: List of completed QuerySteps
        previous_results: List of execution results from completed steps

    Returns:
        Formatted prompt string for iterative planning

    """
    # Format plan progress
    progress_lines = []
    for step in completed_steps:
        status_emoji = "✓" if step.get("status") == "completed" else "✗"
        progress_lines.append(f"{status_emoji} Step {step.get('step_number')}: {step.get('description')} ({step.get('status')})")
    plan_progress = "\n".join(progress_lines) if progress_lines else "No steps completed yet"

    # Format previous results summary
    results_lines = []
    for i, result in enumerate(previous_results, 1):
        if result.get("success"):
            rows_count = result.get("rows_returned", 0)
            results_lines.append(f"Step {i} Result: {rows_count} rows returned")
            # Include a sample of the data structure
            if result.get("rows"):
                sample_row = result["rows"][0] if result["rows"] else {}
                columns = list(sample_row.keys())[:5]  # First 5 columns
                results_lines.append(f"  Columns: {', '.join(columns)}")
        else:
            results_lines.append(f"Step {i} Result: Failed - {result.get('error_message', 'Unknown error')}")

    previous_results_str = "\n".join(results_lines) if results_lines else "No results available yet"

    return PLANNER_ITERATIVE_PROMPT_TEMPLATE.format(
        question=question,
        classification_type=classification_type,
        classification_confidence=classification_confidence,
        plan_progress=plan_progress,
        previous_results=previous_results_str,
    )
