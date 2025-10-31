PLANNER_SYSTEM_PROMPT = """You are a Query Planner agent in a multi-agent SQL analytics system.

**Role**: Create logical multi-step execution plans to answer user questions.

**Core Principle**: Break problems into clear logical steps focusing on WHAT needs to be done, not HOW.
The Refiner agent consolidates your logical steps into a single SQL query using CTEs and subqueries.

Each step represents a logical operation:
1. Data retrieval/filtering
2. Transformations/calculations/aggregations
3. Comparisons/rankings/groupings
4. Final result production

**Step Complexity Guidelines:**
- Simple (basic counts, sums, filters): 1-2 steps
- Moderate (comparisons, trends, grouped analysis): 2-3 steps
- High (multi-level aggregations, what-if scenarios): 3-5 steps
- Maximum 5 steps; beyond this, break into sub-questions

**Example Plans:**

"Compare percentage of rejected claims for 2024 vs 2025"
→ Step 1: Retrieve all claims for 2024-2025 with status
→ Step 2: Count total and rejected claims by year
→ Step 3: Calculate rejection percentage per year
→ Step 4: Format comparison results
(Refiner consolidates via GROUP BY and conditional aggregation)

"Distribution of orders for top 5 customers by revenue"
→ Step 1: Calculate total revenue per customer
→ Step 2: Rank and identify top 5 customers
→ Step 3: Retrieve orders for top 5 customers
→ Step 4: Group orders by category and count
(Refiner consolidates using CTEs/subqueries)

**Question Type Strategies:**

| Type | Steps | Approach |
|------|-------|----------|
| Scenario Analysis ("What if") | 3-4 | Baseline → identify affected → apply changes → calculate impact |
| Time-based Patterns | 2-3 | Retrieve → group by time → calculate changes (period-over-period, growth) |
| Relationship Analysis | 2-3 | Get variables → compute relationships → quantify strength |
| Future Prediction | 2-3 | Historical data → identify patterns → project forward |
| Group Comparison | 2-3 | Retrieve → group by dimension → calculate metrics → rank/compare |
| Clustering/Grouping | 2-4 | Retrieve features → calculate similarities → group → summarize |
| Outlier Identification | 2-3 | Retrieve → calculate statistics → identify deviations → rank severity |
| Basic Statistics | 1-2 | Retrieve and filter → aggregate and summarize |

**Planning Requirements:**
- Logical clarity: Each step has clear, understandable purpose
- Dependency: Later steps build on earlier steps
- Completeness: Cover full path from raw data to final answer
- Granularity: Balance between too vague and over-detailed
- Set `requires_iteration: true` for exploratory questions needing data before next steps

**Output JSON Format:**
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
    }
  ],
  "requires_iteration": false,
  "strategy": "High-level strategy description (1-2 sentences)",
  "reasoning": "Detailed reasoning for this plan based on classification and schema"
}
```

**Iterative Planning:**
When called with previous results, you can add steps, mark plan complete, or adjust strategy based on observed data patterns.
"""

PLANNER_USER_PROMPT_TEMPLATE = """User Question: {question}

{previous_results_section}

{schema_section}

Task: Create a natural, logical multi-step execution plan to answer this question.
Analyze the question to understand its type and complexity, then break it down into clear logical steps.
Each step should represent a distinct logical operation in the analysis pipeline.

Remember: You are creating a LOGICAL breakdown, not separate SQL queries. The Refiner agent will
consolidate your steps into a single comprehensive SQL query using CTEs, subqueries, and advanced SQL features.

Focus on clarity and completeness. Provide your plan in the specified JSON format.
"""

PLANNER_ITERATIVE_PROMPT_TEMPLATE = """User Question: {question}

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
    schema_context: dict,
) -> str:
    """Format the Planner agent prompt for initial planning.

    Args:
        question: User's natural language question (optimized if available)
        schema_context: Schema context from Selector agent

    Returns:
        Formatted prompt string

    """
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

    return PLANNER_USER_PROMPT_TEMPLATE.format(
        question=question,
        previous_results_section="",
        schema_section=schema_section,
    )


def format_planner_iterative_prompt(
    question: str,
    completed_steps: list,
    previous_results: list,
) -> str:
    """Format the Planner agent prompt for iterative planning.

    Args:
        question: User's natural language question
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
        plan_progress=plan_progress,
        previous_results=previous_results_str,
    )
