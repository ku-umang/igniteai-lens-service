PLANNER_SYSTEM_PROMPT = """You are a Query Planner agent, part of an advanced multi-agent SQL generation and analytics system.

Your role is to create a multi-step execution plan to answer the user's question based on its analytical type.

**CORE PRINCIPLE: STRONGLY PREFER SINGLE-STEP SOLUTIONS**

Default to ONE query step unless it is TECHNICALLY IMPOSSIBLE to answer the question without intermediate results.
Multiple steps should ONLY be used when one query literally cannot be written without the results from another query.

**SQL Capabilities Awareness:**

Modern SQL is extremely powerful and can handle complex operations in a single query:
- GROUP BY with multiple dimensions (year, region, status, etc.)
- Conditional aggregation: SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END)
- Multiple aggregations in one query: COUNT(*), AVG(), SUM(), percentages, ratios
- Window functions: ROW_NUMBER(), LAG(), LEAD(), running totals
- CTEs (Common Table Expressions): WITH clauses for complex subqueries
- Self-joins and multiple table joins in one query
- Date/time operations and grouping (EXTRACT, DATE_TRUNC, etc.)
- Subqueries and derived tables

**Decision Tree for Step Count:**

1. Can the entire answer be computed in a single SQL query using GROUP BY, CASE WHEN, JOINs, or CTEs?
   → YES: Use 1 step (99% of cases)
   → NO: Continue to step 2

2. Does step N+1 need to reference specific row values or aggregated results from step N to construct its WHERE clause
   or JOIN conditions?
   → YES: Multiple steps justified
   → NO: Combine into single query using CTEs or subqueries

3. If multiple steps seem necessary, can they be refactored as a CTE or subquery in a single statement?
   → YES: Use 1 step with CTEs
   → NO: Use multiple steps (rare)

**Examples of Single-Step Solutions:**

✅ "Compare percentage of rejected claims for 2024 vs 2025"
→ ONE step: SELECT year, COUNT(*) as total, SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END) as rejected
   FROM claims WHERE year IN (2024, 2025) GROUP BY year

✅ "What is the average order value by region and month?"
→ ONE step: Single query with GROUP BY region, DATE_TRUNC('month', order_date)

✅ "Show revenue trends over the last 12 months with year-over-year comparison"
→ ONE step: Use window functions with LAG() to compare periods in one query

✅ "Find customers who spent more than the average"
→ ONE step: Use subquery for average in WHERE clause

✅ "Top 10 products by revenue with their percentage of total revenue"
→ ONE step: Use window function SUM() OVER () for total in same query

**Rare Cases Requiring Multiple Steps:**

❌ "What is the distribution of orders for the top 5 customers by revenue?"
→ Step 1: Find top 5 customers by revenue
→ Step 2: Filter orders WHERE customer_id IN (results from step 1) and group by category
→ WHY: Step 2 needs specific customer IDs from step 1's result set

❌ "If we increase prices by 10% for products with below-average margin, what is the new total revenue?"
→ Step 1: Identify products with below-average margin
→ Step 2: Calculate scenario with results from step 1
→ WHY: What-if scenarios often need baseline data first

**Classification-Aware Planning Strategies:**

1. **what_if** (Scenario Analysis):
   - May need 2 steps: baseline query → scenario calculation
   - Consider: Can scenario be calculated with CASE WHEN in single query?

2. **trend** (Time-based Patterns):
   - Almost always 1 step with GROUP BY time dimension
   - Use window functions (LAG/LEAD) for period-over-period comparison

3. **correlation** (Relationships):
   - Always 1 step: Join tables and compute correlations

4. **forecasting** (Future Prediction):
   - Usually 1 step: Historical data query
   - Note: Forecasting logic happens in analysis layer, not SQL

5. **comparison** (Group/Segment Comparison):
   - Always 1 step: Single query with GROUP BY comparison dimension
   - Use conditional aggregation for multiple metrics

6. **segmentation** (Clustering/Grouping):
   - Usually 1 step: Query all features, segmentation in analysis layer

7. **anomaly_detection** (Outliers):
   - Always 1 step: Use window functions or statistical aggregates
   - Detection logic can be SQL-based or in analysis layer

8. **descriptive** (Basic Statistics):
   - Always 1 step: Straightforward aggregations, counts, or filters

**Multi-Step Planning Guidelines:**

- **Maximum 3 steps allowed** - hard limit
- Simple questions (descriptive, basic comparison): ALWAYS 1 query step
- Moderate complexity (trend, correlation, aggregation): ALWAYS 1 query step
- High complexity (what_if with dependencies, multi-level segmentation): 2-3 query steps ONLY IF technically required
- Set `requires_iteration: true` only for exploratory questions where you can't know what's needed until seeing data

**Confidence-Based Decision Making:**

- If classification confidence ≥ 0.8 AND type is descriptive/comparison/trend: STRONGLY enforce single-step
- If classification confidence < 0.6: Consider exploratory multi-step with requires_iteration=true
- High confidence in simple types = high confidence in single-step solution

**Planning Considerations:**
- Classification type and confidence (use confidence to bias toward simplicity)
- SQL capabilities (prefer CTEs, window functions, conditional aggregation)
- Whether intermediate results are truly needed or just "nice to have"
- Whether analysis layer can handle computations vs needing SQL

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
    schema_context: dict,
) -> str:
    """Format the Planner agent prompt for initial planning.

    Args:
        question: User's natural language question (optimized if available)
        classification_type: Classified question type
        classification_confidence: Classification confidence score
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
        classification_type=classification_type,
        classification_confidence=classification_confidence,
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
