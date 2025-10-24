"""Prompts for the Decomposer agent in MAC-SQL workflow.

The Decomposer agent breaks down the user's question into logical query steps
and creates a query plan that the Refiner can convert to SQL.
"""

DECOMPOSER_SYSTEM_PROMPT = """You are a Query Decomposer agent, part of a multi-agent SQL generation system.

Your role is to analyze the user's question and the selected schema, then create a logical query plan.

You should:
1. Break complex questions into logical steps
2. Identify necessary operations (filters, joins, aggregations, sorting, etc.)
3. Determine the JOIN strategy if multiple tables are involved
4. Identify any window functions, CTEs, or subqueries needed
5. Estimate query complexity
6. Provide clear reasoning for your decomposition

Output Json Format:
```json
{
  "steps": [
    "Step 1: Filter customers by region",
    "Step 2: Join with orders table",
    "Step 3: Aggregate total sales per customer",
    "Step 4: Sort by total sales descending"
  ],
  "join_strategy": "INNER JOIN between customers and orders on customer_id",
  "aggregations": ["SUM(order_amount)", "COUNT(orders)"],
  "filters": ["region = 'Northeast'", "order_date >= '2024-01-01'"],
  "window_functions": [],
  "needs_cte": false,
  "complexity_score": 0.6,
  "reasoning": "Your detailed reasoning for the decomposition"
}
```
Complexity Scoring (0-1):
- 0.1-0.3: Simple SELECT with basic filters
- 0.3-0.5: JOINs with aggregations
- 0.5-0.7: Multiple JOINs, window functions, or subqueries
- 0.7-1.0: Complex CTEs, recursive queries, or multiple subqueries
"""

DECOMPOSER_USER_PROMPT_TEMPLATE = """User Question: {question}

Selected Schema:
{selected_schema}

Selected Tables: {selected_tables}

Join Paths (if any):
{join_paths}

Task: Create a logical query plan to answer the user's question.
Break it into clear steps and provide your plan in the specified JSON format.

Consider:
- What filters are needed?
- What aggregations are needed?
- What joins are required?
- What sorting or grouping is needed?
- Are window functions or CTEs beneficial?
"""


def format_decomposer_prompt(
    question: str,
    selected_tables: list,
    selected_columns: dict,
    join_paths: list,
) -> str:
    """Format the Decomposer agent prompt with selected schema.

    Args:
        question: User's natural language question
        selected_tables: List of selected table names
        selected_columns: Dict mapping table names to column lists
        join_paths: List of join path definitions

    Returns:
        Formatted prompt string

    """
    # Format selected schema
    schema_lines = []
    for table in selected_tables:
        columns = selected_columns.get(table, [])
        schema_lines.append(f"Table: {table}")
        schema_lines.append(f"  Columns: {', '.join(columns)}")
    schema_str = "\n".join(schema_lines) if schema_lines else "No schema selected"

    # Format join paths
    join_paths_lines = []
    for jp in join_paths:
        from_table = jp.get("from_table", "unknown")
        to_table = jp.get("to_table", "unknown")
        join_cols = jp.get("join_columns", [])
        join_desc = f"{from_table} -> {to_table}"
        if join_cols:
            col_pairs = ", ".join([f"{jc.get('from')}={jc.get('to')}" for jc in join_cols])
            join_desc += f" ON {col_pairs}"
        join_paths_lines.append(f"- {join_desc}")
    join_paths_str = "\n".join(join_paths_lines) if join_paths_lines else "No joins needed"

    return DECOMPOSER_USER_PROMPT_TEMPLATE.format(
        question=question,
        selected_schema=schema_str,
        selected_tables=", ".join(selected_tables),
        join_paths=join_paths_str,
    )
