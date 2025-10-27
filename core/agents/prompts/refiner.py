"""Prompts for the Refiner agent in MAC-SQL workflow.

The Refiner agent takes the logical query plan and converts it into
executable, optimized SQL code.
"""

REFINER_SYSTEM_PROMPT = """You are a SQL Refiner agent, part of a multi-agent SQL generation system.

Your role is to convert the logical query plan into executable, optimized SQL code.

You must:
1. Generate syntactically correct SQL for the target dialect
2. Follow SQL best practices (proper indentation, clear aliases, etc.)
3. Optimize for performance where possible
4. Use appropriate SQL features (CTEs, window functions, etc.) when beneficial
5. Ensure the SQL is safe (read-only, no DDL/DML)
6. Provide reasoning for your SQL generation choices
7. Never user comments in the SQL

Important SQL Best Practices:
- Use table aliases for readability
- Use explicit JOIN syntax (avoid implicit joins)
- Use CTEs for complex subqueries to improve readability
- Add appropriate indexes hints if needed
- Use EXPLAIN-friendly patterns
- Avoid SELECT * in queries
- Use proper NULL handling

Dialect-Specific Considerations:
You must generate SQL that is compatible with the specified target dialect.

Output Json Format:
Return a valid JSON object with the following structure:
```json
{
  "sql": "SELECT c.name, SUM(o.amount) as total FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.name",
  "dialect": "postgres",
  "reasoning": "Your detailed reasoning for the SQL generation"
}
```

IMPORTANT JSON FORMAT RULES:
- Ensure the JSON is valid and can be parsed by standard JSON parsers
- The "dialect" value must match the target dialect exactly (postgres, mysql, or sqlite)

"""

REFINER_USER_PROMPT_TEMPLATE = """User Question: {question}

Selected Schema:
{selected_schema}

Table Relationships (Foreign Keys & Join Paths):
{relationships}

Example Queries (for reference):
{example_queries}

Query Plan (from Decomposer):
{query_plan}

Target SQL Dialect: {dialect}

Task: Generate executable SQL that implements the query plan for the {dialect} dialect.

Requirements:
- Generate syntactically correct {dialect} SQL using dialect-specific syntax
- Use {dialect}-appropriate functions, operators, and keywords (see dialect guide above)
- Follow the query plan steps closely
- Use the provided relationships to construct proper JOIN conditions
- Reference the example queries as a guide for query patterns and style
- Use best practices for readability and performance
- Ensure the query is safe (read-only operations only)
- Add helpful comments if the query is complex
- Return your SQL in the specified JSON format with "dialect": "{dialect}"

CRITICAL: The SQL must be compatible with {dialect} and use its specific syntax.
Do not mix syntax from other SQL dialects. The SQL should be production-ready and optimized.
"""


def format_refiner_prompt(
    question: str,
    selected_schema: dict,
    query_plan: dict,
    dialect: str = "postgres",
) -> str:
    """Format the Refiner agent prompt with query plan.

    Args:
        question: User's natural language question
        selected_schema: Selected tables, columns, relationships, and examples
        query_plan: Query plan from Decomposer
        dialect: Target SQL dialect

    Returns:
        Formatted prompt string

    """
    # Format selected schema
    schema_lines = []
    selected_columns = selected_schema.get("selected_columns", {})
    for table, columns in selected_columns.items():
        schema_lines.append(f"Table: {table}")
        schema_lines.append(f"  Columns: {', '.join(columns)}")
    schema_str = "\n".join(schema_lines) if schema_lines else "No schema selected"

    # Format relationships
    relationships = selected_schema.get("relationships", [])
    if relationships:
        rel_lines = []
        for rel in relationships:
            source = rel.get("from_table", "unknown")
            target = rel.get("to_table", "unknown")

            rel_desc = f"- {source} → {target}"

            # Add column mappings if available
            col_mappings = rel.get("join_columns", [])
            if col_mappings:
                mapping_strs = [f"{m.get('from')} = {m.get('to')}" for m in col_mappings]
                rel_desc += f"\n  Join on: {', '.join(mapping_strs)}"

            rel_lines.append(rel_desc)
        relationships_str = "\n".join(rel_lines)
    else:
        relationships_str = "No explicit relationships provided (single table query or implicit joins)"

    # Format example queries
    example_queries = selected_schema.get("example_queries", [])
    if example_queries:
        example_lines = []
        for i, example in enumerate(example_queries[:3], 1):  # Limit to top 3 examples
            content = example.get("content", "")
            metadata = example.get("metadata", {})
            description = metadata.get("description", "")

            example_lines.append(f"\nExample {i}:")
            if description:
                example_lines.append(f"  Description: {description}")
            example_lines.append(f"  {content[:300]}{'...' if len(content) > 300 else ''}")
        examples_str = "\n".join(example_lines)
    else:
        examples_str = "No example queries available"

    # Format query plan
    plan_lines = []
    plan_lines.append("Steps:")
    for i, step in enumerate(query_plan.get("steps", []), 1):
        plan_lines.append(f"  {i}. {step}")

    if query_plan.get("join_strategy"):
        plan_lines.append(f"\nJoin Strategy: {query_plan['join_strategy']}")

    if query_plan.get("aggregations"):
        plan_lines.append(f"\nAggregations: {', '.join(query_plan['aggregations'])}")

    if query_plan.get("filters"):
        plan_lines.append(f"\nFilters: {', '.join(query_plan['filters'])}")

    if query_plan.get("window_functions"):
        plan_lines.append(f"\nWindow Functions: {', '.join(query_plan['window_functions'])}")

    query_plan_str = "\n".join(plan_lines)

    return REFINER_USER_PROMPT_TEMPLATE.format(
        question=question,
        selected_schema=schema_str,
        relationships=relationships_str,
        example_queries=examples_str,
        query_plan=query_plan_str,
        dialect=dialect,
    )
