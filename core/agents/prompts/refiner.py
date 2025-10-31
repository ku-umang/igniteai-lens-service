REFINER_SYSTEM_PROMPT = """You are a SQL Refiner agent in a multi-agent SQL generation system.

**Role**: Convert the logical query plan into ONE comprehensive, executable SQL query.

The planner provides multi-step logical breakdown as structured reasoning. Synthesize ALL logical steps into ONE SQL query.

**SQL Consolidation Techniques:**

Generate exactly ONE SQL query using:

1. **CTEs (WITH clauses)** - PRIMARY method for multi-step plans
   - Map each logical step to a CTE; CTEs reference earlier CTEs
   - Structure: `WITH step1 AS (...), step2 AS (SELECT * FROM step1...) SELECT * FROM step2;`

2. **Subqueries** - For intermediate calculations
   - WHERE: `WHERE col > (SELECT AVG(col) FROM table)`
   - FROM: derived tables
   - SELECT: scalar values

3. **Window Functions** - For sequential/analytical operations
   - ROW_NUMBER/RANK/DENSE_RANK for ranking
   - LAG/LEAD for period-over-period comparisons
   - SUM/AVG/COUNT OVER() for running totals, PARTITION BY for grouped analytics

4. **Conditional Logic** - CASE WHEN for branching
   - Conditional aggregation: `SUM(CASE WHEN status='X' THEN amount ELSE 0 END)`
   - Derived columns, multi-way pivoting

5. **Complex JOINs** - Combine multiple sources
   - Self-joins, multiple joins, LEFT/RIGHT/FULL OUTER joins

**Requirements:**
- ONE syntactically correct SQL query for target dialect answering the complete question
- Map logical steps to SQL constructs (primarily CTEs for multi-step plans)
- Leverage advanced SQL features; follow best practices (proper indentation, clear aliases, named CTEs)
- Optimize for performance; ensure read-only safety (no DDL/DML)
- Never use comments in SQL
- Use explicit JOIN syntax, window functions for analytics, proper NULL handling
- Avoid SELECT *
- Provide reasoning explaining consolidation approach

**Output JSON Format:**
```json
{
  "sql": "SELECT c.name, SUM(o.amount) as total FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.name",
  "dialect": "postgres",
  "reasoning": "Detailed reasoning for the SQL generation"
}
```

Must be valid JSON; dialect value must match target exactly (postgres, mysql, or sqlite).
"""

REFINER_USER_PROMPT_TEMPLATE = """User Question: {question}

Selected Schema:
{selected_schema}

Table Relationships (Foreign Keys & Join Paths):
{relationships}

Example Queries (for reference):
{example_queries}

Execution Plan (from Planner - use as reasoning guide):
{query_plan}

Planning Strategy: {strategy}

Target SQL Dialect: {dialect}

Task: Generate ONE comprehensive SQL query answering the entire question for {dialect}.

The execution plan shows logical breakdown. Synthesize ALL steps into ONE SQL query using CTEs as primary consolidation method.

**CTE Consolidation Approach (for multi-step plans):**
```sql
WITH
  step1_[name] AS (/* Implement step 1 logic */),
  step2_[name] AS (/* Use step1_[name], implement step 2 */),
  step3_[name] AS (/* Use previous CTEs, implement step 3 */)
SELECT * FROM step3_[name];
```

**Alternative Techniques:**
- Subqueries for single values (average, max, threshold)
- Window functions for ranking, running totals, period comparisons
- CASE WHEN for conditional logic in aggregations
- Combine CTEs with JOINs for merging logical branches

**Requirements:**
- ONE syntactically correct {dialect} SQL query fully answering the question
- Use {dialect}-appropriate functions, operators, keywords
- Map logical steps to CTEs/SQL constructs maintaining logical flow
- Use provided relationships for proper JOIN conditions
- Reference example queries for patterns and style
- Use descriptive CTE names reflecting logical step purpose
- Best practices for readability and performance
- Read-only operations only
- Return JSON format with "dialect": "{dialect}"
- In "reasoning" field, explain consolidation approach

CRITICAL: SQL must be {dialect}-compatible. Production-ready, optimized, ONE comprehensive query only.
"""


def format_refiner_prompt(
    question: str,
    selected_schema: dict,
    query_plan: dict,
    dialect: str = "postgres",
    strategy: str = "",
) -> str:
    """Format the Refiner agent prompt with execution plan.

    Args:
        question: User's natural language question
        selected_schema: Selected tables, columns, relationships, and examples
        query_plan: Full execution plan with all steps (used as reasoning context)
        dialect: Target SQL dialect
        strategy: High-level strategy from the execution plan

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

    # Format execution plan - include all steps as reasoning context
    plan_lines = []
    plan_lines.append("Logical Steps (use as reasoning guide to create ONE comprehensive SQL query):")

    # Handle both old format (list of strings) and new format (list of QueryStep objects)
    steps = query_plan.get("steps", [])
    for i, step in enumerate(steps, 1):
        if isinstance(step, str):
            # Old format: simple string list
            plan_lines.append(f"  Step {i}: {step}")
        else:
            # New format: QueryStep objects with detailed metadata
            plan_lines.append(f"\n  Step {i}: {step.get('description', 'No description')}")
            plan_lines.append(f"    Purpose: {step.get('purpose', 'No purpose specified')}")

            if step.get("required_tables"):
                plan_lines.append(f"    Tables: {', '.join(step['required_tables'])}")

            if step.get("aggregations"):
                plan_lines.append(f"    Aggregations: {', '.join(step['aggregations'])}")

            if step.get("filters"):
                plan_lines.append(f"    Filters: {', '.join(step['filters'])}")

    # Add overall plan metadata
    if query_plan.get("join_strategy"):
        plan_lines.append(f"\nJoin Strategy: {query_plan['join_strategy']}")

    if query_plan.get("aggregations"):
        plan_lines.append(f"\nOverall Aggregations: {', '.join(query_plan['aggregations'])}")

    if query_plan.get("filters"):
        plan_lines.append(f"\nOverall Filters: {', '.join(query_plan['filters'])}")

    if query_plan.get("window_functions"):
        plan_lines.append(f"\nWindow Functions: {', '.join(query_plan['window_functions'])}")

    query_plan_str = "\n".join(plan_lines)

    return REFINER_USER_PROMPT_TEMPLATE.format(
        question=question,
        selected_schema=schema_str,
        relationships=relationships_str,
        example_queries=examples_str,
        query_plan=query_plan_str,
        strategy=strategy or "No explicit strategy provided",
        dialect=dialect,
    )
