SELECTOR_SYSTEM_PROMPT = """You are a Schema Selector agent, part of a multi-agent SQL generation system.

Your role is to analyze the user's natural language question and select the MINIMAL set of tables
and columns needed to answer that question from the provided schema metadata.

Guidelines:
1. Be selective - only include tables and columns that are directly relevant
2. Include related tables if joins are needed to answer the question
3. Pay attention to foreign key relationships to identify join paths
4. Consider example queries as hints for common query patterns
5. Output your reasoning clearly so downstream agents can use your selection

Output Json Format:
```json
{
  "selected_tables": [list of table qualified names],
  "selected_columns": {
    "table_name": [list of column names]
  },
  "relationships": [
    {
      "from_table": "table1",
      "to_table": "table2",
      "join_columns": [{"from": "col1", "to": "col2"}]
    }
  ],
  "reasoning": "Your detailed reasoning for the selection"
}
```
"""

SELECTOR_USER_PROMPT_TEMPLATE = """User Question: {question}

Available Schema (from retrieval):

Tables:
{tables_info}

Columns:
{columns_info}

Relationships (Foreign Keys):
{relationships_info}


Task: Select the minimal set of tables and columns needed to answer the user's question.
Provide your selection in the specified JSON format with clear reasoning.
"""


def format_selector_prompt(question: str, tables: list, columns: list, relationships: list) -> str:
    """Format the Selector agent prompt with retrieved schema.

    Args:
        question: User's natural language question
        tables: List of table metadata from retrieval
        columns: List of column metadata from retrieval
        relationships: List of FK relationships

    Returns:
        Formatted prompt string

    """
    # Format tables
    tables_info = []
    for table in tables:
        tables_info.append(f"- {table['table_qualified_name']}: {table['content']}")
    tables_str = "\n".join(tables_info) if tables_info else "No tables retrieved"

    # Format columns
    columns_info = []
    for col in columns:
        col_meta = col.get("metadata", {})
        columns_info.append(
            f"- {col['table_qualified_name']}.{col_meta.get('column_name', 'unknown')}: "
            f"{col_meta.get('data_type', 'unknown')} - sample_values: {col_meta.get('sample_values', [])}"
        )
    columns_str = "\n".join(columns_info) if columns_info else "No columns retrieved"

    # Format relationships
    relationships_info = []
    for rel in relationships:
        relationships_info.append(
            f"- {rel.get('source_table', 'unknown')} -> {rel.get('target_table', 'unknown')}: {rel.get('join_hint', 'No hint')}"
        )
    relationships_str = "\n".join(relationships_info) if relationships_info else "No relationships found"

    return SELECTOR_USER_PROMPT_TEMPLATE.format(
        question=question,
        tables_info=tables_str,
        columns_info=columns_str,
        relationships_info=relationships_str,
    )
