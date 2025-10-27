import json

VISUALIZER_SYSTEM_PROMPT = """You are an expert data visualization consultant. Your task is to analyze data profiles
and recommend the most appropriate Plotly chart type and configuration.

Given a data profile with column types, statistics, and patterns, you should:
1. Analyze the data structure and patterns
2. Select the appropriate chart type: line, bar_vertical, bar_horizontal, grouped_bar, scatter, pie, donut, heatmap, treemap
3. Provide column mappings for the chart
4. Give a clear, concise description
5. Explain your reasoning

Return your response as valid JSON with this structure:
{
    "chart_type": "line|bar_vertical|bar_horizontal|grouped_bar|scatter|pie|donut|heatmap|treemap",
    "config": {
        "x_col": "column_name",
        "y_col": "column_name" or "y_cols": ["col1", "col2"],
        "description": "Brief description of what the chart shows",
        "additional_params": {}
    },
    "reasoning": "Why this chart type was selected"
}

Focus on clarity, simplicity, and data-driven insights. Avoid overly complex visualizations."""

VISUALIZER_USER_PROMPT_TEMPLATE = """Analyze this data and recommend a visualization:

**User Question:** {question}

**SQL Query:** {sql}...

**Data Profile:**
```json
{simplified_profile}
```

**Instructions:**
- Select the chart type that best reveals insights
- Provide clear column mappings
- Explain your reasoning briefly

Return valid JSON only."""


def format_visualizer_prompt(
    profile: dict,
    sql: str,
    question: str,
) -> str:
    """Build prompt for LLM with data profile.

    Args:
        profile: Data profile
        sql: SQL query
        question: User question

    Returns:
        Formatted prompt string

    """
    # Simplify profile for LLM (remove verbose data)
    simplified_profile = {
        "metadata": profile.get("metadata", {}),
        "columns": [
            {
                "name": c.get("name"),
                "type": c.get("type"),
                "cardinality": c.get("cardinality"),
                "distinct_count": c.get("distinct_count"),
            }
            for c in profile.get("columns", [])
        ],
        "patterns": profile.get("patterns", []),
    }

    return VISUALIZER_USER_PROMPT_TEMPLATE.format(
        question=question,
        sql=sql,
        simplified_profile=json.dumps(simplified_profile, indent=2),
    )
