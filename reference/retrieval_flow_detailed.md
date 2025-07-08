# Detailed Retrieval Flow Documentation

## System Overview

The retrieval system uses a query router to select between three optimized paths based on query intent. Each path is designed for maximum efficiency and relevance.

## Initial Input Parameters

- `user_query`: Natural language question from user
- `bank_name`: Target bank (e.g., "TD Bank")
- `fiscal_year`: Year of interest (e.g., 2024)
- `quarter`: Quarter of interest (e.g., "Q1")

---

## 🚦 Query Router

### Purpose
Analyzes query intent to select optimal retrieval strategy

### Process
```python
router_prompt = f"""
Query: {user_query}

Analyze this query and decide the best retrieval approach:
1. "similarity_search" - For specific topics, metrics, or details that may appear anywhere
2. "section_retrieval" - For queries targeting specific sections (management discussion, Q&A, etc.)
3. "full_transcript" - For comprehensive analysis, summaries, or comparisons

Examples:
- "What was the net interest margin?" → similarity_search
- "What did management say about credit losses?" → section_retrieval
- "Summarize the entire earnings call" → full_transcript

Return: {approach: "[approach_name]"}
"""
```

### Output
- `retrieval_approach`: Selected path

---

## 📍 Path 1: Similarity Search (Optimized Flow)

### Stage 1: Vector Similarity Search

**Purpose**: Find semantically relevant sections across entire transcript

**Input Fields**:
- `query_embedding`: Vector representation of user query
- `bank_name`, `fiscal_year`, `quarter`: Filters

**SQL Query**:
```sql
SELECT 
    id,
    section_content,
    section_order,
    section_tokens,
    importance_score,
    primary_section_type,
    secondary_section_type,
    secondary_section_summary,
    preceding_context_relevance,
    following_context_relevance
FROM transcript_sections 
WHERE bank_name = ? AND fiscal_year = ? AND quarter = ?
ORDER BY section_embedding <-> query_embedding
LIMIT 20;
```

**Output**: Top 20 most similar sections

---

### Stage 2: LLM Relevance Filtering (Moved Earlier)

**Purpose**: Remove sections that are semantically similar but not actually relevant

**Input Fields**:
- `primary_section_type`: Section category
- `secondary_section_type`: Subsection type
- `secondary_section_summary`: Summary for analysis

**Process**:
```python
section_summaries = []
for i, section in enumerate(similarity_results):
    summary_text = f"""
    Index: {i}
    Primary: {section.primary_section_type}
    Secondary: {section.secondary_section_type}
    Summary: {section.secondary_section_summary}
    """
    section_summaries.append(summary_text)

filtering_prompt = f"""
Query: {user_query}

Which of these sections are NOT relevant to answering the query?
Sections:
{'\n'.join(section_summaries)}

Return: {irrelevant_indices: [list of indices]}
"""

irrelevant_indices = llm.identify_irrelevant(filtering_prompt)
filtered_sections = [s for i, s in enumerate(similarity_results) if i not in irrelevant_indices]
```

**Output**: Only relevant sections remain

---

### Stage 3: Importance-Based Reranking

**Purpose**: Prioritize by importance scores and limit to top 10

**Input Fields**:
- `importance_score`: Section importance (0-1)

**Process**:
```python
# Sort by importance score
reranked_sections = sorted(filtered_sections, key=lambda x: x.importance_score, reverse=True)
# Keep only top 10
top_sections = reranked_sections[:10]
```

**Output**: Top 10 most important relevant sections

---

### Stage 4: Context Enhancement

**Purpose**: Add neighboring sections for better context

**Input Fields**:
- `preceding_context_relevance`: Previous section importance (0-1)
- `following_context_relevance`: Next section importance (0-1)
- `section_order`: Position identifier

**Process**:
```python
enhanced_sections = []
processed_orders = set()

for section in top_sections:
    # Check preceding context
    if section.preceding_context_relevance > 0.7:
        prev_order = section.section_order - 1
        if prev_order not in processed_orders:
            prev_section = fetch_section_by_order(bank_name, fiscal_year, quarter, prev_order)
            if prev_section:
                enhanced_sections.append(prev_section)
                processed_orders.add(prev_order)
    
    # Add current section
    enhanced_sections.append(section)
    processed_orders.add(section.section_order)
    
    # Check following context
    if section.following_context_relevance > 0.7:
        next_order = section.section_order + 1
        if next_order not in processed_orders:
            next_section = fetch_section_by_order(bank_name, fiscal_year, quarter, next_order)
            if next_section:
                enhanced_sections.append(next_section)
                processed_orders.add(next_order)
```

**Output**: Original sections plus important neighbors

---

### Stage 5: Section Ordering

**Purpose**: Restore original transcript sequence

**Input Fields**:
- `section_order`: Original position

**Process**:
```python
ordered_sections = sorted(enhanced_sections, key=lambda x: x.section_order)
```

**Output**: Sections in transcript order

---

### Stage 6: Gap Filling

**Purpose**: Fill small gaps for narrative continuity

**Input Fields**:
- `section_order`: To identify gaps
- `section_tokens`: For token budget

**Process**:
```python
MAX_GAP_SIZE = 5
TOKEN_BUDGET = 4000
current_tokens = sum(s.section_tokens for s in ordered_sections)

final_sections = []
for i in range(len(ordered_sections)):
    final_sections.append(ordered_sections[i])
    
    if i < len(ordered_sections) - 1:
        gap_size = ordered_sections[i + 1].section_order - ordered_sections[i].section_order - 1
        
        if 0 < gap_size < MAX_GAP_SIZE:
            gap_sections = fetch_sections_in_range(
                bank_name, fiscal_year, quarter,
                ordered_sections[i].section_order + 1,
                ordered_sections[i + 1].section_order - 1
            )
            
            gap_tokens = sum(s.section_tokens for s in gap_sections)
            if current_tokens + gap_tokens <= TOKEN_BUDGET:
                final_sections.extend(gap_sections)
                current_tokens += gap_tokens
```

**Output**: Complete section sequences

---

## 📂 Path 2: Section Retrieval

### Stage 1: Get Primary Section Summaries

**Purpose**: Fetch available sections for selection

**SQL Query**:
```sql
SELECT DISTINCT 
    primary_section_type,
    primary_section_summary,
    MIN(section_order) as section_start_order
FROM transcript_sections
WHERE bank_name = ? AND fiscal_year = ? AND quarter = ?
GROUP BY primary_section_type, primary_section_summary
ORDER BY MIN(section_order);
```

**Output**: List of primary sections with summaries

---

### Stage 2: LLM Section Selection

**Purpose**: Choose relevant primary sections

**Process**:
```python
selection_prompt = f"""
Query: {user_query}

Available sections:
{format_section_list(primary_sections)}

Which sections contain relevant information?
Return: {selected_sections: ["Section Name 1", "Section Name 2"]}
"""

selected_sections = llm.select_sections(selection_prompt)
```

**Output**: Selected section names

---

### Stage 3: Retrieve Complete Sections

**Purpose**: Get all content from selected sections

**SQL Query**:
```sql
SELECT 
    section_content,
    section_order,
    section_tokens,
    primary_section_type,
    secondary_section_type
FROM transcript_sections
WHERE bank_name = ? AND fiscal_year = ? AND quarter = ?
  AND primary_section_type IN (?)
ORDER BY section_order;
```

**Output**: All subsections from selected primary sections

---

## 📄 Path 3: Full Transcript

### Single Stage: Retrieve Everything

**Purpose**: Get complete transcript

**SQL Query**:
```sql
SELECT 
    section_content,
    section_order,
    section_tokens,
    primary_section_type,
    secondary_section_type
FROM transcript_sections
WHERE bank_name = ? AND fiscal_year = ? AND quarter = ?
ORDER BY section_order;
```

**Output**: Every section in order

---

## 🔄 Final Synthesis (All Paths)

### Purpose
Generate coherent answer from retrieved sections

### Process
```python
content_blocks = []
for section in final_sections:
    content_blocks.append({
        'text': section.section_content,
        'source': f"{section.primary_section_type} > {section.secondary_section_type}",
        'position': section.section_order
    })

synthesis_prompt = f"""
Query: {user_query}

Using these transcript sections, provide a comprehensive answer.
Cite specific sections when making claims.

Sections:
{format_content_blocks(content_blocks)}

Instructions:
1. Answer directly and comprehensively
2. Use specific quotes and data
3. Cite sources using section names
4. Maintain narrative flow
"""

final_response = llm.synthesize(synthesis_prompt)
```

### Output Format
```json
{
    "answer": "Based on the transcript...",
    "sources": ["Management Discussion > Net Interest Income", ...],
    "confidence": 0.92,
    "sections_used": 12,
    "total_tokens": 3500
}
```

## Performance Optimizations

1. **Early Filtering**: LLM relevance check before expensive reranking
2. **Limited Reranking**: Only process top 10 after filtering
3. **Smart Context**: Only pull neighbors with high relevance scores
4. **Token Budget**: Prevent context overflow in gap filling
5. **Path Selection**: Router ensures most efficient path for each query