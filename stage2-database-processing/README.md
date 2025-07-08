# Stage 2: Database Processing Pipeline

## Overview

Stage 2 transforms PDF transcript files from Stage 1 into a searchable PostgreSQL database with intelligent chunking, embeddings, and a multi-path retrieval system.

## System Architecture

### Input
- PDF files from Stage 1: `/database_refresh/YYYY/QX/transcript.pdf`
- Files contain Canadian and US bank earnings call transcripts

### Output
- PostgreSQL database with hierarchical sections
- Vector embeddings for similarity search
- Three retrieval paths optimized for different query types

## Database Schema

### Core Table: `transcript_sections`

```sql
-- File metadata
fiscal_year         INTEGER      -- Year: 2020, 2021, etc.
quarter            VARCHAR(10)   -- Quarter: Q1, Q2, Q3, Q4
bank_name          VARCHAR(255)  -- Bank name: "TD Bank", "JP Morgan"
ticker_region      VARCHAR(100)  -- Combined ticker-region: "TD-CA", "JPM-US"
filepath           TEXT          -- Full NAS path to source PDF
filename           VARCHAR(255)  -- Original PDF filename
date_last_modified TIMESTAMP     -- File modification timestamp

-- Hierarchical classification
primary_section_type      VARCHAR(100)  -- Main sections (6 types)
primary_section_summary   TEXT         -- Summary of entire primary section
secondary_section_type    VARCHAR(255) -- Context-specific subsections
secondary_section_summary TEXT         -- Summary of specific chunk

-- Content and search data
section_content    TEXT          -- Full text of the secondary section
section_order      INTEGER       -- Sequential order across transcript
section_tokens     INTEGER       -- Token count for budget management
section_embedding  vector(2000)  -- OpenAI text-embedding-3-large

-- Reranking scores (0.0 to 1.0)
importance_score              DECIMAL(3,2)  -- Importance within primary section
preceding_context_relevance   DECIMAL(3,2)  -- Importance of previous section
following_context_relevance   DECIMAL(3,2)  -- Importance of next section
```

### Primary Section Types
1. Safe Harbor Statement
2. Introduction
3. Management Discussion
4. Financial Performance
5. Investor Q&A
6. Closing Remarks

## Processing Pipeline

### Phase 1: Section Identification
1. **Primary Section Detection**
   - Send transcript pages to LLM sequentially
   - LLM identifies line numbers where main sections change
   - Creates 6 standard primary sections

2. **Secondary Section Breakdown**
   - Send each primary section to LLM
   - LLM creates contextual subsections (e.g., "Capital Markets Performance")
   - These subsections become our searchable chunks

### Phase 2: Content Enhancement
1. **Summary Generation**
   - Generate summaries for primary sections
   - Generate summaries for each secondary section
   
2. **Embedding Creation**
   - Create 2000-dimension embeddings using text-embedding-3-large
   - Store for similarity search

3. **Score Calculation**
   - Calculate importance scores (0-1) for each section
   - Calculate context relevance scores for neighbors

## Retrieval System

### Query Router
Analyzes user queries and selects optimal retrieval path:
- **Similarity Search**: For specific topics/metrics
- **Section Retrieval**: For section-specific queries
- **Full Transcript**: For comprehensive analysis

### Path 1: Similarity Search (Updated Flow)

1. **Vector Similarity Search**
   - Find top 20 most similar sections using embeddings
   
2. **LLM Relevance Filtering** (NEW POSITION)
   - Send section summaries to LLM
   - Remove irrelevant sections despite semantic similarity
   
3. **Importance-Based Reranking**
   - Reorder remaining sections by importance score
   - Keep only top 10 sections
   
4. **Context Enhancement**
   - Check preceding/following context relevance scores
   - Pull in neighboring sections where relevance > 0.7
   
5. **Section Ordering**
   - Restore original transcript order
   
6. **Gap Filling**
   - Fill gaps < 5 sections if under token budget
   
7. **Synthesis**
   - Generate final answer with source attribution

### Path 2: Section Retrieval

1. **Get Primary Summaries**
   - Fetch all primary section summaries
   
2. **LLM Section Selection**
   - LLM chooses relevant primary sections
   
3. **Retrieve Full Sections**
   - Get all content from selected sections
   
4. **Synthesis**
   - Generate answer from complete sections

### Path 3: Full Transcript

1. **Retrieve Everything**
   - Get all sections in order
   
2. **Synthesis**
   - Generate comprehensive answer

## Key Innovations

1. **No Separate Chunking**: Secondary sections ARE the chunks
2. **Three-Path System**: Optimized for different query types
3. **Smart Filtering**: LLM relevance check before reranking
4. **Context Awareness**: Automatic neighbor inclusion
5. **Gap Filling**: Maintains narrative continuity

## Implementation Status

- ✅ Database schema designed
- ✅ Retrieval flow architected
- 🔄 PDF processing implementation pending
- 🔄 LLM integration pending
- 🔄 API development pending

## Next Steps

1. Implement PDF text extraction
2. Build LLM section identification pipeline
3. Create embedding generation system
4. Develop retrieval API
5. Build synthesis layer

## Configuration

Update NAS credentials and paths in `config.py` before running.