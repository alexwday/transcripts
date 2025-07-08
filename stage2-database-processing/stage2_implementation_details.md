# Stage 2 Implementation Details

## Code Structure Requirements (Based on Stage 1 Pattern)

### 1. Standalone Script Design
- Single Python file that can run in Jupyter notebook or command line
- All configuration at the top of the script
- Clear section separators with comments
- Comprehensive error handling with logging

### 2. Configuration Pattern
```python
# ========================================
# CONFIGURATION - MODIFY AS NEEDED
# ========================================

# NAS Configuration (reuse from Stage 1)
NAS_USERNAME = "your_username"
NAS_PASSWORD = "your_password"
DEST_NAS_IP = "192.168.2.100"
DEST_NAS_PORT = 445

# Database Configuration
DATABASE_HOST = "localhost"
DATABASE_PORT = 5432
DATABASE_NAME = "transcripts"
DATABASE_USER = "postgres"
DATABASE_PASSWORD = "postgres_password"

# OpenAI Configuration
OPENAI_API_KEY = "sk-..."
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 2000
COMPLETION_MODEL = "gpt-4-turbo"

# Processing Configuration
INPUT_PATH = "Finance Data and Analytics/DSA/AEGIS/Transcripts/database_refresh"
MASTER_DB_PATH = "Finance Data and Analytics/DSA/AEGIS/Transcripts/master_database.csv"
BATCH_SIZE = 10  # Number of files to process at once
TOKEN_LIMITS = {
    "primary_section": 2000,
    "secondary_section": 500,
    "summary": 150
}
```

### 3. NAS Connection Reuse
- Use same SMBConnection pattern from Stage 1
- Same connection function with NTLM v2 and direct TCP
- Same error handling and retry logic

### 4. Key Dependencies
```python
# Core imports (same pattern as Stage 1)
import os
import sys
import time
import logging
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import hashlib
from io import BytesIO

# Additional Stage 2 imports
import pandas as pd
import numpy as np
import json
import re
from dataclasses import dataclass

# External libraries with error handling
try:
    from smb.SMBConnection import SMBConnection
    from smb.smb_structs import OperationFailure
except ImportError:
    print("ERROR: pysmb not installed. Please run: pip install pysmb")
    sys.exit(1)

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from pgvector.psycopg2 import register_vector
except ImportError:
    print("ERROR: PostgreSQL libraries not installed. Please run: pip install psycopg2-binary pgvector")
    sys.exit(1)

try:
    import PyPDF2
    from PyPDF2 import PdfReader
except ImportError:
    print("ERROR: PyPDF2 not installed. Please run: pip install PyPDF2")
    sys.exit(1)

try:
    import openai
except ImportError:
    print("ERROR: OpenAI not installed. Please run: pip install openai")
    sys.exit(1)
```

### 5. Main Processing Functions

#### PDF Processing
```python
def extract_pdf_text(pdf_content: BytesIO) -> List[Dict[str, any]]:
    """Extract text from PDF with page numbers and line tracking"""
    # Returns list of pages with line numbers
    
def clean_transcript_text(raw_text: str) -> str:
    """Clean and normalize transcript text"""
    # Remove headers, footers, normalize whitespace
```

#### LLM Integration
```python
def identify_primary_sections(pages: List[Dict]) -> List[Dict]:
    """Use LLM to identify primary section boundaries"""
    # Returns section boundaries with line numbers
    
def create_secondary_sections(primary_section: Dict) -> List[Dict]:
    """Use LLM to break primary section into secondary sections"""
    # Returns list of secondary sections
    
def generate_summaries(section: Dict) -> Dict:
    """Generate primary and secondary summaries"""
    # Returns summaries with token counts
    
def calculate_importance_scores(sections: List[Dict]) -> List[Dict]:
    """Calculate importance and context relevance scores"""
    # Returns sections with scores
```

#### Database Operations
```python
def connect_to_database() -> psycopg2.connection:
    """Create PostgreSQL connection with pgvector"""
    # Returns database connection
    
def insert_sections(conn: psycopg2.connection, sections: List[Dict]):
    """Batch insert sections into database"""
    # Handles transactions and errors
    
def update_master_database(file_info: Dict, status: str):
    """Update CSV master database on NAS"""
    # Track processing status
```

### 6. Error Handling Pattern
- Comprehensive try/except blocks
- Detailed error logging
- Write error logs to NAS (same as Stage 1)
- Graceful degradation on failures
- Transaction rollback on database errors

### 7. Processing Flow
```python
def process_transcript(nas_conn: SMBConnection, db_conn: psycopg2.connection, 
                      file_path: str, file_info: Dict) -> bool:
    """Process single transcript file"""
    # 1. Download PDF from NAS
    # 2. Extract text with page/line numbers
    # 3. Identify primary sections
    # 4. Create secondary sections
    # 5. Generate summaries and embeddings
    # 6. Calculate scores
    # 7. Insert into database
    # 8. Update master database
    # Returns success/failure

def main():
    """Main execution function"""
    # 1. Connect to NAS
    # 2. Connect to database
    # 3. Read master database
    # 4. Scan for new/modified files
    # 5. Process files in batches
    # 6. Generate summary report
    # 7. Write logs to NAS
```

## Key Differences from Original Plan

1. **Single File**: All code in one file instead of multiple modules
2. **NAS Integration**: Read PDFs directly from NAS, not local files
3. **Master Database**: CSV file on NAS, not local
4. **Error Logs**: Written to NAS, not local filesystem
5. **Configuration**: All settings at top, no separate config file

## LLM Prompt Templates

### Primary Section Detection
```python
PRIMARY_SECTION_PROMPT = """
Analyze this earnings call transcript page and identify where primary sections change.

Primary sections are:
1. Safe Harbor Statement
2. Introduction
3. Management Discussion
4. Financial Performance
5. Investor Q&A
6. Closing Remarks

Current page content with line numbers:
{page_content}

If this page contains a section transition, return:
{"transition_found": true, "line_number": X, "from_section": "name", "to_section": "name"}

If no transition, return:
{"transition_found": false}
"""
```

### Secondary Section Creation
```python
SECONDARY_SECTION_PROMPT = """
Break down this {primary_section_type} section into logical subsections.

Section content:
{section_content}

Create subsections that represent distinct topics or themes. For each subsection provide:
1. A descriptive name (e.g., "Capital Markets Performance", "Credit Risk Update")
2. Start and end line numbers
3. Brief rationale for the division

Return as JSON list:
[{"name": "...", "start_line": X, "end_line": Y, "rationale": "..."}]
"""
```

### Summary Generation
```python
SUMMARY_PROMPT = """
Generate a concise summary of this {section_type} section.

Content:
{content}

Create a {token_limit}-token summary that captures key points, metrics, and insights.
Focus on factual information and specific details mentioned.
"""
```

### Importance Scoring
```python
IMPORTANCE_SCORE_PROMPT = """
Rate the importance of this section within the context of {primary_section_type}.

Section: {secondary_section_type}
Summary: {summary}

Provide scores (0.0-1.0) for:
1. importance_score: How critical is this information?
2. preceding_context_relevance: How important is the previous section for understanding this?
3. following_context_relevance: How important is the next section for understanding this?

Consider financial materiality, strategic importance, and investor relevance.

Return: {"importance": X, "preceding": Y, "following": Z}
"""
```

## Complete Dependency List
```bash
# requirements.txt
pysmb==1.2.9
psycopg2-binary==2.9.9
pgvector==0.2.3
PyPDF2==3.0.1
openai==1.12.0
pandas==2.1.4
numpy==1.26.2
python-dateutil==2.8.2
```

## Next Steps for Implementation

1. Create database with schema from `final_schema.sql`
2. Set up PostgreSQL with pgvector extension
3. Create single Python script following Stage 1 pattern
4. Test with sample transcript
5. Deploy and schedule with cron like Stage 1