#!/usr/bin/env python3
"""
Bank Earnings Call Transcript Processor - Stage 2 (Updated)
Processes PDF transcripts from NAS into CSV master database with embeddings

DEBUG MODE USAGE:
To enable debug mode for testing with detailed logging and intermediate file saving:

1. Set DEBUG_MODE = True in the configuration section
2. Optionally set DEBUG_TARGET_FILE to target a specific file (e.g., "2024/Q1/TD_Bank_Q1_2024.pdf")
3. Configure other debug options:
   - DEBUG_SAVE_INTERMEDIATES: Save all intermediate outputs to /tmp/transcript_debug/
   - DEBUG_SKIP_EMBEDDINGS: Skip embedding generation for faster testing
   - DEBUG_MAX_PAGES: Limit number of pages processed (None = all pages)

Debug mode will:
- Process only one file (target file or first available)
- Save detailed logs and intermediate outputs
- Provide step-by-step processing details
- Enable DEBUG-level logging

Example debug files saved:
- 01_extracted_pages.json - PDF extraction results
- 02_first_1000_words.txt - Bank identification input
- 03_page_XX_prompt.txt - Primary section identification prompts
- 03_page_XX_response.json - Primary section identification responses
- And many more intermediate files for analysis
"""

import os
import sys
import time
import logging
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import hashlib
from io import BytesIO
import json
import re
from dataclasses import dataclass, asdict
import csv
import tempfile
import requests

# Core libraries
import pandas as pd
import numpy as np

# pysmb imports
try:
    from smb.SMBConnection import SMBConnection
    from smb.smb_structs import OperationFailure
except ImportError:
    print("ERROR: pysmb not installed. Please run: pip install pysmb")
    sys.exit(1)

# PDF processing
try:
    import PyPDF2
    from PyPDF2 import PdfReader
except ImportError:
    print("ERROR: PyPDF2 not installed. Please run: pip install PyPDF2")
    sys.exit(1)

# OpenAI imports
try:
    import openai
    from openai import OpenAI
except ImportError:
    print("ERROR: OpenAI not installed. Please run: pip install openai")
    sys.exit(1)

# Token counting
try:
    import tiktoken
except ImportError:
    print("WARNING: tiktoken not installed. Using fallback token counting. Run: pip install tiktoken")
    tiktoken = None

# ========================================
# CONFIGURATION - MODIFY AS NEEDED
# ========================================

# NAS Authentication (reused from Stage 1)
NAS_USERNAME = "your_username"
NAS_PASSWORD = "your_password"
CLIENT_MACHINE_NAME = "PYTHON_SCRIPT"
SERVER_MACHINE_NAME = "NAS_SERVER"

# Destination NAS Configuration
DEST_NAS_IP = "192.168.2.100"
DEST_NAS_PORT = 445
DEST_CONFIG = {
    "share": "wrkgrp33",
    "base_path": "Finance Data and Analytics/DSA/AEGIS/Transcripts/database_refresh",
    "master_db_path": "Finance Data and Analytics/DSA/AEGIS/Transcripts/master_database.csv",
    "logs_path": "Finance Data and Analytics/DSA/AEGIS/Transcripts/logs"
}

# OAuth Configuration for OpenAI
OAUTH_CONFIG = {
    "auth_endpoint": "https://your-auth-endpoint.com/oauth/token",
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "base_url": "https://your-openai-proxy.com/v1"
}

# OpenAI Configuration
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 2000
COMPLETION_MODEL = "gpt-4-turbo"
MAX_RETRIES = 3
RETRY_DELAY = 2

# Processing Configuration
BATCH_SIZE = 5  # Number of files to process in one run
VALID_YEAR_RANGE = (2020, 2031)
VALID_QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
FILE_EXTENSIONS = [".pdf", ".PDF"]

# Debug Configuration
DEBUG_MODE = False  # Set to True for detailed single-file testing
DEBUG_TARGET_FILE = None  # Specific file to debug, e.g. "2024/Q1/TD_Bank_Q1_2024.pdf"
DEBUG_SAVE_INTERMEDIATES = False  # Save intermediate outputs to files (can be verbose)
DEBUG_SKIP_EMBEDDINGS = False  # Skip embedding generation in debug mode
DEBUG_MAX_PAGES = None  # Limit pages processed in debug mode (None = all pages)

# QUICK DEBUG ACTIVATION - Uncomment these lines to enable debug mode:
# DEBUG_MODE = True
# DEBUG_TARGET_FILE = "2024/Q1/TD_Bank_Q1_2024.pdf"  # Optional: target specific file
# DEBUG_SKIP_EMBEDDINGS = True  # Optional: skip embeddings for faster testing
# DEBUG_MAX_PAGES = 5  # Optional: limit to first 5 pages
# DEBUG_SAVE_INTERMEDIATES = True  # Optional: save all intermediate files

# Token limits for different section types
TOKEN_LIMITS = {
    "primary_summary": 200,
    "secondary_summary": 150,
    "section_chunk": 800
}

# Primary sections to identify
PRIMARY_SECTIONS = [
    "Safe Harbor Statement",
    "Introduction",
    "Management Discussion",
    "Financial Performance",
    "Investor Q&A",
    "Closing Remarks"
]

# Standardized bank mappings
BANK_MAPPINGS = {
    # Canadian Banks
    "TD Bank": {"name": "TD Bank", "ticker_region": "TD:CAN"},
    "Toronto-Dominion Bank": {"name": "TD Bank", "ticker_region": "TD:CAN"},
    "Royal Bank of Canada": {"name": "RBC", "ticker_region": "RY:CAN"},
    "RBC": {"name": "RBC", "ticker_region": "RY:CAN"},
    "Bank of Montreal": {"name": "BMO", "ticker_region": "BMO:CAN"},
    "BMO": {"name": "BMO", "ticker_region": "BMO:CAN"},
    "Scotiabank": {"name": "Scotia", "ticker_region": "BNS:CAN"},
    "Bank of Nova Scotia": {"name": "Scotia", "ticker_region": "BNS:CAN"},
    "Canadian Imperial Bank of Commerce": {"name": "CIBC", "ticker_region": "CM:CAN"},
    "CIBC": {"name": "CIBC", "ticker_region": "CM:CAN"},
    "National Bank of Canada": {"name": "National Bank", "ticker_region": "NA:CAN"},
    
    # US Banks
    "JPMorgan Chase": {"name": "JPMorgan", "ticker_region": "JPM:US"},
    "JP Morgan": {"name": "JPMorgan", "ticker_region": "JPM:US"},
    "Bank of America": {"name": "Bank of America", "ticker_region": "BAC:US"},
    "Wells Fargo": {"name": "Wells Fargo", "ticker_region": "WFC:US"},
    "Citigroup": {"name": "Citigroup", "ticker_region": "C:US"},
    "Goldman Sachs": {"name": "Goldman Sachs", "ticker_region": "GS:US"},
    "Morgan Stanley": {"name": "Morgan Stanley", "ticker_region": "MS:US"},
    "U.S. Bancorp": {"name": "US Bank", "ticker_region": "USB:US"},
    "PNC Financial": {"name": "PNC", "ticker_region": "PNC:US"},
    "Truist Financial": {"name": "Truist", "ticker_region": "TFC:US"},
}

# ========================================
# LOGGING SETUP
# ========================================

# Set logging level based on debug mode
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress SMB logging which is very verbose and includes expected errors
logging.getLogger('nmb.NetBIOS').setLevel(logging.ERROR)
logging.getLogger('smb.SMBConnection').setLevel(logging.ERROR)
logging.getLogger('smb.smb_structs').setLevel(logging.ERROR)

# Debug output directory
DEBUG_OUTPUT_DIR = "/tmp/transcript_debug" if DEBUG_MODE else None
if DEBUG_MODE and DEBUG_SAVE_INTERMEDIATES:
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Debug mode enabled. Intermediate files will be saved to: {DEBUG_OUTPUT_DIR}")

# ========================================
# DEBUG HELPER FUNCTIONS
# ========================================

def create_summary_table(title: str, data: List[Dict], columns: List[str]) -> str:
    """Create a formatted table for debug output"""
    if not data:
        return f"\n{title}\n{'='*len(title)}\nNo data available\n"
    
    # Calculate column widths
    col_widths = {}
    for col in columns:
        col_widths[col] = max(len(col), max(len(str(row.get(col, ''))) for row in data))
    
    # Create table
    table = f"\n{title}\n{'='*len(title)}\n"
    
    # Header
    header = " | ".join(col.ljust(col_widths[col]) for col in columns)
    table += header + "\n"
    table += "-" * len(header) + "\n"
    
    # Rows
    for row in data:
        row_str = " | ".join(str(row.get(col, '')).ljust(col_widths[col]) for col in columns)
        table += row_str + "\n"
    
    return table

def create_processing_summary(stage: str, stats: Dict) -> str:
    """Create a processing stage summary"""
    summary = f"\n{'='*60}\n"
    summary += f"STAGE SUMMARY: {stage}\n"
    summary += f"{'='*60}\n"
    
    for key, value in stats.items():
        summary += f"{key}: {value}\n"
    
    summary += f"{'='*60}\n"
    return summary

def debug_save_to_file(content: str, filename: str, description: str = ""):
    """Save content to debug file if in debug mode"""
    if not DEBUG_MODE or not DEBUG_SAVE_INTERMEDIATES:
        return
    
    try:
        filepath = os.path.join(DEBUG_OUTPUT_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.debug(f"DEBUG: Saved {description} to {filepath}")
    except Exception as e:
        logger.warning(f"DEBUG: Failed to save {filename}: {str(e)}")

def debug_save_json(data: Any, filename: str, description: str = ""):
    """Save JSON data to debug file if in debug mode"""
    if not DEBUG_MODE or not DEBUG_SAVE_INTERMEDIATES:
        return
    
    try:
        filepath = os.path.join(DEBUG_OUTPUT_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug(f"DEBUG: Saved {description} to {filepath}")
    except Exception as e:
        logger.warning(f"DEBUG: Failed to save {filename}: {str(e)}")

def debug_log_step(step_name: str, details: str = ""):
    """Log debug step with detailed information"""
    if DEBUG_MODE:
        separator = "=" * 60
        logger.debug(f"\n{separator}")
        logger.debug(f"DEBUG STEP: {step_name}")
        if details:
            logger.debug(f"DETAILS: {details}")
        logger.debug(f"{separator}")

def debug_log_llm_call(prompt_type: str, prompt: str, response: str = None):
    """Log LLM call details in debug mode"""
    if not DEBUG_MODE:
        return
    
    timestamp = datetime.now().strftime("%H%M%S")
    
    # Only save files if intermediates are enabled
    if DEBUG_SAVE_INTERMEDIATES:
        # Save prompt
        debug_save_to_file(
            prompt, 
            f"{timestamp}_{prompt_type}_prompt.txt",
            f"{prompt_type} prompt"
        )
        
        # Save response if provided
        if response:
            debug_save_to_file(
                response,
                f"{timestamp}_{prompt_type}_response.txt", 
                f"{prompt_type} response"
            )
    
    # Only log essential info
    logger.debug(f"LLM {prompt_type}: {len(prompt)} chars → {len(response) if response else 'pending'} chars")

# ========================================
# DATA CLASSES
# ========================================

@dataclass
class TranscriptSection:
    """Represents a section in the transcript"""
    fiscal_year: int
    quarter: str
    bank_name: str
    ticker_region: str
    filepath: str
    filename: str
    date_last_modified: datetime
    primary_section_type: str
    primary_section_summary: str
    secondary_section_type: str
    secondary_section_summary: str
    section_content: str
    section_order: int
    section_tokens: int
    section_embedding: Optional[List[float]] = None
    importance_score: Optional[float] = None
    preceding_context_relevance: Optional[float] = None
    following_context_relevance: Optional[float] = None

@dataclass
class ProcessingStatus:
    """Track file processing status"""
    filepath: str
    status: str  # 'new', 'modified', 'processed', 'error'
    last_processed: Optional[datetime] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    section_count: Optional[int] = None
    primary_sections: Optional[List[str]] = None
    total_tokens: Optional[int] = None
    sections: Optional[List[Dict]] = None  # Section records for master database

@dataclass
class BankInfo:
    """Bank identification information"""
    detected_name: str
    standardized_name: str
    ticker_region: str
    confidence: float
    recognized: bool

# ========================================
# HELPER FUNCTIONS - NAS OPERATIONS
# ========================================

def create_smb_connection(server_ip: str, username: str, password: str, port: int = 445) -> SMBConnection:
    """Create and return an SMB connection"""
    try:
        conn = SMBConnection(username, password, CLIENT_MACHINE_NAME, SERVER_MACHINE_NAME, 
                           use_ntlm_v2=True, is_direct_tcp=True)
        connected = conn.connect(server_ip, port=port)
        if connected:
            logger.info(f"Successfully connected to {server_ip}:{port}")
            return conn
        else:
            raise Exception(f"Failed to connect to {server_ip}:{port}")
    except Exception as e:
        logger.error(f"Connection error to {server_ip}:{port}: {str(e)}")
        raise

def download_file_from_nas(conn: SMBConnection, share: str, file_path: str) -> BytesIO:
    """Download file from NAS to BytesIO buffer"""
    try:
        file_obj = BytesIO()
        conn.retrieveFile(share, file_path, file_obj)
        file_obj.seek(0)
        return file_obj
    except Exception as e:
        logger.error(f"Error downloading file {file_path}: {str(e)}")
        raise

def upload_file_to_nas(conn: SMBConnection, share: str, file_path: str, content: bytes):
    """Upload file to NAS"""
    try:
        file_obj = BytesIO(content)
        conn.storeFile(share, file_path, file_obj)
        file_obj.close()
    except Exception as e:
        logger.error(f"Error uploading file {file_path}: {str(e)}")
        raise

def ensure_directory_exists(conn: SMBConnection, share: str, path: str):
    """Ensure directory exists on NAS, create if necessary"""
    parts = path.split('/')
    current_path = ""
    
    for part in parts:
        if not part:
            continue
            
        if current_path:
            current_path = f"{current_path}/{part}"
        else:
            current_path = part
        
        try:
            conn.listPath(share, current_path)
        except OperationFailure:
            try:
                conn.createDirectory(share, current_path)
                logger.info(f"Created directory: {current_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {current_path}: {str(e)}")
                raise

# ========================================
# OAUTH TOKEN GENERATION
# ========================================

def generate_oauth_token() -> str:
    """Generate OAuth token for OpenAI API"""
    try:
        response = requests.post(
            OAUTH_CONFIG["auth_endpoint"],
            data={
                "grant_type": "client_credentials",
                "client_id": OAUTH_CONFIG["client_id"],
                "client_secret": OAUTH_CONFIG["client_secret"]
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code == 200:
            token_data = response.json()
            return token_data.get("access_token")
        else:
            raise Exception(f"OAuth token generation failed: {response.status_code} - {response.text}")
    
    except Exception as e:
        logger.error(f"Error generating OAuth token: {str(e)}")
        raise

def get_openai_client() -> OpenAI:
    """Initialize OpenAI client with OAuth token"""
    try:
        token = generate_oauth_token()
        client = OpenAI(
            api_key=token,
            base_url=OAUTH_CONFIG["base_url"]
        )
        return client
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {str(e)}")
        raise

# ========================================
# HELPER FUNCTIONS - PDF PROCESSING
# ========================================

def extract_pdf_text_indexed(pdf_content: BytesIO) -> List[Dict[str, Any]]:
    """Extract text from PDF with indexed lines and page numbers"""
    debug_log_step("PDF Text Extraction", "Starting PDF text extraction with line indexing")
    
    pages = []
    
    try:
        reader = PdfReader(pdf_content)
        total_pages = len(reader.pages)
        global_line_number = 0
        
        # Apply debug page limit if set
        max_pages = DEBUG_MAX_PAGES if DEBUG_MODE and DEBUG_MAX_PAGES else total_pages
        pages_to_process = min(max_pages, total_pages)
        
        if DEBUG_MODE:
            logger.debug(f"DEBUG: Processing {pages_to_process} of {total_pages} pages")
        
        for page_num, page in enumerate(reader.pages[:pages_to_process]):
            text = page.extract_text()
            
            # Split into lines and create indexed structure
            lines = text.split('\n')
            page_data = {
                'page_number': page_num + 1,
                'lines': [],
                'start_line': global_line_number + 1,
                'end_line': global_line_number + len([l for l in lines if l.strip()])
            }
            
            for line in lines:
                if line.strip():  # Skip empty lines
                    global_line_number += 1
                    page_data['lines'].append({
                        'global_line_number': global_line_number,
                        'page_line_number': len(page_data['lines']) + 1,
                        'text': line.strip()
                    })
            
            pages.append(page_data)
            
            if DEBUG_MODE:
                logger.debug(f"DEBUG: Page {page_num + 1} - {len(page_data['lines'])} lines extracted")
        
        # Create extraction summary
        page_stats = []
        for page in pages:
            page_stats.append({
                'Page': page['page_number'],
                'Lines': len(page['lines']),
                'Start_Line': page['start_line'],
                'End_Line': page['end_line']
            })
        
        if DEBUG_MODE:
            # Print summary table
            table = create_summary_table("PDF EXTRACTION SUMMARY", page_stats, ['Page', 'Lines', 'Start_Line', 'End_Line'])
            logger.info(table)
            
            # Create processing summary
            extraction_stats = {
                'Total Pages': len(pages),
                'Total Lines': global_line_number,
                'Average Lines per Page': f"{global_line_number / len(pages):.1f}",
                'Pages Processed': f"{pages_to_process}/{total_pages}" if DEBUG_MAX_PAGES else f"{total_pages}/{total_pages}"
            }
            summary = create_processing_summary("PDF Text Extraction", extraction_stats)
            logger.info(summary)
            
            # Save files only if intermediates enabled
            if DEBUG_SAVE_INTERMEDIATES:
                debug_save_json(pages, "01_extracted_pages.json", "extracted PDF pages with line indexing")
        
        logger.info(f"Extracted indexed text from {len(pages)} pages, {global_line_number} lines total")
        return pages
        
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        raise

def extract_first_n_words(pages: List[Dict[str, Any]], n: int = 1000) -> str:
    """Extract first N words from transcript"""
    words = []
    word_count = 0
    
    for page in pages:
        for line in page['lines']:
            line_words = line['text'].split()
            for word in line_words:
                if word_count >= n:
                    break
                words.append(word)
                word_count += 1
            if word_count >= n:
                break
        if word_count >= n:
            break
    
    return ' '.join(words)

def clean_transcript_text(text: str) -> str:
    """Clean and normalize transcript text"""
    if not text:
        return ""
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and headers
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{4}', '', text)
    
    # Remove disclaimer footers
    text = re.sub(r'(?i)this transcript.*?accuracy\.?', '', text)
    
    # Normalize quotes and escape problematic characters
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove or replace characters that can cause JSON issues
    text = text.replace('\\', ' ')  # Replace backslashes
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    text = text.replace('\r', ' ')  # Replace carriage returns
    text = text.replace('\t', ' ')  # Replace tabs with spaces
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces again
    
    return text.strip()

# ========================================
# TOKEN COUNTING
# ========================================

def count_tokens_tiktoken(text: str, model: str = "gpt-4") -> int:
    """Count tokens using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error with tiktoken: {str(e)}")
        return count_tokens_fallback(text)

def count_tokens_fallback(text: str) -> int:
    """Fallback token counting (rough approximation)"""
    return len(text) // 4

def count_tokens(text: str) -> int:
    """Count tokens with tiktoken fallback"""
    if tiktoken:
        return count_tokens_tiktoken(text)
    else:
        return count_tokens_fallback(text)

# ========================================
# OPENAI INTEGRATION
# ========================================

def call_openai_with_retry(client: OpenAI, messages: List[Dict], 
                          model: str = COMPLETION_MODEL, 
                          response_format: Optional[Dict] = None,
                          tools: Optional[List] = None,
                          max_tokens: Optional[int] = None) -> str:
    """Call OpenAI API with retry logic and token refresh"""
    for attempt in range(MAX_RETRIES):
        try:
            # Regenerate client with fresh token for each attempt
            if attempt > 0:
                client = get_openai_client()
            
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": max_tokens or 32768
            }
            
            if response_format:
                kwargs["response_format"] = response_format
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "required"
            
            response = client.chat.completions.create(**kwargs)
            
            if tools:
                # Return tool call result
                tool_calls = response.choices[0].message.tool_calls
                if tool_calls:
                    return tool_calls[0].function.arguments
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"OpenAI API error (attempt {attempt + 1}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise

def truncate_text_for_embedding(text: str, max_tokens: int = 8000) -> str:
    """Truncate text to fit within embedding token limits"""
    token_count = count_tokens(text)
    
    if token_count <= max_tokens:
        return text
    
    # Rough estimate: truncate to ~80% of max tokens to be safe
    target_chars = int(len(text) * (max_tokens * 0.8) / token_count)
    truncated = text[:target_chars]
    
    # Try to end at a sentence boundary
    last_period = truncated.rfind('.')
    if last_period > target_chars * 0.8:  # If we can find a period in the last 20%
        truncated = truncated[:last_period + 1]
    
    if DEBUG_MODE:
        logger.debug(f"Truncated text from {token_count} to ~{count_tokens(truncated)} tokens for embedding")
    
    return truncated

def generate_embedding(client: OpenAI, text: str) -> List[float]:
    """Generate embedding for text"""
    if DEBUG_MODE and DEBUG_SKIP_EMBEDDINGS:
        logger.debug("Skipping embedding generation (DEBUG_SKIP_EMBEDDINGS=True)")
        return [0.0] * EMBEDDING_DIMENSIONS  # Return dummy embedding
    
    try:
        # Truncate text if it's too long
        truncated_text = truncate_text_for_embedding(text)
        
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=truncated_text,
            dimensions=EMBEDDING_DIMENSIONS
        )
        
        embedding = response.data[0].embedding
        
        if DEBUG_MODE and len(text) != len(truncated_text):
            logger.debug(f"Generated embedding (truncated {len(text)} → {len(truncated_text)} chars)")
        
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

# ========================================
# BANK IDENTIFICATION
# ========================================

def identify_bank_name(first_words: str, client: OpenAI) -> BankInfo:
    """Identify bank name from first 1000 words using LLM tool call"""
    debug_log_step("Bank Identification", f"Identifying bank from {len(first_words)} characters")
    
    # Save first words in debug mode
    if DEBUG_MODE:
        debug_save_to_file(first_words, "02_first_1000_words.txt", "first 1000 words for bank identification")
    
    # Create tool definition
    bank_identification_tool = {
        "type": "function",
        "function": {
            "name": "identify_bank",
            "description": "Identify the bank name from transcript text",
            "parameters": {
                "type": "object",
                "properties": {
                    "detected_bank_name": {
                        "type": "string",
                        "description": "The bank name as detected in the transcript"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score from 0.0 to 1.0"
                    }
                },
                "required": ["detected_bank_name", "confidence"]
            }
        }
    }
    
    # Create bank list for prompt
    bank_list = "\n".join([f"- {name}" for name in BANK_MAPPINGS.keys()])
    
    prompt = f"""Analyze this earnings call transcript excerpt and identify the bank name.

Expected banks (use exact names from this list if possible):
{bank_list}

Transcript excerpt (first 1000 words):
{first_words}

Look for:
1. Company name in headers/titles
2. Speaker introductions
3. References to "we", "our bank", "our company"
4. Stock ticker mentions
5. Legal entity names

Use the tool to return the detected bank name and your confidence level.
"""

    messages = [
        {"role": "system", "content": "You are an expert at identifying financial institutions from earnings call transcripts."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        # Log the LLM call
        debug_log_llm_call("bank_identification", prompt)
        
        response = call_openai_with_retry(
            client, 
            messages, 
            tools=[bank_identification_tool],
            max_tokens=4096
        )
        
        # Log the response
        debug_log_llm_call("bank_identification", prompt, response)
        
        result = json.loads(response)
        detected_name = result.get("detected_bank_name", "").strip()
        confidence = float(result.get("confidence", 0.0))
        
        if DEBUG_MODE:
            logger.debug(f"DEBUG: Bank identification result - Name: '{detected_name}', Confidence: {confidence}")
        
        # Check if detected name matches our mappings
        standardized_info = None
        for mapped_name, info in BANK_MAPPINGS.items():
            if mapped_name.lower() in detected_name.lower() or detected_name.lower() in mapped_name.lower():
                standardized_info = info
                if DEBUG_MODE:
                    logger.debug(f"DEBUG: Matched '{detected_name}' to '{mapped_name}' -> {info}")
                break
        
        if standardized_info:
            return BankInfo(
                detected_name=detected_name,
                standardized_name=standardized_info["name"],
                ticker_region=standardized_info["ticker_region"],
                confidence=confidence,
                recognized=True
            )
        else:
            return BankInfo(
                detected_name=detected_name,
                standardized_name=detected_name,
                ticker_region="UNKNOWN:UNKNOWN",
                confidence=confidence,
                recognized=False
            )
            
    except Exception as e:
        logger.error(f"Error identifying bank name: {str(e)}")
        return BankInfo(
            detected_name="ERROR",
            standardized_name="ERROR",
            ticker_region="ERROR:ERROR",
            confidence=0.0,
            recognized=False
        )

def identify_bank_with_retry(first_words: str, client: OpenAI) -> BankInfo:
    """Identify bank name with retry logic"""
    # First attempt
    bank_info = identify_bank_name(first_words, client)
    
    # If not recognized, try once more
    if not bank_info.recognized and bank_info.detected_name != "ERROR":
        logger.info(f"Bank not recognized on first attempt: {bank_info.detected_name}. Retrying...")
        bank_info = identify_bank_name(first_words, client)
    
    # Create identification summary
    if DEBUG_MODE:
        bank_stats = {
            'Detected Name': bank_info.detected_name,
            'Standardized Name': bank_info.standardized_name,
            'Ticker:Region': bank_info.ticker_region,
            'Confidence': f"{bank_info.confidence:.2f}",
            'Recognized': 'Yes' if bank_info.recognized else 'No',
            'Input Length': f"{len(first_words)} chars"
        }
        summary = create_processing_summary("Bank Identification", bank_stats)
        logger.info(summary)
    
    return bank_info

# ========================================
# PRIMARY SECTION IDENTIFICATION
# ========================================

def format_page_for_llm(page_data: Dict) -> str:
    """Format page data for LLM processing"""
    formatted_lines = []
    for line in page_data['lines']:
        formatted_lines.append(f"Line {line['global_line_number']}: {line['text']}")
    
    return f"=== Page {page_data['page_number']} ===\n" + "\n".join(formatted_lines)

def identify_primary_sections_progressive(pages: List[Dict], client: OpenAI) -> List[Dict]:
    """Identify primary sections page by page with progressive context"""
    debug_log_step("Primary Section Identification", f"Processing {len(pages)} pages with progressive context")
    
    section_results = []  # Will store results for each page
    identified_sections = []  # Will store final section boundaries
    
    for page_idx, current_page in enumerate(pages):
        # Get context pages
        prev_page = pages[page_idx - 1] if page_idx > 0 else None
        next_page = pages[page_idx + 1] if page_idx < len(pages) - 1 else None
        
        # Build context from previous page results
        previous_context = ""
        if section_results:
            prev_sections = [f"Page {r['page_number']}: {', '.join(r['sections'])}" 
                           for r in section_results]
            previous_context = "Previous page classifications:\n" + "\n".join(prev_sections) + "\n\n"
        
        # Format current page
        current_page_text = format_page_for_llm(current_page)
        
        # Format next page for context
        next_page_context = ""
        if next_page:
            next_page_context = f"\nNext page context (for reference only):\n{format_page_for_llm(next_page)}"
        
        # Create tool for structured section identification
        section_identification_tool = {
            "type": "function",
            "function": {
                "name": "identify_sections",
                "description": "Identify primary transcript sections on the current page",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "page_number": {
                            "type": "integer",
                            "description": "The page number being analyzed"
                        },
                        "sections": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "section_name": {
                                        "type": "string",
                                        "enum": PRIMARY_SECTIONS,
                                        "description": "Name of the primary section"
                                    },
                                    "start_line": {
                                        "type": "integer",
                                        "description": "Global line number where section starts"
                                    },
                                    "end_line": {
                                        "type": "integer", 
                                        "description": "Global line number where section ends"
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "description": "Confidence level 0.0-1.0"
                                    }
                                },
                                "required": ["section_name", "start_line", "end_line", "confidence"]
                            }
                        }
                    },
                    "required": ["page_number", "sections"]
                }
            }
        }

        # Create prompt
        prompt = f"""Analyze page {current_page['page_number']} to identify primary transcript sections.

Expected sections: {', '.join(PRIMARY_SECTIONS)}

{previous_context}Current page content:
{current_page_text}
{next_page_context}

Instructions:
- Only analyze the CURRENT page
- If a section spans multiple pages, mark the portion on THIS page
- Use exact global line numbers from the transcript
- Only identify clear section boundaries, not assumptions"""

        messages = [
            {"role": "system", "content": "You analyze earnings call transcripts to identify section boundaries."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Save debug files only if intermediates enabled
            if DEBUG_MODE and DEBUG_SAVE_INTERMEDIATES:
                debug_save_to_file(
                    prompt, 
                    f"03_page_{current_page['page_number']:02d}_prompt.txt",
                    f"primary section identification prompt for page {current_page['page_number']}"
                )
            
            response = call_openai_with_retry(
                client, 
                messages,
                tools=[section_identification_tool],
                max_tokens=8192
            )
            
            # Save response only if intermediates enabled
            if DEBUG_MODE and DEBUG_SAVE_INTERMEDIATES:
                debug_save_to_file(
                    response,
                    f"03_page_{current_page['page_number']:02d}_response.json",
                    f"primary section identification response for page {current_page['page_number']}"
                )
            
            result = json.loads(response)
            page_result = {
                'page_number': current_page['page_number'],
                'sections': []
            }
            
            for section in result.get('sections', []):
                page_result['sections'].append(section['section_name'])
                
                # Add to identified sections
                identified_sections.append({
                    'section_name': section['section_name'],
                    'page_number': current_page['page_number'],
                    'start_line': section['start_line'],
                    'end_line': section['end_line'],
                    'confidence': section.get('confidence', 0.8)
                })
            
            section_results.append(page_result)
            logger.info(f"Page {current_page['page_number']}: {page_result['sections'] or 'No sections'}")
            
        except Exception as e:
            logger.error(f"Error processing page {current_page['page_number']}: {str(e)}")
            section_results.append({
                'page_number': current_page['page_number'],
                'sections': []
            })
    
    # Consolidate overlapping sections and create final boundaries
    consolidated_sections = consolidate_section_boundaries(identified_sections)
    
    # Create section identification summary
    if DEBUG_MODE:
        # Primary sections table
        section_stats = []
        for section in consolidated_sections:
            line_count = section['end_line'] - section['start_line'] + 1
            section_stats.append({
                'Section': section['section_name'],
                'Start_Line': section['start_line'],
                'End_Line': section['end_line'], 
                'Lines': line_count,
                'Confidence': f"{section['confidence']:.2f}",
                'Pages': f"{section['page_span'][0]}-{section['page_span'][1]}"
            })
        
        table = create_summary_table("PRIMARY SECTIONS IDENTIFIED", section_stats, 
                                   ['Section', 'Start_Line', 'End_Line', 'Lines', 'Confidence', 'Pages'])
        logger.info(table)
        
        # Overall stats
        total_lines = sum(s['end_line'] - s['start_line'] + 1 for s in consolidated_sections)
        primary_stats = {
            'Total Primary Sections': len(consolidated_sections),
            'Total Lines Covered': total_lines,
            'Average Section Length': f"{total_lines / len(consolidated_sections):.1f} lines" if consolidated_sections else "0",
            'Pages Analyzed': len(section_results),
            'Expected Sections': len(PRIMARY_SECTIONS),
            'Coverage': f"{len(consolidated_sections)}/{len(PRIMARY_SECTIONS)}"
        }
        summary = create_processing_summary("Primary Section Identification", primary_stats)
        logger.info(summary)
        
        # Save files only if intermediates enabled
        if DEBUG_SAVE_INTERMEDIATES:
            debug_save_json(consolidated_sections, "03_consolidated_sections.json", "final consolidated section boundaries")
    
    return consolidated_sections

def consolidate_section_boundaries(identified_sections: List[Dict]) -> List[Dict]:
    """Consolidate overlapping section boundaries into clean boundaries"""
    if not identified_sections:
        return []
    
    # Group by section name
    section_groups = defaultdict(list)
    for section in identified_sections:
        section_groups[section['section_name']].append(section)
    
    # For each section, find the overall start and end
    consolidated = []
    for section_name, occurrences in section_groups.items():
        min_start = min(occ['start_line'] for occ in occurrences)
        max_end = max(occ['end_line'] for occ in occurrences)
        avg_confidence = sum(occ['confidence'] for occ in occurrences) / len(occurrences)
        
        consolidated.append({
            'section_name': section_name,
            'start_line': min_start,
            'end_line': max_end,
            'confidence': avg_confidence,
            'page_span': [min(occ['page_number'] for occ in occurrences),
                         max(occ['page_number'] for occ in occurrences)]
        })
    
    # Sort by start line
    consolidated.sort(key=lambda x: x['start_line'])
    
    return consolidated

# ========================================
# SECONDARY SECTION IDENTIFICATION
# ========================================

def extract_section_content_by_lines(pages: List[Dict], start_line: int, end_line: int) -> str:
    """Extract content between global line numbers"""
    content = []
    
    for page in pages:
        for line in page['lines']:
            if start_line <= line['global_line_number'] <= end_line:
                content.append(line['text'])
    
    return ' '.join(content)

def create_secondary_sections(primary_content: str, primary_type: str, client: OpenAI) -> List[Dict]:
    """Break primary section into secondary sections"""
    debug_log_step("Secondary Section Creation", f"Breaking down {primary_type} section into secondary sections")
    
    # Create tool for structured output
    section_breakdown_tool = {
        "type": "function",
        "function": {
            "name": "create_sections",
            "description": "Break down a primary transcript section into logical secondary sections",
            "parameters": {
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Descriptive name for the subsection"
                                },
                                "content": {
                                    "type": "string", 
                                    "description": "Full text content for this subsection"
                                },
                                "rationale": {
                                    "type": "string",
                                    "description": "Brief explanation of why this is a distinct subsection"
                                }
                            },
                            "required": ["name", "content", "rationale"]
                        }
                    }
                },
                "required": ["sections"]
            }
        }
    }
    
    # Truncate content if too long for LLM context
    max_chars = 15000  # Leave room for prompt
    truncated_content = primary_content[:max_chars]
    if len(primary_content) > max_chars:
        # Try to end at sentence boundary
        last_period = truncated_content.rfind('.')
        if last_period > max_chars * 0.8:
            truncated_content = truncated_content[:last_period + 1]
        logger.warning(f"Truncated {primary_type} content from {len(primary_content)} to {len(truncated_content)} chars")
    
    prompt = f"""Break down this {primary_type} section into 3-6 logical secondary sections.

Guidelines:
- Each section should have 50+ words and represent a distinct topic
- Look for natural breaks: speaker changes, topic shifts, new metrics
- Common patterns for {primary_type}: financial metrics, strategic updates, operational highlights

Section content to analyze:
{truncated_content}

Use the tool to return the secondary sections."""

    messages = [
        {"role": "system", "content": "You are an expert at analyzing financial transcript structure."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        # Save debug information
        if DEBUG_MODE and DEBUG_SAVE_INTERMEDIATES:
            debug_save_to_file(truncated_content, f"04_{primary_type.replace(' ', '_').lower()}_content.txt", f"{primary_type} section content")
        
        response = call_openai_with_retry(
            client, 
            messages,
            tools=[section_breakdown_tool],
            max_tokens=32768
        )
        
        # Parse the tool call response with better error handling
        try:
            result = json.loads(response)
            sections_data = result.get('sections', [])
            
            if not sections_data:
                raise ValueError("No sections returned from tool call")
            
            secondary_sections = []
            for section in sections_data:
                # Validate required fields
                if not all(key in section for key in ['name', 'content']):
                    logger.warning(f"Skipping invalid section: {section}")
                    continue
                    
                secondary_sections.append({
                    'name': section['name'][:200],  # Limit name length
                    'content': clean_transcript_text(section['content']),
                    'rationale': section.get('rationale', '')[:500]  # Limit rationale length
                })
            
            if not secondary_sections:
                raise ValueError("No valid sections created after parsing")
            
            # Log summary
            logger.info(f"Created {len(secondary_sections)} secondary sections for {primary_type}")
            return secondary_sections
            
        except json.JSONDecodeError as json_error:
            logger.error(f"JSON parsing error for {primary_type}: {json_error}")
            logger.error(f"Raw response: {response[:500]}...")  # Log first 500 chars
            # Fall through to fallback handling
        except Exception as parse_error:
            logger.error(f"Section parsing error for {primary_type}: {parse_error}")
            # Fall through to fallback handling
        
    except Exception as e:
        logger.error(f"Error creating secondary sections for {primary_type}: {str(e)}")
    # Fallback: split by paragraphs or return as single section
    logger.warning(f"Using fallback section creation for {primary_type}")
    paragraphs = [p.strip() for p in truncated_content.split('\n\n') if p.strip()]
    
    if len(paragraphs) > 1:
        # Create sections from paragraphs
        secondary_sections = []
        for i, para in enumerate(paragraphs[:5], 1):  # Max 5 sections
            if len(para) > 50:  # Only substantial paragraphs
                secondary_sections.append({
                    'name': f"{primary_type} - Part {i}",
                    'content': clean_transcript_text(para),
                    'rationale': f'Paragraph-based division due to LLM parsing error'
                })
        
        if secondary_sections:
            logger.warning(f"Used paragraph fallback for {primary_type}: {len(secondary_sections)} sections")
            return secondary_sections
    
    # Final fallback: single section
    logger.warning(f"Using single section fallback for {primary_type}")
    return [{
        'name': f"{primary_type} - Complete",
        'content': clean_transcript_text(truncated_content),
        'rationale': 'Single section fallback due to LLM parsing error'
    }]

# ========================================
# SUMMARY AND SCORING
# ========================================

def generate_summary(content: str, section_type: str, client: OpenAI) -> str:
    """Generate summary for a section"""
    token_limit = TOKEN_LIMITS.get(section_type, 150)
    
    prompt = f"""TASK: Generate a concise {token_limit}-token summary of this {section_type}.

CONTENT TO SUMMARIZE:
{content}

SUMMARY REQUIREMENTS:
• Capture key financial metrics with specific numbers
• Highlight main themes and strategic messages
• Include forward-looking statements and guidance
• Note significant changes, comparisons, or trends
• Focus on factual information valuable for financial analysis
• Use precise, professional language

FORMAT: Return only the summary text with no additional formatting or preamble.
"""

    messages = [
        {"role": "system", "content": "You are an expert financial analyst creating concise summaries."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = call_openai_with_retry(client, messages, max_tokens=2048)
        return response.strip()
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return f"Summary generation failed for {section_type}"

def calculate_context_relevance_scores(sections: List[Dict], current_index: int, 
                                     primary_summary: str, client: OpenAI) -> Dict:
    """Calculate importance and context relevance scores for a section"""
    
    current_section = sections[current_index]
    prev_section = sections[current_index - 1] if current_index > 0 else None
    next_section = sections[current_index + 1] if current_index < len(sections) - 1 else None
    
    # Create tool for structured scoring
    scoring_tool = {
        "type": "function",
        "function": {
            "name": "score_section",
            "description": "Score transcript section importance and context dependencies",
            "parameters": {
                "type": "object",
                "properties": {
                    "importance_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "How critical this information is for investors (0.0-1.0)"
                    },
                    "preceding_context_relevance": {
                        "type": "number", 
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "How much understanding depends on previous section (0.0-1.0)"
                    },
                    "following_context_relevance": {
                        "type": "number",
                        "minimum": 0.0, 
                        "maximum": 1.0,
                        "description": "How much understanding depends on following section (0.0-1.0)"
                    }
                },
                "required": ["importance_score", "preceding_context_relevance", "following_context_relevance"]
            }
        }
    }
    
    # Build context for scoring
    prompt = f"""Score this transcript section for importance and context dependencies.

PRIMARY SECTION: {primary_summary[:200]}

CURRENT SECTION: {current_section['secondary_section_type']}
Summary: {current_section['secondary_section_summary']}

PREVIOUS: {prev_section['secondary_section_type'] if prev_section else 'None'}
NEXT: {next_section['secondary_section_type'] if next_section else 'None'}

Scoring guidelines:
- Importance: Financial metrics/guidance (0.8-1.0), Operations (0.5-0.7), Procedural (0.0-0.4)
- Context dependency: Critical (0.8-1.0), Helpful (0.5-0.7), Minimal (0.0-0.4)"""

    messages = [
        {"role": "system", "content": "You evaluate financial information importance and context dependencies."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = call_openai_with_retry(
            client, 
            messages,
            tools=[scoring_tool],
            max_tokens=1024
        )
        
        result = json.loads(response)
        return {
            'importance_score': float(result.get('importance_score', 0.5)),
            'preceding_context_relevance': float(result.get('preceding_context_relevance', 0.0)) if prev_section else 0.0,
            'following_context_relevance': float(result.get('following_context_relevance', 0.0)) if next_section else 0.0
        }
        
    except Exception as e:
        logger.error(f"Error calculating context scores: {str(e)}")
        return {
            'importance_score': 0.5,
            'preceding_context_relevance': 0.0 if not prev_section else 0.3,
            'following_context_relevance': 0.0 if not next_section else 0.3
        }

# ========================================
# MASTER DATABASE OPERATIONS
# ========================================

def read_master_database(nas_conn: SMBConnection) -> pd.DataFrame:
    """Read master database CSV from NAS (mirrors PostgreSQL schema)"""
    try:
        # Suppress SMB connection logging for file existence check
        old_level = logging.getLogger('smb.SMBConnection').level
        logging.getLogger('smb.SMBConnection').setLevel(logging.CRITICAL)
        
        csv_content = download_file_from_nas(
            nas_conn, 
            DEST_CONFIG['share'], 
            DEST_CONFIG['master_db_path']
        )
        
        # Restore logging level
        logging.getLogger('smb.SMBConnection').setLevel(old_level)
        
        df = pd.read_csv(csv_content)
        if DEBUG_MODE:
            logger.info(f"✓ Loaded master database: {len(df)} section records")
        else:
            logger.info(f"Read master database with {len(df)} records")
        return df
    except OperationFailure:
        # File doesn't exist, create empty dataframe with PostgreSQL schema
        if DEBUG_MODE:
            logger.info("⚠ Master database not found, creating new one")
        else:
            logger.info("Master database not found, creating new one")
        
        return pd.DataFrame(columns=[
            # File metadata
            'fiscal_year', 'quarter', 'bank_name', 'ticker_region', 'filepath', 'filename', 'date_last_modified',
            # Hierarchical classification  
            'primary_section_type', 'primary_section_summary', 'secondary_section_type', 'secondary_section_summary',
            # Content and search data
            'section_content', 'section_order', 'section_tokens', 'section_embedding',
            # Reranking scores
            'importance_score', 'preceding_context_relevance', 'following_context_relevance',
            # Metadata
            'created_at', 'updated_at'
        ])

def update_master_database(nas_conn: SMBConnection, df: pd.DataFrame):
    """Write master database CSV to NAS"""
    try:
        # Suppress SMB logging for upload
        old_level = logging.getLogger('smb.SMBConnection').level
        logging.getLogger('smb.SMBConnection').setLevel(logging.CRITICAL)
        
        # Convert to CSV
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Upload to NAS
        upload_file_to_nas(
            nas_conn, 
            DEST_CONFIG['share'], 
            DEST_CONFIG['master_db_path'],
            csv_content
        )
        
        # Restore logging level
        logging.getLogger('smb.SMBConnection').setLevel(old_level)
        
        if DEBUG_MODE:
            logger.info(f"✓ Updated master database: {len(df)} section records")
        else:
            logger.info("Updated master database on NAS")
    except Exception as e:
        logger.error(f"Failed to update master database: {str(e)}")

def scan_for_transcripts(nas_conn: SMBConnection) -> Dict[str, Dict]:
    """Scan NAS for transcript files"""
    transcripts = {}
    
    try:
        base_items = nas_conn.listPath(DEST_CONFIG['share'], DEST_CONFIG['base_path'])
        
        for year_item in base_items:
            if year_item.filename in ['.', '..'] or not year_item.isDirectory:
                continue
            
            try:
                year = int(year_item.filename)
                if not (VALID_YEAR_RANGE[0] <= year <= VALID_YEAR_RANGE[1]):
                    continue
            except ValueError:
                continue
            
            year_path = f"{DEST_CONFIG['base_path']}/{year_item.filename}"
            
            try:
                quarter_items = nas_conn.listPath(DEST_CONFIG['share'], year_path)
                
                for quarter_item in quarter_items:
                    if quarter_item.filename in ['.', '..'] or not quarter_item.isDirectory:
                        continue
                    
                    if quarter_item.filename.upper() not in VALID_QUARTERS:
                        continue
                    
                    quarter_path = f"{year_path}/{quarter_item.filename}"
                    
                    try:
                        files = nas_conn.listPath(DEST_CONFIG['share'], quarter_path)
                        
                        for file_item in files:
                            if file_item.isDirectory:
                                continue
                            
                            if any(file_item.filename.endswith(ext) for ext in FILE_EXTENSIONS):
                                file_path = f"{quarter_path}/{file_item.filename}"
                                
                                # Convert last_write_time to datetime for consistency
                                last_modified = file_item.last_write_time
                                if isinstance(last_modified, (int, float)):
                                    last_modified = pd.to_datetime(last_modified, unit='s')
                                else:
                                    last_modified = pd.to_datetime(last_modified)
                                
                                transcripts[file_path] = {
                                    'filename': file_item.filename,
                                    'filepath': file_path,
                                    'fiscal_year': year,
                                    'quarter': quarter_item.filename.upper(),
                                    'last_modified': last_modified,
                                    'size': file_item.file_size
                                }
                    
                    except Exception as e:
                        logger.error(f"Error scanning {quarter_path}: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error scanning {year_path}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error scanning base path: {str(e)}")
    
    logger.info(f"Found {len(transcripts)} transcript files")
    return transcripts

def compare_files_with_master(current_files: Dict, master_df: pd.DataFrame) -> Dict[str, List]:
    """Compare current files with master database"""
    result = {
        'new_files': [],
        'modified_files': [],
        'deleted_files': []
    }
    
    # Get existing files from master database
    existing_files = set(master_df['filepath'].tolist()) if len(master_df) > 0 else set()
    current_file_paths = set(current_files.keys())
    
    # Find new files
    new_file_paths = current_file_paths - existing_files
    for filepath in new_file_paths:
        result['new_files'].append(current_files[filepath])
    
    # Find modified files
    for filepath in current_file_paths.intersection(existing_files):
        row = master_df[master_df['filepath'] == filepath].iloc[0]
        current_modified = current_files[filepath]['last_modified']
        
        if pd.notna(row['date_last_modified']):
            master_modified = pd.to_datetime(row['date_last_modified'])
            # Ensure current_modified is also a datetime for comparison
            current_modified_dt = pd.to_datetime(current_modified)
            
            if current_modified_dt > master_modified:
                result['modified_files'].append(current_files[filepath])
    
    # Find deleted files
    deleted_file_paths = existing_files - current_file_paths
    result['deleted_files'] = list(deleted_file_paths)
    
    return result

# ========================================
# LOGGING FUNCTIONS
# ========================================

def log_unrecognized_bank(nas_conn: SMBConnection, bank_info: BankInfo, filepath: str):
    """Log unrecognized bank names to NAS"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"unrecognized_banks_{timestamp}.log"
        log_path = f"{DEST_CONFIG['logs_path']}/{log_filename}"
        
        # Ensure logs directory exists
        ensure_directory_exists(nas_conn, DEST_CONFIG['share'], DEST_CONFIG['logs_path'])
        
        log_entry = f"""
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
File: {filepath}
Detected Name: {bank_info.detected_name}
Confidence: {bank_info.confidence}
Standardized Name: {bank_info.standardized_name}
Ticker Region: {bank_info.ticker_region}
---
"""
        
        # Try to read existing log
        try:
            existing_log = download_file_from_nas(nas_conn, DEST_CONFIG['share'], log_path)
            existing_content = existing_log.read().decode('utf-8')
            full_content = existing_content + log_entry
        except:
            full_content = "Unrecognized Banks Log\n" + "=" * 50 + "\n" + log_entry
        
        # Write updated log
        upload_file_to_nas(nas_conn, DEST_CONFIG['share'], log_path, full_content.encode('utf-8'))
        logger.info(f"Logged unrecognized bank: {bank_info.detected_name}")
        
    except Exception as e:
        logger.error(f"Failed to log unrecognized bank: {str(e)}")

def write_error_log(nas_conn: SMBConnection, errors: List[str]):
    """Write error log to NAS"""
    if not errors:
        return
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"transcript_processing_errors_{timestamp}.log"
        log_path = f"{DEST_CONFIG['logs_path']}/{log_filename}"
        
        # Ensure logs directory exists
        ensure_directory_exists(nas_conn, DEST_CONFIG['share'], DEST_CONFIG['logs_path'])
        
        log_content = f"Transcript Processing Error Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        log_content += "=" * 60 + "\n\n"
        
        for error in errors:
            log_content += f"{error}\n"
        
        upload_file_to_nas(nas_conn, DEST_CONFIG['share'], log_path, log_content.encode('utf-8'))
        logger.info(f"Error log written to: {log_path}")
        
    except Exception as e:
        logger.error(f"Failed to write error log: {str(e)}")

# ========================================
# MAIN PROCESSING FUNCTION
# ========================================

def process_transcript(nas_conn: SMBConnection, file_info: Dict, client: OpenAI) -> ProcessingStatus:
    """Process a single transcript file"""
    start_time = time.time()
    status = ProcessingStatus(
        filepath=file_info['filepath'],
        status='processing'
    )
    
    try:
        logger.info(f"Processing: {file_info['filename']}")
        
        # Download PDF
        pdf_content = download_file_from_nas(
            nas_conn,
            DEST_CONFIG['share'],
            file_info['filepath']
        )
        
        # Extract text with indexed lines
        pages = extract_pdf_text_indexed(pdf_content)
        
        # Extract first 1000 words for bank identification
        first_words = extract_first_n_words(pages, 1000)
        
        # Identify bank name
        bank_info = identify_bank_with_retry(first_words, client)
        
        # Log unrecognized banks
        if not bank_info.recognized:
            log_unrecognized_bank(nas_conn, bank_info, file_info['filepath'])
        
        # Identify primary sections
        primary_boundaries = identify_primary_sections_progressive(pages, client)
        
        if not primary_boundaries:
            raise Exception("No primary sections identified")
        
        # Process each primary section
        all_sections = []
        section_order = 0
        primary_section_names = []
        
        for i, boundary in enumerate(primary_boundaries):
            # Extract primary section content
            next_boundary = primary_boundaries[i + 1] if i < len(primary_boundaries) - 1 else None
            
            start_line = boundary['start_line']
            end_line = next_boundary['start_line'] - 1 if next_boundary else max(
                line['global_line_number'] for page in pages for line in page['lines']
            )
            
            primary_content = extract_section_content_by_lines(pages, start_line, end_line)
            
            # Generate primary section summary
            primary_summary = generate_summary(
                primary_content, 
                'primary_summary', 
                client
            )
            
            primary_section_names.append(boundary['section_name'])
            
            # Create secondary sections
            secondary_sections = create_secondary_sections(
                primary_content,
                boundary['section_name'],
                client
            )
            
            # Store sections for scoring
            temp_sections = []
            
            # Process each secondary section
            for sec in secondary_sections:
                section_order += 1
                
                # Generate secondary summary
                secondary_summary = generate_summary(
                    sec['content'],
                    'secondary_summary',
                    client
                )
                
                # Create section object
                section = TranscriptSection(
                    fiscal_year=file_info['fiscal_year'],
                    quarter=file_info['quarter'],
                    bank_name=bank_info.standardized_name,
                    ticker_region=bank_info.ticker_region,
                    filepath=file_info['filepath'],
                    filename=file_info['filename'],
                    date_last_modified=file_info['last_modified'],
                    primary_section_type=boundary['section_name'],
                    primary_section_summary=primary_summary,
                    secondary_section_type=sec['name'],
                    secondary_section_summary=secondary_summary,
                    section_content=sec['content'],
                    section_order=section_order,
                    section_tokens=count_tokens(sec['content'])
                )
                
                temp_sections.append(section)
            
            # Calculate context relevance scores for sections in this primary
            sections_dict = [asdict(s) for s in temp_sections]
            
            for idx, section in enumerate(temp_sections):
                scores = calculate_context_relevance_scores(
                    sections_dict, 
                    idx, 
                    primary_summary, 
                    client
                )
                
                section.importance_score = scores['importance_score']
                section.preceding_context_relevance = scores['preceding_context_relevance']
                section.following_context_relevance = scores['following_context_relevance']
                
                # Generate embedding
                section.section_embedding = generate_embedding(client, section.section_content)
                
                all_sections.append(section)
        
        # Calculate totals
        total_tokens = sum(s.section_tokens for s in all_sections)
        
        status.status = 'processed'
        status.last_processed = datetime.now()
        status.processing_time = time.time() - start_time
        status.section_count = len(all_sections)
        status.primary_sections = primary_section_names
        status.total_tokens = total_tokens
        
        logger.info(f"Successfully processed {file_info['filename']} with {len(all_sections)} sections")
        
        # Create final processing summary
        if DEBUG_MODE:
            # Secondary sections table
            secondary_stats = []
            for i, section in enumerate(all_sections, 1):
                secondary_stats.append({
                    'Order': i,
                    'Primary': section.primary_section_type,
                    'Secondary': section.secondary_section_type[:40] + '...' if len(section.secondary_section_type) > 40 else section.secondary_section_type,
                    'Tokens': section.section_tokens,
                    'Importance': f"{section.importance_score:.2f}" if section.importance_score else "N/A",
                    'Prev_Ctx': f"{section.preceding_context_relevance:.2f}" if section.preceding_context_relevance else "0.00",
                    'Next_Ctx': f"{section.following_context_relevance:.2f}" if section.following_context_relevance else "0.00"
                })
            
            table = create_summary_table("FINAL SECONDARY SECTIONS", secondary_stats[:10],  # Show first 10
                                       ['Order', 'Primary', 'Secondary', 'Tokens', 'Importance', 'Prev_Ctx', 'Next_Ctx'])
            logger.info(table)
            
            if len(all_sections) > 10:
                logger.info(f"... and {len(all_sections) - 10} more sections")
            
            # Final processing stats
            primary_breakdown = {}
            for section in all_sections:
                primary_breakdown[section.primary_section_type] = primary_breakdown.get(section.primary_section_type, 0) + 1
            
            final_stats = {
                'Total Processing Time': f"{status.processing_time:.2f} seconds",
                'Bank': f"{bank_info.standardized_name} ({bank_info.ticker_region})",
                'Primary Sections': len(primary_section_names),
                'Secondary Sections': len(all_sections),
                'Total Tokens': total_tokens,
                'Average Tokens/Section': f"{total_tokens / len(all_sections):.0f}" if all_sections else "0",
                'Embedding Status': 'Generated' if not (DEBUG_MODE and DEBUG_SKIP_EMBEDDINGS) else 'Skipped (Debug)'
            }
            
            breakdown_str = ', '.join([f"{k}: {v}" for k, v in primary_breakdown.items()])
            final_stats['Section Breakdown'] = breakdown_str
            
            summary = create_processing_summary("Final Processing Results", final_stats)
            logger.info(summary)
            
            # Save final data only if intermediates enabled
            if DEBUG_SAVE_INTERMEDIATES:
                debug_save_json([asdict(s) for s in all_sections], "05_final_sections.json", "final processed sections")
        
        # Store all sections in master database (mirrors PostgreSQL schema)
        status.section_count = len(all_sections)
        logger.info(f"✓ Processing complete: {len(all_sections)} sections created")
        
        # Convert sections to DataFrame records for master database
        section_records = []
        for section in all_sections:
            # Convert embedding to string for CSV storage
            embedding_str = ','.join(map(str, section.section_embedding)) if section.section_embedding else ''
            
            section_records.append({
                'fiscal_year': section.fiscal_year,
                'quarter': section.quarter,
                'bank_name': section.bank_name,
                'ticker_region': section.ticker_region,
                'filepath': section.filepath,
                'filename': section.filename,
                'date_last_modified': section.date_last_modified,
                'primary_section_type': section.primary_section_type,
                'primary_section_summary': section.primary_section_summary,
                'secondary_section_type': section.secondary_section_type,
                'secondary_section_summary': section.secondary_section_summary,
                'section_content': section.section_content,
                'section_order': section.section_order,
                'section_tokens': section.section_tokens,
                'section_embedding': embedding_str,
                'importance_score': section.importance_score,
                'preceding_context_relevance': section.preceding_context_relevance,
                'following_context_relevance': section.following_context_relevance,
                'created_at': section.created_at,
                'updated_at': section.updated_at
            })
        
        # Return sections for storage in main function
        status.sections = section_records
        
    except Exception as e:
        logger.error(f"Error processing {file_info['filename']}: {str(e)}")
        status.status = 'error'
        status.error_message = str(e)
        status.processing_time = time.time() - start_time
    
    return status

# ========================================
# MAIN EXECUTION
# ========================================

def main():
    """Main execution function"""
    mode = "DEBUG" if DEBUG_MODE else "PRODUCTION"
    logger.info(f"Starting Bank Transcript Processor - Stage 2 ({mode} Mode)")
    logger.info("=" * 60)
    
    if DEBUG_MODE:
        logger.info(f"DEBUG MODE ENABLED:")
        logger.info(f"  - Target file: {DEBUG_TARGET_FILE or 'First available file'}")
        logger.info(f"  - Save intermediates: {DEBUG_SAVE_INTERMEDIATES}")
        logger.info(f"  - Skip embeddings: {DEBUG_SKIP_EMBEDDINGS}")
        logger.info(f"  - Max pages: {DEBUG_MAX_PAGES or 'All pages'}")
        logger.info(f"  - Output directory: {DEBUG_OUTPUT_DIR}")
        logger.info("=" * 60)
    
    errors = []
    stats = {
        'new_files': 0,
        'modified_files': 0,
        'deleted_files': 0,
        'processed_files': 0,
        'failed_files': 0,
        'total_sections': 0
    }
    
    nas_conn = None
    
    try:
        # Initialize OpenAI client
        openai_client = get_openai_client()
        
        # Connect to NAS
        logger.info(f"Connecting to NAS ({DEST_NAS_IP})...")
        nas_conn = create_smb_connection(DEST_NAS_IP, NAS_USERNAME, NAS_PASSWORD, DEST_NAS_PORT)
        
        # Read master database
        master_df = read_master_database(nas_conn)
        
        # Scan for transcript files
        current_files = scan_for_transcripts(nas_conn)
        
        # Compare with master database
        file_comparison = compare_files_with_master(current_files, master_df)
        
        # Update stats
        stats['new_files'] = len(file_comparison['new_files'])
        stats['modified_files'] = len(file_comparison['modified_files'])
        stats['deleted_files'] = len(file_comparison['deleted_files'])
        
        logger.info(f"File comparison results:")
        logger.info(f"  - New files: {stats['new_files']}")
        logger.info(f"  - Modified files: {stats['modified_files']}")
        logger.info(f"  - Deleted files: {stats['deleted_files']}")
        
        # Remove deleted files from master database
        if file_comparison['deleted_files']:
            logger.info(f"Removing {len(file_comparison['deleted_files'])} deleted files from master database")
            master_df = master_df[~master_df['filepath'].isin(file_comparison['deleted_files'])]
        
        # Process new and modified files
        files_to_process = file_comparison['new_files'] + file_comparison['modified_files']
        
        # In debug mode, filter to target file or take first file
        if DEBUG_MODE and files_to_process:
            if DEBUG_TARGET_FILE:
                # Look for specific target file
                target_files = [f for f in files_to_process if DEBUG_TARGET_FILE in f['filepath']]
                if target_files:
                    files_to_process = [target_files[0]]
                    logger.info(f"DEBUG: Found target file: {files_to_process[0]['filepath']}")
                else:
                    logger.warning(f"DEBUG: Target file '{DEBUG_TARGET_FILE}' not found in files to process")
                    if files_to_process:
                        files_to_process = [files_to_process[0]]
                        logger.info(f"DEBUG: Using first available file: {files_to_process[0]['filepath']}")
                    else:
                        logger.error("DEBUG: No files available for processing")
                        return
            else:
                # Take first file
                files_to_process = [files_to_process[0]]
                logger.info(f"DEBUG: Using first available file: {files_to_process[0]['filepath']}")
        
        if files_to_process:
            logger.info(f"\nProcessing {len(files_to_process)} files (batch size: {BATCH_SIZE})...")
            
            # Remove existing records for modified files
            if file_comparison['modified_files']:
                modified_paths = [f['filepath'] for f in file_comparison['modified_files']]
                master_df = master_df[~master_df['filepath'].isin(modified_paths)]
            
            # Process files in batches
            for i in range(0, len(files_to_process), BATCH_SIZE):
                batch = files_to_process[i:i + BATCH_SIZE]
                logger.info(f"\nProcessing batch {i//BATCH_SIZE + 1} of {(len(files_to_process) + BATCH_SIZE - 1)//BATCH_SIZE}")
                
                for file_info in batch:
                    # Process file
                    status = process_transcript(nas_conn, file_info, openai_client)
                    
                    # Add sections to master database if processing succeeded
                    if status.status == 'processed' and status.sections:
                        # Add all section records to master database
                        sections_df = pd.DataFrame(status.sections)
                        master_df = pd.concat([master_df, sections_df], ignore_index=True)
                        
                        if DEBUG_MODE:
                            logger.info(f"✓ Added {len(status.sections)} section records to master database")
                    else:
                        # For failed processing, log error
                        if status.status == 'error':
                            logger.error(f"✗ Failed to process {file_info['filename']}: {status.error_message}")
                    
                    # Update stats
                    if status.status == 'processed':
                        stats['processed_files'] += 1
                        stats['total_sections'] += status.section_count or 0
                    else:
                        stats['failed_files'] += 1
                        errors.append(f"{status.filepath}: {status.error_message}")
                    
                    # Save master database after each file
                    update_master_database(nas_conn, master_df)
        
        else:
            logger.info("\nNo files to process - all transcripts are up to date")
        
        # Write error log if needed
        if errors:
            write_error_log(nas_conn, errors)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING COMPLETE - Summary:")
        logger.info(f"  - New files processed: {stats['new_files'] - stats['failed_files']}")
        logger.info(f"  - Modified files processed: {stats['modified_files']}")
        logger.info(f"  - Deleted files removed: {stats['deleted_files']}")
        logger.info(f"  - Failed files: {stats['failed_files']}")
        logger.info(f"  - Total processed successfully: {stats['processed_files']}")
        logger.info(f"  - Total sections created: {stats['total_sections']}")
        
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        errors.append(f"Critical error: {str(e)}")
        
    finally:
        # Clean up connections
        if nas_conn:
            nas_conn.close()
        
        logger.info("\nScript execution completed")

if __name__ == "__main__":
    main()