#!/usr/bin/env python3
"""
Bank Earnings Call Transcript Processor - Stage 2: File Management
Handles file comparison and master database initialization

This stage:
1. Checks if master database exists, creates if not
2. Scans NAS for transcript files
3. Compares files with master database
4. Outputs organized file lists for processing stages

Outputs to database_refresh folder:
- 01_master_database.csv (created if not exists)
- 02_files_to_add.json (new + modified files)  
- 03_files_to_delete.json (files to remove from master)
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from io import BytesIO

# Core libraries
import pandas as pd

# pysmb imports
try:
    from smb.SMBConnection import SMBConnection
    from smb.smb_structs import OperationFailure
except ImportError:
    print("ERROR: pysmb not installed. Please run: pip install pysmb")
    sys.exit(1)

# ========================================
# CONFIGURATION
# ========================================

# NAS Authentication
NAS_USERNAME = "your_username"
NAS_PASSWORD = "your_password"
CLIENT_MACHINE_NAME = "PYTHON_SCRIPT"
SERVER_MACHINE_NAME = "NAS_SERVER"

# NAS Configuration
NAS_IP = "192.168.2.100"
NAS_PORT = 445
NAS_CONFIG = {
    "share": "wrkgrp33",
    "source_path": "Finance Data and Analytics/DSA/AEGIS/Transcripts/database_refresh",
    "output_path": "Finance Data and Analytics/DSA/AEGIS/Transcripts",
    "master_db_filename": "master_database.csv",
    "refresh_folder": "database_refresh"
}

# Processing Configuration
VALID_YEAR_RANGE = (2020, 2031)
VALID_QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
FILE_EXTENSIONS = [".pdf", ".PDF"]

# ========================================
# LOGGING SETUP
# ========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress SMB logging
logging.getLogger('nmb.NetBIOS').setLevel(logging.ERROR)
logging.getLogger('smb.SMBConnection').setLevel(logging.ERROR)
logging.getLogger('smb.smb_structs').setLevel(logging.ERROR)

# ========================================
# NAS OPERATIONS
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
        logger.info(f"Uploaded file: {file_path}")
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
# MASTER DATABASE OPERATIONS
# ========================================

def get_master_database_path() -> str:
    """Get the full path to master database"""
    return f"{NAS_CONFIG['output_path']}/{NAS_CONFIG['master_db_filename']}"

def check_master_database_exists(conn: SMBConnection) -> bool:
    """Check if master database exists"""
    try:
        master_path = get_master_database_path()
        conn.getAttributes(NAS_CONFIG['share'], master_path)
        return True
    except OperationFailure:
        return False

def create_master_database(conn: SMBConnection) -> pd.DataFrame:
    """Create empty master database with proper schema"""
    logger.info("Creating new master database...")
    
    # Create empty dataframe with PostgreSQL schema
    df = pd.DataFrame(columns=[
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
    
    # Save to NAS
    master_path = get_master_database_path()
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    upload_file_to_nas(conn, NAS_CONFIG['share'], master_path, csv_content)
    logger.info(f"Master database created at: {master_path}")
    
    return df

def read_master_database(conn: SMBConnection) -> pd.DataFrame:
    """Read master database, create if it doesn't exist"""
    if not check_master_database_exists(conn):
        logger.info("Master database does not exist")
        return create_master_database(conn)
    
    try:
        master_path = get_master_database_path()
        csv_content = download_file_from_nas(conn, NAS_CONFIG['share'], master_path)
        df = pd.read_csv(csv_content)
        logger.info(f"Loaded master database: {len(df)} section records")
        return df
    except Exception as e:
        logger.error(f"Error reading master database: {str(e)}")
        raise

# ========================================
# FILE SCANNING
# ========================================

def scan_for_transcripts(conn: SMBConnection) -> Dict[str, Dict]:
    """Scan NAS for transcript files"""
    logger.info("Scanning NAS for transcript files...")
    transcripts = {}
    
    try:
        base_items = conn.listPath(NAS_CONFIG['share'], NAS_CONFIG['source_path'])
        
        for year_item in base_items:
            if year_item.filename in ['.', '..'] or not year_item.isDirectory:
                continue
            
            try:
                year = int(year_item.filename)
                if not (VALID_YEAR_RANGE[0] <= year <= VALID_YEAR_RANGE[1]):
                    continue
            except ValueError:
                continue
            
            year_path = f"{NAS_CONFIG['source_path']}/{year_item.filename}"
            
            try:
                quarter_items = conn.listPath(NAS_CONFIG['share'], year_path)
                
                for quarter_item in quarter_items:
                    if quarter_item.filename in ['.', '..'] or not quarter_item.isDirectory:
                        continue
                    
                    if quarter_item.filename.upper() not in VALID_QUARTERS:
                        continue
                    
                    quarter_path = f"{year_path}/{quarter_item.filename}"
                    
                    try:
                        files = conn.listPath(NAS_CONFIG['share'], quarter_path)
                        
                        for file_item in files:
                            if file_item.isDirectory:
                                continue
                            
                            if any(file_item.filename.endswith(ext) for ext in FILE_EXTENSIONS):
                                file_path = f"{quarter_path}/{file_item.filename}"
                                
                                # Convert last_write_time to datetime
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

# ========================================
# FILE COMPARISON
# ========================================

def compare_files_with_master(current_files: Dict, master_df: pd.DataFrame) -> Dict[str, List]:
    """Compare current files with master database"""
    logger.info("Comparing files with master database...")
    
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
            current_modified_dt = pd.to_datetime(current_modified)
            
            if current_modified_dt > master_modified:
                result['modified_files'].append(current_files[filepath])
    
    # Find deleted files
    deleted_file_paths = existing_files - current_file_paths
    result['deleted_files'] = list(deleted_file_paths)
    
    logger.info(f"Comparison results:")
    logger.info(f"  - New files: {len(result['new_files'])}")
    logger.info(f"  - Modified files: {len(result['modified_files'])}")
    logger.info(f"  - Deleted files: {len(result['deleted_files'])}")
    
    return result

# ========================================
# OUTPUT GENERATION
# ========================================

def save_output_to_nas(conn: SMBConnection, filename: str, data: any, data_type: str = "json"):
    """Save output file to database_refresh folder"""
    refresh_path = f"{NAS_CONFIG['output_path']}/{NAS_CONFIG['refresh_folder']}"
    
    # Ensure refresh directory exists
    ensure_directory_exists(conn, NAS_CONFIG['share'], refresh_path)
    
    # Prepare content
    if data_type == "json":
        content = json.dumps(data, indent=2, default=str).encode('utf-8')
    elif data_type == "csv":
        csv_buffer = BytesIO()
        data.to_csv(csv_buffer, index=False)
        content = csv_buffer.getvalue()
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    # Upload file
    file_path = f"{refresh_path}/{filename}"
    upload_file_to_nas(conn, NAS_CONFIG['share'], file_path, content)

def copy_master_to_refresh(conn: SMBConnection, master_df: pd.DataFrame):
    """Copy master database to refresh folder"""
    save_output_to_nas(conn, "01_master_database.csv", master_df, "csv")

# ========================================
# MAIN EXECUTION
# ========================================

def main():
    """Main execution function"""
    logger.info("Starting Stage 2: File Management")
    logger.info("=" * 50)
    
    nas_conn = None
    
    try:
        # Connect to NAS
        logger.info(f"Connecting to NAS ({NAS_IP})...")
        nas_conn = create_smb_connection(NAS_IP, NAS_USERNAME, NAS_PASSWORD, NAS_PORT)
        
        # Step 1: Read/Create master database
        logger.info("\nStep 1: Master Database Check")
        master_df = read_master_database(nas_conn)
        
        # Step 2: Scan for transcript files
        logger.info("\nStep 2: Scanning for Transcript Files")
        current_files = scan_for_transcripts(nas_conn)
        
        # Step 3: Compare files with master
        logger.info("\nStep 3: Comparing Files with Master Database")
        file_comparison = compare_files_with_master(current_files, master_df)
        
        # Step 4: Generate output files
        logger.info("\nStep 4: Generating Output Files")
        
        # Copy master database to refresh folder
        copy_master_to_refresh(nas_conn, master_df)
        logger.info("✓ Copied master database to refresh folder")
        
        # Save files to add (new + modified)
        files_to_add = file_comparison['new_files'] + file_comparison['modified_files']
        save_output_to_nas(nas_conn, "02_files_to_add.json", files_to_add)
        logger.info(f"✓ Saved {len(files_to_add)} files to add")
        
        # Save files to delete
        save_output_to_nas(nas_conn, "03_files_to_delete.json", file_comparison['deleted_files'])
        logger.info(f"✓ Saved {len(file_comparison['deleted_files'])} files to delete")
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 2 COMPLETE - Summary:")
        logger.info(f"  - Master database: {'Created' if len(master_df) == 0 else 'Loaded'}")
        logger.info(f"  - Files to add: {len(files_to_add)}")
        logger.info(f"  - Files to delete: {len(file_comparison['deleted_files'])}")
        logger.info(f"  - Current files in NAS: {len(current_files)}")
        logger.info(f"  - Files in master database: {len(set(master_df['filepath'].tolist()) if len(master_df) > 0 else [])}")
        
        # Output files created
        logger.info("\nOutput files created in database_refresh folder:")
        logger.info("  - 01_master_database.csv")
        logger.info("  - 02_files_to_add.json")
        logger.info("  - 03_files_to_delete.json")
        
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        raise
        
    finally:
        # Clean up connections
        if nas_conn:
            nas_conn.close()
        
        logger.info("\nStage 2 execution completed")

if __name__ == "__main__":
    main()