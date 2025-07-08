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
- 01_files_to_add.json (new + modified files)  
- 02_files_to_delete.json (files to remove from master)

Master database location: database/master_database.csv (created if not exists)
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
    "source_path": "Finance Data and Analytics/DSA/AEGIS/Transcripts/data",
    "output_path": "Finance Data and Analytics/DSA/AEGIS/Transcripts",
    "master_db_filename": "master_database.csv",
    "database_folder": "database",
    "refresh_outputs_folder": "database_refresh",
    "logs_folder": "database_refresh/logs"
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
    return f"{NAS_CONFIG['output_path']}/{NAS_CONFIG['database_folder']}/{NAS_CONFIG['master_db_filename']}"

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
    
    # Ensure database directory exists
    database_path = f"{NAS_CONFIG['output_path']}/{NAS_CONFIG['database_folder']}"
    ensure_directory_exists(conn, NAS_CONFIG['share'], database_path)
    
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
    """Save output file to refresh_outputs folder"""
    refresh_outputs_path = f"{NAS_CONFIG['output_path']}/{NAS_CONFIG['refresh_outputs_folder']}"
    
    # Ensure refresh_outputs directory exists
    ensure_directory_exists(conn, NAS_CONFIG['share'], refresh_outputs_path)
    
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
    file_path = f"{refresh_outputs_path}/{filename}"
    upload_file_to_nas(conn, NAS_CONFIG['share'], file_path, content)

# Removed copy_master_to_refresh_outputs function - master DB should stay in database/ folder only

# ========================================
# ERROR LOGGING
# ========================================

def write_error_log(conn: SMBConnection, errors: List[str], warnings: List[str] = None):
    """Write error and warning log to NAS logs folder"""
    if not errors and not warnings:
        return
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"stage2_file_management_{timestamp}.log"
        logs_path = f"{NAS_CONFIG['output_path']}/{NAS_CONFIG['logs_folder']}"
        log_file_path = f"{logs_path}/{log_filename}"
        
        # Ensure logs directory exists
        ensure_directory_exists(conn, NAS_CONFIG['share'], logs_path)
        
        # Build log content
        log_content = f"Stage 2 File Management Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        log_content += "=" * 60 + "\n\n"
        
        if errors:
            log_content += f"ERRORS ({len(errors)}):\n"
            log_content += "-" * 20 + "\n"
            for i, error in enumerate(errors, 1):
                log_content += f"{i}. {error}\n"
            log_content += "\n"
        
        if warnings:
            log_content += f"WARNINGS ({len(warnings)}):\n"
            log_content += "-" * 20 + "\n"
            for i, warning in enumerate(warnings, 1):
                log_content += f"{i}. {warning}\n"
            log_content += "\n"
        
        # Upload log file
        upload_file_to_nas(conn, NAS_CONFIG['share'], log_file_path, log_content.encode('utf-8'))
        logger.info(f"Error log written to: {log_file_path}")
        
    except Exception as e:
        logger.error(f"Failed to write error log: {str(e)}")

def write_summary_log(conn: SMBConnection, summary_stats: Dict):
    """Write processing summary log to NAS logs folder"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"stage2_summary_{timestamp}.log"
        logs_path = f"{NAS_CONFIG['output_path']}/{NAS_CONFIG['logs_folder']}"
        log_file_path = f"{logs_path}/{log_filename}"
        
        # Ensure logs directory exists
        ensure_directory_exists(conn, NAS_CONFIG['share'], logs_path)
        
        # Build summary content
        log_content = f"Stage 2 File Management Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        log_content += "=" * 60 + "\n\n"
        
        log_content += "PROCESSING RESULTS:\n"
        log_content += "-" * 20 + "\n"
        for key, value in summary_stats.items():
            log_content += f"{key}: {value}\n"
        log_content += "\n"
        
        log_content += "FILES PROCESSED:\n"
        log_content += "-" * 20 + "\n"
        if 'files_to_add' in summary_stats:
            log_content += f"Files to add: {summary_stats['files_to_add']}\n"
        if 'files_to_delete' in summary_stats:
            log_content += f"Files to delete: {summary_stats['files_to_delete']}\n"
        if 'total_nas_files' in summary_stats:
            log_content += f"Total NAS files: {summary_stats['total_nas_files']}\n"
        if 'master_db_records' in summary_stats:
            log_content += f"Master DB records: {summary_stats['master_db_records']}\n"
        
        # Upload summary log
        upload_file_to_nas(conn, NAS_CONFIG['share'], log_file_path, log_content.encode('utf-8'))
        logger.info(f"Summary log written to: {log_file_path}")
        
    except Exception as e:
        logger.error(f"Failed to write summary log: {str(e)}")
        # Don't raise here as this is just logging

# ========================================
# MAIN EXECUTION
# ========================================

def main():
    """Main execution function"""
    start_time = datetime.now()
    logger.info("Starting Stage 2: File Management")
    logger.info("=" * 50)
    
    nas_conn = None
    errors = []
    warnings = []
    
    try:
        # Connect to NAS
        logger.info(f"Connecting to NAS ({NAS_IP})...")
        try:
            nas_conn = create_smb_connection(NAS_IP, NAS_USERNAME, NAS_PASSWORD, NAS_PORT)
            logger.info("✓ NAS connection established successfully")
        except Exception as e:
            error_msg = f"Failed to connect to NAS: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise
        
        # Step 1: Read/Create master database
        logger.info("\nStep 1: Master Database Check")
        try:
            master_df = read_master_database(nas_conn)
            if len(master_df) == 0:
                logger.info("✓ Master database created (was empty)")
            else:
                logger.info(f"✓ Master database loaded with {len(master_df)} records")
        except Exception as e:
            error_msg = f"Failed to read/create master database: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise
        
        # Step 2: Scan for transcript files
        logger.info("\nStep 2: Scanning for Transcript Files")
        try:
            current_files = scan_for_transcripts(nas_conn)
            logger.info(f"✓ Found {len(current_files)} transcript files in NAS")
            
            if len(current_files) == 0:
                warning_msg = "No transcript files found in NAS database_refresh folder"
                logger.warning(warning_msg)
                warnings.append(warning_msg)
                
        except Exception as e:
            error_msg = f"Failed to scan transcript files: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise
        
        # Step 3: Compare files with master
        logger.info("\nStep 3: Comparing Files with Master Database")
        try:
            file_comparison = compare_files_with_master(current_files, master_df)
            logger.info("✓ File comparison completed successfully")
            
            # Log detailed comparison results
            logger.info(f"  - New files found: {len(file_comparison['new_files'])}")
            logger.info(f"  - Modified files found: {len(file_comparison['modified_files'])}")
            logger.info(f"  - Deleted files found: {len(file_comparison['deleted_files'])}")
            
            if len(file_comparison['new_files']) == 0 and len(file_comparison['modified_files']) == 0:
                logger.info("  - No files require processing")
            
        except Exception as e:
            error_msg = f"Failed to compare files with master database: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise
        
        # Step 4: Generate output files
        logger.info("\nStep 4: Generating Output Files")
        try:
            # Save files to add (new + modified)
            files_to_add = file_comparison['new_files'] + file_comparison['modified_files']
            save_output_to_nas(nas_conn, "01_files_to_add.json", files_to_add)
            logger.info(f"✓ Saved {len(files_to_add)} files to add")
            
            # Save files to delete
            save_output_to_nas(nas_conn, "02_files_to_delete.json", file_comparison['deleted_files'])
            logger.info(f"✓ Saved {len(file_comparison['deleted_files'])} files to delete")
            
            logger.info("✓ All output files generated successfully")
            
        except Exception as e:
            error_msg = f"Failed to generate output files: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise
        
        # Calculate execution time
        execution_time = datetime.now() - start_time
        
        # Prepare summary statistics
        summary_stats = {
            'execution_time': str(execution_time),
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'master_database_status': 'Created' if len(master_df) == 0 else 'Loaded',
            'master_db_records': len(set(master_df['filepath'].tolist()) if len(master_df) > 0 else []),
            'total_nas_files': len(current_files),
            'files_to_add': len(files_to_add),
            'files_to_delete': len(file_comparison['deleted_files']),
            'new_files': len(file_comparison['new_files']),
            'modified_files': len(file_comparison['modified_files']),
            'errors': len(errors),
            'warnings': len(warnings)
        }
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 2 COMPLETE - Summary:")
        logger.info(f"  - Execution time: {execution_time}")
        logger.info(f"  - Master database: {summary_stats['master_database_status']}")
        logger.info(f"  - Files to add: {len(files_to_add)} (New: {len(file_comparison['new_files'])}, Modified: {len(file_comparison['modified_files'])})")
        logger.info(f"  - Files to delete: {len(file_comparison['deleted_files'])}")
        logger.info(f"  - Current files in NAS: {len(current_files)}")
        logger.info(f"  - Files in master database: {len(set(master_df['filepath'].tolist()) if len(master_df) > 0 else [])}")
        logger.info(f"  - Errors: {len(errors)}")
        logger.info(f"  - Warnings: {len(warnings)}")
        
        # Output files created
        logger.info("\nOutput files created in refresh_outputs folder:")
        logger.info("  - 01_files_to_add.json")
        logger.info("  - 02_files_to_delete.json")
        logger.info(f"\nMaster database location: {NAS_CONFIG['database_folder']}/{NAS_CONFIG['master_db_filename']}")
        
        # Write summary log
        write_summary_log(nas_conn, summary_stats)
        
    except Exception as e:
        error_msg = f"Critical error: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        
        # Try to write error log even if there was a critical error
        if nas_conn:
            write_error_log(nas_conn, errors, warnings)
        
        raise
        
    finally:
        # Write error log if there were any issues
        if nas_conn and (errors or warnings):
            write_error_log(nas_conn, errors, warnings)
        
        # Clean up connections
        if nas_conn:
            nas_conn.close()
        
        logger.info("\nStage 2 execution completed")

if __name__ == "__main__":
    main()