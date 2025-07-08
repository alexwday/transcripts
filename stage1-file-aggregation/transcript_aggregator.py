#!/usr/bin/env python3
"""
Bank Earnings Call Transcript Aggregator
Monitors NAS locations for new transcript files and copies them to a centralized location
"""

import os
import sys
import time
import logging
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import hashlib

# pysmb imports
try:
    from smb.SMBConnection import SMBConnection
    from smb.smb_structs import OperationFailure
except ImportError:
    print("ERROR: pysmb not installed. Please run: pip install pysmb")
    sys.exit(1)

# ========================================
# CONFIGURATION - MODIFY AS NEEDED
# ========================================

# NAS Authentication (same for all connections)
NAS_USERNAME = "your_username"
NAS_PASSWORD = "your_password"
CLIENT_MACHINE_NAME = "PYTHON_SCRIPT"  # Can be any name
SERVER_MACHINE_NAME = "NAS_SERVER"     # Can be any name

# Source NAS Configuration
SOURCE_NAS_IP = "192.168.1.100"  # Replace with actual IP
SOURCE_NAS_PORT = 445

# Canadian Banks Source
CANADIAN_CONFIG = {
    "name": "Canadian Banks",
    "share": "wrkgrp30",
    "base_path": "Investor Relations/5. Benchmarking/Peer Benchmarking/Canadian Peer Benchmarking",
    "target_folders": ["final transcripts"]  # Case-insensitive
}

# US Banks Source  
US_CONFIG = {
    "name": "US Banks",
    "share": "wrkgrp30",
    "base_path": "Investor Relations/5. Benchmarking/Peer Benchmarking/US Peer Benchmarking",
    "target_folders": ["clean transcript", "clean transcripts"]  # Case-insensitive
}

# Destination NAS Configuration
DEST_NAS_IP = "192.168.2.100"  # Replace with actual IP
DEST_NAS_PORT = 445
DEST_CONFIG = {
    "share": "wrkgrp33",
    "base_path": "Finance Data and Analytics/DSA/AEGIS/Transcripts/data",
    "output_path": "Finance Data and Analytics/DSA/AEGIS/Transcripts",
    "logs_folder": "database_refresh/logs"
}

# Processing Configuration
VALID_YEAR_RANGE = (2020, 2031)  # Updated to start from 2020
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

# Suppress SMB logging which is very verbose
logging.getLogger('nmb.NetBIOS').setLevel(logging.ERROR)
logging.getLogger('smb.SMBConnection').setLevel(logging.ERROR)
logging.getLogger('smb.smb_structs').setLevel(logging.ERROR)

# ========================================
# HELPER FUNCTIONS
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

def is_valid_year_folder(folder_name: str) -> bool:
    """Check if folder name is a valid fiscal year"""
    try:
        year = int(folder_name)
        return VALID_YEAR_RANGE[0] <= year <= VALID_YEAR_RANGE[1]
    except ValueError:
        return False

def is_quarter_folder(folder_name: str) -> bool:
    """Check if folder matches any quarter pattern (flexible matching)"""
    folder_lower = folder_name.lower()
    # Match Q1, Q121, Q1 2020, Q1 2022, etc.
    return (folder_lower.startswith('q1') or folder_lower.startswith('q2') or 
            folder_lower.startswith('q3') or folder_lower.startswith('q4'))

def get_normalized_quarter(folder_name: str) -> str:
    """Get normalized quarter name (Q1, Q2, Q3, Q4) from any quarter folder name"""
    folder_lower = folder_name.lower()
    if folder_lower.startswith('q1'):
        return 'Q1'
    elif folder_lower.startswith('q2'):
        return 'Q2'
    elif folder_lower.startswith('q3'):
        return 'Q3'
    elif folder_lower.startswith('q4'):
        return 'Q4'
    return folder_name  # fallback

def is_target_folder(folder_name: str, target_patterns: List[str]) -> bool:
    """Check if folder matches any of the target patterns (case-insensitive and flexible)"""
    folder_lower = folder_name.lower()
    
    # First check exact matches (for backward compatibility)
    if any(folder_lower == pattern.lower() for pattern in target_patterns):
        return True
    
    # Then check flexible patterns for transcript folders
    # Matches: "clean final transcripts", "final transcripts", etc.
    if ('transcript' in folder_lower and 
        ('final' in folder_lower or 'clean' in folder_lower)):
        return True
    
    return False

def get_file_info(conn: SMBConnection, share: str, file_path: str) -> Dict:
    """Get file information including last modified time"""
    try:
        attributes = conn.getAttributes(share, file_path)
        return {
            'path': file_path,
            'last_modified': attributes.last_write_time,
            'size': attributes.file_size
        }
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {str(e)}")
        return None

def scan_destination(conn: SMBConnection, share: str, base_path: str) -> Dict[str, Dict]:
    """Scan destination folder and catalog existing files"""
    logger.info("Scanning destination folder for existing files...")
    existing_files = {}
    files_count = 0
    
    try:
        # Recursively scan the destination
        def scan_folder(path):
            nonlocal files_count
            try:
                items = conn.listPath(share, path)
                for item in items:
                    if item.filename in ['.', '..']:
                        continue
                    
                    item_path = f"{path}/{item.filename}" if path else item.filename
                    
                    if item.isDirectory:
                        scan_folder(item_path)
                    else:
                        if any(item.filename.endswith(ext) for ext in FILE_EXTENSIONS):
                            # Create a standardized key: year/quarter/filename
                            path_parts = item_path.split('/')
                            if len(path_parts) >= 3:
                                year = None
                                quarter = None
                                
                                # Find year and quarter in path
                                for part in path_parts:
                                    if is_valid_year_folder(part):
                                        year = part
                                    elif is_quarter_folder(part):
                                        quarter = get_normalized_quarter(part)
                                
                                if year and quarter:
                                    key = f"{year}/{quarter}/{item.filename}"
                                    existing_files[key] = {
                                        'path': item_path,
                                        'last_modified': item.last_write_time,
                                        'size': item.file_size
                                    }
                                    files_count += 1
            except Exception as e:
                logger.error(f"Error scanning folder {path}: {str(e)}")
        
        scan_folder(base_path)
        logger.info(f"Found {files_count} existing transcript files in destination")
        
    except Exception as e:
        logger.error(f"Error scanning destination: {str(e)}")
    
    return existing_files

def scan_source(conn: SMBConnection, config: Dict, target_folders: List[str]) -> Dict[str, Dict]:
    """Scan source folder for transcript files"""
    logger.info(f"Scanning {config['name']} source folder...")
    source_files = {}
    files_count = 0
    
    try:
        base_items = conn.listPath(config['share'], config['base_path'])
        
        for year_item in base_items:
            if year_item.filename in ['.', '..'] or not year_item.isDirectory:
                continue
            
            if not is_valid_year_folder(year_item.filename):
                continue
            
            year = year_item.filename
            year_path = f"{config['base_path']}/{year}"
            
            try:
                quarter_items = conn.listPath(config['share'], year_path)
                
                for quarter_item in quarter_items:
                    if quarter_item.filename in ['.', '..'] or not quarter_item.isDirectory:
                        continue
                    
                    if not is_quarter_folder(quarter_item.filename):
                        continue
                    
                    quarter = get_normalized_quarter(quarter_item.filename)
                    quarter_path = f"{year_path}/{quarter_item.filename}"  # Use actual folder name for path
                    
                    try:
                        transcript_folders = conn.listPath(config['share'], quarter_path)
                        
                        for folder_item in transcript_folders:
                            if folder_item.filename in ['.', '..'] or not folder_item.isDirectory:
                                continue
                            
                            if is_target_folder(folder_item.filename, target_folders):
                                transcript_path = f"{quarter_path}/{folder_item.filename}"
                                
                                try:
                                    files = conn.listPath(config['share'], transcript_path)
                                    
                                    for file_item in files:
                                        if file_item.filename in ['.', '..'] or file_item.isDirectory:
                                            continue
                                        
                                        if any(file_item.filename.endswith(ext) for ext in FILE_EXTENSIONS):
                                            file_path = f"{transcript_path}/{file_item.filename}"
                                            key = f"{year}/{quarter}/{file_item.filename}"
                                            
                                            source_files[key] = {
                                                'path': file_path,
                                                'share': config['share'],
                                                'last_modified': file_item.last_write_time,
                                                'size': file_item.file_size,
                                                'source': config['name']
                                            }
                                            files_count += 1
                                            
                                except Exception as e:
                                    logger.error(f"Error listing transcript files in {transcript_path}: {str(e)}")
                                    
                    except Exception as e:
                        logger.error(f"Error listing folders in {quarter_path}: {str(e)}")
                        
            except Exception as e:
                logger.error(f"Error listing quarters in {year_path}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error scanning {config['name']} source: {str(e)}")
    
    logger.info(f"Found {files_count} transcript files in {config['name']} source")
    return source_files

def ensure_directory_exists(conn: SMBConnection, share: str, path: str):
    """Ensure directory exists, create if necessary"""
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
            # Check if directory exists
            conn.listPath(share, current_path)
        except OperationFailure:
            # Directory doesn't exist, create it
            try:
                conn.createDirectory(share, current_path)
                logger.info(f"Created directory: {current_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {current_path}: {str(e)}")
                raise

def copy_file(source_conn: SMBConnection, dest_conn: SMBConnection, 
              source_share: str, dest_share: str, source_path: str, dest_path: str) -> bool:
    """Copy file from source to destination"""
    try:
        # Create temporary file-like object to hold the data
        from io import BytesIO
        temp_file = BytesIO()
        
        # Read file from source
        source_conn.retrieveFile(source_share, source_path, temp_file)
        
        # Ensure destination directory exists
        dest_dir = '/'.join(dest_path.split('/')[:-1])
        ensure_directory_exists(dest_conn, dest_share, dest_dir)
        
        # Reset file pointer to beginning
        temp_file.seek(0)
        
        # Write file to destination
        dest_conn.storeFile(dest_share, dest_path, temp_file)
        
        # Clean up
        temp_file.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to copy {source_path} to {dest_path}: {str(e)}")
        return False

def write_error_log(dest_conn: SMBConnection, errors: List[str], warnings: List[str] = None):
    """Write error and warning log to NAS logs folder"""
    if not errors and not warnings:
        return
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"stage1_transcript_aggregator_{timestamp}.log"
        logs_path = f"{DEST_CONFIG['output_path']}/{DEST_CONFIG['logs_folder']}"
        log_file_path = f"{logs_path}/{log_filename}"
        
        # Ensure logs directory exists
        ensure_directory_exists(dest_conn, DEST_CONFIG['share'], logs_path)
        
        # Build log content
        log_content = f"Stage 1 Transcript Aggregator Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
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
        from io import BytesIO
        log_file = BytesIO(log_content.encode('utf-8'))
        dest_conn.storeFile(DEST_CONFIG['share'], log_file_path, log_file)
        log_file.close()
        logger.info(f"Error log written to: {log_file_path}")
        
    except Exception as e:
        logger.error(f"Failed to write error log: {str(e)}")

def write_summary_log(dest_conn: SMBConnection, summary_stats: Dict):
    """Write processing summary log to NAS logs folder"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"stage1_summary_{timestamp}.log"
        logs_path = f"{DEST_CONFIG['output_path']}/{DEST_CONFIG['logs_folder']}"
        log_file_path = f"{logs_path}/{log_filename}"
        
        # Ensure logs directory exists
        ensure_directory_exists(dest_conn, DEST_CONFIG['share'], logs_path)
        
        # Build summary content
        log_content = f"Stage 1 Transcript Aggregator Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        log_content += "=" * 60 + "\n\n"
        
        log_content += "PROCESSING RESULTS:\n"
        log_content += "-" * 20 + "\n"
        for key, value in summary_stats.items():
            log_content += f"{key}: {value}\n"
        log_content += "\n"
        
        log_content += "FILE TRANSFER SUMMARY:\n"
        log_content += "-" * 20 + "\n"
        if 'new_files_copied' in summary_stats:
            log_content += f"New files copied: {summary_stats['new_files_copied']}\n"
        if 'updated_files_copied' in summary_stats:
            log_content += f"Updated files copied: {summary_stats['updated_files_copied']}\n"
        if 'total_source_files' in summary_stats:
            log_content += f"Total source files: {summary_stats['total_source_files']}\n"
        if 'existing_destination_files' in summary_stats:
            log_content += f"Existing destination files: {summary_stats['existing_destination_files']}\n"
        
        # Upload summary log
        from io import BytesIO
        log_file = BytesIO(log_content.encode('utf-8'))
        dest_conn.storeFile(DEST_CONFIG['share'], log_file_path, log_file)
        log_file.close()
        logger.info(f"Summary log written to: {log_file_path}")
        
    except Exception as e:
        logger.error(f"Failed to write summary log: {str(e)}")
        # Don't raise here as this is just logging

def main():
    """Main execution function"""
    start_time = datetime.now()
    logger.info("Starting Stage 1: Bank Transcript Aggregator")
    logger.info("=" * 50)
    
    errors = []
    warnings = []
    stats = {
        'new_files': 0,
        'updated_files': 0,
        'failed_copies': 0,
        'total_processed': 0
    }
    
    source_conn = None
    dest_conn = None
    
    try:
        # Step 1: Connect to source NAS
        logger.info("Step 1: Connecting to Source NAS")
        try:
            logger.info(f"Connecting to source NAS ({SOURCE_NAS_IP})...")
            source_conn = create_smb_connection(SOURCE_NAS_IP, NAS_USERNAME, NAS_PASSWORD, SOURCE_NAS_PORT)
            logger.info("✓ Source NAS connection established successfully")
        except Exception as e:
            error_msg = f"Failed to connect to source NAS: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise
        
        # Step 2: Connect to destination NAS
        logger.info("\nStep 2: Connecting to Destination NAS")
        try:
            logger.info(f"Connecting to destination NAS ({DEST_NAS_IP})...")
            dest_conn = create_smb_connection(DEST_NAS_IP, NAS_USERNAME, NAS_PASSWORD, DEST_NAS_PORT)
            logger.info("✓ Destination NAS connection established successfully")
        except Exception as e:
            error_msg = f"Failed to connect to destination NAS: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise
        
        # Step 3: Scan destination for existing files
        logger.info("\nStep 3: Scanning Destination for Existing Files")
        try:
            existing_files = scan_destination(dest_conn, DEST_CONFIG['share'], DEST_CONFIG['base_path'])
            logger.info(f"✓ Found {len(existing_files)} existing files in destination")
        except Exception as e:
            error_msg = f"Failed to scan destination: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise
        
        # Step 4: Scan source locations
        logger.info("\nStep 4: Scanning Source Locations")
        all_source_files = {}
        
        try:
            # Canadian banks
            logger.info("Scanning Canadian banks...")
            canadian_files = scan_source(source_conn, CANADIAN_CONFIG, CANADIAN_CONFIG['target_folders'])
            all_source_files.update(canadian_files)
            logger.info(f"✓ Found {len(canadian_files)} Canadian bank files")
            
            # US banks
            logger.info("Scanning US banks...")
            us_files = scan_source(source_conn, US_CONFIG, US_CONFIG['target_folders'])
            all_source_files.update(us_files)
            logger.info(f"✓ Found {len(us_files)} US bank files")
            
            logger.info(f"✓ Total source files found: {len(all_source_files)}")
            
            if len(all_source_files) == 0:
                warning_msg = "No transcript files found in any source locations"
                logger.warning(warning_msg)
                warnings.append(warning_msg)
                
        except Exception as e:
            error_msg = f"Failed to scan source locations: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise
        
        # Step 5: Compare and identify files to process
        logger.info("\nStep 5: Comparing Files and Identifying Changes")
        try:
            files_to_copy = []
            
            for key, source_info in all_source_files.items():
                if key not in existing_files:
                    # New file
                    files_to_copy.append((key, source_info, 'new'))
                    stats['new_files'] += 1
                elif source_info['last_modified'] > existing_files[key]['last_modified']:
                    # Updated file
                    files_to_copy.append((key, source_info, 'updated'))
                    stats['updated_files'] += 1
            
            logger.info("✓ File comparison completed successfully")
            logger.info(f"  - Files to process: {len(files_to_copy)}")
            logger.info(f"  - New files: {stats['new_files']}")
            logger.info(f"  - Updated files: {stats['updated_files']}")
            
            if len(files_to_copy) == 0:
                logger.info("  - No files require copying")
            
        except Exception as e:
            error_msg = f"Failed to compare files: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise
        
        # Step 6: Copy files
        if files_to_copy:
            logger.info("\nStep 6: Copying Files")
            try:
                logger.info(f"Starting file copy process for {len(files_to_copy)} files...")
                
                for i, (key, source_info, status) in enumerate(files_to_copy, 1):
                    year, quarter, filename = key.split('/')
                    dest_path = f"{DEST_CONFIG['base_path']}/{year}/{quarter}/{filename}"
                    
                    logger.info(f"[{i}/{len(files_to_copy)}] Copying {status} file: {filename} ({source_info['source']})")
                    
                    success = copy_file(
                        source_conn, dest_conn,
                        source_info['share'], DEST_CONFIG['share'],
                        source_info['path'], dest_path
                    )
                    
                    if success:
                        stats['total_processed'] += 1
                        logger.info(f"  ✓ Successfully copied {filename}")
                    else:
                        stats['failed_copies'] += 1
                        error_msg = f"Failed to copy: {source_info['path']} -> {dest_path}"
                        logger.error(f"  ✗ {error_msg}")
                        errors.append(error_msg)
                
                logger.info("✓ File copy process completed")
                
            except Exception as e:
                error_msg = f"Failed during file copy process: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                raise
        else:
            logger.info("\nStep 6: No Files to Copy")
            logger.info("✓ Destination is up to date")
        
        # Calculate execution time
        execution_time = datetime.now() - start_time
        
        # Calculate successful copies
        new_files_copied = max(0, stats['new_files'] - stats['failed_copies'])
        updated_files_copied = max(0, stats['updated_files'] - stats['failed_copies'])
        
        # Prepare summary statistics
        summary_stats = {
            'execution_time': str(execution_time),
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_source_files': len(all_source_files),
            'existing_destination_files': len(existing_files),
            'files_to_copy': len(files_to_copy),
            'new_files': stats['new_files'],
            'updated_files': stats['updated_files'],
            'new_files_copied': new_files_copied,
            'updated_files_copied': updated_files_copied,
            'failed_copies': stats['failed_copies'],
            'total_processed': stats['total_processed'],
            'errors': len(errors),
            'warnings': len(warnings)
        }
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 1 COMPLETE - Summary:")
        logger.info(f"  - Execution time: {execution_time}")
        logger.info(f"  - Total source files: {len(all_source_files)}")
        logger.info(f"  - Existing destination files: {len(existing_files)}")
        logger.info(f"  - New files copied: {new_files_copied}")
        logger.info(f"  - Updated files copied: {updated_files_copied}")
        logger.info(f"  - Failed copies: {stats['failed_copies']}")
        logger.info(f"  - Total processed successfully: {stats['total_processed']}")
        logger.info(f"  - Errors: {len(errors)}")
        logger.info(f"  - Warnings: {len(warnings)}")
        
        # Write summary log
        write_summary_log(dest_conn, summary_stats)
        
    except Exception as e:
        error_msg = f"Critical error: {str(e)}"
        logger.error(error_msg)
        errors.append(error_msg)
        
        # Try to write error log even if there was a critical error
        if dest_conn:
            write_error_log(dest_conn, errors, warnings)
        
        raise
        
    finally:
        # Write error log if there were any issues
        if dest_conn and (errors or warnings):
            write_error_log(dest_conn, errors, warnings)
        
        # Clean up connections
        if source_conn:
            source_conn.close()
        if dest_conn:
            dest_conn.close()
        
        logger.info("\nStage 1 execution completed")

if __name__ == "__main__":
    main()