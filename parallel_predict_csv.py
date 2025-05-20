#!/usr/bin/env python
import os
import argparse
import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import time
import subprocess
import tempfile
from pathlib import Path
import sys
import logging
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parallel processing of CSV latency predictions')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--num_processes', type=int, default=None, 
                        help='Number of parallel processes (default: number of CPU cores)')
    parser.add_argument('--chunk_size', type=int, default=None, 
                        help='Number of rows per chunk (default: automatically calculated)')
    parser.add_argument('--flash', action='store_true', help='Use flash attention')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timeout in seconds for processing each chunk (default: 3600)')
    parser.add_argument('--retry', type=int, default=2,
                        help='Number of retry attempts for failed chunks (default: 2)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    return parser.parse_args()

def split_csv(input_file, num_chunks):
    """Split input CSV into multiple chunks"""
    logger.info(f"Reading input CSV file: {input_file}")
    try:
        df = pd.read_csv(input_file)
        total_rows = len(df)
        
        if total_rows <= 1:
            logger.warning("Input file has only 1 or 0 rows, no splitting needed")
            return [(input_file, 0, total_rows)]
        
        chunk_size = max(1, total_rows // num_chunks)
        chunks = []
        
        # Create a temporary directory to store the chunk files
        # 确保tmp目录存在（相对路径）
        os.makedirs("tmp", exist_ok=True)

        # 在tmp目录下创建临时文件夹
        temp_dir = tempfile.mkdtemp(dir="tmp")
        logger.info(f"Created temporary directory: {temp_dir}")
        
        # Split the dataframe and save chunks
        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)
            chunk_df = df.iloc[i:end_idx]
            chunk_file = os.path.join(temp_dir, f"chunk_{i}_{end_idx}.csv")
            chunk_df.to_csv(chunk_file, index=False)
            chunks.append((chunk_file, i, end_idx))
        
        logger.info(f"Split input file into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting CSV file: {e}")
        logger.debug(traceback.format_exc())
        raise

def process_chunk(args):
    """Process a single chunk using predict_csv_latencies.py"""
    chunk_file, start_idx, end_idx, use_flash, timeout, retry_count = args
    
    # Create output file path for this chunk
    output_file = f"{chunk_file}.out"
    logger.info(f"Output file for chunk {start_idx}:{end_idx} will be {output_file}")
    
    # Build command
    cmd = ["python", "predict_csv_latencies.py", "--input", chunk_file, "--output", output_file]
    if use_flash:
        cmd.append("--flash")
    
    logger.info(f"Processing chunk {start_idx}:{end_idx}")
    start_time = time.time()
    
    # Try processing with retries
    for attempt in range(retry_count + 1):
        try:
            if attempt > 0:
                logger.info(f"Retry attempt {attempt} for chunk {start_idx}:{end_idx}")
            
            # Run the prediction script with timeout
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            
            # Check if the command was successful
            if result.returncode == 0:
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    end_time = time.time()
                    logger.info(f"Finished chunk {start_idx}:{end_idx} in {end_time - start_time:.2f} seconds")
                    return output_file, start_idx, end_idx
                else:
                    logger.error(f"Output file is empty or missing for chunk {start_idx}:{end_idx}")
            else:
                logger.error(f"Error processing chunk {start_idx}:{end_idx} (return code: {result.returncode})")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                
            # If we get here, either the return code was non-zero or the output file is missing/empty
            if attempt == retry_count:
                logger.error(f"Failed to process chunk {start_idx}:{end_idx} after {retry_count + 1} attempts")
                return None
        
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout processing chunk {start_idx}:{end_idx} after {timeout} seconds")
            if attempt == retry_count:
                return None
        
        except Exception as e:
            logger.error(f"Exception processing chunk {start_idx}:{end_idx}: {e}")
            logger.debug(traceback.format_exc())
            if attempt == retry_count:
                return None
    
    # This should not be reached, but just in case
    return None

def merge_results(chunk_results, output_file):
    """Merge processed chunks into a single output file"""
    logger.info(f"Merging {len(chunk_results)} chunk results into {output_file}")
    
    try:
        # Sort chunks by their start index to maintain original order
        chunk_results.sort(key=lambda x: x[1])
        
        # Read and concatenate all chunk dataframes
        all_dfs = []
        for chunk_file, start_idx, end_idx in chunk_results:
            logger.info(f"Reading results from chunk {start_idx}:{end_idx}")
            try:
                df = pd.read_csv(chunk_file)
                if df.empty:
                    logger.warning(f"Chunk {start_idx}:{end_idx} has empty results")
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading chunk result file {chunk_file}: {e}")
        
        # Combine all chunks
        if all_dfs:
            result_df = pd.concat(all_dfs, ignore_index=True)
            result_df.to_csv(output_file, index=False)
            logger.info(f"Successfully merged results to {output_file} with {len(result_df)} rows")
            return True
        else:
            logger.error("No valid results to merge")
            return False
    except Exception as e:
        logger.error(f"Error merging results: {e}")
        logger.debug(traceback.format_exc())
        return False

def cleanup_temp_files(chunks, chunk_results):
    """Remove temporary files"""
    logger.info("Cleaning up temporary files")
    
    # Clean up chunk files
    for chunk_file, _, _ in chunks:
        if os.path.exists(chunk_file):
            try:
                os.remove(chunk_file)
                logger.debug(f"Removed temporary file: {chunk_file}")
            except Exception as e:
                logger.debug(f"Failed to remove {chunk_file}: {e}")
    
    # Clean up output files
    for out_file, _, _ in chunk_results:
        if out_file and os.path.exists(out_file):
            try:
                os.remove(out_file)
                logger.debug(f"Removed temporary file: {out_file}")
            except Exception as e:
                logger.debug(f"Failed to remove {out_file}: {e}")
    
    # Try to remove the temp directory
    if chunks and len(chunks) > 0:
        temp_dir = os.path.dirname(chunks[0][0])
        try:
            os.rmdir(temp_dir)
            logger.debug(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            logger.debug(f"Failed to remove temporary directory {temp_dir}: {e}")

def main():
    args = parse_arguments()
    input_file = args.input
    output_file = args.output
    
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Set number of processes
    num_processes = args.num_processes or max(1, mp.cpu_count() - 1)  # Leave one core free
    logger.info(f"Using {num_processes} parallel processes")
    
    try:
        # Determine chunk count - if chunk_size is specified use that,
        # otherwise create chunks based on the number of processes
        if args.chunk_size:
            # Read just the number of rows first to calculate chunks
            nrows = len(pd.read_csv(input_file))
            num_chunks = max(1, nrows // args.chunk_size)
            logger.info(f"Using specified chunk size: {args.chunk_size} rows per chunk")
        else:
            # Create at least one chunk per process, plus some extra to ensure
            # all processors stay busy (some chunks might finish faster than others)
            num_chunks = num_processes * 2
            logger.info(f"Automatically determining chunks: {num_chunks} chunks")
        
        # Split the input file
        chunks = split_csv(input_file, num_chunks)
        
        if not chunks:
            logger.error("No chunks were created. Exiting.")
            return 1
        
        # Prepare arguments for each process
        process_args = [
            (chunk_file, start_idx, end_idx, args.flash, args.timeout, args.retry) 
            for chunk_file, start_idx, end_idx in chunks
        ]
        
        # Process chunks in parallel
        start_time = time.time()
        with mp.Pool(processes=num_processes) as pool:
            chunk_results = list(tqdm(
                pool.imap(process_chunk, process_args), 
                total=len(process_args),
                desc="Processing chunks"
            ))
        
        # Filter out any failed chunks
        valid_results = [r for r in chunk_results if r is not None]
        
        if not valid_results:
            logger.error("All chunks failed processing. Check logs for errors.")
            return 1
        
        # Merge results
        success = merge_results(valid_results, output_file)
        
        if success:
            logger.info(f"Successfully processed {len(valid_results)} out of {len(chunks)} chunks")
        else:
            logger.error("Failed to merge results")
            return 1
        
        end_time = time.time()
        logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
        
        # Cleanup temporary files
        cleanup_temp_files(chunks, valid_results)
        
        # Report success rate
        if len(valid_results) < len(chunks):
            logger.warning(f"Warning: {len(chunks) - len(valid_results)} chunks failed processing")
            return 0
        else:
            logger.info("All chunks processed successfully")
            return 0
    except KeyboardInterrupt:
        logger.error("Processing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 