#!/usr/bin/env python3
"""
CONCENTRATOR v3.0 - Unified Hashcat Rule Processor
Enhanced with comprehensive rule filtering options available after extraction/generation.
"""

import sys
import os
import re
import glob
import signal
import argparse
import math
import itertools
import multiprocessing
import tempfile
import subprocess
import random
import datetime
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Callable, Any, Set

# Third-party imports with fallbacks
try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Create a simple progress bar replacement
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.unit = unit
            self.n = 0
            
        def __iter__(self):
            if self.desc:
                print(f"{self.desc}...", end="", flush=True)
            for item in self.iterable:
                yield item
                self.n += 1
            if self.desc:
                print(" done")
                
        def update(self, n=1):
            self.n += n
            
        def close(self):
            pass

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ==============================================================================
# GLOBAL VARIABLES AND CONSTANTS
# ==============================================================================

# Global variables for cleanup
_temp_files_to_cleanup = []
_TEMP_DIR_PATH = None
_IN_MEMORY_MODE = False
_OPENCL_CONTEXT = None
_OPENCL_QUEUE = None
_OPENCL_PROGRAM = None

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    RESET = '\033[0m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'

# Hashcat Rule Syntax Definitions
ALL_RULE_CHARS = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:,.lu.#()=%!?|~+*-^$sStTiIoOcCrRyYzZeEfFxXdDpPbBqQ`[]><@&vV")
OPERATORS_REQUIRING_ARGS = {
    's': 2, 'S': 2, 't': 2, 'T': 2, 'i': 2, 'I': 2, 'o': 2, 'O': 2, 'c': 2, 'C': 2, 'r': 2, 'R': 2, 'y': 2, 'Y': 2, 'z': 2, 'Z': 2, 'e': 2, 'E': 2,
    'f': 1, 'F': 1, 'x': 1, 'X': 1, 'd': 1, 'D': 1, 'p': 1, 'P': 1, 'b': 1, 'B': 1, 'q': 1, 'Q': 1, '`': 1, '[': 1, ']': 1, '>': 1, '<': 1, '@': 1, '&': 1,
    'v': 3, 'V': 3,
}
SIMPLE_OPERATORS = [
    ':', ',', 'l', 'u', '.', '#', '(', ')', '=', '%', '!', '?', '|', '~', '+', '*', '-', '^', '$']

ALL_OPERATORS = list(OPERATORS_REQUIRING_ARGS.keys()) + SIMPLE_OPERATORS
for i in range(10):
    ALL_OPERATORS.append(f'${i}')
    ALL_OPERATORS.append(f'^{i}')

# Build regex for operator parsing
operators_to_escape = [op for op in ALL_OPERATORS if not (op.startswith('$') and len(op) > 1 and op[1].isdigit()) and not (op.startswith('^') and len(op) > 1 and op[1].isdigit())]
REGEX_OPERATORS = [re.escape(op) for op in operators_to_escape]
REGEX_OPERATORS.append(r'\$[0-9]') 
REGEX_OPERATORS.append(r'\^[0-9]') 
COMPILED_REGEX = re.compile('|'.join(filter(None, sorted(list(set(REGEX_OPERATORS)), key=len, reverse=True))))

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def print_banner():
    """Print the colorized program banner"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}" + "="*80)
    print("          CONCENTRATOR v3.0 - Unified Hashcat Rule Processor")
    print("="*80 + f"{Colors.END}")
    print(f"{Colors.YELLOW}Combined Features:{Colors.END}")
    print(f"  {Colors.GREEN}•{Colors.END} OpenCL GPU Acceleration for validation and generation")
    print(f"  {Colors.GREEN}•{Colors.END} Three Processing Modes: Extraction, Combinatorial, Markov")
    print(f"  {Colors.GREEN}•{Colors.END} Hashcat Rule Engine Simulation & Functional Minimization")
    print(f"  {Colors.GREEN}•{Colors.END} Rule Validation and Cleanup (CPU/GPU compatible)")
    print(f"  {Colors.GREEN}•{Colors.END} Levenshtein Distance Filtering")
    print(f"  {Colors.GREEN}•{Colors.END} Smart Processing Selection & Memory Safety")
    print(f"  {Colors.GREEN}•{Colors.END} Interactive & CLI Modes with Colorized Output")
    print(f"{Colors.CYAN}{Colors.BOLD}" + "="*80 + f"{Colors.END}\n")

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE}{text:^80}{Colors.RESET}")
    print(f"{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE}{'='*80}{Colors.RESET}")

def print_section(text: str):
    """Print a section header"""
    print(f"\n{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE} {text} {Colors.RESET}")

def print_warning(text: str):
    """Print a warning message"""
    print(f"{Colors.BG_YELLOW}{Colors.BOLD}{Colors.BLUE}⚠️  WARNING:{Colors.RESET} {Colors.YELLOW}{text}{Colors.RESET}")

def print_error(text: str):
    """Print an error message"""
    print(f"{Colors.BG_RED}{Colors.BOLD}{Colors.WHITE}❌ ERROR:{Colors.RESET} {Colors.RED}{text}{Colors.RESET}")

def print_success(text: str):
    """Print a success message"""
    print(f"{Colors.BG_GREEN}{Colors.BOLD}{Colors.WHITE}✅ SUCCESS:{Colors.RESET} {Colors.GREEN}{text}{Colors.RESET}")

def print_info(text: str):
    """Print an info message"""
    print(f"{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE}ℹ️  INFO:{Colors.RESET} {Colors.BLUE}{text}{Colors.RESET}")

def colorize(text: str, color: str) -> str:
    """Wrap text with color codes"""
    return f"{color}{text}{Colors.RESET}"

def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input from user with default"""
    choices = "Y/n" if default else "y/N"
    response = input(f"{prompt} [{choices}]: ").strip().lower()
    
    if not response:
        return default
    return response in ['y', 'yes']

# ==============================================================================
# MEMORY MANAGEMENT AND SAFETY
# ==============================================================================

def signal_handler(sig, frame):
    """Handle Ctrl+C and other termination signals to clean up temporary files."""
    print(f"\n{Colors.RED}⚠️  INTERRUPT RECEIVED - Cleaning up...{Colors.RESET}")
    
    # Clean up temporary files
    if _temp_files_to_cleanup:
        print(f"{Colors.YELLOW}Cleaning up temporary files...{Colors.RESET}")
        for temp_file in _temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"{Colors.GREEN}✓ Removed temporary file: {temp_file}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.RED}✗ Error removing {temp_file}: {e}{Colors.RESET}")
    
    print(f"{Colors.RED}Script terminated by user.{Colors.RESET}")
    sys.exit(1)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_memory_usage():
    """Get current memory usage statistics."""
    if not PSUTIL_AVAILABLE:
        return None
        
    try:
        virtual_mem = psutil.virtual_memory()
        swap_mem = psutil.swap_memory()
        
        return {
            'ram_used': virtual_mem.used,
            'ram_total': virtual_mem.total,
            'ram_percent': virtual_mem.percent,
            'swap_used': swap_mem.used,
            'swap_total': swap_mem.total,
            'swap_percent': swap_mem.percent,
            'total_used': virtual_mem.used + swap_mem.used,
            'total_available': virtual_mem.total + swap_mem.total,
            'total_percent': (virtual_mem.used + swap_mem.used) / (virtual_mem.total + swap_mem.total) * 100
        }
    except Exception as e:
        print_error(f"Could not monitor memory usage: {e}")
        return None

def format_bytes(bytes_size):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def check_memory_safety(threshold_percent=85):
    """
    Check if memory usage is below safety threshold.
    Returns True if safe, False if approaching limits.
    """
    mem_info = get_memory_usage()
    if not mem_info:
        return True  # Assume safe if we can't monitor
    
    total_percent = mem_info['total_percent']
    
    if total_percent >= threshold_percent:
        print_warning(f"System memory usage at {total_percent:.1f}% (threshold: {threshold_percent}%)")
        print(f"   {Colors.CYAN}RAM:{Colors.RESET} {format_bytes(mem_info['ram_used'])} / {format_bytes(mem_info['ram_total'])} ({mem_info['ram_percent']:.1f}%)")
        print(f"   {Colors.CYAN}Swap:{Colors.RESET} {format_bytes(mem_info['swap_used'])} / {format_bytes(mem_info['swap_total'])} ({mem_info['swap_percent']:.1f}%)")
        return False
    return True

def memory_safe_operation(operation_name, threshold_percent=85):
    """
    Decorator to check memory safety before running memory-intensive operations.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            print_section(f"Memory Check before {operation_name}")
            
            if not check_memory_safety(threshold_percent):
                print_error(f"{operation_name} requires significant memory.")
                print(f"   Current memory usage exceeds {threshold_percent}% threshold.")
                
                response = input(f"{Colors.YELLOW}Continue with {operation_name} anyway? (y/N): {Colors.RESET}").strip().lower()
                if response not in ['y', 'yes']:
                    print_error(f"{operation_name} cancelled due to memory constraints.")
                    return None
            
            print_success(f"Starting {operation_name}...")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def estimate_memory_usage(rules_count, avg_rule_length=50):
    """
    Estimate memory usage for rule processing operations.
    Returns estimated size in bytes.
    """
    # Rough estimation: each rule string + overhead
    estimated_bytes = rules_count * (avg_rule_length + 50)  # 50 bytes overhead per rule
    return estimated_bytes

def print_memory_status():
    """Print current memory status with colors"""
    mem_info = get_memory_usage()
    if not mem_info:
        return
        
    ram_color = Colors.GREEN
    if mem_info['ram_percent'] > 85:
        ram_color = Colors.RED
    elif mem_info['ram_percent'] > 70:
        ram_color = Colors.YELLOW
        
    print(f"{Colors.CYAN}Memory Status:{Colors.END} {ram_color}RAM {mem_info['ram_percent']:.1f}% ({format_bytes(mem_info['ram_used'])}/{format_bytes(mem_info['ram_total'])}){Colors.END}", end="")
    
    if mem_info['swap_total'] > 0:
        if mem_info['swap_used'] > 0:
            swap_color = Colors.YELLOW if mem_info['swap_percent'] < 50 else Colors.RED
            print(f" | {Colors.CYAN}SWAP:{Colors.END} {swap_color}{mem_info['swap_percent']:.1f}% ({format_bytes(mem_info['swap_used'])}/{format_bytes(mem_info['swap_total'])}){Colors.END}")
        else:
            print(f" | {Colors.CYAN}Swap:{Colors.END} {Colors.GREEN}available ({format_bytes(mem_info['swap_total'])}){Colors.END}")
    else:
        print(f" | {Colors.CYAN}Swap:{Colors.END} {Colors.RED}not available{Colors.END}")

def memory_intensive_operation_warning(operation_name):
    """Warn user about memory-intensive operations and check if they want to continue"""
    mem_info = get_memory_usage()
    if not mem_info:
        return True
        
    if mem_info['ram_percent'] > 85:
        print(f"{Colors.RED}{Colors.BOLD}WARNING:{Colors.END} {Colors.YELLOW}High RAM usage detected ({mem_info['ram_percent']:.1f}%) for {operation_name}{Colors.END}")
        print_memory_status()
        
        if mem_info['swap_total'] == 0:
            print(f"{Colors.RED}CRITICAL: No swap space available. System may become unstable.{Colors.END}")
            response = input(f"{Colors.YELLOW}Continue with memory-intensive operation? (y/N): {Colors.END}").strip().lower()
            return response in ('y', 'yes')
        else:
            print(f"{Colors.YELLOW}System will use swap space. Performance may be slower.{Colors.END}")
            response = input(f"{Colors.YELLOW}Continue with memory-intensive operation? (Y/n): {Colors.END}").strip().lower()
            return response not in ('n', 'no')
    
    return True

# ==============================================================================
# INTERACTIVE MODE FUNCTION - UPDATED WITH INPUT ANALYSIS
# ==============================================================================

def interactive_mode():
    """Interactive mode for user-friendly configuration"""
    print_header("CONCENTRATOR v3.0 - INTERACTIVE MODE")
    
    settings = {}
    
    # Get input paths
    print(f"\n{Colors.CYAN}Input Configuration:{Colors.END}")
    while True:
        paths_input = input(f"{Colors.YELLOW}Enter rule files/directories (space separated): {Colors.END}").strip()
        if paths_input:
            paths = paths_input.split()
            # Check if paths exist
            valid_paths = []
            for path in paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    print_warning(f"Path not found: {path}")
            
            if valid_paths:
                settings['paths'] = valid_paths
                break
            else:
                print_error("No valid paths provided. Please try again.")
        else:
            print_error("Please provide at least one path.")
    
    # Now analyze the input to provide recommendations
    print(f"\n{Colors.CYAN}Analyzing Input Data...{Colors.END}")
    
    # Quick analysis of input files
    try:
        # Find rule files
        all_filepaths = find_rule_files_recursive(settings['paths'], max_depth=3)
        
        if not all_filepaths:
            print_error("No rule files found in the provided paths.")
            return None
        
        print_success(f"Found {len(all_filepaths)} rule files.")
        
        # Quick scan for analysis
        total_rules = 0
        unique_rules = set()
        max_rule_len = 0
        
        for filepath in all_filepaths[:10]:  # Sample first 10 files for quick analysis
            try:
                with open(filepath, 'r', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#') or len(line) > 100:
                            continue
                        total_rules += 1
                        unique_rules.add(line)
                        max_rule_len = max(max_rule_len, len(line))
            except:
                continue
        
        # Estimate total rules (extrapolate from sample)
        estimated_total_rules = total_rules * max(1, len(all_filepaths) // 10)
        
        print(f"{Colors.CYAN}Quick Analysis Results:{Colors.END}")
        print(f"  Files found: {len(all_filepaths)}")
        print(f"  Sampled rules: {total_rules}")
        print(f"  Estimated total rules: {estimated_total_rules:,}")
        print(f"  Unique rules in sample: {len(unique_rules)}")
        print(f"  Max rule length observed: {max_rule_len}")
        
        # Mode recommendations based on analysis
        print(f"\n{Colors.CYAN}Mode Recommendations:{Colors.END}")
        
        if estimated_total_rules < 1000:
            print(f"  {Colors.GREEN}•{Colors.END} Small dataset: Consider {Colors.YELLOW}Combinatorial Generation{Colors.END} mode")
            print(f"  {Colors.GREEN}•{Colors.END} You have room to expand with new rule combinations")
            recommended_mode = 'combo'
        elif len(unique_rules) / max(1, total_rules) < 0.3:
            print(f"  {Colors.GREEN}•{Colors.END} Low uniqueness: Consider {Colors.YELLOW}Extraction{Colors.END} mode")
            print(f"  {Colors.GREEN}•{Colors.END} Focus on extracting the most effective existing rules")
            recommended_mode = 'extraction'
        else:
            print(f"  {Colors.GREEN}•{Colors.END} Good dataset diversity: Consider {Colors.YELLOW}Markov{Colors.END} mode")
            print(f"  {Colors.GREEN}•{Colors.END} Generate statistically probable new rules")
            recommended_mode = 'markov'
            
        # Additional recommendations based on rule length
        if max_rule_len > 20:
            print(f"  {Colors.GREEN}•{Colors.END} Long rules detected: Enable {Colors.YELLOW}functional minimization{Colors.END}")
        
    except Exception as e:
        print_warning(f"Quick analysis failed: {e}")
        print(f"  {Colors.YELLOW}Continuing with manual mode selection...{Colors.END}")
        recommended_mode = None
    
    # Choose mode
    print(f"\n{Colors.CYAN}Processing Mode:{Colors.END}")
    print(f"  {Colors.GREEN}1{Colors.END} - Extract top existing rules")
    print(f"  {Colors.GREEN}2{Colors.END} - Generate combinatorial rules") 
    print(f"  {Colors.GREEN}3{Colors.END} - Generate Markov rules")
    
    if recommended_mode:
        mode_display = {'extraction': '1', 'combo': '2', 'markov': '3'}
        print(f"{Colors.YELLOW}  Recommended based on analysis: Mode {mode_display[recommended_mode]}{Colors.END}")
    
    while True:
        mode_choice = input(f"{Colors.YELLOW}Select mode (1-3): {Colors.END}").strip()
        if mode_choice == '1':
            settings['mode'] = 'extraction'
            break
        elif mode_choice == '2':
            settings['mode'] = 'combo'
            break
        elif mode_choice == '3':
            settings['mode'] = 'markov'
            break
        else:
            print_error("Invalid choice. Please enter 1, 2, or 3.")
    
    # Mode-specific settings
    if settings['mode'] == 'extraction':
        while True:
            try:
                top_rules = int(input(f"{Colors.YELLOW}Number of top rules to extract (default 10000): {Colors.END}") or "10000")
                if top_rules > 0:
                    settings['top_rules'] = top_rules
                    break
                else:
                    print_error("Please enter a positive number.")
            except ValueError:
                print_error("Please enter a valid number.")
        
        settings['statistical_sort'] = get_yes_no(f"{Colors.YELLOW}Use statistical sort instead of frequency?{Colors.END}", False)
    
    else:  # combo or markov
        while True:
            try:
                target_rules = int(input(f"{Colors.YELLOW}Target number of rules to generate (default 10000): {Colors.END}") or "10000")
                if target_rules > 0:
                    settings['target_rules'] = target_rules
                    break
                else:
                    print_error("Please enter a positive number.")
            except ValueError:
                print_error("Please enter a valid number.")
        
        while True:
            try:
                min_len = int(input(f"{Colors.YELLOW}Minimum rule length (default 1): {Colors.END}") or "1")
                max_len = int(input(f"{Colors.YELLOW}Maximum rule length (default 3): {Colors.END}") or "3")
                if 1 <= min_len <= max_len:
                    settings['min_len'] = min_len
                    settings['max_len'] = max_len
                    break
                else:
                    print_error("Minimum must be <= maximum and both >= 1.")
            except ValueError:
                print_error("Please enter valid numbers.")
    
    # Global settings
    print(f"\n{Colors.CYAN}Global Settings:{Colors.END}")
    
    settings['output_base_name'] = input(f"{Colors.YELLOW}Output base name (default 'concentrator_output'): {Colors.END}") or "concentrator_output"
    
    while True:
        try:
            max_length = int(input(f"{Colors.YELLOW}Maximum rule length to process (default 31): {Colors.END}") or "31")
            if max_length > 0:
                settings['max_length'] = max_length
                break
            else:
                print_error("Please enter a positive number.")
        except ValueError:
            print_error("Please enter a valid number.")
    
    settings['no_gpu'] = not get_yes_no(f"{Colors.YELLOW}Enable GPU acceleration?{Colors.END}", True)
    settings['in_memory'] = get_yes_no(f"{Colors.YELLOW}Process entirely in RAM?{Colors.END}", False)
    
    # Set temp_dir based on in_memory choice
    if not settings['in_memory']:
        temp_dir = input(f"{Colors.YELLOW}Temporary directory (default: system temp): {Colors.END}").strip()
        settings['temp_dir'] = temp_dir if temp_dir else None
    else:
        settings['temp_dir'] = None  # Explicitly set to None for in-memory mode
    
    # Ensure all required keys are present with defaults
    required_keys = {
        'temp_dir': None,
        'no_gpu': False,
        'in_memory': False,
        'max_length': 31,
        'output_base_name': 'concentrator_output'
    }
    
    # Add mode-specific defaults
    if settings['mode'] == 'extraction':
        required_keys.update({
            'top_rules': 10000,
            'statistical_sort': False
        })
    else:
        required_keys.update({
            'target_rules': 10000,
            'min_len': 1,
            'max_len': 3
        })
    
    # Ensure all required keys are present
    for key, default_value in required_keys.items():
        if key not in settings:
            settings[key] = default_value
    
    # Show summary
    print(f"\n{Colors.CYAN}Configuration Summary:{Colors.END}")
    print(f"  Mode: {settings['mode']}")
    print(f"  Input paths: {len(settings['paths'])} locations")
    print(f"  Output: {settings['output_base_name']}")
    print(f"  Max rule length: {settings['max_length']}")
    print(f"  GPU: {'Enabled' if not settings['no_gpu'] else 'Disabled'}")
    print(f"  In-memory: {'Yes' if settings['in_memory'] else 'No'}")
    print(f"  Temp directory: {settings['temp_dir'] or 'System default'}")
    
    if settings['mode'] == 'extraction':
        print(f"  Top rules: {settings['top_rules']}")
        print(f"  Statistical sort: {'Yes' if settings['statistical_sort'] else 'No'}")
    else:
        print(f"  Target rules: {settings['target_rules']}")
        print(f"  Rule length: {settings['min_len']}-{settings['max_len']}")
    
    proceed = get_yes_no(f"\n{Colors.YELLOW}Start processing with these settings?{Colors.END}", True)
    
    if proceed:
        return settings
    else:
        print_info("Configuration cancelled.")
        return None

# ==============================================================================
# FILE MANAGEMENT AND DISCOVERY
# ==============================================================================

def find_rule_files_recursive(paths, max_depth=3):
    """Find rule files recursively with depth limit"""
    all_filepaths = []
    rule_extensions = {'.rule', '.rules', '.hr', '.hashcat', '.txt', '.lst'}
    
    for path in paths:
        if os.path.isfile(path):
            file_ext = os.path.splitext(path.lower())[1]
            if file_ext in rule_extensions:
                all_filepaths.append(path)
                print_success(f"Rule file: {path}")
            else:
                print_warning(f"Not a rule file (wrong extension): {path}")
        elif os.path.isdir(path):
            print_info(f"Scanning directory: {path} (max depth: {max_depth})")
            found_in_dir = 0
            
            for root, dirs, files in os.walk(path):
                # Calculate current depth
                current_depth = root[len(path):].count(os.sep)
                if current_depth >= max_depth:
                    # Don't go deeper into subdirectories at max depth
                    dirs.clear()
                    continue
                    
                for file in files:
                    file_ext = os.path.splitext(file.lower())[1]
                    if file_ext in rule_extensions:
                        full_path = os.path.join(root, file)
                        all_filepaths.append(full_path)
                        found_in_dir += 1
                        if current_depth == 0:
                            print_success(f"Rule file: {full_path}")
                        else:
                            print_success(f"Rule file (depth {current_depth}): {full_path}")
            
            if found_in_dir == 0:
                print_warning(f"No rule files found in: {path}")
            else:
                print_success(f"Found {found_in_dir} rule files in: {path}")
        else:
            print_error(f"Path not found: {path}")
    
    return sorted(list(set(all_filepaths)))

def set_global_flags(temp_dir_path, in_memory_mode):
    """Sets the global flags required by worker processes."""
    global _TEMP_DIR_PATH, _IN_MEMORY_MODE
    _IN_MEMORY_MODE = in_memory_mode

    if temp_dir_path and not in_memory_mode:
        _TEMP_DIR_PATH = temp_dir_path
        if not os.path.isdir(_TEMP_DIR_PATH):
            try:
                os.makedirs(_TEMP_DIR_PATH, exist_ok=True)
                print_info(f"Using temporary directory: {_TEMP_DIR_PATH}")
            except OSError as e:
                print_warning(f"Could not create temporary directory at {temp_dir_path}. Falling back to default system temp directory. Error: {e}")
                _TEMP_DIR_PATH = None
    elif in_memory_mode:
        print_info("In-Memory Mode activated. Temporary files will be skipped.")

def cleanup_temp_file(temp_file: str):
    """Clean up a single temporary file and remove it from the cleanup list."""
    try:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print_info(f"Cleaned up temporary file: {temp_file}")
        if temp_file in _temp_files_to_cleanup:
            _temp_files_to_cleanup.remove(temp_file)
    except OSError as e:
        print_error(f"Error deleting temporary file {temp_file}: {e}")

def cleanup_temp_files():
    """Clean up all registered temporary files."""
    if not _temp_files_to_cleanup:
        return
    
    print_info("Cleaning up temporary files...")
    for temp_file in _temp_files_to_cleanup[:]:  # Use slice copy to avoid modification during iteration
        cleanup_temp_file(temp_file)

# ==============================================================================
# HASHCAT RULE ENGINE SIMULATION
# ==============================================================================

def i36(string):
    '''Shorter way of converting base 36 string to integer'''
    return int(string, 36)

# --- FUNCTS DICTIONARY ---
FUNCTS: Dict[str, Callable] = {}
FUNCTS[':'] = lambda x, i: x
FUNCTS['l'] = lambda x, i: x.lower()
FUNCTS['u'] = lambda x, i: x.upper()
FUNCTS['c'] = lambda x, i: x.capitalize()
FUNCTS['C'] = lambda x, i: x.capitalize().swapcase()
FUNCTS['t'] = lambda x, i: x.swapcase()

def T(x, i):
    number = i36(i)
    if number >= len(x): return x
    return ''.join((x[:number], x[number].swapcase(), x[number + 1:]))
FUNCTS['T'] = T

FUNCTS['r'] = lambda x, i: x[::-1]
FUNCTS['d'] = lambda x, i: x+x
FUNCTS['p'] = lambda x, i: x*(i36(i)+1)
FUNCTS['f'] = lambda x, i: x+x[::-1]
FUNCTS['{'] = lambda x, i: x[1:]+x[0] if x else x
FUNCTS['}'] = lambda x, i: x[-1]+x[:-1] if x else x
FUNCTS['$'] = lambda x, i: x+i
FUNCTS['^'] = lambda x, i: i+x
FUNCTS['['] = lambda x, i: x[1:]
FUNCTS[']'] = lambda x, i: x[:-1]

def D(x, i):
    idx = i36(i)
    if idx >= len(x): return x
    return x[:idx]+x[idx+1:]
FUNCTS['D'] = D

def x(x, i):
    # Hashcat x functions takes two arguments (start, end)
    start = i36(i[0])
    end = i36(i[1])
    if start < 0 or end < 0 or start > len(x) or end > len(x) or start > end: return "" 
    return x[start:end]
FUNCTS['x'] = x

def O(x, i):
    # Hashcat O functions takes two arguments (start, end)
    start = i36(i[0])
    end = i36(i[1])
    if start < 0 or end < 0 or start > len(x) or end > len(x): return x
    if start > end: return x
    return x[:start]+x[end+1:]
FUNCTS['O'] = O

def i(x, i):
    # Hashcat i functions takes two arguments (pos, char)
    pos = i36(i[0])
    char = i[1]
    if pos > len(x): pos = len(x)
    return x[:pos]+char+x[pos:]
FUNCTS['i'] = i

def o(x, i):
    # Hashcat o functions takes two arguments (pos, char)
    pos = i36(i[0])
    char = i[1]
    if pos >= len(x): return x
    return x[:pos]+char+x[pos+1:]
FUNCTS['o'] = o

FUNCTS["'"] = lambda x, i: x[:i36(i)]
FUNCTS['s'] = lambda x, i: x.replace(i[0], i[1])
FUNCTS['@'] = lambda x, i: x.replace(i, '')

def z(x, i):
    num = i36(i)
    if x: return x[0]*num+x
    return ''
FUNCTS['z'] = z

def Z(x, i):
    num = i36(i)
    if x: return x+x[-1]*num
    return ''
FUNCTS['Z'] = Z
FUNCTS['q'] = lambda x, i: ''.join([a*2 for a in x])

__memorized__ = ['']

def extract_memory(string, args):
    '''Insert section of stored string into current string'''
    if not __memorized__[0]: return string
    try:
        # Note: Your implementation of X uses three arguments, matching hashcat's memory extraction
        pos, length, i = map(i36, args)
        string_list = list(string)
        mem_segment = __memorized__[0][pos:pos+length]
        string_list.insert(i, mem_segment)
        return ''.join(string_list)
    except Exception:
        # Fallback to original string if arguments fail
        return string 
FUNCTS['X'] = extract_memory
FUNCTS['4'] = lambda x, i: x+__memorized__[0]
FUNCTS['6'] = lambda x, i: __memorized__[0]+x

def memorize(string, _):
    ''''Store current string in memory'''
    __memorized__[0] = string
    return string
FUNCTS['M'] = memorize

def rule_regex_gen():
    ''''Generates regex to parse rules'''
    __rules__ = [
        ':', 'l', 'u', 'c', 'C', 't', r'T\w', 'r', 'd', r'p\w', 'f', '{',
        '}', '$.', '^.', '[', ']', r'D\w', r'x\w\w', r'O\w\w', r'i\w.',
        r'o\w.', r"'\w", 's..', '@.', r'z\w', r'Z\w', 'q',
        r'X\w\w\w', '4', '6', 'M'
        ]
    # Build regex, escaping the first character but using raw regex for the arguments
    for i, func in enumerate(__rules__):
        __rules__[i] = re.escape(func[0]) + func[1:].replace(r'\w', '[a-zA-Z0-9]')
    ruleregex = '|'.join(__rules__)
    return re.compile(ruleregex)
__ruleregex__ = rule_regex_gen()

class RuleEngine(object):
    ''' Simplified Rule Engine for functional simulation '''
    def __init__(self, rules: List[str]):
        # Parse all rule strings into a list of lists of function strings
        self.rules = tuple(map(__ruleregex__.findall, rules))

    def apply(self, string: str) -> str:
        ''' 
        Apply all functions in the rule string to a single string and return the result.
        '''
        for rule_functions in self.rules: # self.rules contains one parsed rule (list of functions)
            word = string
            __memorized__[0] = ''
            
            for function in rule_functions: # Iterate over functions in the rule
                try:
                    word = FUNCTS[function[0]](word, function[1:])
                except Exception:
                    pass
            
            # This returns the result after ALL functions in the rule have been applied.
            return word 
        
        return string

# ==============================================================================
# RULE VALIDATION AND PROCESSING
# ==============================================================================

def is_valid_hashcat_rule(rule: str, op_reqs: dict = OPERATORS_REQUIRING_ARGS, valid_chars: set = ALL_RULE_CHARS) -> bool:
    """
    Checks if a generated rule string has valid Hashcat syntax, specifically 
    ensuring operators have the correct number of arguments.
    """
    i = 0
    while i < len(rule):
        op = rule[i]
        
        if op in ('$', '^') and i + 1 < len(rule) and rule[i+1].isdigit():
            # Handle positional operators $X and ^X
            i += 2 
            continue
        
        if op not in op_reqs:
            if op not in valid_chars:
                return False
            i += 1
            continue
            
        required_args = op_reqs.get(op, 0)
        
        if i + 1 + required_args > len(rule):
            return False
            
        args_segment = rule[i+1 : i + 1 + required_args]
        if not all(arg in valid_chars for arg in args_segment):
            return False
            
        i += 1 + required_args 
            
    return True

# ==============================================================================
# OPENCL SETUP AND KERNELS
# ==============================================================================

OPENCL_VALIDATION_KERNEL = """
// OpenCL Rule Validation Kernel
__kernel void validate_rules_batch(
    __global const uchar* rules,
    __global uchar* results,
    const uint rule_stride,
    const uint max_rule_len,
    const uint num_rules)
{
    uint rule_idx = get_global_id(0);
    if (rule_idx >= num_rules) return;
    
    // Simple validation - check if rule contains only valid characters
    __global const uchar* rule = rules + rule_idx * rule_stride;
    bool valid = true;
    
    for (uint i = 0; i < max_rule_len && rule[i] != 0; i++) {
        uchar c = rule[i];
        // Basic validation - check if character is in valid set
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || 
              (c >= 'A' && c <= 'Z') || c == ':' || c == ',' || c == '.' || 
              c == 'l' || c == 'u' || c == '#' || c == '(' || c == ')' || 
              c == '=' || c == '%' || c == '!' || c == '?' || c == '|' || 
              c == '~' || c == '+' || c == '*' || c == '-' || c == '^' || 
              c == '$' || c == 's' || c == 'S' || c == 't' || c == 'T' || 
              c == 'i' || c == 'I' || c == 'o' || c == 'O' || c == 'c' || 
              c == 'C' || c == 'r' || c == 'R' || c == 'y' || c == 'Y' || 
              c == 'z' || c == 'Z' || c == 'e' || c == 'E' || c == 'f' || 
              c == 'F' || c == 'x' || c == 'X' || c == 'd' || c == 'D' || 
              c == 'p' || c == 'P' || c == 'b' || c == 'B' || c == 'q' || 
              c == 'Q' || c == '`' || c == '[' || c == ']' || c == '>' || 
              c == '<' || c == '@' || c == '&' || c == 'v' || c == 'V')) {
            valid = false;
            break;
        }
    }
    
    results[rule_idx] = valid ? 1 : 0;
}
"""

def setup_opencl():
    """Initialize OpenCL context and compile kernels"""
    global _OPENCL_CONTEXT, _OPENCL_QUEUE, _OPENCL_PROGRAM
    
    if not OPENCL_AVAILABLE:
        return False
        
    try:
        platforms = cl.get_platforms()
        if not platforms:
            print_warning("No OpenCL platforms found")
            return False
            
        devices = platforms[0].get_devices(cl.device_type.GPU)
        if not devices:
            print_warning("No GPU devices found, trying CPU")
            devices = platforms[0].get_devices(cl.device_type.CPU)
            
        if not devices:
            print_warning("No OpenCL devices found")
            return False
            
        _OPENCL_CONTEXT = cl.Context(devices)
        _OPENCL_QUEUE = cl.CommandQueue(_OPENCL_CONTEXT)
        _OPENCL_PROGRAM = cl.Program(_OPENCL_CONTEXT, OPENCL_VALIDATION_KERNEL).build()
        
        print_success(f"OpenCL initialized on: {devices[0].name}")
        return True
        
    except Exception as e:
        print_error(f"OpenCL initialization failed: {e}")
        return False

def gpu_validate_rules(rules_list, max_rule_length=64):
    """Validate thousands of rules in parallel on GPU"""
    if not _OPENCL_CONTEXT or not rules_list:
        return [False] * len(rules_list)
    
    try:
        # Prepare data
        num_rules = len(rules_list)
        rule_stride = ((max_rule_length + 15) // 16) * 16  # 16-byte alignment
        
        rules_buffer = np.zeros((num_rules, rule_stride), dtype=np.uint8)
        
        # Fill buffer with rules
        for i, rule in enumerate(rules_list):
            rule_bytes = rule.encode('ascii', 'ignore')
            length = min(len(rule_bytes), rule_stride)
            rules_buffer[i, :length] = np.frombuffer(rule_bytes[:length], dtype=np.uint8)
        
        results = np.zeros(num_rules, dtype=np.uint8)
        
        # Create GPU buffers
        mf = cl.mem_flags
        rules_gpu = cl.Buffer(_OPENCL_CONTEXT, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rules_buffer)
        results_gpu = cl.Buffer(_OPENCL_CONTEXT, mf.WRITE_ONLY, results.nbytes)
        
        # Execute kernel
        global_size = (num_rules,)
        _OPENCL_PROGRAM.validate_rules_batch(_OPENCL_QUEUE, global_size, None,
                                           rules_gpu, results_gpu,
                                           np.uint32(rule_stride),
                                           np.uint32(max_rule_length),
                                           np.uint32(num_rules))
        
        # Get results
        cl.enqueue_copy(_OPENCL_QUEUE, results, results_gpu)
        _OPENCL_QUEUE.finish()
        
        return [bool(result) for result in results]
        
    except Exception as e:
        print_error(f"GPU validation failed: {e}, falling back to CPU")
        return [is_valid_hashcat_rule(rule) for rule in rules_list]

# ==============================================================================
# PARALLEL FILE PROCESSING
# ==============================================================================

def process_single_file(filepath, max_rule_length):
    """Processes a single rule file and counts frequencies."""
    operator_counts = defaultdict(int)
    full_rule_counts = defaultdict(int)
    clean_rules_list = []
    temp_rule_filepath = None
    
    global _IN_MEMORY_MODE, _TEMP_DIR_PATH

    try:
        with open(filepath, 'r', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or len(line) > max_rule_length:
                    continue
                    
                clean_line = ''.join(c for c in line if c in ALL_RULE_CHARS)
                if not clean_line: continue
                    
                # 1. Frequency Count
                full_rule_counts[clean_line] += 1
                    
                # 2. Store rule
                clean_rules_list.append(clean_line)
                    
                # 3. Operator Count
                i = 0
                while i < len(clean_line):
                    match = COMPILED_REGEX.match(clean_line, i)
                    if match:
                        op = match.group(0)
                        if op in ('$', '^') and i + 1 < len(clean_line) and clean_line[i+1].isdigit():
                            # Positional operators are counted by their symbol ($ or ^)
                            operator_counts[op] += 1
                            i += 2
                        elif op in OPERATORS_REQUIRING_ARGS.keys():
                            # Operator with arguments
                            operator_counts[op] += 1
                            i += 1 + OPERATORS_REQUIRING_ARGS[op]
                        else:
                            # Simple operator
                            operator_counts[op] += 1
                            i += len(op)
                    else:
                        # Character is an argument or not an operator
                        i += 1
            
        if not _IN_MEMORY_MODE:
            # --- FILE MODE: Write collected rules to a temporary file ---
            temp_rule_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8', dir=_TEMP_DIR_PATH)
            temp_rule_filepath = temp_rule_file.name
            _temp_files_to_cleanup.append(temp_rule_filepath)  # Register for cleanup
            for rule in clean_rules_list:
                temp_rule_file.write(rule + '\n')
            temp_rule_file.close()  
            print_success(f"File analysis complete: {filepath}. Temp rules saved to {temp_rule_filepath}")
            return operator_counts, full_rule_counts, [], temp_rule_filepath
        else:
            # --- IN-MEMORY MODE: Return the list of rules directly ---
            print_success(f"File analysis complete: {filepath}. Rules returned in memory.")
            return operator_counts, full_rule_counts, clean_rules_list, None
            
    except Exception as e:
        print_error(f"An error occurred while processing {filepath}: {e}")
        if temp_rule_filepath and os.path.exists(temp_rule_filepath):
            cleanup_temp_file(temp_rule_filepath)
        return defaultdict(int), defaultdict(int), [], None

def analyze_rule_files_parallel(filepaths, max_rule_length):
    """Parallel file analysis using multiprocessing.Pool."""
    total_operator_counts = defaultdict(int)
    total_full_rule_counts = defaultdict(int) 
    
    temp_files_to_merge = []
    total_all_clean_rules = []
    
    global _IN_MEMORY_MODE
    
    existing_filepaths = [fp for fp in filepaths if os.path.exists(fp) and os.path.isfile(fp)]
    
    if not existing_filepaths:
        print_warning("No valid rule files found to process.")
        return defaultdict(int), defaultdict(int), [], None

    num_processes = min(os.cpu_count() or 1, len(existing_filepaths))
    tasks = [(filepath, max_rule_length) for filepath in existing_filepaths]
    
    print_info(f"Starting parallel analysis of {len(existing_filepaths)} files using {num_processes} processes...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_single_file, tasks)
        
    for op_counts, rule_counts_worker, clean_rules_worker, temp_filepath in results:
        for op, count in op_counts.items():
            total_operator_counts[op] += count
            
        for rule, count in rule_counts_worker.items():
            total_full_rule_counts[rule] += count
            
        if _IN_MEMORY_MODE:
            total_all_clean_rules.extend(clean_rules_worker)
        else:
            if temp_filepath:
                temp_files_to_merge.append(temp_filepath)
            
    if not _IN_MEMORY_MODE:
        print_info("Merging temporary rule files into memory for Markov processing...")
        for temp_filepath in temp_files_to_merge:
            try:
                with open(temp_filepath, 'r', encoding='utf-8') as f:
                    total_all_clean_rules.extend([line.strip() for line in f])
                cleanup_temp_file(temp_filepath)
            except Exception as e:
                print_error(f"Error merging temp file {temp_filepath}: {e}")
            
    print_success(f"Total unique rules loaded into memory: {len(total_full_rule_counts)}")
    
    sorted_op_counts = sorted(total_operator_counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_op_counts, total_full_rule_counts, total_all_clean_rules

# ==============================================================================
# MARKOV AND STATISTICAL FUNCTIONS
# ==============================================================================

def get_markov_model(unique_rules):
    """Builds the Markov model (counts) and transition probabilities."""
    if not memory_intensive_operation_warning("Markov model building"):
        return None, None
        
    print_section("Building Markov Sequence Probability Model")
    markov_model_counts = defaultdict(lambda: defaultdict(int))
    START_CHAR = '^'             
    
    # 1. Build the Markov Model (Bigrams and Trigrams)
    for rule in unique_rules.keys():
        # Bigram from start
        markov_model_counts[START_CHAR][rule[0]] += 1
        
        # Bigrams O(i) -> O(i+1)
        for i in range(len(rule) - 1):
            markov_model_counts[rule[i]][rule[i+1]] += 1
            
        # Trigrams O(i-1)O(i) -> O(i+1)
        for i in range(len(rule) - 2):
            prefix = rule[i:i+2]
            suffix = rule[i+2]
            markov_model_counts[prefix][suffix] += 1
            
    total_transitions = {char: sum(counts.values()) for char, counts in markov_model_counts.items()}
    
    # 2. Calculate Probabilities
    markov_probabilities = defaultdict(lambda: defaultdict(float))
    for prefix, next_counts in markov_model_counts.items():
        total = total_transitions[prefix]
        for next_op, count in next_counts.items():
            markov_probabilities[prefix][next_op] = count / total
            
    return markov_probabilities, total_transitions

def get_markov_weighted_rules(unique_rules, markov_probabilities, total_transitions):
    """Calculates the log-probability weight for each unique rule based on the model."""
    if not memory_intensive_operation_warning("Markov weighting"):
        return []
        
    weighted_rules = []

    # 3. Calculate Log-Probability Weight for each rule
    for rule in unique_rules.keys():
        log_probability_sum = 0.0
        
        # P(O1 | Start)
        current_prefix = '^'
        next_char = rule[0]
        if next_char in markov_probabilities[current_prefix]:
            probability = markov_probabilities[current_prefix][next_char]
            log_probability_sum += math.log(probability)
        else:
            continue 
            
        # P(Oi | O_i-1) or P(Oi | O_i-2 O_i-1)
        for i in range(len(rule) - 1):
            # Try Trigram (O_i-1 O_i -> O_i+1) first
            if i >= 1:
                current_prefix = rule[i-1:i+1] # Trigram prefix
                next_char = rule[i+1]
                if current_prefix in markov_probabilities and next_char in markov_probabilities[current_prefix]:
                    probability = markov_probabilities[current_prefix][next_char]
                    log_probability_sum += math.log(probability)
                    continue 
            
            # Fallback to Bigram (O_i -> O_i+1)
            current_prefix = rule[i]  
            next_char = rule[i+1]
            if current_prefix in markov_probabilities and next_char in markov_probabilities[current_prefix]:
                probability = markov_probabilities[current_prefix][next_char]
                log_probability_sum += math.log(probability)
            else:
                log_probability_sum = -float('inf')  
                break
            
        if log_probability_sum > -float('inf'):
            weighted_rules.append((rule, log_probability_sum))
            
    sorted_weighted_rules = sorted(weighted_rules, key=lambda item: item[1], reverse=True)
    return sorted_weighted_rules

def generate_rules_from_markov_model(markov_probabilities, target_rules, min_len, max_len):
    """
    Generates new rules by traversing the Markov model, prioritizing high-probability transitions.
    """
    if not memory_intensive_operation_warning("Markov rule generation"):
        return []
        
    print_section(f"Generating Rules via Markov Model Traversal ({min_len}-{max_len} Operators, Target: {target_rules})")
    generated_rules = set()
    START_CHAR = '^'
    
    def get_next_operator(current_prefix):
        """Returns the next operator based on probability distribution (weighted random choice)."""
        if current_prefix not in markov_probabilities:
            return None
        
        choices = list(markov_probabilities[current_prefix].keys())
        weights = list(markov_probabilities[current_prefix].values())
        
        if not choices:
            return None
        
        # Weighted random choice (simulating the probability)
        return random.choices(choices, weights=weights, k=1)[0]
    
    # Use a maximum number of attempts (e.g., 5 times the target) to prevent infinite loops
    generation_attempts = target_rules * 5
    
    for attempt in range(generation_attempts):
        # Stop once the target number of unique rules is reached
        if len(generated_rules) >= target_rules:
            break

        # Start rule with the most probable starting operator (or weighted random)
        current_rule = get_next_operator(START_CHAR)
        if not current_rule: continue
        
        # Traverse until max_len
        while len(current_rule) < max_len:
            last_op = current_rule[-1]
            last_two_ops = current_rule[-2:] if len(current_rule) >= 2 else None
            
            next_op = None
            
            # 1. Try Trigram transition (more specific context)
            if last_two_ops and last_two_ops in markov_probabilities:
                next_op = get_next_operator(last_two_ops)
            
            # 2. Fallback to Bigram transition
            if not next_op and last_op in markov_probabilities:
                next_op = get_next_operator(last_op)
                
            if not next_op:
                break # Cannot continue the sequence
            
            current_rule += next_op
            
            # Check for completion based on min_len
            if len(current_rule) >= min_len and len(current_rule) <= max_len:
                if is_valid_hashcat_rule(current_rule):
                    generated_rules.add(current_rule)

    print_success(f"Generated {len(generated_rules)} statistically probable and syntactically valid rules (Target: {target_rules}).")
    
    # Calculate weights for the generated rules for final sorting
    if generated_rules:
        # Create a dummy frequency count for the generated rules
        generated_rule_counts = {rule: 1 for rule in generated_rules}
        
        # Use the pre-built model for weighting
        weighted_output = get_markov_weighted_rules(generated_rule_counts, markov_probabilities, {}) 
        
        # Crucial: Trim the final output to the target number of rules after sorting by weight
        return weighted_output[:target_rules]
        
    return []

# ==============================================================================
# COMBINATORIAL GENERATION FUNCTIONS
# ==============================================================================

def find_min_operators_for_target(sorted_operators, target_rules, min_len, max_len):
    """Finds the minimum number of top operators needed to generate the target number of rules."""
    current_rule_count = 0
    num_operators = 0
    while current_rule_count < target_rules and num_operators < len(sorted_operators):
        num_operators += 1
        top_ops = [op for op, count in sorted_operators[:num_operators]]
        current_rule_count = 0
        for length in range(min_len, max_len + 1):
            current_rule_count += (len(top_ops) ** length)
            
    return [op for op, count in sorted_operators[:num_operators]]

def generate_rules_for_length_validated(args):
    """Worker function to generate rules for a single length (L) with syntax validation."""
    top_operators, length, op_reqs, valid_chars = args
    generated_rules = set()
    
    for combo in itertools.product(top_operators, repeat=length):
        new_rule = ''.join(combo)
        
        if is_valid_hashcat_rule(new_rule, op_reqs, valid_chars):
            generated_rules.add(new_rule)
            
    return generated_rules

def generate_rules_parallel(top_operators, min_len, max_len):
    """
    Generates all VALID combinatorial rules in parallel based on a list of operators and a length range.
    """
    if not memory_intensive_operation_warning("combinatorial generation"):
        return set()
        
    all_lengths = list(range(min_len, max_len + 1))
    
    tasks = [(top_operators, length, OPERATORS_REQUIRING_ARGS, ALL_RULE_CHARS) for length in all_lengths]
    
    num_processes = min(os.cpu_count() or 1, len(all_lengths))
    print_info(f"Generating new VALID rules of length {min_len} to {max_len} using {len(top_operators)} operators across {num_processes} processes...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(generate_rules_for_length_validated, tasks)
        
    generated_rules = set().union(*results)
    
    print_success(f"Generated and validated {len(generated_rules)} syntactically correct rules.")
    return generated_rules

# ==============================================================================
# FUNCTIONAL MINIMIZATION
# ==============================================================================

# Test vector for functional minimization
TEST_VECTOR = [
    "Password", "123456", "ADMIN", "1aB", "QWERTY", 
    "longword", "spec!", "!spec", "a", "b", "c", "0123", 
    "xYz!", "TEST", "tEST", "test", "0", "1", "$^", "lorem", "ipsum"
]

def worker_generate_signature(rule_data: Tuple[str, int]) -> Tuple[str, Tuple[str, int]]:
    """Worker function for multiprocessing pool."""
    rule_text, count = rule_data
    # Re-initialize RuleEngine for each rule
    engine = RuleEngine([rule_text])
    signature_parts: List[str] = []
    
    for test_word in TEST_VECTOR:
        result = engine.apply(test_word)
        signature_parts.append(result)

    signature = '|'.join(signature_parts)
    return signature, (rule_text, count)

@memory_safe_operation("Functional Minimization", 85)
def functional_minimization(data: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """
    Functional minimization using actual hashcat rule engine simulation.
    This removes rules that produce identical outputs for all test vectors.
    """
    print_section("Functional Minimization")
    print_warning("This operation is RAM intensive and may take significant time for large datasets.")
    
    if not data:
        return data
    
    # For very large datasets, warn the user
    if len(data) > 10000:
        print_warning(f"Large dataset detected ({len(data):,} rules).")
        
        # Estimate memory usage
        estimated_mem = estimate_memory_usage(len(data))
        print(f"{Colors.CYAN}[MEMORY]{Colors.RESET} Estimated memory usage: {format_bytes(estimated_mem)}")
        
        response = input(f"{Colors.YELLOW}Continue with functional minimization? (y/N): {Colors.RESET}").strip().lower()
        if response not in ['y', 'yes']:
            print_info("Skipping functional minimization.")
            return data
    
    print_info(f"Using hashcat rule engine simulation with test vector (Length: {len(TEST_VECTOR)})")
    
    signature_map: Dict[str, List[Tuple[str, int]]] = {}
    
    num_processes = multiprocessing.cpu_count()
    print(f"{Colors.CYAN}[MP]{Colors.RESET} Using {num_processes} processes for functional simulation.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(worker_generate_signature, data),
            total=len(data),
            desc="Simulating rules",
            unit=" rules"
        ))
    
    for signature, rule_data in results:
        if signature not in signature_map:
            signature_map[signature] = []
        signature_map[signature].append(rule_data)

    final_best_rules_list: List[Tuple[str, int]] = []
    
    for signature, rules_list in signature_map.items():
        # Sort by count (highest first) to pick the most common rule as the representative
        rules_list.sort(key=lambda x: x[1], reverse=True)
        best_rule_text, _ = rules_list[0]
        # Sum all counts to get the total functional value
        total_count = sum(count for _, count in rules_list)
        final_best_rules_list.append((best_rule_text, total_count))
        
    final_best_rules_list.sort(key=lambda x: x[1], reverse=True)
    
    removed_count = len(data) - len(final_best_rules_list)
    print_success(f"Removed {removed_count:,} functionally redundant rules.")
    print_success(f"Final count: {len(final_best_rules_list):,} unique functional rules.")
    
    return final_best_rules_list

# ==============================================================================
# LEVENSHTEIN FILTERING
# ==============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

@memory_safe_operation("Levenshtein Filtering", 85)
def levenshtein_filter(data: List[Tuple[str, int]], max_distance: int = 2) -> List[Tuple[str, int]]:
    """
    Filter rules based on Levenshtein distance to remove similar rules.
    """
    print_section("Levenshtein Filtering")
    print_warning("This operation can be slow for large datasets.")
    
    if not data:
        return data
    
    if len(data) > 5000:
        print_warning(f"Large dataset ({len(data):,} rules). This may take a while.")
        response = input(f"{Colors.YELLOW}Continue with Levenshtein filtering? (y/N): {Colors.RESET}").strip().lower()
        if response not in ['y', 'yes']:
            return data
    
    # Ask for distance threshold
    while True:
        try:
            distance_input = input(f"{Colors.YELLOW}Enter maximum Levenshtein distance (1-10) [{max_distance}]: {Colors.RESET}").strip()
            if not distance_input:
                break
            max_distance = int(distance_input)
            if 1 <= max_distance <= 10:
                break
            else:
                print_error("Please enter a value between 1 and 10.")
        except ValueError:
            print_error("Please enter a valid number.")
    
    unique_rules = []
    removed_count = 0
    
    for i, (rule, count) in tqdm(enumerate(data), total=len(data), desc="Levenshtein filtering"):
        is_similar = False
        
        # Compare with already accepted rules
        for existing_rule, _ in unique_rules:
            if levenshtein_distance(rule, existing_rule) <= max_distance:
                is_similar = True
                removed_count += 1
                break
        
        if not is_similar:
            unique_rules.append((rule, count))
    
    print_success(f"Removed {removed_count:,} similar rules.")
    print_success(f"Final count: {len(unique_rules):,} unique rules.")
    
    return unique_rules

# ==============================================================================
# PARETO ANALYSIS AND CURVE DISPLAY
# ==============================================================================

def display_pareto_curve(data: List[Tuple[str, int]]):
    """Display a detailed Pareto analysis with curve visualization in terminal."""
    if not data:
        print_error("No data to analyze.")
        return
        
    total_value = sum(count for _, count in data)
    cumulative_count = 0
    cumulative_value = 0
    
    print_header("PARETO ANALYSIS - CUMULATIVE VALUE DISTRIBUTION")
    print(f"Total rules: {colorize(f'{len(data):,}', Colors.CYAN)}")
    print(f"Total occurrences: {colorize(f'{total_value:,}', Colors.CYAN)}")
    print()
    
    # Calculate milestones
    milestones = []
    target_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    next_target = 0
    
    print(f"{Colors.BOLD}{'Rank':>6} {'Rule':<30} {'Count':>10} {'Cumulative':>12} {'% Total':>8}{Colors.RESET}")
    print(f"{Colors.BOLD}{'-'*70}{Colors.RESET}")
    
    for i, (rule, count) in enumerate(data):
        cumulative_value += count
        current_percentage = (cumulative_value / total_value) * 100
        
        # Show first 10, then milestones
        if i < 10 or (next_target < len(target_percentages) and current_percentage >= target_percentages[next_target]):
            color = Colors.GREEN if i < 10 else Colors.YELLOW if current_percentage <= 80 else Colors.RED
            print(f"{color}{i+1:>6} {rule:<30} {count:>10,} {cumulative_value:>12,} {current_percentage:>7.1f}%{Colors.RESET}")
            
            if next_target < len(target_percentages) and current_percentage >= target_percentages[next_target]:
                milestones.append((target_percentages[next_target], i + 1, current_percentage))
                next_target += 1
        
        if i >= 10 and next_target >= len(target_percentages):
            break
    
    print(f"{Colors.BOLD}{'-'*70}{Colors.RESET}")
    
    # Show milestone summary
    print(f"\n{Colors.BOLD}PARETO MILESTONES:{Colors.RESET}")
    for target, rules_needed, actual_percent in milestones:
        rules_percentage = (rules_needed / len(data)) * 100
        color = Colors.GREEN if target <= 50 else Colors.YELLOW if target <= 80 else Colors.RED
        print(f"  {color}{target:>2}% of value:{Colors.RESET} {rules_needed:>6,} rules ({rules_percentage:5.1f}% of total) - Actual: {actual_percent:5.1f}%")
    
    # ASCII art Pareto curve
    print(f"\n{Colors.BOLD}PARETO CURVE (ASCII Visualization):{Colors.RESET}")
    print("  100% ┤")
    
    # Create simplified curve with 20 points
    curve_points = 20
    step = len(data) // curve_points
    
    for i in range(curve_points + 1):
        idx = min(i * step, len(data) - 1)
        cum_value = sum(count for _, count in data[:idx + 1])
        percentage = (cum_value / total_value) * 100
        
        # Create bar visualization
        bar_length = int(percentage / 5)  # Scale to 20 chars for 100%
        bar = "█" * bar_length
        
        y_axis = 100 - (i * 5)
        if y_axis % 20 == 0 or i == 0 or i == curve_points:
            print(f"{y_axis:>4}% ┤ {bar}")
    
    print(f"    0% ┼{'─' * 20}")
    print(f"       {'0%':<9}{'50%':<10}{'100%':<5}")
    print(f"       {'Cumulative % of rules':<24}")

def analyze_cumulative_value(sorted_data: List[Tuple[str, int]], total_lines: int):
    """Performs Pareto analysis and prints suggestions for MAX_COUNT filtering."""
    if not sorted_data:
        print_error("No data to analyze.")
        return
        
    total_value = sum(count for _, count in sorted_data)
    cumulative_count = 0
    milestones: List[Tuple[int, int]] = []
    target_percentages = [50, 80, 90, 95] 
    next_target = 0
    
    for i, (_, count) in enumerate(sorted_data):
        cumulative_count += count
        current_percentage = (cumulative_count / total_value) * 100
        if next_target < len(target_percentages) and current_percentage >= target_percentages[next_target]:
            milestones.append((target_percentages[next_target], i + 1))
            next_target += 1
        if next_target >= len(target_percentages): 
            break
            
    print_header("CUMULATIVE VALUE ANALYSIS (PARETO) - SUGGESTED CUTOFF LIMITS")
    print(f"Total value (line occurrences) after consolidation: {colorize(f'{total_value:,}', Colors.CYAN)}")
    print(f"Total number of unique rules: {colorize(f'{len(sorted_data):,}', Colors.CYAN)}")

    for target, rules_needed in milestones:
        rules_percentage = (rules_needed / len(sorted_data)) * 100
        color = Colors.GREEN if target <= 80 else Colors.YELLOW if target <= 90 else Colors.RED
        print(f"{color}[{target}% OF VALUE]:{Colors.RESET} Reached with {colorize(f'{rules_needed:,}', Colors.CYAN)} rules. ({rules_percentage:.2f}% of unique rules)")
    
    print(f"{Colors.BOLD}{'-'*60}{Colors.RESET}")
    if milestones:
        last_milestone_rules = milestones[-1][1]
        print(f"{Colors.GREEN}[SUGGESTION]{Colors.RESET} Consider using a limit of: {colorize(f'{last_milestone_rules:,}', Colors.CYAN)} or {colorize(f'{int(last_milestone_rules * 1.1):,}', Colors.CYAN)} for safety.")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")

# ==============================================================================
# FILTERING FUNCTIONS
# ==============================================================================

def filter_by_min_occurrence(data: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Filter rules by minimum occurrence count."""
    if not data:
        return data
    max_count = data[0][1]
    suggested = max(1, sum(count for _, count in data) // 1000)
    
    while True:
        try:
            threshold = int(input(f"{Colors.YELLOW}Enter MINIMUM occurrence count (1-{max_count:,}, suggested: {suggested:,}): {Colors.RESET}"))
            if 1 <= threshold <= max_count:
                filtered = [(rule, count) for rule, count in data if count >= threshold]
                print_success(f"Kept {len(filtered):,} rules (min count: {threshold:,})")
                return filtered
            else:
                print_error(f"Please enter a value between 1 and {max_count:,}")
        except ValueError:
            print_error("Please enter a valid number.")

def filter_by_max_rules(data: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Filter rules by maximum number to keep."""
    if not data:
        return data
    max_possible = len(data)
    
    while True:
        try:
            limit = int(input(f"{Colors.YELLOW}Enter MAXIMUM number of rules to keep (1-{max_possible:,}): {Colors.RESET}"))
            if 1 <= limit <= max_possible:
                filtered = data[:limit]
                print_success(f"Kept top {len(filtered):,} rules")
                return filtered
            else:
                print_error(f"Please enter a value between 1 and {max_possible:,}")
        except ValueError:
            print_error("Please enter a valid number.")

def inverse_mode_filter(data: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Inverse mode - keep rules below a certain rank."""
    if not data:
        return data
    max_possible = len(data)
    
    while True:
        try:
            cutoff = int(input(f"{Colors.YELLOW}Enter cutoff rank (rules BELOW this rank will be kept, 1-{max_possible:,}): {Colors.RESET}"))
            if 1 <= cutoff <= max_possible:
                filtered = data[cutoff:]
                print_success(f"Kept {len(filtered):,} rules below rank {cutoff:,}")
                return filtered
            else:
                print_error(f"Please enter a value between 1 and {max_possible:,}")
        except ValueError:
            print_error("Please enter a valid number.")

# ==============================================================================
# OUTPUT FORMATTING AND SAVING
# ==============================================================================

def save_rules_to_file(data: List[Tuple[str, int]], filename: str = None, mode: str = 'filtered'):
    """Save current rules to file in the desired format."""
    if not data:
        print_error("No rules to save!")
        return False
        
    if filename is None:
        # Generate a filename based on current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"concentrator_{mode}_{len(data)}rules_{timestamp}.rule"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # Add header with metadata
            f.write(f"# Concentrator v3.0 - {mode.upper()} Rules\n")
            f.write(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total rules: {len(data):,}\n")
            f.write(f"# Mode: {mode}\n")
            f.write("#\n")
            
            # Write rules in the desired format (rule only, no counts)
            for rule, count in data:
                f.write(f"{rule}\n")
                
        print_success(f"{len(data):,} rules saved to: {filename}")
        return True
        
    except IOError as e:
        print_error(f"Failed to save file: {e}")
        return False

def save_concentrator_rules(rules_data, output_filename, mode_name):
    """Save rules in the proper format for concentrator output."""
    if not rules_data:
        print_error("No rules to save!")
        return False
        
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"# CONCENTRATOR v3.0 - {mode_name.upper()} MODE OUTPUT\n")
            f.write(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total rules: {len(rules_data)}\n")
            f.write("# Format: rule_string\n")
            f.write("#\n")
            
            # Write rules in the desired format (just the rule strings)
            for rule_item in rules_data:
                if isinstance(rule_item, tuple):
                    rule = rule_item[0]  # Extract rule from (rule, count) tuple
                else:
                    rule = rule_item
                f.write(f"{rule}\n")
                
        print_success(f"Saved {len(rules_data)} rules to: {output_filename}")
        return True
        
    except Exception as e:
        print_error(f"Failed to save rules: {e}")
        return False

def gpu_extract_and_validate_rules(full_rule_counts, top_rules, gpu_enabled):
    """Extract and validate top rules using GPU if available."""
    # Sort by frequency
    sorted_rules = sorted(full_rule_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Take top rules for validation
    rules_to_validate = [rule for rule, count in sorted_rules[:top_rules*2]]
    
    if gpu_enabled:
        validation_results = gpu_validate_rules(rules_to_validate)
        valid_rules = []
        for rule, is_valid in zip(rules_to_validate, validation_results):
            if is_valid:
                valid_rules.append((rule, full_rule_counts[rule]))
        return valid_rules[:top_rules]
    else:
        # CPU validation
        valid_rules = []
        for rule in rules_to_validate:
            if is_valid_hashcat_rule(rule):
                valid_rules.append((rule, full_rule_counts[rule]))
        return valid_rules[:top_rules]

# ==============================================================================
# HASHCAT RULE CLEANUP IMPLEMENTATION
# ==============================================================================

class HashcatRuleCleaner:
    """
    Implements hashcat's rule validation and cleanup logic.
    Based on the official cleanup-rules.c from hashcat.
    """
    
    # Rule operation constants (from hashcat)
    RULE_OP_MANGLE_NOOP             = ':'
    RULE_OP_MANGLE_LREST            = 'l'
    RULE_OP_MANGLE_UREST            = 'u'
    RULE_OP_MANGLE_LREST_UFIRST     = 'c'
    RULE_OP_MANGLE_UREST_LFIRST     = 'C'
    RULE_OP_MANGLE_TREST            = 't'
    RULE_OP_MANGLE_TOGGLE_AT        = 'T'
    RULE_OP_MANGLE_REVERSE          = 'r'
    RULE_OP_MANGLE_DUPEWORD         = 'd'
    RULE_OP_MANGLE_DUPEWORD_TIMES   = 'p'
    RULE_OP_MANGLE_REFLECT          = 'f'
    RULE_OP_MANGLE_ROTATE_LEFT      = '{'
    RULE_OP_MANGLE_ROTATE_RIGHT     = '}'
    RULE_OP_MANGLE_APPEND           = '$'
    RULE_OP_MANGLE_PREPEND          = '^'
    RULE_OP_MANGLE_DELETE_FIRST     = '['
    RULE_OP_MANGLE_DELETE_LAST      = ']'
    RULE_OP_MANGLE_DELETE_AT        = 'D'
    RULE_OP_MANGLE_EXTRACT          = 'x'
    RULE_OP_MANGLE_INSERT           = 'i'
    RULE_OP_MANGLE_OVERSTRIKE       = 'o'
    RULE_OP_MANGLE_TRUNCATE_AT      = "'"
    RULE_OP_MANGLE_REPLACE          = 's'
    RULE_OP_MANGLE_PURGECHAR        = '@'
    RULE_OP_MANGLE_TOGGLECASE_REC   = 'a'
    RULE_OP_MANGLE_DUPECHAR_FIRST   = 'z'
    RULE_OP_MANGLE_DUPECHAR_LAST    = 'Z'
    RULE_OP_MANGLE_DUPECHAR_ALL     = 'q'
    RULE_OP_MANGLE_EXTRACT_MEMORY   = 'X'
    RULE_OP_MANGLE_APPEND_MEMORY    = '4'
    RULE_OP_MANGLE_PREPEND_MEMORY   = '6'
    RULE_OP_MEMORIZE_WORD           = 'M'
    RULE_OP_REJECT_LESS             = '<'
    RULE_OP_REJECT_GREATER          = '>'
    RULE_OP_REJECT_CONTAIN          = '!'
    RULE_OP_REJECT_NOT_CONTAIN      = '/'
    RULE_OP_REJECT_EQUAL_FIRST      = '('
    RULE_OP_REJECT_EQUAL_LAST       = ')'
    RULE_OP_REJECT_EQUAL_AT         = '='
    RULE_OP_REJECT_CONTAINS         = '%'
    RULE_OP_REJECT_MEMORY           = 'Q'
    # hashcat only
    RULE_OP_MANGLE_SWITCH_FIRST     = 'k'
    RULE_OP_MANGLE_SWITCH_LAST      = 'K'
    RULE_OP_MANGLE_SWITCH_AT        = '*'
    RULE_OP_MANGLE_CHR_SHIFTL       = 'L'
    RULE_OP_MANGLE_CHR_SHIFTR       = 'R'
    RULE_OP_MANGLE_CHR_INCR         = '+'
    RULE_OP_MANGLE_CHR_DECR         = '-'
    RULE_OP_MANGLE_REPLACE_NP1      = '.'
    RULE_OP_MANGLE_REPLACE_NM1      = ','
    RULE_OP_MANGLE_DUPEBLOCK_FIRST  = 'y'
    RULE_OP_MANGLE_DUPEBLOCK_LAST   = 'Y'
    RULE_OP_MANGLE_TITLE            = 'E'

    # Maximum rules per line
    MAX_CPU_RULES = 255
    MAX_GPU_RULES = 255

    def __init__(self, mode: int = 1):
        """
        Initialize the rule cleaner.
        mode: 1 = CPU rules, 2 = GPU rules
        """
        if mode not in [1, 2]:
            raise ValueError("Mode must be 1 (CPU) or 2 (GPU)")
        self.mode = mode
        self.max_rules = self.MAX_CPU_RULES if mode == 1 else self.MAX_GPU_RULES

    @staticmethod
    def class_num(c: str) -> bool:
        """Check if character is a digit."""
        return c >= '0' and c <= '9'

    @staticmethod
    def class_upper(c: str) -> bool:
        """Check if character is uppercase letter."""
        return c >= 'A' and c <= 'Z'

    @staticmethod
    def conv_ctoi(c: str) -> int:
        """Convert character to integer (base36)."""
        if HashcatRuleCleaner.class_num(c):
            return ord(c) - ord('0')
        elif HashcatRuleCleaner.class_upper(c):
            return ord(c) - ord('A') + 10
        return -1

    def is_gpu_denied_op(self, op: str) -> bool:
        """Check if operation is denied on GPU."""
        gpu_denied_ops = {
            self.RULE_OP_MANGLE_EXTRACT_MEMORY,
            self.RULE_OP_MANGLE_APPEND_MEMORY,
            self.RULE_OP_MANGLE_PREPEND_MEMORY,
            self.RULE_OP_MEMORIZE_WORD,
            self.RULE_OP_REJECT_LESS,
            self.RULE_OP_REJECT_GREATER,
            self.RULE_OP_REJECT_CONTAIN,
            self.RULE_OP_REJECT_NOT_CONTAIN,
            self.RULE_OP_REJECT_EQUAL_FIRST,
            self.RULE_OP_REJECT_EQUAL_LAST,
            self.RULE_OP_REJECT_EQUAL_AT,
            self.RULE_OP_REJECT_CONTAINS,
            self.RULE_OP_REJECT_MEMORY
        }
        return op in gpu_denied_ops

    def validate_rule(self, rule_line: str) -> bool:
        """
        Validate a single rule line according to hashcat standards.
        Returns True if rule is valid, False otherwise.
        """
        # Remove spaces and check if empty
        clean_line = rule_line.replace(' ', '')
        if not clean_line:
            return False

        rc = 0
        cnt = 0
        pos = 0
        line_len = len(clean_line)

        while pos < line_len:
            op = clean_line[pos]
            
            # Skip spaces (though we already removed them)
            if op == ' ':
                pos += 1
                continue

            # Validate operation and parameters
            try:
                if op == self.RULE_OP_MANGLE_NOOP:
                    pass
                elif op == self.RULE_OP_MANGLE_LREST:
                    pass
                elif op == self.RULE_OP_MANGLE_UREST:
                    pass
                elif op == self.RULE_OP_MANGLE_LREST_UFIRST:
                    pass
                elif op == self.RULE_OP_MANGLE_UREST_LFIRST:
                    pass
                elif op == self.RULE_OP_MANGLE_TREST:
                    pass
                elif op == self.RULE_OP_MANGLE_TOGGLE_AT:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_REVERSE:
                    pass
                elif op == self.RULE_OP_MANGLE_DUPEWORD:
                    pass
                elif op == self.RULE_OP_MANGLE_DUPEWORD_TIMES:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_REFLECT:
                    pass
                elif op == self.RULE_OP_MANGLE_ROTATE_LEFT:
                    pass
                elif op == self.RULE_OP_MANGLE_ROTATE_RIGHT:
                    pass
                elif op == self.RULE_OP_MANGLE_APPEND:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_PREPEND:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_DELETE_FIRST:
                    pass
                elif op == self.RULE_OP_MANGLE_DELETE_LAST:
                    pass
                elif op == self.RULE_OP_MANGLE_DELETE_AT:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_EXTRACT:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_INSERT:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_OVERSTRIKE:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_TRUNCATE_AT:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_REPLACE:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_PURGECHAR:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_TOGGLECASE_REC:
                    pass
                elif op == self.RULE_OP_MANGLE_DUPECHAR_FIRST:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_DUPECHAR_LAST:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_DUPECHAR_ALL:
                    pass
                elif op == self.RULE_OP_MANGLE_DUPEBLOCK_FIRST:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_DUPEBLOCK_LAST:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_SWITCH_FIRST:
                    pass
                elif op == self.RULE_OP_MANGLE_SWITCH_LAST:
                    pass
                elif op == self.RULE_OP_MANGLE_SWITCH_AT:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_CHR_SHIFTL:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_CHR_SHIFTR:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_CHR_INCR:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_CHR_DECR:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_REPLACE_NP1:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_REPLACE_NM1:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_MANGLE_TITLE:
                    pass
                elif op == self.RULE_OP_MANGLE_EXTRACT_MEMORY:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_MANGLE_APPEND_MEMORY:
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_MANGLE_PREPEND_MEMORY:
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_MEMORIZE_WORD:
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_LESS:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_GREATER:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_CONTAIN:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_NOT_CONTAIN:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_EQUAL_FIRST:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_EQUAL_LAST:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_EQUAL_AT:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_CONTAINS:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                    if self.mode == 2:  # GPU mode
                        rc = -1
                elif op == self.RULE_OP_REJECT_MEMORY:
                    if self.mode == 2:  # GPU mode
                        rc = -1
                else:
                    rc = -1  # Unknown operation
            except IndexError:
                rc = -1

            if rc == -1:
                break

            cnt += 1
            pos += 1

            # Check rule count limits
            if cnt > self.max_rules:
                rc = -1
                break

        return rc == 0

    def clean_rules(self, rules_data: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """
        Clean and validate rules according to hashcat standards.
        Returns only valid rules.
        """
        print_section(f"Hashcat Rule Validation ({'GPU' if self.mode == 2 else 'CPU'} Mode)")
        print(f"Validating {colorize(f'{len(rules_data):,}', Colors.CYAN)} rules for {'GPU' if self.mode == 2 else 'CPU'} compatibility...")
        
        valid_rules = []
        invalid_count = 0
        
        for rule, count in tqdm(rules_data, desc="Validating rules"):
            if self.validate_rule(rule):
                valid_rules.append((rule, count))
            else:
                invalid_count += 1
        
        print_success(f"Removed {invalid_count:,} invalid rules. {len(valid_rules):,} valid rules remaining.")
        return valid_rules

def hashcat_rule_cleanup(data: List[Tuple[str, int]], mode: int = 1) -> List[Tuple[str, int]]:
    """Clean rules using hashcat's validation standards."""
    print_section(f"Hashcat Rule Cleanup ({'GPU' if mode == 2 else 'CPU'} Mode)")
    cleaner = HashcatRuleCleaner(mode)
    cleaned_data = cleaner.clean_rules(data)
    return cleaned_data

# ==============================================================================
# ENHANCED INTERACTIVE PROCESSING WITH FILTERING OPTIONS
# ==============================================================================

def enhanced_interactive_processing_loop(original_data: List[Tuple[str, int]], total_lines: int, args, initial_mode: str = "extracted"):
    """
    Enhanced interactive processing loop with comprehensive filtering options.
    """
    current_data = original_data
    unique_count = len(current_data)
    original_count = unique_count
    
    print_header("ENHANCED RULE PROCESSING - INTERACTIVE MENU")
    print(f"Initial dataset: {colorize(f'{original_count:,}', Colors.CYAN)} unique rules")
    print(f"Current dataset: {colorize(f'{unique_count:,}', Colors.CYAN)} unique rules")
    
    try:
        while True:
            print(f"\n{Colors.BOLD}{'-'*80}{Colors.RESET}")
            print(f"{Colors.BOLD}ADVANCED FILTERING OPTIONS:{Colors.RESET}")
            print(f" {Colors.GREEN}(1){Colors.RESET} Filter by {Colors.CYAN}MINIMUM OCCURRENCE{Colors.RESET}")
            print(f" {Colors.GREEN}(2){Colors.RESET} Filter by {Colors.CYAN}MAXIMUM NUMBER OF RULES{Colors.RESET} (Statistical Cutoff - TOP N)")
            print(f" {Colors.GREEN}(3){Colors.RESET} Filter by {Colors.CYAN}FUNCTIONAL REDUNDANCY{Colors.RESET} (Logic Minimization) [RAM INTENSIVE]")
            print(f" {Colors.GREEN}(4){Colors.RESET} {Colors.YELLOW}**INVERSE MODE**{Colors.RESET} - Save rules *BELOW* the MAX_COUNT limit")
            print(f" {Colors.GREEN}(5){Colors.RESET} {Colors.MAGENTA}**HASHCAT CLEANUP**{Colors.RESET} - Validate and clean rules (CPU/GPU compatible)")
            print(f" {Colors.GREEN}(6){Colors.RESET} {Colors.MAGENTA}**LEVENSHTEIN FILTER**{Colors.RESET} - Remove similar rules")
            
            print(f"\n{Colors.BOLD}ANALYSIS & UTILITIES:{Colors.RESET}")
            print(f" {Colors.BLUE}(p){Colors.RESET} Show {Colors.CYAN}PARETO analysis{Colors.RESET} with detailed curve")
            print(f" {Colors.BLUE}(s){Colors.RESET} {Colors.GREEN}SAVE{Colors.RESET} current rules to file")
            print(f" {Colors.BLUE}(r){Colors.RESET} {Colors.YELLOW}RESET{Colors.RESET} to original dataset")
            print(f" {Colors.BLUE}(i){Colors.RESET} Show {Colors.CYAN}dataset information{Colors.RESET}")
            print(f" {Colors.BLUE}(q){Colors.RESET} {Colors.RED}QUIT{Colors.RESET} program")
            print(f"{Colors.BOLD}{'-'*80}{Colors.RESET}")
            
            choice = input(f"{Colors.YELLOW}Enter your choice: {Colors.RESET}").strip().lower()
            
            if choice == 'q':
                print_header("THANK YOU FOR USING CONCENTRATOR v3.0!")
                break
                
            elif choice == 'p':
                display_pareto_curve(current_data)
                continue
                
            elif choice == 's':
                # Offer multiple save options
                print(f"\n{Colors.CYAN}Save Options:{Colors.RESET}")
                print(f" {Colors.GREEN}(1){Colors.RESET} Save with auto-generated filename")
                print(f" {Colors.GREEN}(2){Colors.RESET} Specify custom filename")
                print(f" {Colors.GREEN}(3){Colors.RESET} Cancel")
                
                save_choice = input(f"{Colors.YELLOW}Choose save option (1-3): {Colors.RESET}").strip()
                
                if save_choice == '1':
                    save_rules_to_file(current_data, mode=f"{initial_mode}_filtered")
                elif save_choice == '2':
                    custom_name = input(f"{Colors.YELLOW}Enter filename: {Colors.RESET}").strip()
                    if custom_name:
                        if not custom_name.endswith(('.rule', '.txt')):
                            custom_name += '.rule'
                        save_rules_to_file(current_data, custom_name, f"{initial_mode}_filtered")
                continue
                
            elif choice == 'r':
                current_data = original_data
                unique_count = len(current_data)
                print_success(f"Restored original dataset: {unique_count:,} rules")
                continue
                
            elif choice == 'i':
                print_section("DATASET INFORMATION")
                print(f"Original rules: {colorize(f'{original_count:,}', Colors.CYAN)}")
                print(f"Current rules: {colorize(f'{unique_count:,}', Colors.CYAN)}")
                print(f"Reduction: {colorize(f'{((original_count - unique_count) / original_count * 100):.1f}%', Colors.GREEN if unique_count < original_count else Colors.YELLOW)}")
                
                if current_data:
                    max_count = current_data[0][1]
                    min_count = current_data[-1][1]
                    avg_count = sum(count for _, count in current_data) / len(current_data)
                    print(f"Max occurrence: {colorize(f'{max_count:,}', Colors.CYAN)}")
                    print(f"Min occurrence: {colorize(f'{min_count:,}', Colors.CYAN)}")
                    print(f"Avg occurrence: {colorize(f'{avg_count:.1f}', Colors.CYAN)}")
                continue
                
            elif choice == '1':
                current_data = filter_by_min_occurrence(current_data)
                unique_count = len(current_data)
                
            elif choice == '2':
                current_data = filter_by_max_rules(current_data)
                unique_count = len(current_data)
                
            elif choice == '3':
                current_data = functional_minimization(current_data)
                unique_count = len(current_data)
                
            elif choice == '4':
                current_data = inverse_mode_filter(current_data)
                unique_count = len(current_data)
                
            elif choice == '5':
                # Ask for CPU or GPU compatibility
                print(f"\n{Colors.MAGENTA}[HASHCAT CLEANUP]{Colors.RESET} Choose compatibility mode:")
                print(f" {Colors.CYAN}(1){Colors.RESET} CPU compatibility (all rules allowed)")
                print(f" {Colors.CYAN}(2){Colors.RESET} GPU compatibility (memory/reject rules disabled)")
                mode_choice = input(f"{Colors.YELLOW}Enter mode (1 or 2): {Colors.RESET}").strip()
                mode = 1 if mode_choice == '1' else 2
                current_data = hashcat_rule_cleanup(current_data, mode)
                unique_count = len(current_data)
                
            elif choice == '6':
                current_data = levenshtein_filter(current_data, getattr(args, 'levenshtein_max_dist', 2))
                unique_count = len(current_data)
                
            else:
                print_error("Invalid choice. Please try again.")
                continue
            
            # Show updated stats after each operation
            if choice in ['1', '2', '3', '4', '5', '6']:
                reduction = ((original_count - unique_count) / original_count * 100) if original_count > 0 else 0
                print_success(f"Dataset updated: {unique_count:,} unique rules ({reduction:.1f}% reduction)")
                
                # Ask if user wants to see Pareto analysis
                if unique_count > 0:
                    show_pareto = input(f"{Colors.YELLOW}Show Pareto analysis? (Y/n): {Colors.RESET}").strip().lower()
                    if show_pareto not in ['n', 'no']:
                        display_pareto_curve(current_data)
                
                # Ask if user wants to save
                save_now = input(f"{Colors.YELLOW}Save current dataset? (y/N): {Colors.RESET}").strip().lower()
                if save_now in ['y', 'yes']:
                    save_rules_to_file(current_data, mode=f"{initial_mode}_filtered")
                    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interactive menu interrupted by user.{Colors.RESET}")
        
        # Ask if user wants to save before exiting
        save_before_exit = input(f"{Colors.YELLOW}Save current dataset before exiting? (y/N): {Colors.RESET}").strip().lower()
        if save_before_exit in ['y', 'yes']:
            save_rules_to_file(current_data, mode=f"{initial_mode}_filtered")
        
        print(f"{Colors.CYAN}Returning to main program...{Colors.RESET}")

    return current_data

# ==============================================================================
# MAIN PROCESSING FUNCTIONS
# ==============================================================================

def process_multiple_files_concentrator(args):
    """Process multiple files using the concentrator approach."""
    print_header("PROCESSING MODE - Interactive Rule Minimization")
    
    # Find rule files
    all_filepaths = find_rule_files_recursive(args.paths, max_depth=3)
    
    if not all_filepaths:
        print_error("No rule files found to process.")
        return
    
    # Set global flags
    set_global_flags(args.temp_dir, args.in_memory)
    
    # Analyze files
    sorted_op_counts, full_rule_counts, all_clean_rules = analyze_rule_files_parallel(all_filepaths, args.max_length)
    
    if not full_rule_counts:
        print_error("No rules found in files.")
        return
    
    # Convert to list of tuples for processing
    rules_data = list(full_rule_counts.items())
    rules_data.sort(key=lambda x: x[1], reverse=True)
    
    print_success(f"Loaded {len(rules_data):,} unique rules for processing.")
    
    # Enter interactive processing loop
    final_data = enhanced_interactive_processing_loop(rules_data, sum(full_rule_counts.values()), args, "processed")
    
    # Save final results
    if final_data:
        output_file = args.output_base_name + "_processed.rule"
        save_rules_to_file(final_data, output_file, "processed")

def concentrator_main_processing(args):
    """Main processing logic for Concentrator with enhanced interactive options"""
    # Determine active mode and output filename
    if args.extract_rules:
        active_mode = 'extraction'
        output_suffix = '_extracted.rule'
        mode_color = Colors.GREEN
        initial_mode = "extracted"
    elif args.generate_combo:
        active_mode = 'combo'
        output_suffix = '_combo.rule'
        mode_color = Colors.BLUE
        initial_mode = "combo"
    elif args.generate_markov_rules:
        active_mode = 'markov'
        output_suffix = '_markov.rule'
        mode_color = Colors.MAGENTA
        initial_mode = "markov"
        
    output_file_name = args.output_base_name + output_suffix
    
    print(f"\n{Colors.CYAN}Active Mode:{Colors.END} {mode_color}{Colors.BOLD}{active_mode.upper()}{Colors.END}")
    print(f"{Colors.CYAN}Output File:{Colors.END} {Colors.WHITE}{output_file_name}{Colors.END}")
    
    # Set length defaults
    if active_mode == 'markov':
        markov_min_len = args.markov_length[0]
        markov_max_len = args.markov_length[-1]
    elif active_mode == 'combo':
        combo_min_len = args.combo_length[0]
        combo_max_len = args.combo_length[-1]

    # Initialize OpenCL
    gpu_enabled = False
    if not args.no_gpu:
        gpu_enabled = setup_opencl()
        if gpu_enabled:
            print_success("GPU Acceleration: ENABLED")
        else:
            print_warning("GPU Acceleration: Disabled (falling back to CPU)")
    else:
        print_warning("GPU Acceleration: Manually disabled")
    
    # Find rule files
    print_section("Collecting Rule Files (Recursive Search, Max Depth 3)")
    all_filepaths = find_rule_files_recursive(args.paths, max_depth=3)
    
    # Ensure the determined output file is excluded from analysis
    output_files_to_exclude = {os.path.basename(output_file_name)}
    all_filepaths = [fp for fp in all_filepaths if os.path.basename(fp) not in output_files_to_exclude]

    if not all_filepaths:
        print_error("No rule files found to process. Exiting.")
        return
    
    print_success(f"Found {len(all_filepaths)} rule files to analyze.")
    
    set_global_flags(args.temp_dir, args.in_memory)

    # Parallel Rule File Analysis
    print_section("Starting Parallel Rule File Analysis")
    sorted_op_counts, full_rule_counts, all_clean_rules = analyze_rule_files_parallel(all_filepaths, args.max_length)
    
    if not sorted_op_counts:
        print_error("No operators found in files. Exiting.")
        return

    # Markov Model Building (conditional)
    markov_probabilities, total_transitions = None, None
    
    build_markov_model = False
    if active_mode == 'extraction' and hasattr(args, 'statistical_sort') and args.statistical_sort:
        build_markov_model = True
        print_section("Building Markov Model for Statistical Sort")
    elif active_mode == 'markov':
        build_markov_model = True
        print_section("Building Markov Model for Rule Generation")
    else:
        print_info("Skipping Markov Model Build (Not needed for current mode)")
    
    if build_markov_model:
        markov_probabilities, total_transitions = get_markov_model(full_rule_counts)

    # Execute active mode
    result_data = None
    
    if active_mode == 'extraction':
        print_section("GPU-Accelerated Rule Extraction and Validation")
        
        if args.statistical_sort:
            mode = 'statistical'
            print_info("Sort Mode: Statistical Sort (Markov Weight)")
            
            if markov_probabilities is None:
                print_error("Statistical sort (-s) requires the Markov model, but it was skipped.")
                return
                
            sorted_rule_data = get_markov_weighted_rules(full_rule_counts, markov_probabilities, total_transitions)
            
            # GPU validate the statistically sorted rules
            if gpu_enabled and sorted_rule_data:
                rules_to_validate = [rule for rule, weight in sorted_rule_data[:args.top_rules*2]]
                validation_results = gpu_validate_rules(rules_to_validate)
                
                # Re-sort validated rules by statistical weight
                validated_with_weights = []
                for rule in rules_to_validate:
                    for original_rule, weight in sorted_rule_data:
                        if rule == original_rule:
                            validated_with_weights.append((rule, weight))
                            break
                
                result_data = validated_with_weights[:args.top_rules]
                print_success(f"GPU validated {len(result_data)} statistically sorted rules")
            else:
                result_data = sorted_rule_data[:args.top_rules]
                
        else:
            mode = 'frequency'
            print_info("Sort Mode: Frequency Sort (Raw Count) with GPU Validation")
            result_data = gpu_extract_and_validate_rules(full_rule_counts, args.top_rules, gpu_enabled)
            
        print_success(f"Extracted {len(result_data)} top unique rules (max length: {args.max_length} characters).")
        
    elif active_mode == 'markov':
        print_section("Starting STATISTICAL Markov Rule Generation (Validated)")
        
        markov_rules_data = generate_rules_from_markov_model(
            markov_probabilities, 
            args.generate_target, 
            markov_min_len, 
            markov_max_len
        )
        
        # GPU validation for Markov-generated rules
        if gpu_enabled and markov_rules_data:
            markov_rules = [rule for rule, weight in markov_rules_data]
            validation_results = gpu_validate_rules(markov_rules, args.max_length)
            
            # Filter and re-weight valid rules
            valid_markov_rules = []
            for (rule, weight), is_valid in zip(markov_rules_data, validation_results):
                if is_valid:
                    valid_markov_rules.append((rule, weight))
            
            print_success(f"GPU validated {len(valid_markov_rules)}/{len(markov_rules_data)} Markov rules as syntactically valid")
            result_data = valid_markov_rules[:args.generate_target]
        else:
            result_data = markov_rules_data
        
    elif active_mode == 'combo':
        print_section("Starting COMBINATORIAL Rule Generation (Validated)")
        
        # Find minimum number of top operators needed
        top_operators_needed = find_min_operators_for_target(
            sorted_op_counts, 
            args.combo_target, 
            combo_min_len, 
            combo_max_len
        )
        
        print_info(f"Using the top {len(top_operators_needed)} operators to approximate {args.combo_target} rules.")
        
        # Generate rules with GPU acceleration if available
        if gpu_enabled:
            # For now, fall back to CPU generation for simplicity
            generated_rules_set = generate_rules_parallel(
                top_operators_needed, 
                combo_min_len, 
                combo_max_len
            )
        else:
            generated_rules_set = generate_rules_parallel(
                top_operators_needed, 
                combo_min_len, 
                combo_max_len
            )
        
        # Convert set to list of tuples for consistency
        result_data = [(rule, 1) for rule in generated_rules_set]
        print_success(f"Generated {len(result_data)} combinatorial rules.")

    # Ask user if they want to enter enhanced interactive mode
    print(f"\n{Colors.CYAN}{Colors.BOLD}" + "="*60)
    print("ENHANCED PROCESSING OPTIONS")
    print("="*60 + f"{Colors.END}")
    print(f"{Colors.YELLOW}Would you like to apply additional filtering and optimization?{Colors.END}")
    print(f"{Colors.CYAN}Available options:{Colors.END}")
    print(f"  {Colors.GREEN}•{Colors.END} Filter by occurrence count")
    print(f"  {Colors.GREEN}•{Colors.END} Remove functionally redundant rules") 
    print(f"  {Colors.GREEN}•{Colors.END} Apply Levenshtein distance filtering")
    print(f"  {Colors.GREEN}•{Colors.END} Hashcat rule validation (CPU/GPU compatibility)")
    print(f"  {Colors.GREEN}•{Colors.END} Pareto analysis for optimal cutoff selection")
    
    enter_interactive = input(f"\n{Colors.YELLOW}Enter enhanced interactive mode? (Y/n): {Colors.END}").strip().lower()
    
    if enter_interactive not in ['n', 'no']:
        # Use the total lines from file analysis as an estimate
        total_lines_estimate = sum(full_rule_counts.values())
        final_data = enhanced_interactive_processing_loop(result_data, total_lines_estimate, args, initial_mode)
        
        # Save the final processed rules after interactive filtering
        if final_data:
            save_concentrator_rules(final_data, output_file_name, active_mode)
            print_success(f"Final processed rules saved to: {output_file_name}")
    else:
        # Save the initial results without enhanced processing
        if result_data:
            save_concentrator_rules(result_data, output_file_name, active_mode)
            print_success(f"Rules saved to: {output_file_name}")

    print_success("Processing Complete")
    
    # Show usage minimizer information
    print_header("CONCENTRATOR USAGE STATEMENT")
    print(f"{Colors.YELLOW}These tool can significantly reduce rule file size while")
    print(f"maintaining or even improving cracking effectiveness.")
    print(f"For even better results, it is recommended to debug rules obtained by using Concentrator.{Colors.END}")
    
    if gpu_enabled:
        print_success("GPU Acceleration was used for improved performance")
    
    # Final RAM usage check
    print_memory_status()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def print_usage():
    """Print colorized usage information"""
    print(f"{Colors.BOLD}{Colors.CYAN}USAGE:{Colors.END}")
    print(f"  {Colors.WHITE}python concentrator.py [OPTIONS] FILE_OR_DIRECTORY [FILE_OR_DIRECTORY...]{Colors.END}")
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}MODES (choose one):{Colors.END}")
    print(f"  {Colors.GREEN}-e, --extract-rules{Colors.END}     Extract top existing rules from input files")
    print(f"  {Colors.GREEN}-g, --generate-combo{Colors.END}    Generate combinatorial rules from top operators") 
    print(f"  {Colors.GREEN}-gm, --generate-markov-rules{Colors.END} Generate statistically probable Markov rules")
    print(f"  {Colors.GREEN}-p, --process-rules{Colors.END}     Interactive rule processing and minimization")
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}EXTRACTION MODE (-e):{Colors.END}")
    print(f"  {Colors.YELLOW}-t, --top-rules INT{Colors.END}     Number of top rules to extract (default: 10000)")
    print(f"  {Colors.YELLOW}-s, --statistical-sort{Colors.END}  Sort by statistical weight instead of frequency")
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}COMBINATORIAL MODE (-g):{Colors.END}")
    print(f"  {Colors.YELLOW}-n, --combo-target INT{Colors.END}  Target number of rules (default: 100000)")
    print(f"  {Colors.YELLOW}-l, --combo-length MIN MAX{Colors.END} Rule length range (default: 1 3)")
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}MARKOV MODE (-gm):{Colors.END}")
    print(f"  {Colors.YELLOW}-gt, --generate-target INT{Colors.END} Target rules (default: 10000)")
    print(f"  {Colors.YELLOW}-ml, --markov-length MIN MAX{Colors.END} Rule length range (default: 1 3)")
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}PROCESSING MODE (-p):{Colors.END}")
    print(f"  {Colors.YELLOW}-d, --use-disk{Colors.END}         Use disk for large datasets to save RAM")
    print(f"  {Colors.YELLOW}-ld, --levenshtein-max-dist INT{Colors.END} Max Levenshtein distance (default: 2)")
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}GLOBAL OPTIONS:{Colors.END}")
    print(f"  {Colors.MAGENTA}-ob, --output-base-name NAME{Colors.END} Base name for output file")
    print(f"  {Colors.MAGENTA}-m, --max-length INT{Colors.END}    Maximum rule length to process (default: 31)")
    print(f"  {Colors.MAGENTA}--temp-dir DIR{Colors.END}        Temporary directory for file mode")
    print(f"  {Colors.MAGENTA}--in-memory{Colors.END}           Process entirely in RAM")
    print(f"  {Colors.MAGENTA}--no-gpu{Colors.END}             Disable GPU acceleration")
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}INTERACTIVE MODE:{Colors.END}")
    print(f"  {Colors.WHITE}python concentrator.py{Colors.END}   (run without arguments for interactive mode)")
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}EXAMPLES:{Colors.END}")
    print(f"  {Colors.WHITE}# Extract top 5000 rules with GPU acceleration{Colors.END}")
    print(f"  {Colors.WHITE}python concentrator.py -e -t 5000 --no-gpu rules/*.rule{Colors.END}")
    print()
    print(f"  {Colors.WHITE}# Generate 50k combinatorial rules{Colors.END}")
    print(f"  {Colors.WHITE}python concentrator.py -g -n 50000 -l 2 4 hashcat/rules/{Colors.END}")
    print()
    print(f"  {Colors.WHITE}# Process rules interactively with functional minimization{Colors.END}")
    print(f"  {Colors.WHITE}python concentrator.py -p -d rules/{Colors.END}")
    print()

if __name__ == '__main__':
    multiprocessing.freeze_support()  
    
    # Print banner
    print_banner()
    
    # Check RAM usage at startup
    print_memory_status()
    mem_info = get_memory_usage()
    
    if mem_info and mem_info['ram_percent'] > 85:
        print_warning(f"High RAM usage detected ({mem_info['ram_percent']:.1f}%)")
        if mem_info['swap_total'] == 0:
            print_error("CRITICAL: No swap space available. System may become unstable.")
            proceed = get_yes_no(f"{Colors.YELLOW}Continue anyway? (y/N): {Colors.END}", default=False)
            if not proceed:
                sys.exit(1)
        else:
            print_warning("System will use swap space. Performance may be slower.")
    
    # Check if we should use interactive mode or show help
    if len(sys.argv) == 1:
        # Interactive mode
        settings = interactive_mode()
        if not settings:
            sys.exit(0)
        
        # Convert interactive settings to argparse-like structure
        class Args:
            def __init__(self, settings):
                self.paths = settings['paths']
                self.output_base_name = settings['output_base_name']
                self.max_length = settings['max_length']
                self.no_gpu = settings['no_gpu']
                self.in_memory = settings['in_memory']
                self.temp_dir = settings['temp_dir']
                
                # Mode flags
                self.extract_rules = (settings['mode'] == 'extraction')
                self.generate_combo = (settings['mode'] == 'combo')
                self.generate_markov_rules = (settings['mode'] == 'markov')
                self.process_rules = False
                
                # Mode-specific settings
                if self.extract_rules:
                    self.top_rules = settings['top_rules']
                    self.statistical_sort = settings['statistical_sort']
                elif self.generate_combo:
                    self.combo_target = settings['target_rules']
                    self.combo_length = [settings['min_len'], settings['max_len']]
                elif self.generate_markov_rules:
                    self.generate_target = settings['target_rules']
                    self.markov_length = [settings['min_len'], settings['max_len']]
        
        args = Args(settings)
        concentrator_main_processing(args)
        
    elif len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']:
        # Show colorized help
        print_usage()
        sys.exit(0)
    else:
        # CLI mode
        parser = argparse.ArgumentParser(
            description=f'{Colors.CYAN}Unified Hashcat Rule Processor with OpenCL support.{Colors.END}',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=f"""{Colors.CYAN}Examples:{Colors.END}
  {Colors.WHITE}# Extract top rules{Colors.END}
  python concentrator.py -e -t 5000 rules/*.rule

  {Colors.WHITE}# Generate combinatorial rules{Colors.END}  
  python concentrator.py -g -n 50000 -l 2 4 hashcat/rules/

  {Colors.WHITE}# Process rules with functional minimization{Colors.END}
  python concentrator.py -p -d rules/

  {Colors.WHITE}# Run interactive mode{Colors.END}
  python concentrator.py
"""
        )
        
        parser.add_argument('paths', nargs='+', help='Paths to rule files or directories to analyze recursively (max depth 3)')
        
        # --- GLOBAL OUTPUT FILENAME ---
        parser.add_argument('-ob', '--output_base_name', type=str, default='concentrator_output', 
                            help='The base name for the output file. The script will append a suffix based on the mode.')
        
        # --- MODE ENFORCEMENT GROUP (The Four Modes) ---
        output_group = parser.add_mutually_exclusive_group(required=True)

        # 1. Extraction Mode
        output_group.add_argument('-e', '--extract_rules', action='store_true', help='Enables rule extraction and sorting from input files. Uses -t for count.')
        parser.add_argument('-t', '--top_rules', type=int, default=10000, help='The number of top existing rules to extract and save (used with -e).')
        parser.add_argument('-s', '--statistical_sort', action='store_true', help='Sorts EXTRACTED rules by Markov sequence probability instead of raw frequency (used with -e).')
        
        # 2. Combinatorial Generation Mode
        output_group.add_argument('-g', '--generate_combo', action='store_true', help='Enables generating combinatorial rules. Uses -n for target count.')
        parser.add_argument('-n', '--combo_target', type=int, default=100000, help='The approximate number of rules to generate in combinatorial mode (used with -g).')
        parser.add_argument('-l', '--combo_length', nargs='+', type=int, default=[1, 3], help='The range of rule chain lengths for combinatorial mode (e.g., 1 3) (used with -g).')
        
        # 3. Statistical (Markov) Generation Mode
        output_group.add_argument('-gm', '--generate_markov_rules', action='store_true', help='Enables generating statistically probable Markov rules. Uses -gt for target count.')
        parser.add_argument('-gt', '--generate_target', type=int, default=10000, help='The target number of rules to generate in Markov mode (used with -gm).')
        parser.add_argument('-ml', '--markov_length', nargs='+', type=int, default=None, help='The range of rule chain lengths for Markov mode (e.g., 1 5) (used with -gm). Defaults to [1, 3].')

        # 4. Processing Mode
        output_group.add_argument('-p', '--process_rules', action='store_true', help='Enables interactive rule processing and minimization.')
        parser.add_argument('-d', '--use_disk', action='store_true', help='Use disk (temp files) for initial consolidation to save RAM.')
        parser.add_argument('-ld', '--levenshtein_max_dist', type=int, default=2, help='Maximum Levenshtein distance for similarity filtering.')

        # Global/Utility Flags
        parser.add_argument('-m', '--max_length', type=int, default=31, help='The maximum length for rules to be extracted/considered in analysis. Default is 31.')
        parser.add_argument('--temp-dir', type=str, default=None, help='Optional: Specify a directory for temporary files.')
        parser.add_argument('--in-memory', action='store_true', help='Process all rules entirely in RAM.')
        parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration even if available.')
        
        args = parser.parse_args()
        
        # Set Markov length defaults if needed
        if hasattr(args, 'markov_length') and args.markov_length is None:
            args.markov_length = [1, 3]
        
        # Execute appropriate mode
        if args.process_rules:
            process_multiple_files_concentrator(args)
        else:
            concentrator_main_processing(args)
    
    # Cleanup temporary files
    cleanup_temp_files()
    sys.exit(0)
