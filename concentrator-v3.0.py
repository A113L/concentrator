#!/usr/bin/env python3
"""
CONCENTRATOR v3.0 - Unified Hashcat Rule Processor
Enhanced with complete OpenCL rule validation and formatted output (spaces between operators)
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
import threading
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Callable, Any, Set, Optional, Union
from dataclasses import dataclass
from enum import Enum
import hashlib

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
    # Simple progress bar replacement
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
_cleanup_lock = threading.Lock()
_cleanup_in_progress = False

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

# ==============================================================================
# CONFIGURATION AND CONSTANTS
# ==============================================================================

class RuleType(Enum):
    """Hashcat rule types for validation"""
    SIMPLE = 1
    POSITIONAL = 2
    SUBSTITUTION = 3
    INSERTION = 4
    DELETION = 5
    CASE = 6
    MEMORY = 7
    REJECT = 8
    COMPLEX = 9

@dataclass
class RuleOperator:
    """Definition of a Hashcat rule operator"""
    symbol: str
    rule_type: RuleType
    num_args: int
    description: str
    gpu_compatible: bool = True
    arg_types: List[str] = None  # 'digit', 'char'

# Complete Hashcat rule definitions (based on official docs)
RULE_OPERATORS = {
    ':': RuleOperator(':', RuleType.SIMPLE, 0, "No operation"),
    'l': RuleOperator('l', RuleType.SIMPLE, 0, "Lowercase all letters"),
    'u': RuleOperator('u', RuleType.SIMPLE, 0, "Uppercase all letters"),
    'c': RuleOperator('c', RuleType.SIMPLE, 0, "Capitalize first letter"),
    'C': RuleOperator('C', RuleType.SIMPLE, 0, "Lowercase first, uppercase rest"),
    't': RuleOperator('t', RuleType.SIMPLE, 0, "Toggle case all letters"),
    'T': RuleOperator('T', RuleType.CASE, 1, "Toggle case at position N", arg_types=['digit']),
    'r': RuleOperator('r', RuleType.SIMPLE, 0, "Reverse word"),
    'd': RuleOperator('d', RuleType.SIMPLE, 0, "Duplicate word"),
    'p': RuleOperator('p', RuleType.COMPLEX, 1, "Duplicate word N times", arg_types=['digit']),
    'f': RuleOperator('f', RuleType.SIMPLE, 0, "Reflect word"),
    '{': RuleOperator('{', RuleType.COMPLEX, 1, "Rotate left N positions", arg_types=['digit']),
    '}': RuleOperator('}', RuleType.COMPLEX, 1, "Rotate right N positions", arg_types=['digit']),
    '$': RuleOperator('$', RuleType.POSITIONAL, 1, "Append character", arg_types=['char']),
    '^': RuleOperator('^', RuleType.POSITIONAL, 1, "Prepend character", arg_types=['char']),
    '[': RuleOperator('[', RuleType.DELETION, 1, "Delete first N characters", arg_types=['digit']),
    ']': RuleOperator(']', RuleType.DELETION, 1, "Delete last N characters", arg_types=['digit']),
    'D': RuleOperator('D', RuleType.DELETION, 1, "Delete character at position N", arg_types=['digit']),
    'x': RuleOperator('x', RuleType.COMPLEX, 2, "Extract substring N to M", arg_types=['digit', 'digit']),
    'O': RuleOperator('O', RuleType.COMPLEX, 2, "Delete range N to M", arg_types=['digit', 'digit']),
    'i': RuleOperator('i', RuleType.INSERTION, 2, "Insert character at position", arg_types=['digit', 'char']),
    'o': RuleOperator('o', RuleType.INSERTION, 2, "Overwrite character at position", arg_types=['digit', 'char']),
    "'": RuleOperator("'", RuleType.COMPLEX, 1, "Truncate at position N", arg_types=['digit']),
    's': RuleOperator('s', RuleType.SUBSTITUTION, 2, "Substitute X with Y", arg_types=['char', 'char']),
    '@': RuleOperator('@', RuleType.SUBSTITUTION, 1, "Purge character X", arg_types=['char']),
    'z': RuleOperator('z', RuleType.COMPLEX, 1, "Duplicate first character N times", arg_types=['digit']),
    'Z': RuleOperator('Z', RuleType.COMPLEX, 1, "Duplicate last character N times", arg_types=['digit']),
    'q': RuleOperator('q', RuleType.COMPLEX, 0, "Duplicate each character"),
    'k': RuleOperator('k', RuleType.SIMPLE, 0, "Swap first two characters"),
    'K': RuleOperator('K', RuleType.COMPLEX, 2, "Swap ranges N and M", arg_types=['digit', 'digit']),
    '*': RuleOperator('*', RuleType.COMPLEX, 2, "Swap characters at positions N and M", arg_types=['digit', 'digit']),
    'L': RuleOperator('L', RuleType.DELETION, 1, "Delete left of position N", arg_types=['digit']),
    'R': RuleOperator('R', RuleType.DELETION, 1, "Delete right of position N", arg_types=['digit']),
    '+': RuleOperator('+', RuleType.COMPLEX, 1, "ASCII increment at position N", arg_types=['digit']),
    '-': RuleOperator('-', RuleType.COMPLEX, 1, "ASCII decrement at position N", arg_types=['digit']),
    '.': RuleOperator('.', RuleType.SUBSTITUTION, 1, "Replace with dot at position N", arg_types=['digit']),
    ',': RuleOperator(',', RuleType.SUBSTITUTION, 1, "Replace with comma at position N", arg_types=['digit']),
    'y': RuleOperator('y', RuleType.COMPLEX, 1, "Duplicate first N characters", arg_types=['digit']),
    'Y': RuleOperator('Y', RuleType.COMPLEX, 1, "Duplicate last N characters", arg_types=['digit']),
    'E': RuleOperator('E', RuleType.CASE, 1, "Title case with separator X", arg_types=['char']),
    'e': RuleOperator('e', RuleType.CASE, 1, "Title case (capitalize after separator)", arg_types=['char']),
    'v': RuleOperator('v', RuleType.INSERTION, 2, "Insert char every N positions", arg_types=['digit', 'char']),
    'V': RuleOperator('V', RuleType.INSERTION, 2, "Insert char before position N", arg_types=['digit', 'char']),
    'M': RuleOperator('M', RuleType.MEMORY, 0, "Memorize word", gpu_compatible=False),
    'X': RuleOperator('X', RuleType.MEMORY, 3, "Extract from memory", arg_types=['digit', 'digit', 'digit'], gpu_compatible=False),
    '4': RuleOperator('4', RuleType.MEMORY, 0, "Append memory", gpu_compatible=False),
    '6': RuleOperator('6', RuleType.MEMORY, 0, "Prepend memory", gpu_compatible=False),
    '_': RuleOperator('_', RuleType.MEMORY, 0, "Memory no-op", gpu_compatible=False),
    '!': RuleOperator('!', RuleType.REJECT, 1, "Reject if contains X", arg_types=['char']),
    '/': RuleOperator('/', RuleType.REJECT, 1, "Reject if doesn't contain X", arg_types=['char']),
    '<': RuleOperator('<', RuleType.REJECT, 1, "Reject if length < N", arg_types=['digit']),
    '>': RuleOperator('>', RuleType.REJECT, 1, "Reject if length > N", arg_types=['digit']),
    '(': RuleOperator('(', RuleType.REJECT, 1, "Reject unless length < N", arg_types=['digit']),
    ')': RuleOperator(')', RuleType.REJECT, 1, "Reject unless length > N", arg_types=['digit']),
    '=': RuleOperator('=', RuleType.REJECT, 2, "Reject unless char at N equals X", arg_types=['digit', 'char']),
    '%': RuleOperator('%', RuleType.REJECT, 2, "Reject unless char at N not X", arg_types=['digit', 'char']),
    '?': RuleOperator('?', RuleType.REJECT, 2, "Reject if char at N equals X", arg_types=['digit', 'char']),
    'Q': RuleOperator('Q', RuleType.REJECT, 0, "Reject if memory empty", gpu_compatible=False),
}

# Valid characters in Hashcat rules (for fallback)
VALID_CHARS = set('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:,.lu.#()=%!?|~+*-^$sStTiIoOcCrRyYzZeEfFxXdDpPbBqQ`[]><@&vV')

# ==============================================================================
# OPENCL VALIDATION KERNEL (Complete, fixed)
# ==============================================================================

OPENCL_VALIDATION_KERNEL = """
// ============================================================================
// hashcat_rules_validation.cl - Complete Hashcat Rule Validation Kernel
// ============================================================================

int is_digit(unsigned char c) { return (c >= '0' && c <= '9'); }
int is_upper(unsigned char c) { return (c >= 'A' && c <= 'Z'); }
int is_lower(unsigned char c) { return (c >= 'a' && c <= 'z'); }
int is_valid_arg(unsigned char c) { return is_digit(c) || is_upper(c) || is_lower(c); }

// Validate a complete rule string (no spaces)
__kernel void validate_rule_full(
    __global const unsigned char* rules,
    __global int* results,
    const int rule_stride,
    const int num_rules)
{
    int rule_idx = get_global_id(0);
    if (rule_idx >= num_rules) return;
    
    __global const unsigned char* rule = rules + rule_idx * rule_stride;
    int valid = 1;
    int pos = 0;
    int rule_len = 0;
    
    while (pos < rule_stride && rule[pos] != 0) { rule_len++; pos++; }
    if (rule_len == 0) { results[rule_idx] = 0; return; }
    
    pos = 0;
    while (pos < rule_len && valid) {
        unsigned char op = rule[pos];
        if (op == ' ') { pos++; continue; } // shouldn't happen in raw rules
        
        switch (op) {
            case ':': case 'l': case 'u': case 'c': case 'C': case 't':
            case 'r': case 'd': case 'f': case '{': case '}': case 'q':
            case 'k': case 'M': case '4': case '6': case '_': case 'Q':
                pos++; break;
            
            case 'T': case 'D': case 'L': case 'R': case '+': case '-':
            case '.': case ',': case 'z': case 'Z': case 'y': case 'Y':
            case '<': case '>': case '(': case ')':
                pos++; if (pos >= rule_len || !is_digit(rule[pos])) valid = 0; else pos++;
                break;
            
            case '$': case '^': case '@': case '!': case '/': case '\\'':
                pos++; if (pos >= rule_len) valid = 0; else pos++;
                break;
            
            case 'x': case 'O': case '*': case 'K':
                pos++; if (pos >= rule_len || !is_digit(rule[pos])) valid = 0; else pos++;
                if (pos >= rule_len || !is_digit(rule[pos])) valid = 0; else pos++;
                break;
            
            case 'i': case 'o': case 'v': case 'V': case '=': case '%': case '?':
                pos++; if (pos >= rule_len || !is_digit(rule[pos])) valid = 0; else pos++;
                if (pos >= rule_len) valid = 0; else pos++;
                break;
            
            case 's':
                pos++; if (pos >= rule_len) valid = 0; else pos++;
                if (pos >= rule_len) valid = 0; else pos++;
                break;
            
            case 'X':
                pos++; if (pos >= rule_len || !is_digit(rule[pos])) valid = 0; else pos++;
                if (pos >= rule_len || !is_digit(rule[pos])) valid = 0; else pos++;
                if (pos >= rule_len || !is_digit(rule[pos])) valid = 0; else pos++;
                break;
            
            case 'E': case 'e':
                pos++; if (pos >= rule_len) valid = 0; else pos++;
                break;
            
            default: valid = 0; break;
        }
    }
    results[rule_idx] = valid ? 1 : 0;
}

// Validate a formatted rule (with spaces between operators)
__kernel void validate_formatted_rule(
    __global const unsigned char* rule,
    __global int* result)
{
    int pos = 0, valid = 1, rule_len = 0;
    while (pos < 256 && rule[pos] != 0) { rule_len++; pos++; }
    if (rule_len == 0) { *result = 0; return; }
    
    pos = 0;
    int in_operator = 0, args_needed = 0, args_got = 0;
    unsigned char current_op = 0;
    
    while (pos < rule_len && valid) {
        unsigned char c = rule[pos];
        
        if (c == ' ') {
            if (in_operator && args_got < args_needed) valid = 0;
            in_operator = 0; args_needed = 0; args_got = 0;
            pos++; continue;
        }
        
        if (!in_operator) {
            in_operator = 1; args_got = 0; current_op = c;
            switch (c) {
                case ':': case 'l': case 'u': case 'c': case 'C': case 't':
                case 'r': case 'd': case 'f': case '{': case '}': case 'q':
                case 'k': case 'M': case '4': case '6': case '_': case 'Q':
                    args_needed = 0; pos++; break;
                
                case 'T': case 'D': case 'L': case 'R': case '+': case '-':
                case '.': case ',': case 'z': case 'Z': case 'y': case 'Y':
                case '<': case '>': case '(': case ')': case '\\'':
                    args_needed = 1; pos++; break;
                
                case '$': case '^': case '@': case '!': case '/':
                    args_needed = 1; pos++; break;
                
                case 'x': case 'O': case '*': case 'K':
                    args_needed = 2; pos++; break;
                
                case 'i': case 'o': case 'v': case 'V': case '=': case '%': case '?':
                    args_needed = 2; pos++; break;
                
                case 's':
                    args_needed = 2; pos++; break;
                
                case 'X':
                    args_needed = 3; pos++; break;
                
                case 'E': case 'e':
                    args_needed = 1; pos++; break;
                
                default: valid = 0; pos++; break;
            }
        } else {
            args_got++;
            pos++;
            if (args_got >= args_needed) in_operator = 0;
        }
    }
    if (in_operator && args_got < args_needed) valid = 0;
    *result = valid ? 1 : 0;
}
"""

# ==============================================================================
# GPU RULE VALIDATOR CLASS
# ==============================================================================

class GPURuleValidator:
    """Handles GPU-based rule validation with complete Hashcat syntax checking"""
    
    def __init__(self, device_type=cl.device_type.GPU):
        self.context = None
        self.queue = None
        self.program = None
        self.available = False
        self.device_name = "None"
        self._init_opencl(device_type)
    
    def _init_opencl(self, device_type):
        if not OPENCL_AVAILABLE:
            print_warning("PyOpenCL not available - GPU validation disabled")
            return
        try:
            platforms = cl.get_platforms()
            if not platforms:
                print_warning("No OpenCL platforms found")
                return
            devices = []
            for platform in platforms:
                try:
                    devices = platform.get_devices(device_type)
                    if devices:
                        break
                except:
                    continue
            if not devices:
                print_warning(f"No {device_type} devices found, trying CPU")
                for platform in platforms:
                    try:
                        devices = platform.get_devices(cl.device_type.CPU)
                        if devices:
                            break
                    except:
                        continue
            if not devices:
                print_warning("No OpenCL devices found")
                return
            self.context = cl.Context(devices)
            self.queue = cl.CommandQueue(self.context)
            self.program = cl.Program(self.context, OPENCL_VALIDATION_KERNEL).build()
            self.device_name = devices[0].name
            self.available = True
            print_success(f"OpenCL initialized on: {self.device_name}")
        except Exception as e:
            print_error(f"OpenCL initialization failed: {e}")
    
    def validate_rules_batch(self, rules: List[str], max_rule_len: int = 64) -> List[bool]:
        """Validate multiple rules in parallel on GPU"""
        if not self.available or not rules:
            return [False] * len(rules)
        try:
            num_rules = len(rules)
            rule_stride = ((max_rule_len + 15) // 16) * 16
            rules_buffer = np.zeros((num_rules, rule_stride), dtype=np.uint8)
            for i, rule in enumerate(rules):
                rule_bytes = rule.encode('ascii', 'ignore')[:max_rule_len-1]
                rules_buffer[i, :len(rule_bytes)] = np.frombuffer(rule_bytes, dtype=np.uint8)
                if len(rule_bytes) < rule_stride:
                    rules_buffer[i, len(rule_bytes)] = 0
            results = np.zeros(num_rules, dtype=np.int32)
            mf = cl.mem_flags
            rules_gpu = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rules_buffer)
            results_gpu = cl.Buffer(self.context, mf.WRITE_ONLY, results.nbytes)
            self.program.validate_rule_full(
                self.queue, (num_rules,), None,
                rules_gpu, results_gpu,
                np.int32(rule_stride), np.int32(num_rules)
            )
            cl.enqueue_copy(self.queue, results, results_gpu)
            self.queue.finish()
            return [bool(r) for r in results]
        except Exception as e:
            print_error(f"GPU validation failed: {e}")
            return [self.validate_rule_cpu(rule) for rule in rules]
    
    def validate_formatted_rule(self, rule: str) -> bool:
        """Validate a single formatted rule (with spaces)"""
        if not self.available:
            return self.validate_rule_cpu(rule, formatted=True)
        try:
            rule_buffer = np.zeros(256, dtype=np.uint8)
            rule_bytes = rule.encode('ascii', 'ignore')[:255]
            rule_buffer[:len(rule_bytes)] = np.frombuffer(rule_bytes, dtype=np.uint8)
            result = np.zeros(1, dtype=np.int32)
            mf = cl.mem_flags
            rule_gpu = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rule_buffer)
            result_gpu = cl.Buffer(self.context, mf.WRITE_ONLY, result.nbytes)
            self.program.validate_formatted_rule(self.queue, (1,), None, rule_gpu, result_gpu)
            cl.enqueue_copy(self.queue, result, result_gpu)
            self.queue.finish()
            return bool(result[0])
        except Exception:
            return self.validate_rule_cpu(rule, formatted=True)
    
    @staticmethod
    def validate_rule_cpu(rule: str, formatted: bool = False) -> bool:
        """CPU fallback validation using RULE_OPERATORS"""
        if formatted:
            parts = rule.strip().split()
            for part in parts:
                if not GPURuleValidator._validate_single_operator(part):
                    return False
            return True
        else:
            return GPURuleValidator._validate_single_operator(rule)
    
    @staticmethod
    def _validate_single_operator(op_string: str) -> bool:
        if not op_string:
            return False
        op = op_string[0]
        if op not in RULE_OPERATORS:
            return False
        operator_def = RULE_OPERATORS[op]
        if len(op_string) - 1 < operator_def.num_args:
            return False
        if operator_def.arg_types:
            for i, arg_type in enumerate(operator_def.arg_types):
                if i + 1 >= len(op_string):
                    return False
                arg = op_string[i + 1]
                if arg_type == 'digit' and not arg.isdigit():
                    return False
                elif arg_type == 'char' and arg not in VALID_CHARS:
                    return False
        return True

# ==============================================================================
# RULE FORMATTER CLASS
# ==============================================================================

class RuleFormatter:
    """Handles formatting of Hashcat rules with proper spacing"""
    
    @staticmethod
    def format_rule(rule: str) -> str:
        """Convert raw rule (e.g., 'o2Ki0T') to formatted ('o2K i0T')"""
        if not rule:
            return ""
        rule = rule.replace(' ', '')
        operators = []
        pos = 0
        rule_len = len(rule)
        while pos < rule_len:
            op = rule[pos]
            if op not in RULE_OPERATORS:
                pos += 1
                continue
            operator_def = RULE_OPERATORS[op]
            num_args = operator_def.num_args
            end_pos = min(pos + 1 + num_args, rule_len)
            operator_str = rule[pos:end_pos]
            if len(operator_str) - 1 >= num_args:
                operators.append(operator_str)
            pos = end_pos
        return ' '.join(operators)
    
    @staticmethod
    def parse_formatted_rule(formatted_rule: str) -> str:
        """Convert formatted rule back to raw"""
        return formatted_rule.replace(' ', '')

# ==============================================================================
# HASHCAT RULE CLEANER (ENHANCED)
# ==============================================================================

class HashcatRuleCleaner:
    """Enhanced rule cleaner with GPU validation"""
    def __init__(self, mode: int = 1, gpu_validator: Optional[GPURuleValidator] = None):
        self.mode = mode  # 1=CPU, 2=GPU
        self.gpu_validator = gpu_validator
    
    def clean_rules(self, rules_data: List[Tuple[str, int]], use_gpu: bool = True) -> List[Tuple[str, int]]:
        print_section(f"Hashcat Rule Validation ({'GPU' if use_gpu and self.gpu_validator and self.gpu_validator.available else 'CPU'} Mode)")
        print(f"Validating {colorize(f'{len(rules_data):,}', Colors.CYAN)} rules...")
        
        valid_rules = []
        invalid_count = 0
        rule_strings = [rule for rule, _ in rules_data]
        
        if use_gpu and self.gpu_validator and self.gpu_validator.available:
            validation_results = self.gpu_validator.validate_rules_batch(rule_strings)
            for (rule, count), is_valid in zip(rules_data, validation_results):
                if is_valid:
                    valid_rules.append((rule, count))
                else:
                    invalid_count += 1
        else:
            for rule, count in tqdm(rules_data, desc="Validating rules"):
                if GPURuleValidator.validate_rule_cpu(rule, formatted=False):
                    valid_rules.append((rule, count))
                else:
                    invalid_count += 1
        
        print_success(f"Removed {invalid_count:,} invalid rules. {len(valid_rules):,} valid rules remaining.")
        return valid_rules

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
    print(f"  {Colors.GREEN}•{Colors.END} Formatted Rule Output (spaces between operators)")
    print(f"{Colors.CYAN}{Colors.BOLD}" + "="*80 + f"{Colors.END}\n")

def print_header(text: str):
    print(f"\n{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE}{text:^80}{Colors.RESET}")
    print(f"{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE}{'='*80}{Colors.RESET}")

def print_section(text: str):
    print(f"\n{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE} {text} {Colors.RESET}")

def print_warning(text: str):
    print(f"{Colors.BG_YELLOW}{Colors.BOLD}{Colors.BLUE}⚠️  WARNING:{Colors.RESET} {Colors.YELLOW}{text}{Colors.RESET}")

def print_error(text: str):
    print(f"{Colors.BG_RED}{Colors.BOLD}{Colors.WHITE}❌ ERROR:{Colors.RESET} {Colors.RED}{text}{Colors.RESET}")

def print_success(text: str):
    print(f"{Colors.BG_GREEN}{Colors.BOLD}{Colors.WHITE}✅ SUCCESS:{Colors.RESET} {Colors.GREEN}{text}{Colors.RESET}")

def print_info(text: str):
    print(f"{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE}ℹ️  INFO:{Colors.RESET} {Colors.BLUE}{text}{Colors.RESET}")

def colorize(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"

def get_yes_no(prompt: str, default: bool = True) -> bool:
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
    global _cleanup_in_progress
    
    with _cleanup_lock:
        if _cleanup_in_progress:
            return
        _cleanup_in_progress = True
    
    if multiprocessing.current_process().name != 'MainProcess':
        sys.exit(0)
    
    print(f"\n{Colors.RED}⚠️  INTERRUPT RECEIVED - Cleaning up...{Colors.RESET}")
    
    if _temp_files_to_cleanup:
        print(f"{Colors.YELLOW}Cleaning up temporary files...{Colors.RESET}")
        for temp_file in _temp_files_to_cleanup[:]:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"{Colors.GREEN}✓ Removed temporary file: {temp_file}{Colors.RESET}")
            except Exception:
                pass
    
    print(f"{Colors.RED}Script terminated by user.{Colors.RESET}")
    sys.exit(1)

def get_memory_usage():
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
    except Exception:
        return None

def format_bytes(bytes_size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def check_memory_safety(threshold_percent=85):
    mem_info = get_memory_usage()
    if not mem_info:
        return True
    total_percent = mem_info['total_percent']
    if total_percent >= threshold_percent:
        print_warning(f"System memory usage at {total_percent:.1f}% (threshold: {threshold_percent}%)")
        print(f"   {Colors.CYAN}RAM:{Colors.RESET} {format_bytes(mem_info['ram_used'])} / {format_bytes(mem_info['ram_total'])} ({mem_info['ram_percent']:.1f}%)")
        print(f"   {Colors.CYAN}Swap:{Colors.RESET} {format_bytes(mem_info['swap_used'])} / {format_bytes(mem_info['swap_total'])} ({mem_info['swap_percent']:.1f}%)")
        return False
    return True

def memory_safe_operation(operation_name, threshold_percent=85):
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
    estimated_bytes = rules_count * (avg_rule_length + 50)
    return estimated_bytes

def print_memory_status():
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
    mem_info = get_memory_usage()
    if not mem_info:
        return True
    if mem_info['ram_percent'] > 85:
        print(f"{Colors.RED}{Colors.BOLD}WARNING:{Colors.END} {Colors.YELLOW}High RAM usage detected ({mem_info['ram_percent']:.1f}%) for {operation_name}{Colors.END}")
        print_memory_status()
        if mem_info['swap_total'] == 0:
            print(f"{Colors.RED}CRITICAL: No swap space available. System may become unstable.{Colors.END}")
            response = input(f"{Colors.YELLOW}Continue with memory-intensive operation? (y/N): {Colors.RESET}").strip().lower()
            return response in ('y', 'yes')
        else:
            print(f"{Colors.YELLOW}System will use swap space. Performance may be slower.{Colors.END}")
            response = input(f"{Colors.YELLOW}Continue with memory-intensive operation? (Y/n): {Colors.RESET}").strip().lower()
            return response not in ('n', 'no')
    return True

def set_global_flags(temp_dir_path, in_memory_mode):
    global _TEMP_DIR_PATH, _IN_MEMORY_MODE
    _IN_MEMORY_MODE = in_memory_mode
    if temp_dir_path and not in_memory_mode:
        _TEMP_DIR_PATH = temp_dir_path
        if not os.path.isdir(_TEMP_DIR_PATH):
            try:
                os.makedirs(_TEMP_DIR_PATH, exist_ok=True)
                print_info(f"Using temporary directory: {_TEMP_DIR_PATH}")
            except OSError:
                print_warning(f"Could not create temporary directory at {temp_dir_path}. Falling back to default system temp directory.")
                _TEMP_DIR_PATH = None
    elif in_memory_mode:
        print_info("In-Memory Mode activated. Temporary files will be skipped.")

def cleanup_temp_file(temp_file: str):
    try:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print_info(f"Cleaned up temporary file: {temp_file}")
        if temp_file in _temp_files_to_cleanup:
            _temp_files_to_cleanup.remove(temp_file)
    except OSError as e:
        print_error(f"Error deleting temporary file {temp_file}: {e}")

def cleanup_temp_files():
    if not _temp_files_to_cleanup:
        return
    print_info("Cleaning up temporary files...")
    for temp_file in _temp_files_to_cleanup[:]:
        cleanup_temp_file(temp_file)

# ==============================================================================
# FILE MANAGEMENT AND DISCOVERY
# ==============================================================================

def find_rule_files_recursive(paths, max_depth=3):
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
                current_depth = root[len(path):].count(os.sep)
                if current_depth >= max_depth:
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

# ==============================================================================
# PARALLEL FILE PROCESSING
# ==============================================================================

def process_single_file(filepath, max_rule_length):
    operator_counts = defaultdict(int)
    full_rule_counts = defaultdict(int)
    clean_rules_list = []
    temp_rule_filepath = None
    
    global _IN_MEMORY_MODE, _TEMP_DIR_PATH, _temp_files_to_cleanup

    try:
        with open(filepath, 'r', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or len(line) > max_rule_length:
                    continue
                clean_line = ''.join(c for c in line if c in VALID_CHARS)
                if not clean_line: continue
                full_rule_counts[clean_line] += 1
                clean_rules_list.append(clean_line)
                i = 0
                while i < len(clean_line):
                    if clean_line[i] in ('$', '^') and i+1 < len(clean_line) and clean_line[i+1].isdigit():
                        operator_counts[clean_line[i]] += 1
                        i += 2
                    elif clean_line[i] in RULE_OPERATORS:
                        op_def = RULE_OPERATORS[clean_line[i]]
                        operator_counts[clean_line[i]] += 1
                        i += 1 + op_def.num_args
                    else:
                        i += 1
            
        if not _IN_MEMORY_MODE:
            import tempfile
            temp_rule_file = tempfile.NamedTemporaryFile(
                mode='w+', delete=False, encoding='utf-8', dir=_TEMP_DIR_PATH,
                prefix='concentrator_', suffix='.tmp'
            )
            temp_rule_filepath = temp_rule_file.name
            for rule in clean_rules_list:
                temp_rule_file.write(rule + '\n')
            temp_rule_file.close()
            with _cleanup_lock:
                _temp_files_to_cleanup.append(temp_rule_filepath)
            return operator_counts, full_rule_counts, [], temp_rule_filepath
        else:
            return operator_counts, full_rule_counts, clean_rules_list, None
    except Exception as e:
        print_error(f"An error occurred while processing {filepath}: {e}")
        if temp_rule_filepath and os.path.exists(temp_rule_filepath):
            try:
                os.remove(temp_rule_filepath)
                with _cleanup_lock:
                    if temp_rule_filepath in _temp_files_to_cleanup:
                        _temp_files_to_cleanup.remove(temp_rule_filepath)
            except:
                pass
        return defaultdict(int), defaultdict(int), [], None

def analyze_rule_files_parallel(filepaths, max_rule_length):
    total_operator_counts = defaultdict(int)
    total_full_rule_counts = defaultdict(int) 
    temp_files_to_merge = []
    total_all_clean_rules = []
    
    global _IN_MEMORY_MODE, _cleanup_lock
    
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
                if os.path.exists(temp_filepath):
                    with open(temp_filepath, 'r', encoding='utf-8') as f:
                        total_all_clean_rules.extend([line.strip() for line in f])
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                    with _cleanup_lock:
                        if temp_filepath in _temp_files_to_cleanup:
                            _temp_files_to_cleanup.remove(temp_filepath)
            except Exception as e:
                print_error(f"Error merging temp file {temp_filepath}: {e}")
            
    print_success(f"Total unique rules loaded into memory: {len(total_full_rule_counts)}")
    
    sorted_op_counts = sorted(total_operator_counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_op_counts, total_full_rule_counts, total_all_clean_rules

# ==============================================================================
# MARKOV AND STATISTICAL FUNCTIONS
# ==============================================================================

def get_markov_model(unique_rules):
    if not memory_intensive_operation_warning("Markov model building"):
        return None, None
    print_section("Building Markov Sequence Probability Model")
    markov_model_counts = defaultdict(lambda: defaultdict(int))
    START_CHAR = '^'             
    for rule in unique_rules.keys():
        markov_model_counts[START_CHAR][rule[0]] += 1
        for i in range(len(rule) - 1):
            markov_model_counts[rule[i]][rule[i+1]] += 1
        for i in range(len(rule) - 2):
            prefix = rule[i:i+2]
            suffix = rule[i+2]
            markov_model_counts[prefix][suffix] += 1
    total_transitions = {char: sum(counts.values()) for char, counts in markov_model_counts.items()}
    markov_probabilities = defaultdict(lambda: defaultdict(float))
    for prefix, next_counts in markov_model_counts.items():
        total = total_transitions[prefix]
        for next_op, count in next_counts.items():
            markov_probabilities[prefix][next_op] = count / total
    return markov_probabilities, total_transitions

def get_markov_weighted_rules(unique_rules, markov_probabilities, total_transitions):
    if not memory_intensive_operation_warning("Markov weighting"):
        return []
    weighted_rules = []
    for rule in unique_rules.keys():
        log_probability_sum = 0.0
        current_prefix = '^'
        next_char = rule[0]
        if next_char in markov_probabilities[current_prefix]:
            probability = markov_probabilities[current_prefix][next_char]
            log_probability_sum += math.log(probability)
        else:
            continue 
        for i in range(len(rule) - 1):
            if i >= 1:
                current_prefix = rule[i-1:i+1]
                next_char = rule[i+1]
                if current_prefix in markov_probabilities and next_char in markov_probabilities[current_prefix]:
                    probability = markov_probabilities[current_prefix][next_char]
                    log_probability_sum += math.log(probability)
                    continue 
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
    if not memory_intensive_operation_warning("Markov rule generation"):
        return []
    print_section(f"Generating Rules via Markov Model Traversal ({min_len}-{max_len} Operators, Target: {target_rules})")
    generated_rules = set()
    START_CHAR = '^'
    
    def get_next_operator(current_prefix):
        if current_prefix not in markov_probabilities:
            return None
        choices = list(markov_probabilities[current_prefix].keys())
        weights = list(markov_probabilities[current_prefix].values())
        if not choices:
            return None
        return random.choices(choices, weights=weights, k=1)[0]
    
    generation_attempts = target_rules * 5
    for attempt in range(generation_attempts):
        if len(generated_rules) >= target_rules:
            break
        current_rule = get_next_operator(START_CHAR)
        if not current_rule: continue
        while len(current_rule) < max_len:
            last_op = current_rule[-1]
            last_two_ops = current_rule[-2:] if len(current_rule) >= 2 else None
            next_op = None
            if last_two_ops and last_two_ops in markov_probabilities:
                next_op = get_next_operator(last_two_ops)
            if not next_op and last_op in markov_probabilities:
                next_op = get_next_operator(last_op)
            if not next_op:
                break
            current_rule += next_op
            if len(current_rule) >= min_len and len(current_rule) <= max_len:
                if GPURuleValidator.validate_rule_cpu(current_rule):
                    generated_rules.add(current_rule)
    print_success(f"Generated {len(generated_rules)} statistically probable and syntactically valid rules (Target: {target_rules}).")
    if generated_rules:
        generated_rule_counts = {rule: 1 for rule in generated_rules}
        weighted_output = get_markov_weighted_rules(generated_rule_counts, markov_probabilities, {}) 
        return weighted_output[:target_rules]
    return []

# ==============================================================================
# COMBINATORIAL GENERATION FUNCTIONS
# ==============================================================================

def find_min_operators_for_target(sorted_operators, target_rules, min_len, max_len):
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
    top_operators, length = args
    generated_rules = set()
    for combo in itertools.product(top_operators, repeat=length):
        new_rule = ''.join(combo)
        if GPURuleValidator.validate_rule_cpu(new_rule):
            generated_rules.add(new_rule)
    return generated_rules

def generate_rules_parallel(top_operators, min_len, max_len):
    if not memory_intensive_operation_warning("combinatorial generation"):
        return set()
    all_lengths = list(range(min_len, max_len + 1))
    tasks = [(top_operators, length) for length in all_lengths]
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

TEST_VECTOR = [
    "Password", "123456", "ADMIN", "1aB", "QWERTY", 
    "longword", "spec!", "!spec", "a", "b", "c", "0123", 
    "xYz!", "TEST", "tEST", "test", "0", "1", "$^", "lorem", "ipsum"
]

def worker_generate_signature(rule_data: Tuple[str, int]) -> Tuple[str, Tuple[str, int]]:
    rule_text, count = rule_data
    from hashcat_rule_engine import RuleEngine  # Assuming original RuleEngine is defined
    engine = RuleEngine([rule_text])
    signature_parts: List[str] = []
    for test_word in TEST_VECTOR:
        result = engine.apply(test_word)
        signature_parts.append(result)
    signature = '|'.join(signature_parts)
    return signature, (rule_text, count)

@memory_safe_operation("Functional Minimization", 85)
def functional_minimization(data: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    print_section("Functional Minimization")
    print_warning("This operation is RAM intensive and may take significant time for large datasets.")
    if not data:
        return data
    if len(data) > 10000:
        print_warning(f"Large dataset detected ({len(data):,} rules).")
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
        rules_list.sort(key=lambda x: x[1], reverse=True)
        best_rule_text, _ = rules_list[0]
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
    print_section("Levenshtein Filtering")
    print_warning("This operation can be slow for large datasets.")
    if not data:
        return data
    if len(data) > 5000:
        print_warning(f"Large dataset ({len(data):,} rules). This may take a while.")
        response = input(f"{Colors.YELLOW}Continue with Levenshtein filtering? (y/N): {Colors.RESET}").strip().lower()
        if response not in ['y', 'yes']:
            return data
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
    milestones = []
    target_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    next_target = 0
    print(f"{Colors.BOLD}{'Rank':>6} {'Rule':<30} {'Count':>10} {'Cumulative':>12} {'% Total':>8}{Colors.RESET}")
    print(f"{Colors.BOLD}{'-'*70}{Colors.RESET}")
    for i, (rule, count) in enumerate(data):
        cumulative_value += count
        current_percentage = (cumulative_value / total_value) * 100
        if i < 10 or (next_target < len(target_percentages) and current_percentage >= target_percentages[next_target]):
            color = Colors.GREEN if i < 10 else Colors.YELLOW if current_percentage <= 80 else Colors.RED
            print(f"{color}{i+1:>6} {rule:<30} {count:>10,} {cumulative_value:>12,} {current_percentage:>7.1f}%{Colors.RESET}")
            if next_target < len(target_percentages) and current_percentage >= target_percentages[next_target]:
                milestones.append((target_percentages[next_target], i + 1, current_percentage))
                next_target += 1
        if i >= 10 and next_target >= len(target_percentages):
            break
    print(f"{Colors.BOLD}{'-'*70}{Colors.RESET}")
    print(f"\n{Colors.BOLD}PARETO MILESTONES:{Colors.RESET}")
    for target, rules_needed, actual_percent in milestones:
        rules_percentage = (rules_needed / len(data)) * 100
        color = Colors.GREEN if target <= 50 else Colors.YELLOW if target <= 80 else Colors.RED
        print(f"  {color}{target:>2}% of value:{Colors.RESET} {rules_needed:>6,} rules ({rules_percentage:5.1f}% of total) - Actual: {actual_percent:5.1f}%")
    print(f"\n{Colors.BOLD}PARETO CURVE (ASCII Visualization):{Colors.RESET}")
    print("  100% ┤")
    curve_points = 20
    step = len(data) // curve_points
    for i in range(curve_points + 1):
        idx = min(i * step, len(data) - 1)
        cum_value = sum(count for _, count in data[:idx + 1])
        percentage = (cum_value / total_value) * 100
        bar_length = int(percentage / 5)
        bar = "█" * bar_length
        y_axis = 100 - (i * 5)
        if y_axis % 20 == 0 or i == 0 or i == curve_points:
            print(f"{y_axis:>4}% ┤ {bar}")
    print(f"    0% ┼{'─' * 20}")
    print(f"       {'0%':<9}{'50%':<10}{'100%':<5}")
    print(f"       {'Cumulative % of rules':<24}")

def analyze_cumulative_value(sorted_data: List[Tuple[str, int]], total_lines: int):
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
    if not data:
        print_error("No rules to save!")
        return False
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"concentrator_{mode}_{len(data)}rules_{timestamp}.rule"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Concentrator v3.0 - {mode.upper()} Rules\n")
            f.write(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total rules: {len(data):,}\n")
            f.write("#\n")
            for rule, count in data:
                f.write(f"{rule}\n")
        print_success(f"{len(data):,} rules saved to: {filename}")
        return True
    except IOError as e:
        print_error(f"Failed to save file: {e}")
        return False

def save_concentrator_rules(rules_data, output_filename, mode_name):
    if not rules_data:
        print_error("No rules to save!")
        return False
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"# CONCENTRATOR v3.0 - {mode_name.upper()} MODE OUTPUT\n")
            f.write(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total rules: {len(rules_data)}\n")
            f.write("# Format: rule_string\n")
            f.write("#\n")
            for rule_item in rules_data:
                if isinstance(rule_item, tuple):
                    rule = rule_item[0]
                else:
                    rule = rule_item
                f.write(f"{rule}\n")
        print_success(f"Saved {len(rules_data)} rules to: {output_filename}")
        return True
    except Exception as e:
        print_error(f"Failed to save rules: {e}")
        return False

def save_formatted_rules(rules_data: List[Tuple[str, str, int]], filename: str = None):
    """Save rules in formatted style with spaces between operators"""
    if not rules_data:
        print_error("No rules to save!")
        return False
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"concentrator_formatted_{len(rules_data)}rules_{timestamp}.rule"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Concentrator v3.0 - Formatted Rules\n")
            f.write(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total rules: {len(rules_data):,}\n")
            f.write("# Format: operators separated by spaces\n")
            f.write("# Example: o2K i0T\n")
            f.write("#\n")
            for _, formatted_rule, _ in rules_data:
                f.write(f"{formatted_rule}\n")
        print_success(f"{len(rules_data):,} formatted rules saved to: {filename}")
        raw_filename = filename.replace('.rule', '_raw.rule')
        with open(raw_filename, 'w', encoding='utf-8') as f:
            f.write(f"# Concentrator v3.0 - Raw Rules (from formatted)\n")
            f.write(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total rules: {len(rules_data):,}\n")
            f.write("# Format: raw rule strings (no spaces)\n")
            f.write("#\n")
            for raw_rule, _, _ in rules_data:
                f.write(f"{raw_rule}\n")
        print_success(f"{len(rules_data):,} raw rules saved to: {raw_filename}")
        return True
    except IOError as e:
        print_error(f"Failed to save file: {e}")
        return False

# ==============================================================================
# ENHANCED INTERACTIVE PROCESSING WITH FORMATTING
# ==============================================================================

def enhanced_interactive_processing_loop_with_formatting(
    original_data: List[Tuple[str, int]], 
    total_lines: int, 
    args, 
    initial_mode: str = "extracted",
    gpu_validator: Optional[GPURuleValidator] = None
):
    current_data = original_data
    unique_count = len(current_data)
    original_count = unique_count
    
    print_header("ENHANCED RULE PROCESSING - WITH FORMATTING")
    print(f"Initial dataset: {colorize(f'{original_count:,}', Colors.CYAN)} unique rules")
    print(f"Current dataset: {colorize(f'{unique_count:,}', Colors.CYAN)} unique rules")
    
    formatter = RuleFormatter()
    
    try:
        while True:
            print(f"\n{Colors.BOLD}{'-'*80}{Colors.RESET}")
            print(f"{Colors.BOLD}ADVANCED FILTERING OPTIONS:{Colors.RESET}")
            print(f" {Colors.GREEN}(1){Colors.RESET} Filter by {Colors.CYAN}MINIMUM OCCURRENCE{Colors.RESET}")
            print(f" {Colors.GREEN}(2){Colors.RESET} Filter by {Colors.CYAN}MAXIMUM NUMBER OF RULES{Colors.RESET}")
            print(f" {Colors.GREEN}(3){Colors.RESET} Filter by {Colors.CYAN}FUNCTIONAL REDUNDANCY{Colors.RESET}")
            print(f" {Colors.GREEN}(4){Colors.RESET} {Colors.YELLOW}**INVERSE MODE**{Colors.RESET}")
            print(f" {Colors.GREEN}(5){Colors.RESET} {Colors.MAGENTA}**HASHCAT CLEANUP**{Colors.RESET}")
            print(f" {Colors.GREEN}(6){Colors.RESET} {Colors.MAGENTA}**LEVENSHTEIN FILTER**{Colors.RESET}")
            print(f" {Colors.GREEN}(7){Colors.RESET} {Colors.CYAN}**FORMAT RULES**{Colors.RESET} - Add spaces between operators")
            
            print(f"\n{Colors.BOLD}ANALYSIS & UTILITIES:{Colors.RESET}")
            print(f" {Colors.BLUE}(p){Colors.RESET} Show {Colors.CYAN}PARETO analysis{Colors.RESET}")
            print(f" {Colors.BLUE}(s){Colors.RESET} {Colors.GREEN}SAVE{Colors.RESET} current rules (raw)")
            print(f" {Colors.BLUE}(f){Colors.RESET} {Colors.CYAN}SAVE FORMATTED{Colors.RESET} (with spaces)")
            print(f" {Colors.BLUE}(r){Colors.RESET} {Colors.YELLOW}RESET{Colors.RESET} to original dataset")
            print(f" {Colors.BLUE}(i){Colors.RESET} Show {Colors.CYAN}dataset information{Colors.RESET}")
            print(f" {Colors.BLUE}(q){Colors.RESET} {Colors.RED}QUIT{Colors.RESET}")
            print(f"{Colors.BOLD}{'-'*80}{Colors.RESET}")
            
            choice = input(f"{Colors.YELLOW}Enter your choice: {Colors.RESET}").strip().lower()
            
            if choice == 'q':
                print_header("THANK YOU FOR USING CONCENTRATOR v3.0!")
                break
            
            elif choice == 'p':
                display_pareto_curve(current_data)
                continue
            
            elif choice == 's':
                save_rules_to_file(current_data, mode=f"{initial_mode}_filtered")
                continue
            
            elif choice == 'f':
                if current_data:
                    formatted_data = []
                    for rule, count in current_data:
                        formatted = formatter.format_rule(rule)
                        formatted_data.append((rule, formatted, count))
                    save_formatted_rules(formatted_data)
                continue
            
            elif choice == '7':
                print_section("Rule Formatting Preview")
                print(f"Showing first 10 formatted rules:")
                print()
                for i, (rule, count) in enumerate(current_data[:10]):
                    formatted = formatter.format_rule(rule)
                    color = Colors.GREEN if i % 2 == 0 else Colors.CYAN
                    print(f"{color}{formatted:<30}{Colors.RESET}  (raw: {rule})")
                print()
                save_now = input(f"{Colors.YELLOW}Save all formatted rules? (y/N): {Colors.RESET}").strip().lower()
                if save_now in ['y', 'yes']:
                    formatted_data = []
                    for rule, count in current_data:
                        formatted = formatter.format_rule(rule)
                        formatted_data.append((rule, formatted, count))
                    save_formatted_rules(formatted_data)
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
                cleaner = HashcatRuleCleaner(1, gpu_validator)
                current_data = cleaner.clean_rules(current_data, use_gpu=True)
                unique_count = len(current_data)
            elif choice == '6':
                current_data = levenshtein_filter(current_data, getattr(args, 'levenshtein_max_dist', 2))
                unique_count = len(current_data)
            else:
                print_error("Invalid choice. Please try again.")
                continue
            
            if choice in ['1', '2', '3', '4', '5', '6']:
                reduction = ((original_count - unique_count) / original_count * 100) if original_count > 0 else 0
                print_success(f"Dataset updated: {unique_count:,} unique rules ({reduction:.1f}% reduction)")
                if unique_count > 0:
                    show_pareto = input(f"{Colors.YELLOW}Show Pareto analysis? (Y/n): {Colors.RESET}").strip().lower()
                    if show_pareto not in ['n', 'no']:
                        display_pareto_curve(current_data)
                save_now = input(f"{Colors.YELLOW}Save current dataset? (y/N): {Colors.RESET}").strip().lower()
                if save_now in ['y', 'yes']:
                    save_rules_to_file(current_data, mode=f"{initial_mode}_filtered")
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interactive menu interrupted by user.{Colors.RESET}")
        save_before_exit = input(f"{Colors.YELLOW}Save current dataset before exiting? (y/N): {Colors.RESET}").strip().lower()
        if save_before_exit in ['y', 'yes']:
            save_rules_to_file(current_data, mode=f"{initial_mode}_filtered")
        print(f"{Colors.CYAN}Returning to main program...{Colors.RESET}")
    return current_data

# ==============================================================================
# MODIFIED MAIN PROCESSING FUNCTION
# ==============================================================================

def concentrator_main_processing(args):
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
    
    if active_mode == 'markov':
        markov_min_len = args.markov_length[0]
        markov_max_len = args.markov_length[-1]
    elif active_mode == 'combo':
        combo_min_len = args.combo_length[0]
        combo_max_len = args.combo_length[-1]

    gpu_validator = None
    if not args.no_gpu:
        gpu_validator = GPURuleValidator()
        if gpu_validator.available:
            print_success("GPU Acceleration: ENABLED")
        else:
            print_warning("GPU Acceleration: Disabled (falling back to CPU)")
    else:
        print_warning("GPU Acceleration: Manually disabled")
    
    print_section("Collecting Rule Files (Recursive Search, Max Depth 3)")
    all_filepaths = find_rule_files_recursive(args.paths, max_depth=3)
    
    output_files_to_exclude = {os.path.basename(output_file_name)}
    all_filepaths = [fp for fp in all_filepaths if os.path.basename(fp) not in output_files_to_exclude]

    if not all_filepaths:
        print_error("No rule files found to process. Exiting.")
        return
    
    print_success(f"Found {len(all_filepaths)} rule files to analyze.")
    
    set_global_flags(args.temp_dir, args.in_memory)

    print_section("Starting Parallel Rule File Analysis")
    sorted_op_counts, full_rule_counts, all_clean_rules = analyze_rule_files_parallel(all_filepaths, args.max_length)
    
    if not sorted_op_counts:
        print_error("No operators found in files. Exiting.")
        return

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
            if gpu_validator and gpu_validator.available and sorted_rule_data:
                rules_to_validate = [rule for rule, weight in sorted_rule_data[:args.top_rules*2]]
                validation_results = gpu_validator.validate_rules_batch(rules_to_validate)
                validated_with_weights = []
                for rule, is_valid in zip(rules_to_validate, validation_results):
                    if is_valid:
                        for original_rule, weight in sorted_rule_data:
                            if rule == original_rule:
                                validated_with_weights.append((original_rule, weight))
                                break
                result_data = validated_with_weights[:args.top_rules]
                print_success(f"GPU validated {len(result_data)} statistically sorted rules")
            else:
                result_data = sorted_rule_data[:args.top_rules]
        else:
            mode = 'frequency'
            print_info("Sort Mode: Frequency Sort (Raw Count) with GPU Validation")
            if gpu_validator and gpu_validator.available:
                rules_list = list(full_rule_counts.keys())[:args.top_rules*2]
                validation_results = gpu_validator.validate_rules_batch(rules_list)
                valid_rules = []
                for rule, is_valid in zip(rules_list, validation_results):
                    if is_valid:
                        valid_rules.append((rule, full_rule_counts[rule]))
                result_data = valid_rules[:args.top_rules]
            else:
                result_data = list(full_rule_counts.items())[:args.top_rules]
        print_success(f"Extracted {len(result_data)} top unique rules.")
        
    elif active_mode == 'markov':
        print_section("Starting STATISTICAL Markov Rule Generation (Validated)")
        markov_rules_data = generate_rules_from_markov_model(
            markov_probabilities, 
            args.generate_target, 
            markov_min_len, 
            markov_max_len
        )
        if gpu_validator and gpu_validator.available and markov_rules_data:
            markov_rules = [rule for rule, weight in markov_rules_data]
            validation_results = gpu_validator.validate_rules_batch(markov_rules, args.max_length)
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
        top_operators_needed = find_min_operators_for_target(
            sorted_op_counts, 
            args.combo_target, 
            combo_min_len, 
            combo_max_len
        )
        print_info(f"Using the top {len(top_operators_needed)} operators to approximate {args.combo_target} rules.")
        generated_rules_set = generate_rules_parallel(
            top_operators_needed, 
            combo_min_len, 
            combo_max_len
        )
        result_data = [(rule, 1) for rule in generated_rules_set]
        if gpu_validator and gpu_validator.available and result_data:
            rules_to_validate = [rule for rule, _ in result_data]
            validation_results = gpu_validator.validate_rules_batch(rules_to_validate)
            valid_rules = []
            for (rule, count), is_valid in zip(result_data, validation_results):
                if is_valid:
                    valid_rules.append((rule, count))
            print_success(f"GPU validated {len(valid_rules)}/{len(result_data)} combinatorial rules as syntactically valid")
            result_data = valid_rules
        else:
            print_success(f"Generated {len(result_data)} combinatorial rules.")

    print(f"\n{Colors.CYAN}{Colors.BOLD}" + "="*60)
    print("ENHANCED PROCESSING OPTIONS")
    print("="*60 + f"{Colors.END}")
    print(f"{Colors.YELLOW}Would you like to apply additional filtering and optimization?{Colors.END}")
    print(f"{Colors.CYAN}Available options include:{Colors.END}")
    print(f"  {Colors.GREEN}•{Colors.END} Filter by occurrence count")
    print(f"  {Colors.GREEN}•{Colors.END} Remove functionally redundant rules") 
    print(f"  {Colors.GREEN}•{Colors.END} Apply Levenshtein distance filtering")
    print(f"  {Colors.GREEN}•{Colors.END} Hashcat rule validation (CPU/GPU compatibility)")
    print(f"  {Colors.GREEN}•{Colors.END} Pareto analysis for optimal cutoff selection")
    print(f"  {Colors.GREEN}•{Colors.END} Format rules with spaces between operators")
    
    enter_interactive = input(f"\n{Colors.YELLOW}Enter enhanced interactive mode? (Y/n): {Colors.END}").strip().lower()
    
    if enter_interactive not in ['n', 'no']:
        total_lines_estimate = sum(full_rule_counts.values())
        final_data = enhanced_interactive_processing_loop_with_formatting(
            result_data, total_lines_estimate, args, initial_mode, gpu_validator
        )
        if final_data:
            save_concentrator_rules(final_data, output_file_name, active_mode)
            print_success(f"Final processed rules saved to: {output_file_name}")
    else:
        if result_data:
            save_concentrator_rules(result_data, output_file_name, active_mode)
            print_success(f"Rules saved to: {output_file_name}")

    print_success("Processing Complete")
    
    print_header("CONCENTRATOR USAGE STATEMENT")
    print(f"{Colors.YELLOW}This tool can significantly reduce rule file size while")
    print(f"maintaining or even improving cracking effectiveness.")
    print(f"For even better results, it is recommended to debug rules obtained by using Concentrator.{Colors.END}")
    
    if gpu_validator and gpu_validator.available:
        print_success("GPU Acceleration was used for improved performance")
    
    print_memory_status()

def process_multiple_files_concentrator(args):
    """Original processing mode - kept for compatibility"""
    print_header("PROCESSING MODE - Interactive Rule Minimization")
    all_filepaths = find_rule_files_recursive(args.paths, max_depth=3)
    if not all_filepaths:
        print_error("No rule files found to process.")
        return
    set_global_flags(args.temp_dir, args.in_memory)
    sorted_op_counts, full_rule_counts, all_clean_rules = analyze_rule_files_parallel(all_filepaths, args.max_length)
    if not full_rule_counts:
        print_error("No rules found in files.")
        return
    rules_data = list(full_rule_counts.items())
    rules_data.sort(key=lambda x: x[1], reverse=True)
    print_success(f"Loaded {len(rules_data):,} unique rules for processing.")
    final_data = enhanced_interactive_processing_loop_with_formatting(rules_data, sum(full_rule_counts.values()), args, "processed")
    if final_data:
        output_file = args.output_base_name + "_processed.rule"
        save_rules_to_file(final_data, output_file, "processed")

def interactive_mode():
    """Original interactive mode - kept for compatibility"""
    print_header("CONCENTRATOR v3.0 - INTERACTIVE MODE")
    settings = {}
    print(f"\n{Colors.CYAN}Input Configuration:{Colors.END}")
    while True:
        paths_input = input(f"{Colors.YELLOW}Enter rule files/directories (space separated): {Colors.END}").strip()
        if paths_input:
            paths = paths_input.split()
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
    # Quick analysis (simplified)
    all_filepaths = find_rule_files_recursive(settings['paths'], max_depth=3)
    if not all_filepaths:
        print_error("No rule files found in the provided paths.")
        return None
    print_success(f"Found {len(all_filepaths)} rule files.")
    # Mode selection
    print(f"\n{Colors.CYAN}Processing Mode:{Colors.END}")
    print(f"  {Colors.GREEN}1{Colors.END} - Extract top existing rules")
    print(f"  {Colors.GREEN}2{Colors.END} - Generate combinatorial rules") 
    print(f"  {Colors.GREEN}3{Colors.END} - Generate Markov rules")
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
    else:
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
    if not settings['in_memory']:
        temp_dir = input(f"{Colors.YELLOW}Temporary directory (default: system temp): {Colors.END}").strip()
        settings['temp_dir'] = temp_dir if temp_dir else None
    else:
        settings['temp_dir'] = None
    # Defaults
    required_keys = {'temp_dir': None, 'no_gpu': False, 'in_memory': False, 'max_length': 31, 'output_base_name': 'concentrator_output'}
    if settings['mode'] == 'extraction':
        required_keys.update({'top_rules': 10000, 'statistical_sort': False})
    else:
        required_keys.update({'target_rules': 10000, 'min_len': 1, 'max_len': 3})
    for key, default_value in required_keys.items():
        if key not in settings:
            settings[key] = default_value
    # Summary
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

def print_usage():
    print(f"{Colors.BOLD}{Colors.CYAN}USAGE:{Colors.END}")
    print(f"  {Colors.WHITE}python concentrator.py [OPTIONS] FILE_OR_DIRECTORY [FILE_OR_DIRECTORY...]{Colors.END}")
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}MODES (choose one):{Colors.END}")
    print(f"  {Colors.GREEN}-e, --extract-rules{Colors.END}     Extract top existing rules")
    print(f"  {Colors.GREEN}-g, --generate-combo{Colors.END}    Generate combinatorial rules") 
    print(f"  {Colors.GREEN}-gm, --generate-markov-rules{Colors.END} Generate Markov rules")
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

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    multiprocessing.freeze_support()  
    
    print_banner()
    
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
    
    if len(sys.argv) == 1:
        settings = interactive_mode()
        if not settings:
            sys.exit(0)
        
        class Args:
            def __init__(self, settings):
                self.paths = settings['paths']
                self.output_base_name = settings['output_base_name']
                self.max_length = settings['max_length']
                self.no_gpu = settings['no_gpu']
                self.in_memory = settings['in_memory']
                self.temp_dir = settings['temp_dir']
                self.extract_rules = (settings['mode'] == 'extraction')
                self.generate_combo = (settings['mode'] == 'combo')
                self.generate_markov_rules = (settings['mode'] == 'markov')
                self.process_rules = False
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
        print_usage()
        sys.exit(0)
    else:
        parser = argparse.ArgumentParser(
            description=f'{Colors.CYAN}Unified Hashcat Rule Processor with OpenCL support.{Colors.END}',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=""
        )
        parser.add_argument('paths', nargs='+', help='Paths to rule files or directories to analyze recursively (max depth 3)')
        parser.add_argument('-ob', '--output_base_name', type=str, default='concentrator_output', help='Base name for output file')
        output_group = parser.add_mutually_exclusive_group(required=True)
        output_group.add_argument('-e', '--extract_rules', action='store_true', help='Enables rule extraction and sorting.')
        parser.add_argument('-t', '--top_rules', type=int, default=10000, help='Number of top existing rules to extract.')
        parser.add_argument('-s', '--statistical_sort', action='store_true', help='Sort EXTRACTED rules by Markov probability.')
        output_group.add_argument('-g', '--generate_combo', action='store_true', help='Enables generating combinatorial rules.')
        parser.add_argument('-n', '--combo_target', type=int, default=100000, help='Approximate number of rules to generate.')
        parser.add_argument('-l', '--combo_length', nargs='+', type=int, default=[1, 3], help='Range of rule chain lengths.')
        output_group.add_argument('-gm', '--generate_markov_rules', action='store_true', help='Enables generating Markov rules.')
        parser.add_argument('-gt', '--generate_target', type=int, default=10000, help='Target number of rules to generate.')
        parser.add_argument('-ml', '--markov_length', nargs='+', type=int, default=None, help='Range of rule chain lengths.')
        output_group.add_argument('-p', '--process_rules', action='store_true', help='Enables interactive rule processing.')
        parser.add_argument('-d', '--use_disk', action='store_true', help='Use disk for initial consolidation.')
        parser.add_argument('-ld', '--levenshtein_max_dist', type=int, default=2, help='Max Levenshtein distance.')
        parser.add_argument('-m', '--max_length', type=int, default=31, help='Maximum rule length.')
        parser.add_argument('--temp-dir', type=str, default=None, help='Directory for temporary files.')
        parser.add_argument('--in-memory', action='store_true', help='Process entirely in RAM.')
        parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration.')
        
        args = parser.parse_args()
        
        if hasattr(args, 'markov_length') and args.markov_length is None:
            args.markov_length = [1, 3]
        
        if args.process_rules:
            process_multiple_files_concentrator(args)
        else:
            concentrator_main_processing(args)
    
    cleanup_temp_files()
    sys.exit(0)
