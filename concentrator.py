#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations  # Python 3.8+ type-hint compatibility
"""
CONCENTRATOR v3.3 - Unified Hashcat Rule Processor

Changes from v3.2 → v3.3
─────────────────────────
1. Full token-level Markov model (get_markov_model)
   The model is now built on atomic hashcat TOKENS produced by TOKEN_REGEX
   (e.g. 'l', '$5', 'sae', 'T3') instead of raw characters.  Three
   transition tables are maintained:

     START → first_token          (key: '^')
     token_i → token_{i+1}        (key: token string, unigram context)
     (token_{i-1}, token_i) →     (key: 2-tuple of token strings, bigram ctx)
         token_{i+1}

   Only rules that pass the round-trip tokenisation check are used for
   training, so the model is never confused by partially-tokenisable input.
   The function now prints a summary of how many start/unigram/bigram
   contexts were learned.

2. Token-level rule scoring (get_markov_weighted_rules)
   Log-probability scoring now operates on TOKEN sequences, not characters.
   Scoring strategy: P(tok[0]|START) × ∏ P(tok[i]|bigram or unigram ctx).
   Rules that do not tokenise cleanly are silently skipped.

3. Token-level Markov walk (generate_rules_from_markov_model)
   The random walk now samples full TOKENS at every step, replacing the old
   character-by-character walk.  Key improvements:

   • min_len / max_len count TOKENS (operators), not raw bytes.  This is the
     natural measure of rule complexity for hashcat (a rule with 3 tokens is
     exactly 3 chained operators regardless of argument byte lengths).
   • Every walk candidate is structurally valid — concatenating valid tokens
     always produces a valid rule — so the rejection rate is near zero
     compared to the old character-level walk.
   • Banned operators (NEVER_PRODUCE_OPS) are excluded at sampling time;
     the _has_banned_op() paranoia check and the TOKEN_REGEX round-trip gate
     are still applied before any rule is accepted.
   • Budget increased to target × 20 attempts (was × 5) to compensate for
     potentially smaller Markov graphs when only real operators are nodes.

Changes from v3.1 → v3.2
─────────────────────────
1. Full-token analysis (process_single_file)
   Operator counting now uses TOKEN_REGEX.findall so that each full token
   (e.g. '$5', 'sae', 'T3') is counted as one atomic unit instead of
   counting the operator byte and its argument bytes independently.
   This means find_min_operators_for_target works on the correct token pool
   and combinatorial counts accurately reflect producible rule-chains.

2. Round-trip re-parse validation gate (_generate_for_length)
   After joining a token combination into a rule string, the string is
   re-tokenised with TOKEN_REGEX.  The result must equal the original token
   list exactly.  This catches any case where two adjacent tokens accidentally
   merge into a different longer operator at the string boundary.
   Any combination that fails this round-trip check is silently dropped.

3. Python 3.8+ compatibility
   Added `from __future__ import annotations` so that PEP-604 / PEP-585
   style hints (list[str], dict[str, int] …) work on Python 3.8 and 3.9
   without raising TypeError at import time.

Changes from v3.0 → v3.1
─────────────────────────
1. NEVER_PRODUCE_OPS  — single, authoritative constant that lists every operator
   (memory: M 4 6 X  and  reject: < > ! / ( ) = % Q) that must NEVER appear in
   any produced/extracted/generated output, regardless of CPU/GPU mode.
   Previously these were only blocked in GPU mode via _GPU_DENIED_OPS; now they
   are filtered at *every* processing stage:
     • process_single_file   – skips any loaded rule that contains a banned op
     • gpu_extract_and_validate_rules – post-validation filter
     • find_min_operators_for_target / combinatorial generation – banned ops
       stripped from the operator pool before combinations are built
     • generate_rules_from_markov_model – uses NEVER_PRODUCE_OPS (replaces the
       now-removed EXCLUDED_MARKOV_OPERATORS constant)
     • HashcatRuleCleaner.validate_rule (both modes) – always rejects banned ops
     • save_rules – final safety net, drops any slipped-through rule

2. Expanded TEST_VECTOR (21 → 52 words)
   The minimizer deduplicates rules whose functional signatures are identical on
   the test vector.  With only 21 short words many position-specific operators
   (D5, i7X, O3A …) produced identical signatures and were incorrectly merged.
   The new vector covers:
     • lengths 1-32  (exercises every base-36 position operand 0-V)
     • all-lowercase, all-uppercase, mixed, digits-only
     • strings with special characters
     • strings with internal spaces, underscores, hyphens
   This makes false-positive collisions far less likely.

3. Minor clean-ups
   • _GPU_DENIED_OPS removed (replaced by NEVER_PRODUCE_OPS)
   • EXCLUDED_MARKOV_OPERATORS removed (was identical to _GPU_DENIED_OPS)
   • _has_banned_op() helper added
"""

import sys
import os
import re
import signal
import argparse
import math
import itertools
import multiprocessing
import tempfile
import random
import datetime
import threading
import functools
import sqlite3
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable, Any, Set, Optional

# ---------------------------------------------------------------------------
# Third-party imports with fallbacks
# ---------------------------------------------------------------------------
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

    class tqdm:  # minimal shim
        def __init__(self, iterable=None, total=None, desc=None, unit=None, **_):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0

        def __iter__(self):
            if self.desc:
                print(f"{self.desc}...", end="", flush=True)
            for item in self.iterable:
                yield item
                self.n += 1
            if self.desc:
                print(" done")

        def update(self, n: int = 1) -> None:
            self.n += n

        def close(self) -> None:
            pass

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ==============================================================================
# GLOBAL STATE (minimal — prefer AppState below)
# ==============================================================================

_cleanup_lock = threading.Lock()
_cleanup_in_progress = False
_temp_files_to_cleanup: List[str] = []


@dataclass
class AppState:
    """Replaces scattered module-level mutable globals."""
    temp_dir_path: Optional[str] = None
    in_memory_mode: bool = False
    output_format: str = "line"          # 'line' | 'expanded'
    gpu_mode_enabled: bool = False
    opencl_context: Any = None
    opencl_queue: Any = None
    opencl_program: Any = None


# Singleton application state
STATE = AppState()


# ==============================================================================
# COLORS
# ==============================================================================

class Colors:
    RED       = '\033[91m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    BLUE      = '\033[94m'
    MAGENTA   = '\033[95m'
    CYAN      = '\033[96m'
    WHITE     = '\033[97m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    END       = '\033[0m'
    RESET     = '\033[0m'
    BG_RED    = '\033[41m'
    BG_GREEN  = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE   = '\033[44m'
    BG_MAGENTA= '\033[45m'
    BG_CYAN   = '\033[46m'


# ==============================================================================
# OPERATOR DEFINITIONS – COMPREHENSIVE HASHCAT RULE SYNTAX
# ==============================================================================

BASE36_CHARS  = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
PRINTABLE_ASCII = set(chr(i) for i in range(32, 127))

# Maps operator character → list of argument types ('num' or 'char')
OPERATOR_ARGS: Dict[str, List[str]] = {
    # No arguments
    ':': [], 'l': [], 'u': [], 'c': [], 'C': [], 't': [], 'r': [], 'd': [], 'f': [],
    '{': [], '}': [], '[': [], ']': [], 'q': [], 'M': [], '4': [], '6': [], 'k': [], 'K': [],
    'Q': [], 'E': [],
    # One base-36 number
    'T': ['num'], 'p': ['num'], 'D': ['num'], 'z': ['num'], 'Z': ['num'], "'": ['num'],
    '+': ['num'], '-': ['num'], '.': ['num'], ',': ['num'], 'L': ['num'], 'R': ['num'],
    'y': ['num'], 'Y': ['num'], '<': ['num'], '>': ['num'], '_': ['num'],
    # Two base-36 numbers
    'x': ['num', 'num'], 'O': ['num', 'num'], '*': ['num', 'num'],
    # One literal character
    '$': ['char'], '^': ['char'], '@': ['char'], '!': ['char'], '/': ['char'],
    '(': ['char'], ')': ['char'],
    # Two literal characters
    's': ['char', 'char'],
    # One base-36 number + one literal character
    'i': ['num', 'char'], 'o': ['num', 'char'], '=': ['num', 'char'], '%': ['num', 'char'],
    '3': ['num', 'char'],
    # Three base-36 numbers
    'X': ['num', 'num', 'num'],
    # One literal character (eX)
    'e': ['char'],
}

ALL_OPERATORS = list(OPERATOR_ARGS.keys())

ALL_RULE_CHARS = set(
    '0123456789abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ':,.lu.#()=%!?|~+*-^$sStTiIoOcCrRyYzZeEfFxXdDpPbBqQ`[]><@&vV'
)

# ---------------------------------------------------------------------------
# NEVER_PRODUCE_OPS  (v3.1)
# ---------------------------------------------------------------------------
NEVER_PRODUCE_OPS: Set[str] = frozenset({'M', '4', '6', 'X', '<', '>', '!', '/', '(', ')', '=', '%', 'Q'})


def _has_banned_op(rule: str) -> bool:
    """Return True if *rule* contains any operator from NEVER_PRODUCE_OPS."""
    tokens = TOKEN_REGEX.findall(rule)
    return any(t[0] in NEVER_PRODUCE_OPS for t in tokens)


def _build_token_regex() -> re.Pattern:
    """Compile regex that tokenizes a hashcat rule into operator+arg chunks."""
    patterns = []
    for op, args in OPERATOR_ARGS.items():
        escaped = re.escape(op)
        arg_pat = ''.join('[0-9A-Z]' if a == 'num' else '[ -~]' for a in args)
        patterns.append(escaped + arg_pat)
    patterns.sort(key=len, reverse=True)
    return re.compile('|'.join(patterns))


def _build_count_regex() -> re.Pattern:
    """Compile regex that matches any operator character (for counting)."""
    patterns = sorted([re.escape(op) for op in ALL_OPERATORS], key=len, reverse=True)
    return re.compile('|'.join(patterns))


TOKEN_REGEX    = _build_token_regex()
OPERATOR_REGEX = _build_count_regex()

OPERATORS_REQUIRING_ARGS: Dict[str, int] = {
    op: len(args) for op, args in OPERATOR_ARGS.items() if args
}


# ==============================================================================
# UTILITY / PRINT HELPERS
# ==============================================================================

def print_banner() -> None:
    print(f"\n{Colors.CYAN}{Colors.BOLD}" + "=" * 80)
    print("          CONCENTRATOR v3.3 - Unified Hashcat Rule Processor")
    print("=" * 80 + f"{Colors.END}")
    features = [
        "OpenCL GPU Acceleration for validation and generation",
        "Three Processing Modes: Extraction, Combinatorial, Markov",
        "Hashcat Rule Engine Simulation & Functional Minimization",
        "Rule Validation and Cleanup (CPU/GPU compatible)",
        "Levenshtein Distance Filtering",
        "Smart Processing Selection & Memory Safety",
        "Interactive & CLI Modes with Colorized Output",
        "Multiple output formats: line, expanded",
        "Memory/Reject operator guard — filtered at every pipeline stage",
    ]
    for f in features:
        print(f"  {Colors.GREEN}•{Colors.END} {f}")
    print(f"{Colors.CYAN}{Colors.BOLD}" + "=" * 80 + f"{Colors.END}\n")


def print_header(text: str) -> None:
    print(f"\n{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE}{text:^80}{Colors.RESET}")
    print(f"{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE}{'=' * 80}{Colors.RESET}")


def print_section(text: str) -> None:
    print(f"\n{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE} {text} {Colors.RESET}")


def print_warning(text: str) -> None:
    print(f"{Colors.BG_YELLOW}{Colors.BOLD}{Colors.BLUE}⚠️  WARNING:{Colors.RESET} {Colors.YELLOW}{text}{Colors.RESET}")


def print_error(text: str) -> None:
    print(f"{Colors.BG_RED}{Colors.BOLD}{Colors.WHITE}❌ ERROR:{Colors.RESET} {Colors.RED}{text}{Colors.RESET}")


def print_success(text: str) -> None:
    print(f"{Colors.BG_GREEN}{Colors.BOLD}{Colors.WHITE}✅ SUCCESS:{Colors.RESET} {Colors.GREEN}{text}{Colors.RESET}")


def print_info(text: str) -> None:
    print(f"{Colors.BG_BLUE}{Colors.BOLD}{Colors.WHITE}ℹ️  INFO:{Colors.RESET} {Colors.BLUE}{text}{Colors.RESET}")


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


def get_yes_no(prompt: str, default: bool = True) -> bool:
    choices = "Y/n" if default else "y/N"
    response = input(f"{prompt} [{choices}]: ").strip().lower()
    if not response:
        return default
    return response in ('y', 'yes')


# ==============================================================================
# MEMORY MANAGEMENT
# ==============================================================================

def signal_handler(sig, frame) -> None:
    global _cleanup_in_progress
    with _cleanup_lock:
        if _cleanup_in_progress:
            return
        _cleanup_in_progress = True
    if multiprocessing.current_process().name != 'MainProcess':
        sys.exit(0)
    print(f"\n{Colors.RED}⚠️  INTERRUPT RECEIVED - Cleaning up...{Colors.RESET}")
    _remove_temp_files()
    print(f"{Colors.RED}Script terminated by user.{Colors.RESET}")
    sys.exit(1)


def _remove_temp_files() -> None:
    for fp in list(_temp_files_to_cleanup):
        try:
            if os.path.exists(fp):
                os.remove(fp)
                print(f"{Colors.GREEN}✓ Removed: {fp}{Colors.RESET}")
        except OSError:
            pass


def cleanup_temp_files() -> None:
    if not _temp_files_to_cleanup:
        return
    print_info("Cleaning up temporary files...")
    for fp in list(_temp_files_to_cleanup):
        try:
            if os.path.exists(fp):
                os.remove(fp)
                print_info(f"Cleaned up: {fp}")
            _temp_files_to_cleanup.remove(fp)
        except OSError:
            pass


def get_memory_usage() -> Optional[Dict[str, float]]:
    if not PSUTIL_AVAILABLE:
        return None
    try:
        vm   = psutil.virtual_memory()
        swap = psutil.swap_memory()
        total_used  = vm.used  + swap.used
        total_avail = vm.total + swap.total
        return {
            'ram_used':      vm.used,
            'ram_total':     vm.total,
            'ram_percent':   vm.percent,
            'swap_used':     swap.used,
            'swap_total':    swap.total,
            'swap_percent':  swap.percent,
            'total_used':    total_used,
            'total_available': total_avail,
            'total_percent': total_used / total_avail * 100,
        }
    except Exception as exc:
        print_error(f"Could not monitor memory: {exc}")
        return None


def format_bytes(n: float) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def check_memory_safety(threshold_percent: float = 85.0) -> bool:
    mem = get_memory_usage()
    if not mem:
        return True
    pct = mem['total_percent']
    if pct >= threshold_percent:
        print_warning(f"System memory at {pct:.1f}% (threshold: {threshold_percent}%)")
        print(f"   {Colors.CYAN}RAM:{Colors.RESET}  {format_bytes(mem['ram_used'])} / "
              f"{format_bytes(mem['ram_total'])} ({mem['ram_percent']:.1f}%)")
        print(f"   {Colors.CYAN}Swap:{Colors.RESET} {format_bytes(mem['swap_used'])} / "
              f"{format_bytes(mem['swap_total'])} ({mem['swap_percent']:.1f}%)")
        return False
    return True


def memory_safe_operation(operation_name: str, threshold_percent: float = 85.0):
    """Decorator that warns about memory before running an expensive function."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print_section(f"Memory Check before {operation_name}")
            if not check_memory_safety(threshold_percent):
                print_error(f"{operation_name} requires significant memory.")
                resp = input(
                    f"{Colors.YELLOW}Continue with {operation_name} anyway? (y/N): {Colors.RESET}"
                ).strip().lower()
                if resp not in ('y', 'yes'):
                    print_error(f"{operation_name} cancelled due to memory constraints.")
                    return None
            print_success(f"Starting {operation_name}...")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def estimate_memory_usage(rules_count: int, avg_rule_length: int = 50) -> int:
    return rules_count * (avg_rule_length + 50)


def print_memory_status() -> None:
    mem = get_memory_usage()
    if not mem:
        return
    ram_color = Colors.RED if mem['ram_percent'] > 85 else (
        Colors.YELLOW if mem['ram_percent'] > 70 else Colors.GREEN
    )
    print(
        f"{Colors.CYAN}Memory Status:{Colors.END} "
        f"{ram_color}RAM {mem['ram_percent']:.1f}% "
        f"({format_bytes(mem['ram_used'])}/{format_bytes(mem['ram_total'])}){Colors.END}",
        end="",
    )
    if mem['swap_total'] > 0:
        if mem['swap_used'] > 0:
            sc = Colors.YELLOW if mem['swap_percent'] < 50 else Colors.RED
            print(f" | {Colors.CYAN}SWAP:{Colors.END} {sc}{mem['swap_percent']:.1f}%"
                  f" ({format_bytes(mem['swap_used'])}/{format_bytes(mem['swap_total'])}){Colors.END}")
        else:
            print(f" | {Colors.CYAN}Swap:{Colors.END} {Colors.GREEN}available"
                  f" ({format_bytes(mem['swap_total'])}){Colors.END}")
    else:
        print(f" | {Colors.CYAN}Swap:{Colors.END} {Colors.RED}not available{Colors.END}")


def memory_intensive_operation_warning(operation_name: str) -> bool:
    mem = get_memory_usage()
    if not mem:
        return True
    if mem['ram_percent'] > 85:
        print(f"{Colors.RED}{Colors.BOLD}WARNING:{Colors.END} {Colors.YELLOW}"
              f"High RAM usage ({mem['ram_percent']:.1f}%) for {operation_name}{Colors.END}")
        print_memory_status()
        if mem['swap_total'] == 0:
            print(f"{Colors.RED}CRITICAL: No swap space available.{Colors.END}")
            resp = input(
                f"{Colors.YELLOW}Continue with memory-intensive operation? (y/N): {Colors.END}"
            ).strip().lower()
            return resp not in ('n', 'no')
        else:
            print(f"{Colors.YELLOW}System will use swap. Performance may degrade.{Colors.END}")
    return True


# ==============================================================================
# FILE MANAGEMENT
# ==============================================================================

def find_rule_files_recursive(paths: List[str], max_depth: int = 3) -> List[str]:
    extensions = {'.rule', '.rules', '.hr', '.hashcat', '.txt', '.lst'}
    found: List[str] = []
    for path in paths:
        if os.path.isfile(path):
            if os.path.splitext(path.lower())[1] in extensions:
                found.append(path)
                print_success(f"Rule file: {path}")
            else:
                print_warning(f"Not a rule file (wrong extension): {path}")
        elif os.path.isdir(path):
            print_info(f"Scanning directory: {path} (max depth: {max_depth})")
            count = 0
            for root, dirs, files in os.walk(path):
                depth = root[len(path):].count(os.sep)
                if depth >= max_depth:
                    dirs.clear()
                    continue
                for name in files:
                    if os.path.splitext(name.lower())[1] in extensions:
                        fp = os.path.join(root, name)
                        found.append(fp)
                        count += 1
                        suffix = f" (depth {depth})" if depth else ""
                        print_success(f"Rule file{suffix}: {fp}")
            if count == 0:
                print_warning(f"No rule files found in: {path}")
            else:
                print_success(f"Found {count} rule files in: {path}")
        else:
            print_error(f"Path not found: {path}")
    return sorted(set(found))


def set_global_flags(temp_dir_path: Optional[str], in_memory_mode: bool) -> None:
    STATE.in_memory_mode = in_memory_mode
    if temp_dir_path and not in_memory_mode:
        STATE.temp_dir_path = temp_dir_path
        try:
            os.makedirs(STATE.temp_dir_path, exist_ok=True)
            print_info(f"Using temporary directory: {STATE.temp_dir_path}")
        except OSError as exc:
            print_warning(f"Could not create temp dir {temp_dir_path}: {exc}. Using system temp.")
            STATE.temp_dir_path = None
    elif in_memory_mode:
        print_info("In-Memory Mode activated.")


# ==============================================================================
# RULE VALIDATION (CPU)
# ==============================================================================

def is_valid_hashcat_rule(rule: str) -> bool:
    """Return True if *rule* is syntactically valid according to OPERATOR_ARGS."""
    tokens = TOKEN_REGEX.findall(rule)
    if ''.join(tokens) != rule:
        return False
    for token in tokens:
        op = token[0]
        if op not in OPERATOR_ARGS:
            return False
        if len(token) != 1 + len(OPERATOR_ARGS[op]):
            return False
        for idx, arg_type in enumerate(OPERATOR_ARGS[op]):
            if arg_type == 'num' and token[1 + idx] not in BASE36_CHARS:
                return False
    return True


# ==============================================================================
# OPENCL SETUP
# ==============================================================================

OPENCL_VALIDATION_KERNEL = r"""
__kernel void validate_rules_batch(
    __global const uchar* rules,
    __global uchar* results,
    const uint rule_stride,
    const uint max_rule_len,
    const uint num_rules)
{
    uint idx = get_global_id(0);
    if (idx >= num_rules) return;
    __global const uchar* rule = rules + idx * rule_stride;
    bool valid = true;
    for (uint i = 0; i < max_rule_len && rule[i] != 0; i++) {
        uchar c = rule[i];
        bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
                || c == ':' || c == ',' || c == '.' || c == '(' || c == ')' || c == '='
                || c == '%' || c == '!' || c == '?' || c == '|' || c == '~' || c == '+'
                || c == '*' || c == '-' || c == '^' || c == '$' || c == '[' || c == ']'
                || c == '>' || c == '<' || c == '@' || c == '&' || c == 'v' || c == 'V'
                || c == '#' || c == '`' || c == '/';
        if (!ok) { valid = false; break; }
    }
    results[idx] = valid ? 1 : 0;
}
"""


def setup_opencl() -> bool:
    if not OPENCL_AVAILABLE:
        return False
    try:
        platforms = cl.get_platforms()
        if not platforms:
            print_warning("No OpenCL platforms found.")
            return False
        devices = platforms[0].get_devices(cl.device_type.GPU)
        if not devices:
            print_warning("No GPU devices; trying CPU.")
            devices = platforms[0].get_devices(cl.device_type.CPU)
        if not devices:
            print_warning("No OpenCL devices found.")
            return False
        STATE.opencl_context = cl.Context(devices)
        STATE.opencl_queue   = cl.CommandQueue(STATE.opencl_context)
        STATE.opencl_program = cl.Program(STATE.opencl_context, OPENCL_VALIDATION_KERNEL).build()
        print_success(f"OpenCL initialised on: {devices[0].name}")
        return True
    except Exception as exc:
        print_error(f"OpenCL initialisation failed: {exc}")
        return False


def gpu_validate_rules(rules_list: List[str], max_rule_length: int = 64) -> List[bool]:
    if not STATE.opencl_context or not rules_list:
        return [False] * len(rules_list)
    if not NUMPY_AVAILABLE:
        return [is_valid_hashcat_rule(r) for r in rules_list]
    try:
        n = len(rules_list)
        stride = ((max_rule_length + 15) // 16) * 16
        buf = np.zeros((n, stride), dtype=np.uint8)
        for i, rule in enumerate(rules_list):
            rb = rule.encode('ascii', 'ignore')
            ln = min(len(rb), stride)
            buf[i, :ln] = np.frombuffer(rb[:ln], dtype=np.uint8)
        results = np.zeros(n, dtype=np.uint8)
        mf = cl.mem_flags
        rules_gpu   = cl.Buffer(STATE.opencl_context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=buf)
        results_gpu = cl.Buffer(STATE.opencl_context, mf.WRITE_ONLY, results.nbytes)
        STATE.opencl_program.validate_rules_batch(
            STATE.opencl_queue, (n,), None,
            rules_gpu, results_gpu,
            np.uint32(stride), np.uint32(max_rule_length), np.uint32(n),
        )
        cl.enqueue_copy(STATE.opencl_queue, results, results_gpu)
        STATE.opencl_queue.finish()
        return [bool(r) for r in results]
    except Exception as exc:
        print_error(f"GPU validation failed: {exc}; falling back to CPU.")
        return [is_valid_hashcat_rule(r) for r in rules_list]


# ==============================================================================
# PARALLEL FILE PROCESSING
# ==============================================================================

def process_single_file(filepath: str, max_rule_length: int) -> Tuple:
    """
    Read one rule file and return:
      (operator_counts, rule_counts, clean_rules_list, temp_filepath_or_None)

    v3.1: rules containing any operator from NEVER_PRODUCE_OPS are silently
    dropped at this stage so they never enter the processing pipeline.

    v3.2: operator counting now uses TOKEN_REGEX.findall so that full tokens
    (e.g. '$5', 'sae', 'T3') are counted as atomic units instead of counting
    the operator character and its argument bytes separately.
    """
    operator_counts:  Dict[str, int] = defaultdict(int)
    full_rule_counts: Dict[str, int] = defaultdict(int)
    clean_rules:      List[str]      = []
    tmp_path:         Optional[str]  = None

    try:
        with open(filepath, 'r', errors='ignore') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#') or len(line) > max_rule_length:
                    continue
                clean = ''.join(c for c in line if c in ALL_RULE_CHARS)
                if not clean:
                    continue
                # --- v3.1: drop memory/reject operators at load time ----------
                if _has_banned_op(clean):
                    continue
                # --------------------------------------------------------------
                full_rule_counts[clean] += 1
                clean_rules.append(clean)
                # v3.2: count full tokens (e.g. '$5', 'sae') not bare op chars
                for token in TOKEN_REGEX.findall(clean):
                    operator_counts[token] += 1

        if not STATE.in_memory_mode:
            with tempfile.NamedTemporaryFile(
                mode='w+', delete=False, encoding='utf-8',
                dir=STATE.temp_dir_path, prefix='concentrator_', suffix='.tmp',
            ) as tf:
                tmp_path = tf.name
                tf.writelines(r + '\n' for r in clean_rules)
            with _cleanup_lock:
                _temp_files_to_cleanup.append(tmp_path)
            print_success(f"Processed: {filepath} → {tmp_path}")
            return operator_counts, full_rule_counts, [], tmp_path
        else:
            print_success(f"Processed (in-memory): {filepath}")
            return operator_counts, full_rule_counts, clean_rules, None

    except Exception as exc:
        print_error(f"Error processing {filepath}: {exc}")
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                with _cleanup_lock:
                    if tmp_path in _temp_files_to_cleanup:
                        _temp_files_to_cleanup.remove(tmp_path)
            except OSError:
                pass
        return defaultdict(int), defaultdict(int), [], None


def analyze_rule_files_parallel(
    filepaths: List[str], max_rule_length: int
) -> Tuple[List, Dict, List]:
    valid_fps = [fp for fp in filepaths if os.path.isfile(fp)]
    if not valid_fps:
        print_warning("No valid rule files to process.")
        return [], defaultdict(int), []

    total_op_counts:   Dict[str, int] = defaultdict(int)
    total_rule_counts: Dict[str, int] = defaultdict(int)
    temp_files: List[str] = []
    all_rules:  List[str] = []

    n_procs = min(os.cpu_count() or 1, len(valid_fps))
    tasks   = [(fp, max_rule_length) for fp in valid_fps]
    print_info(f"Parallel analysis of {len(valid_fps)} files using {n_procs} processes...")

    with multiprocessing.Pool(processes=n_procs) as pool:
        for op_c, rule_c, rules, tmp in pool.starmap(process_single_file, tasks):
            for op, cnt in op_c.items():
                total_op_counts[op] += cnt
            for rule, cnt in rule_c.items():
                total_rule_counts[rule] += cnt
            if STATE.in_memory_mode:
                all_rules.extend(rules)
            elif tmp:
                temp_files.append(tmp)

    if not STATE.in_memory_mode and temp_files:
        print_info("Merging temporary rule files...")
        for tmp in temp_files:
            try:
                with open(tmp, 'r', encoding='utf-8') as fh:
                    all_rules.extend(ln.strip() for ln in fh)
                os.remove(tmp)
                with _cleanup_lock:
                    if tmp in _temp_files_to_cleanup:
                        _temp_files_to_cleanup.remove(tmp)
            except OSError as exc:
                print_error(f"Error merging {tmp}: {exc}")

    print_success(f"Total unique rules loaded: {len(total_rule_counts):,}")
    sorted_op_counts = sorted(total_op_counts.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_op_counts, total_rule_counts, all_rules


# ==============================================================================
# MARKOV MODEL
# ==============================================================================

def get_markov_model(
    unique_rules: Dict[str, int]
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Build a second-order token-level Markov model from a rule corpus.

    The model operates on atomic hashcat TOKENS (as produced by TOKEN_REGEX),
    NOT on raw characters.  A token is the smallest meaningful unit of a
    hashcat rule — e.g. 'l', 'u', '$5', 'sae', 'T3', 'i2X'.  Using tokens
    instead of characters ensures that:

      • Transitions reflect semantically meaningful operator sequences.
      • Every walk through the model produces structurally valid rules because
        every token is itself a complete, valid operator+argument unit.
      • min_len / max_len refer to operator counts, not byte lengths, which is
        the natural measure of rule complexity for hashcat.

    Three transition tables are stored under the same 'probs' dict:

      START → first_token          key: '^'  (string)
      token_i → token_{i+1}        key: token_i  (string, unigram context)
      (token_{i-1}, token_i) →     key: (token_{i-1}, token_i)  (tuple, bigram context)
          token_{i+1}

    The bigram context is tried first during generation/scoring; the unigram
    is used as a fallback when no bigram entry exists.
    """
    if not memory_intensive_operation_warning("Markov model building"):
        return None, None
    print_section("Building Token-Level Markov Sequence Probability Model")
    counts: Dict = defaultdict(lambda: defaultdict(int))
    START = '^'
    skipped = 0
    for rule in unique_rules:
        if not rule:
            continue
        tokens = TOKEN_REGEX.findall(rule)
        # Only train on rules that tokenize cleanly (round-trip check)
        if not tokens or ''.join(tokens) != rule:
            skipped += 1
            continue
        # START → first token
        counts[START][tokens[0]] += 1
        # Unigram transitions: tokens[i] → tokens[i+1]
        for i in range(len(tokens) - 1):
            counts[tokens[i]][tokens[i + 1]] += 1
        # Bigram transitions: (tokens[i], tokens[i+1]) → tokens[i+2]
        for i in range(len(tokens) - 2):
            bigram_key = (tokens[i], tokens[i + 1])
            counts[bigram_key][tokens[i + 2]] += 1

    if skipped:
        print_warning(f"Markov training: skipped {skipped:,} rules that did not tokenize cleanly.")

    totals = {k: sum(v.values()) for k, v in counts.items()}
    probs: Dict = defaultdict(lambda: defaultdict(float))
    for prefix, next_counts in counts.items():
        t = totals[prefix]
        for nxt, cnt in next_counts.items():
            probs[prefix][nxt] = cnt / t

    unique_first_tokens = len(probs.get(START, {}))
    unique_unigrams = sum(1 for k in probs if isinstance(k, str) and k != START)
    unique_bigrams  = sum(1 for k in probs if isinstance(k, tuple))
    print_success(
        f"Token-level Markov model built: "
        f"{unique_first_tokens} start tokens, "
        f"{unique_unigrams} unigram contexts, "
        f"{unique_bigrams} bigram contexts."
    )
    return probs, totals


def get_markov_weighted_rules(
    unique_rules:         Dict[str, int],
    markov_probabilities: Dict,
    total_transitions:    Dict,
) -> List[Tuple[str, float]]:
    """Score each rule by its log-probability under the token-level Markov model.

    Scoring strategy (mirrors the generation walk):
      1. P(tokens[0] | START)
      2. For each subsequent token tokens[i] (i ≥ 1):
           a. Try bigram context (tokens[i-2], tokens[i-1]) → tokens[i]   (if i ≥ 2)
           b. Fall back to unigram context tokens[i-1] → tokens[i]
           c. If neither context exists the rule is assigned -∞ and skipped.

    Rules that do not tokenize cleanly (round-trip check fails) are silently
    dropped — they cannot have been produced by the model.
    """
    if not memory_intensive_operation_warning("Markov weighting"):
        return []
    START    = '^'
    weighted: List[Tuple[str, float]] = []

    for rule in unique_rules:
        if not rule:
            continue
        tokens = TOKEN_REGEX.findall(rule)
        if not tokens or ''.join(tokens) != rule:
            continue  # rule does not tokenize cleanly

        logp  = 0.0
        valid = True

        # ── Score first token ─────────────────────────────────────────────
        if tokens[0] not in markov_probabilities.get(START, {}):
            continue
        logp += math.log(markov_probabilities[START][tokens[0]])

        # ── Score subsequent tokens ───────────────────────────────────────
        for i in range(1, len(tokens)):
            scored = False
            # Try bigram context first (available from position i=2 onward)
            if i >= 2:
                bigram_key = (tokens[i - 2], tokens[i - 1])
                if (bigram_key in markov_probabilities
                        and tokens[i] in markov_probabilities[bigram_key]):
                    logp += math.log(markov_probabilities[bigram_key][tokens[i]])
                    scored = True
            if not scored:
                # Fall back to unigram context
                prev = tokens[i - 1]
                if prev in markov_probabilities and tokens[i] in markov_probabilities[prev]:
                    logp += math.log(markov_probabilities[prev][tokens[i]])
                else:
                    valid = False
                    break

        if valid:
            weighted.append((rule, logp))

    return sorted(weighted, key=lambda kv: kv[1], reverse=True)


# ==============================================================================
# MARKOV GENERATION
# ==============================================================================

def generate_rules_from_markov_model(
    markov_probabilities: Dict,
    target:               int,
    min_len:              int,
    max_len:              int,
    gpu_mode:             bool = False,
    excluded_operators:   Optional[Set[str]] = None,
) -> List[Tuple[str, float]]:
    """Generate *target* valid hashcat rules via a token-level Markov walk.

    ── Key design decisions ────────────────────────────────────────────────

    TOKEN-LEVEL walk (v3.3 improvement over v3.2 character-level walk):
      The walk samples full hashcat TOKENS at every step instead of single
      characters.  A token is an atomic operator+argument unit (e.g. 'l',
      '$5', 'sae', 'i3X') as produced by TOKEN_REGEX.  This guarantees that
      every candidate rule is structurally valid and eliminates the massive
      rejection rate of the old character-level approach.

    LENGTH SEMANTICS:
      min_len / max_len now count TOKENS (operators), not raw bytes.
      min_len=1, max_len=3 means rules with 1, 2, or 3 chained operators.
      This is the natural measure of rule complexity for hashcat.

    SAMPLING STRATEGY:
      1. Sample first token from P(token | START).
      2. At each subsequent step, try the bigram context
         (tokens[-2], tokens[-1]) first; fall back to unigram tokens[-1].
      3. Accept the current sequence as a rule whenever
         min_len ≤ len(token_seq) ≤ max_len.
      4. Continue extending until max_len is reached or no transition exists.

    SAFETY NETS:
      • Tokens whose leading operator is in excluded_operators are never
        sampled (NEVER_PRODUCE_OPS by default).
      • After joining, _has_banned_op() is checked as a paranoia guard.
      • TOKEN_REGEX round-trip validation confirms the joined string re-
        tokenises back to the exact same token list.
      • is_valid_hashcat_rule() / HashcatRuleCleaner.validate_rule() provide
        a final syntactic gate.

    v3.1: excluded_operators defaults to NEVER_PRODUCE_OPS.
    """
    if not memory_intensive_operation_warning("Markov rule generation"):
        return []
    if excluded_operators is None:
        excluded_operators = NEVER_PRODUCE_OPS

    print_section(
        f"Generating Token-Level Markov Rules "
        f"({min_len}–{max_len} tokens/operators, target: {target:,})"
    )
    print_info(f"Excluding operators: {', '.join(sorted(excluded_operators))}")

    START = '^'

    def _sample_next(context) -> Optional[str]:
        """Sample the next token given a context key (str or 2-tuple of str).

        Only tokens whose leading operator character is NOT in
        excluded_operators are eligible.  The returned value is a full token
        string such as 'l', '$5', or 'sae'.
        """
        if context not in markov_probabilities:
            return None
        choices: List[str]   = []
        weights: List[float] = []
        for tok, prob in markov_probabilities[context].items():
            if tok[0] not in excluded_operators:
                choices.append(tok)
                weights.append(prob)
        if not choices:
            return None
        total = sum(weights)
        if total == 0:
            return None
        return random.choices(choices, weights=[w / total for w in weights], k=1)[0]

    generated: Set[str] = set()
    max_attempts = target * 20  # generous budget; most walks succeed

    for _ in range(max_attempts):
        if len(generated) >= target:
            break

        # ── Step 1: Sample the first token ────────────────────────────────
        first = _sample_next(START)
        if not first:
            continue
        token_seq: List[str] = [first]

        # ── Step 2: Extend the token chain up to max_len ──────────────────
        while len(token_seq) < max_len:
            # Prefer bigram context; fall back to unigram
            if len(token_seq) >= 2:
                bigram_key = (token_seq[-2], token_seq[-1])
                nxt = _sample_next(bigram_key)
                if not nxt:
                    nxt = _sample_next(token_seq[-1])
            else:
                nxt = _sample_next(token_seq[-1])

            if not nxt:
                break  # dead end — accept what we have if in range

            token_seq.append(nxt)

            # ── Step 3: Accept if token count is in [min_len, max_len] ────
            if min_len <= len(token_seq) <= max_len:
                rule = ''.join(token_seq)

                # Paranoia guard: no banned operator must have slipped in
                if _has_banned_op(rule):
                    continue
                # Round-trip validation: joined string must re-tokenise to
                # the exact same token list (catches accidental merges)
                if TOKEN_REGEX.findall(rule) != token_seq:
                    continue
                # Final syntactic gate
                if gpu_mode:
                    if HashcatRuleCleaner(2).validate_rule(rule):
                        generated.add(rule)
                elif is_valid_hashcat_rule(rule):
                    generated.add(rule)

        # Also accept chain at min_len even if we broke out before max_len
        if len(token_seq) >= min_len:
            rule = ''.join(token_seq[:min_len])
            if (not _has_banned_op(rule)
                    and TOKEN_REGEX.findall(rule) == token_seq[:min_len]):
                if gpu_mode:
                    if HashcatRuleCleaner(2).validate_rule(rule):
                        generated.add(rule)
                elif is_valid_hashcat_rule(rule):
                    generated.add(rule)

    print_success(f"Generated {len(generated):,} valid token-level Markov rules.")
    if not generated:
        return []
    dummy = {r: 1 for r in generated}
    return get_markov_weighted_rules(dummy, markov_probabilities, {})[:target]


# ==============================================================================
# COMBINATORIAL GENERATION
# ==============================================================================

def find_min_operators_for_target(
    sorted_operators: List[Tuple[str, int]],
    target:           int,
    min_len:          int,
    max_len:          int,
) -> List[str]:
    """Return the fewest top operators whose cartesian product covers *target* rules.

    v3.1: operators from NEVER_PRODUCE_OPS are excluded from the candidate pool.
    v3.2: sorted_operators now contains full tokens (e.g. '$5', 'sae') so the
    count accurately reflects how many distinct rule-chains are producible.
    """
    # Strip banned operators from the candidate pool at the source
    safe_operators = [(op, cnt) for op, cnt in sorted_operators if op not in NEVER_PRODUCE_OPS]

    current = 0
    n       = 0
    while current < target and n < len(safe_operators):
        n      += 1
        top_ops = [op for op, _ in safe_operators[:n]]
        current = sum(len(top_ops) ** length for length in range(min_len, max_len + 1))
    return [op for op, _ in safe_operators[:n]]


def _generate_for_length(args: Tuple) -> Set[str]:
    top_ops, length, gpu_mode = args
    generated: Set[str] = set()
    invalid_concat = 0
    for combo in itertools.product(top_ops, repeat=length):
        rule = ''.join(combo)
        # Paranoia check: never emit a rule with a banned operator
        if _has_banned_op(rule):
            continue
        if not is_valid_hashcat_rule(rule):
            continue
        # Round-trip re-parse gate (v3.2):
        # After joining tokens into a rule string, re-tokenise it.  The
        # result must equal the original token list exactly.  This catches
        # cases where two adjacent tokens accidentally merge into a different
        # longer operator — e.g. token 's' followed by 'a' followed by 'b'
        # must re-parse back as exactly ['s','a','b'] not ['sab'].
        # Any ambiguous concatenation is silently dropped.
        reparsed = TOKEN_REGEX.findall(rule)
        if reparsed != list(combo):
            invalid_concat += 1
            continue
        if gpu_mode:
            if not HashcatRuleCleaner(2).validate_rule(rule):
                continue
        generated.add(rule)
    return generated


def generate_rules_parallel(
    top_operators: List[str],
    min_len:       int,
    max_len:       int,
    gpu_mode:      bool = False,
) -> Set[str]:
    if not memory_intensive_operation_warning("combinatorial generation"):
        return set()
    # Extra safety: strip any banned operator that might have sneaked in
    safe_ops = [op for op in top_operators if op not in NEVER_PRODUCE_OPS]
    lengths  = list(range(min_len, max_len + 1))
    tasks    = [(safe_ops, ln, gpu_mode) for ln in lengths]
    n_procs  = min(os.cpu_count() or 1, len(lengths))
    print_info(
        f"Generating rules of length {min_len}–{max_len} "
        f"using {len(safe_ops)} operators, {n_procs} processes..."
    )
    with multiprocessing.Pool(processes=n_procs) as pool:
        sets = pool.map(_generate_for_length, tasks)
    generated = set().union(*sets)
    print_success(f"Generated {len(generated):,} valid rules.")
    return generated


# ==============================================================================
# HASHCAT RULE ENGINE SIMULATION
# ==============================================================================

def _i36(s: str) -> int:
    return int(s, 36)


class RuleEngine:
    """
    Simulates hashcat's rule application on a test string.
    """

    def __init__(self, rules: List[str]) -> None:
        self._token_lists = [TOKEN_REGEX.findall(r) for r in rules]
        self.memorized    = ''

    def apply(self, string: str) -> str:
        """Apply each rule in sequence, passing output of one as input to the next."""
        word = string
        self.memorized = ''
        for tokens in self._token_lists:
            for token in tokens:
                try:
                    word = self._dispatch(token, word)
                except Exception:
                    pass
        return word

    def _dispatch(self, token: str, word: str) -> str:
        op   = token[0]
        args = token[1:]

        if op == ':':
            return word
        elif op == 'l':
            return word.lower()
        elif op == 'u':
            return word.upper()
        elif op == 'c':
            return word.capitalize()
        elif op == 'C':
            return word.capitalize().swapcase()
        elif op == 't':
            return word.swapcase()
        elif op == 'T':
            n = _i36(args[0])
            if n >= len(word):
                return word
            return word[:n] + word[n].swapcase() + word[n + 1:]
        elif op == 'r':
            return word[::-1]
        elif op == 'd':
            return word + word
        elif op == 'p':
            return word * (_i36(args[0]) + 1)
        elif op == 'f':
            return word + word[::-1]
        elif op == '{':
            return (word[1:] + word[0]) if word else word
        elif op == '}':
            return (word[-1] + word[:-1]) if word else word
        elif op == '$':
            return word + args[0]
        elif op == '^':
            return args[0] + word
        elif op == '[':
            return word[1:]
        elif op == ']':
            return word[:-1]
        elif op == 'D':
            n = _i36(args[0])
            return word[:n] + word[n + 1:] if n < len(word) else word
        elif op == 'x':
            s, e = _i36(args[0]), _i36(args[1])
            if s < 0 or e < 0 or s > len(word) or e > len(word) or s > e:
                return ''
            return word[s:e]
        elif op == 'O':
            s, e = _i36(args[0]), _i36(args[1])
            if s < 0 or e < 0 or s > len(word) or e > len(word) or s > e:
                return word
            return word[:s] + word[e + 1:]
        elif op == 'i':
            pos  = min(_i36(args[0]), len(word))
            char = args[1]
            return word[:pos] + char + word[pos:]
        elif op == 'o':
            pos  = _i36(args[0])
            char = args[1]
            return word[:pos] + char + word[pos + 1:] if pos < len(word) else word
        elif op == "'":
            return word[:_i36(args[0])]
        elif op == 's':
            return word.replace(args[0], args[1])
        elif op == '@':
            return word.replace(args[0], '')
        elif op == 'z':
            n = _i36(args[0])
            return word[0] * n + word if word else ''
        elif op == 'Z':
            n = _i36(args[0])
            return word + word[-1] * n if word else ''
        elif op == 'q':
            return ''.join(c * 2 for c in word)
        elif op == 'X':
            if not self.memorized:
                return word
            pos, ln, ins = _i36(args[0]), _i36(args[1]), _i36(args[2])
            seg = self.memorized[pos:pos + ln]
            lst = list(word)
            lst.insert(ins, seg)
            return ''.join(lst)
        elif op == '4':
            return word + self.memorized
        elif op == '6':
            return self.memorized + word
        elif op == 'M':
            self.memorized = word
            return word
        elif op == 'k':
            if len(word) >= 2:
                return word[1] + word[0] + word[2:]
            return word
        elif op == 'K':
            if len(word) >= 2:
                return word[:-2] + word[-1] + word[-2]
            return word
        elif op == '*':
            a, b = _i36(args[0]), _i36(args[1])
            if a >= len(word) or b >= len(word):
                return word
            lst    = list(word)
            lst[a], lst[b] = lst[b], lst[a]
            return ''.join(lst)
        elif op == 'L':
            n = _i36(args[0])
            if n >= len(word):
                return word
            return word[:n] + chr(ord(word[n]) << 1) + word[n + 1:]
        elif op == 'R':
            n = _i36(args[0])
            if n >= len(word):
                return word
            return word[:n] + chr(ord(word[n]) >> 1) + word[n + 1:]
        elif op == '+':
            n = _i36(args[0])
            if n >= len(word):
                return word
            return word[:n] + chr(ord(word[n]) + 1) + word[n + 1:]
        elif op == '-':
            n = _i36(args[0])
            if n >= len(word):
                return word
            return word[:n] + chr(ord(word[n]) - 1) + word[n + 1:]
        elif op == '.':
            n = _i36(args[0])
            if n + 1 >= len(word):
                return word
            return word[:n] + word[n + 1] + word[n + 1:]
        elif op == ',':
            n = _i36(args[0])
            if n == 0 or n >= len(word):
                return word
            return word[:n] + word[n - 1] + word[n + 1:]
        elif op == 'y':
            n = _i36(args[0])
            return word[:n] + word if word else word
        elif op == 'Y':
            n = _i36(args[0])
            return word + word[-n:] if word else word
        elif op == 'E':
            return ' '.join(w.capitalize() for w in word.split(' '))
        elif op == 'e':
            sep  = args[0]
            return sep.join(w.capitalize() for w in word.split(sep))
        else:
            return word


# ==============================================================================
# FUNCTIONAL MINIMIZATION
# ==============================================================================
#
# Changes (minimizer.py integration)
# ────────────────────────────────────────────────────────────────────────────
# 1. Byte-level rule engine (_min_apply_single / _min_apply_chain)
#    Replaces the RuleEngine-based approach with a latin-1 byte-level
#    implementation ported from minimizer.py.  Key improvements:
#      • Byte-level processing avoids Python Unicode artefacts and matches
#        hashcat's GPU kernel behaviour exactly.
#      • \xNN hex-escape notation in argument positions is handled correctly.
#      • Rules with unsupported opcodes return _UNSUPPORTED_SIG instead of
#        silently no-oping, so they are tracked and kept separately.
#      • Both space-separated and concatenated rule formats are accepted.
#
# 2. Tuple-based signatures (_compute_signature)
#    Signatures are now tuples of per-word outputs rather than joined strings.
#    This eliminates false collisions caused by output values that happen to
#    contain the separator character used in the old '|'.join() approach.
#
# 3. Extended probe vector (TEST_VECTOR)
#    Merged concentrator's original 52-word vector with minimizer.py's
#    BUILTIN_PROBES.  Critical additions: "password" (was missing!), short
#    common base words (pass, root, admin, letmein …), mixed-case compounds
#    (AdminUser, HelloWorld …), and embedded-digit words (pass123, user9999).
#
# 4. SQLite-backed deduplication (_functional_minimization_sqlite)
#    For rulesets > _MIN_SQLITE_THRESHOLD (1 M rules) the signature map lives
#    in a temporary on-disk SQLite database instead of an in-memory dict.
#    This prevents OOM on very large datasets.  The temp file is removed
#    unconditionally on completion (success or error).

# ---------------------------------------------------------------------------
# Byte-level hashcat rule engine (ported from minimizer.py)
# ---------------------------------------------------------------------------

_ZERO_ARG_OPS_MIN = frozenset(':lucCtErdfkK{}[]q')
_ONE_ARG_OPS_MIN  = frozenset([
    '^', '$', '@', 'p', 'T', 'D', 'L', 'R',
    '+', '-', '.', ',', "'", 'z', 'Z', 'y', 'Y', 'e',
])
_TWO_ARG_OPS_MIN  = frozenset(['s', 'i', 'o', 'x', 'O', '*', '3'])

# Sentinel returned when a rule contains an unsupported opcode.
# All such rules share this signature and are kept intact (not deduplicated).
_UNSUPPORTED_SIG: tuple = ('__UNSUPPORTED__',)

# Rulesets above this size use a SQLite temp-DB instead of an in-memory dict
# to avoid OOM on very large inputs.
_MIN_SQLITE_THRESHOLD = 1_000_000


def _min_arg_ord(token: str, pos: int) -> int:
    """Return the integer code-point of the argument character at *pos*,
    resolving \\xNN hex-escape sequences transparently."""
    if (pos < len(token)
            and token[pos] == '\\'
            and pos + 3 < len(token)
            and token[pos + 1] == 'x'
            and all(c in '0123456789abcdefABCDEF' for c in token[pos + 2:pos + 4])):
        return int(token[pos + 2:pos + 4], 16)
    return ord(token[pos]) if pos < len(token) else 0


def _min_apply_single(rule: str, word: str) -> Optional[str]:
    """Apply one hashcat rule atom to *word* at the byte level (latin-1).

    Returns None if the opcode is unsupported — the caller must treat the
    whole rule as having signature *_UNSUPPORTED_SIG*.
    """
    if not rule:
        return word
    w   = list(word.encode('latin-1'))
    cmd = rule[0]

    def dg(c: str) -> int:
        return ord(c) - 48 if '0' <= c <= '9' else -1

    try:
        if cmd == ':':
            pass
        elif cmd == 'l':
            w = [c | 0x20 if 65 <= c <= 90 else c for c in w]
        elif cmd == 'u':
            w = [c & ~0x20 if 97 <= c <= 122 else c for c in w]
        elif cmd == 'c':
            if w:
                w[0] = w[0] & ~0x20 if 97 <= w[0] <= 122 else w[0]
                w[1:] = [c | 0x20 if 65 <= c <= 90 else c for c in w[1:]]
        elif cmd == 'C':
            if w:
                w[0] = w[0] | 0x20 if 65 <= w[0] <= 90 else w[0]
                w[1:] = [c & ~0x20 if 97 <= c <= 122 else c for c in w[1:]]
        elif cmd == 't':
            w = [c | 0x20 if 65 <= c <= 90 else
                 (c & ~0x20 if 97 <= c <= 122 else c) for c in w]
        elif cmd == 'E':
            out: list = []; cap = True
            for c in w:
                out.append(c & ~0x20 if cap and 97 <= c <= 122 else c)
                cap = c in (32, 45, 95)
            w = out
        elif cmd == 'r':
            w = w[::-1]
        elif cmd == 'd':
            w = w + w
        elif cmd == 'f':
            w = w + w[::-1]
        elif cmd == '{':
            if len(w) > 1: w = w[1:] + [w[0]]
        elif cmd == '}':
            if len(w) > 1: w = [w[-1]] + w[:-1]
        elif cmd == '[':
            if w: w = w[1:]
        elif cmd == ']':
            if w: w = w[:-1]
        elif cmd == 'k':
            if len(w) >= 2: w[0], w[1] = w[1], w[0]
        elif cmd == 'K':
            if len(w) >= 2: w[-1], w[-2] = w[-2], w[-1]
        elif cmd == 'q':
            out = []
            for c in w: out += [c, c]
            w = out
        elif cmd == '^' and len(rule) >= 2:
            w = [_min_arg_ord(rule, 1)] + w
        elif cmd == '$' and len(rule) >= 2:
            w = w + [_min_arg_ord(rule, 1)]
        elif cmd == '@' and len(rule) >= 2:
            ch = _min_arg_ord(rule, 1)
            w  = [c for c in w if c != ch]
        elif cmd == 'p' and len(rule) >= 2:
            n = dg(rule[1])
            if n > 0:
                orig = w[:]
                for _ in range(n): w += orig
        elif cmd == 'T' and len(rule) >= 2:
            p = dg(rule[1])
            if 0 <= p < len(w):
                c = w[p]
                w[p] = (c | 0x20 if 65 <= c <= 90
                        else (c & ~0x20 if 97 <= c <= 122 else c))
        elif cmd == 'D' and len(rule) >= 2:
            p = dg(rule[1])
            if 0 <= p < len(w): w.pop(p)
        elif cmd == 'L' and len(rule) >= 2:
            p = dg(rule[1])
            if 0 <= p < len(w): w[p] = (w[p] << 1) & 0xFF
        elif cmd == 'R' and len(rule) >= 2:
            p = dg(rule[1])
            if 0 <= p < len(w): w[p] = (w[p] >> 1) & 0xFF
        elif cmd == '+' and len(rule) >= 2:
            p = dg(rule[1])
            if 0 <= p < len(w): w[p] = (w[p] + 1) & 0xFF
        elif cmd == '-' and len(rule) >= 2:
            p = dg(rule[1])
            if 0 <= p < len(w): w[p] = (w[p] - 1) & 0xFF
        elif cmd in ('.', ',') and len(rule) >= 2:
            p     = dg(rule[1])
            delta = 1 if cmd == '.' else -1
            if 0 <= p < len(w): w[p] = (w[p] + delta) & 0xFF
        elif cmd == "'" and len(rule) >= 2:
            p = dg(rule[1])
            if 0 <= p < len(w): w = w[:p + 1]
        elif cmd == 'z' and len(rule) >= 2:
            n = dg(rule[1])
            if n > 0 and w: w = [w[0]] * n + w
        elif cmd == 'Z' and len(rule) >= 2:
            n = dg(rule[1])
            if n > 0 and w: w = w + [w[-1]] * n
        elif cmd == 'y' and len(rule) >= 2:
            n = dg(rule[1])
            if n > 0: w = w[:n] + w
        elif cmd == 'Y' and len(rule) >= 2:
            n = dg(rule[1])
            if n > 0 and len(w) >= n: w = w + w[-n:]
        elif cmd == 's' and len(rule) >= 3:
            a = _min_arg_ord(rule, 1)
            b = _min_arg_ord(rule, 2 if rule[1] != '\\' else 5)
            w = [b if c == a else c for c in w]
        elif cmd == 'i' and len(rule) >= 3:
            p, ch = dg(rule[1]), _min_arg_ord(rule, 2)
            if 0 <= p <= len(w): w.insert(p, ch)
        elif cmd == 'o' and len(rule) >= 3:
            p, ch = dg(rule[1]), _min_arg_ord(rule, 2)
            if 0 <= p < len(w): w[p] = ch
        elif cmd == 'e' and len(rule) >= 2:
            sep = _min_arg_ord(rule, 1); out = []; cap = True
            for c in w:
                out.append(c & ~0x20 if cap and 97 <= c <= 122 else c)
                cap = (c == sep)
            w = out
        elif cmd == 'x' and len(rule) >= 3:
            a, b = dg(rule[1]), dg(rule[2])
            if a > b: a, b = b, a
            w = w[a:b + 1]
        elif cmd == 'O' and len(rule) >= 3:
            p, m = dg(rule[1]), dg(rule[2])
            if 0 <= p < len(w) and m > 0: w = w[:p] + w[p + m:]
        elif cmd == '*' and len(rule) >= 3:
            a, b = dg(rule[1]), dg(rule[2])
            if 0 <= a < len(w) and 0 <= b < len(w) and a != b:
                w[a], w[b] = w[b], w[a]
        elif cmd == '3' and len(rule) >= 3:
            n, sep = dg(rule[1]), _min_arg_ord(rule, 2)
            cnt = 0
            for i, c in enumerate(w):
                if c == sep:
                    cnt += 1
                    if cnt == n and i + 1 < len(w):
                        ci = w[i + 1]
                        w[i + 1] = (ci | 0x20 if 65 <= ci <= 90
                                    else (ci & ~0x20 if 97 <= ci <= 122 else ci))
                        break
        else:
            return None  # unsupported opcode
    except Exception:
        return None

    try:
        return bytes(w).decode('latin-1')
    except Exception:
        return None


def _min_read_arg_char(chain: str, pos: int) -> Tuple[str, int]:
    """Read one argument character from *chain* at *pos*,
    handling \\xNN hex-escape notation."""
    if pos >= len(chain):
        return ('', pos)
    if (chain[pos] == '\\'
            and pos + 3 < len(chain)
            and chain[pos + 1] == 'x'
            and all(c in '0123456789abcdefABCDEF' for c in chain[pos + 2:pos + 4])):
        return (chain[pos:pos + 4], pos + 4)
    return (chain[pos], pos + 1)


def _min_tokenize_rule(chain: str) -> List[str]:
    """Split a hashcat rule line into individual opcode atoms.

    Handles space-separated (``l r $1``), concatenated (``lr$1``), and
    mixed formats, as well as \\xNN hex-escape notation in argument positions.
    """
    tokens: List[str] = []
    i = 0
    n = len(chain)
    while i < n:
        c = chain[i]
        if c == ' ':
            i += 1
            continue
        if c in _ZERO_ARG_OPS_MIN:
            tokens.append(c)
            i += 1
        elif c in _ONE_ARG_OPS_MIN:
            arg, i2 = _min_read_arg_char(chain, i + 1)
            tokens.append(c + arg)
            i = i2
        elif c in _TWO_ARG_OPS_MIN:
            arg1, i2 = _min_read_arg_char(chain, i + 1)
            arg2, i3 = _min_read_arg_char(chain, i2)
            tokens.append(c + arg1 + arg2)
            i = i3
        else:
            tokens.append(chain[i:])   # unknown — consume rest; _apply_single returns None
            break
    return tokens


def _min_apply_chain(chain: str, word: str) -> Optional[str]:
    """Apply a full hashcat rule chain (any format) to *word*.

    Returns None if any atom contains an unsupported opcode, which causes
    the caller to assign _UNSUPPORTED_SIG to the entire rule.
    """
    cur: Optional[str] = word
    for atom in _min_tokenize_rule(chain):
        cur = _min_apply_single(atom, cur)  # type: ignore[arg-type]
        if cur is None:
            return None
    return cur


def _min_compute_signature(rule: str, probe_words: List[str]) -> tuple:
    """Return the functional signature of *rule* as a tuple of per-word outputs.

    Returns _UNSUPPORTED_SIG if any opcode in the rule is unsupported.
    Using a tuple (not a joined string) eliminates false collisions from
    output values that contain the separator character.
    """
    outputs = []
    for word in probe_words:
        out = _min_apply_chain(rule, word)
        if out is None:
            return _UNSUPPORTED_SIG
        outputs.append(out)
    return tuple(outputs)


# ---------------------------------------------------------------------------
# Probe vector (TEST_VECTOR)
# ---------------------------------------------------------------------------
# Concentrator's original 52-word vector merged with minimizer.py's
# BUILTIN_PROBES.  Critical additions: "password" (was missing from the
# original), short common base words (pass, root, admin, letmein …),
# mixed-case compounds (AdminUser, HelloWorld …), embedded-digit words
# (pass123, user9999 …), and "bbbb" for repeated-char coverage.

TEST_VECTOR: List[str] = [
    # ── very short — edge cases for k, K, {, }, [, ] ────────────────────
    "ab", "abc", "abcd",
    # ── single chars and 2-char words ────────────────────────────────────
    "a", "Z", "1", "aB", "42", "AB", "0",
    # ── short alphanumeric (len 4–6) ─────────────────────────────────────
    "pass", "root", "test", "admin", "login",
    # ── length 5–8 (original concentrator set) ───────────────────────────
    "hello", "World", "ADMIN", "12345", "abc12", "P@ss1", "TEST", "Test",
    # ── typical password base words (len 7–9) ────────────────────────────
    "letmein",          # len 7
    "welcome",          # len 7
    "password",         # len 8  ← THE critical probe word (was missing!)
    "sunshine",         # len 8
    "football",         # len 8
    "baseball",         # len 8
    "princess",         # len 8
    "dragon12",         # len 8, ends with digits
    # ── length 9–12 (original concentrator set) ──────────────────────────
    "Password1", "qwertyuio", "ABCDEFGHIJK", "0123456789",
    "pass_word", "hello-you",
    # ── longer words (len 10+) — truncation / repeat ops ─────────────────
    "qwertyuiop",       # len 10
    "iloveyou12",       # len 10, trailing digits
    "monkey12345",      # len 11
    "superman123",      # len 11
    "mustang2024",      # len 11
    # ── mixed-case — l/u/c/C/t/E/T/k/K ──────────────────────────────────
    "Password", "AdminUser", "MySecret", "HelloWorld",
    # ── length 13–16 (original concentrator set) ─────────────────────────
    "Password12345", "administrator", "Summer2023pass",
    "QWERTYUIOPLKJ", "correctHorse!1",
    # ── length 17–20 (original concentrator set) ─────────────────────────
    "verylongpassword1!", "A1b2C3d4E5f6G7h8",
    "abcdefghijklmnopq", "ABCDEFGHIJKLMNOPQ",
    # ── length 21–24 (original concentrator set) ─────────────────────────
    "thisisaverylongstring", "A1b2C3d4E5f6G7h8I9j0",
    "abcdefghijklmnopqrstu",
    # ── length 25–28 (original concentrator set) ─────────────────────────
    "averylongpasswordindeed12", "ABCDEFGHIJKLMNOPQRSTUVWXY",
    "abcdefghijklmnopqrstuvwxy",
    # ── length 29–32 (original concentrator set) ─────────────────────────
    "aVerylongPasswordWithNumbers12", "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234",
    "abcdefghijklmnopqrstuvwxyz1234",
    # ── embedded digits — s, o, @, T ops ─────────────────────────────────
    "pass123", "admin2024", "test1234", "user9999",
    # ── special characters — @ removal, s substitution ───────────────────
    "spec!", "!spec", "pa$$word", "s3cr3t!", "P@ssw0rd!",
    "p@ssw0rd", "s3cur1ty",
    # ── with spaces / underscores / hyphens ──────────────────────────────
    "hello world", "foo_bar", "test-case",
    # ── tricky case-change patterns ──────────────────────────────────────
    "hElLo", "pAsSwOrD", "tEsT", "xYz!",
    # ── pure digits ──────────────────────────────────────────────────────
    "01", "000", "99999", "1234567890",
    # ── repeated chars (exercises z, Z, q) ───────────────────────────────
    "aaaa", "ZZZZ", "1111", "bbbb",
]

# Deduplicate while preserving order
_tv_seen: set = set()
_tv_deduped: List[str] = []
for _tv_w in TEST_VECTOR:
    if _tv_w not in _tv_seen:
        _tv_seen.add(_tv_w)
        _tv_deduped.append(_tv_w)
TEST_VECTOR = _tv_deduped
del _tv_seen, _tv_deduped, _tv_w


# ---------------------------------------------------------------------------
# Worker infrastructure (multiprocessing)
# ---------------------------------------------------------------------------

_worker_test_vector: List[str] = []


def _worker_init(test_vec: List[str]) -> None:
    """Pool initializer: store the probe vector in each worker process."""
    global _worker_test_vector
    _worker_test_vector = test_vec


def _compute_signature(rule_data: Tuple[str, int]) -> Tuple[tuple, Tuple[str, int]]:
    """Compute a tuple-based functional signature using the byte-level engine.

    Replaces the old RuleEngine + joined-string approach:
      • latin-1 byte-level processing mirrors hashcat's GPU kernel exactly.
      • Returns a *tuple* — eliminates false collisions from separator chars.
      • Rules with unsupported opcodes return _UNSUPPORTED_SIG so they are
        tracked separately rather than silently no-oped.
      • \\xNN hex-escape arguments and both space-separated / concatenated
        rule formats are handled correctly.
    """
    rule_text, count = rule_data
    sig = _min_compute_signature(rule_text, _worker_test_vector)
    return sig, (rule_text, count)


# ---------------------------------------------------------------------------
# SQLite-backed deduplication for very large rulesets
# ---------------------------------------------------------------------------

def _functional_minimization_sqlite(
    data: List[Tuple[str, int]],
) -> List[Tuple[str, int]]:
    """Signature-deduplication for rulesets > _MIN_SQLITE_THRESHOLD rules.

    The signature map lives entirely in a temporary
    ``concentrator_minsig_<pid>.db`` file in the configured temp directory,
    which is deleted unconditionally on completion (success or error).

    Deduplication strategy (same as the in-memory path):
      • Two rules that share a signature → keep the one with the higher
        individual occurrence count and accumulate both counts.
      • Rules with unsupported opcodes (_UNSUPPORTED_SIG) are kept intact
        and are not deduplicated against each other.

    Commit batching (every 10 000 rows) keeps SQLite write throughput high.
    """
    db_path = os.path.join(
        STATE.temp_dir_path or tempfile.gettempdir(),
        f"concentrator_minsig_{os.getpid()}.db",
    )
    if os.path.exists(db_path):
        os.remove(db_path)

    print_info(
        f"Ruleset exceeds {_MIN_SQLITE_THRESHOLD:,} rules — "
        "using SQLite backing store for signature deduplication."
    )

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.executescript("""
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous  = OFF;
            PRAGMA temp_store   = MEMORY;
            PRAGMA cache_size   = -65536;
        """)
        cur.execute("""
            CREATE TABLE sigs (
                sig   BLOB    PRIMARY KEY,
                rule  TEXT    NOT NULL,
                count INTEGER NOT NULL
            )
        """)
        conn.commit()

        _BATCH             = 10_000
        pending            = 0
        unsupported_rules: List[Tuple[str, int]] = []
        conn.execute("BEGIN")

        for rule_text, count in tqdm(data, desc="Sig (SQLite)", unit=" rules"):
            sig = _min_compute_signature(rule_text, TEST_VECTOR)
            if sig == _UNSUPPORTED_SIG:
                unsupported_rules.append((rule_text, count))
                continue
            sig_blob = pickle.dumps(sig, protocol=4)
            # INSERT new sig; on collision keep the higher-count rule and sum totals
            cur.execute("""
                INSERT INTO sigs (sig, rule, count) VALUES (?, ?, ?)
                ON CONFLICT(sig) DO UPDATE SET
                    rule  = CASE WHEN excluded.count > sigs.count
                                 THEN excluded.rule ELSE sigs.rule END,
                    count = sigs.count + excluded.count
            """, (sig_blob, rule_text, count))
            pending += 1
            if pending >= _BATCH:
                conn.commit()
                conn.execute("BEGIN")
                pending = 0

        conn.commit()
        cur.execute("SELECT rule, count FROM sigs")
        final: List[Tuple[str, int]] = list(cur.fetchall())

    finally:
        conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)
            print_info("Temporary signature database removed.")

    # Unsupported-opcode rules cannot be compared — append all of them as-is
    final.extend(unsupported_rules)
    final.sort(key=lambda kv: kv[1], reverse=True)

    removed = len(data) - len(final)
    print_success(f"Removed {removed:,} functionally redundant rules (SQLite path).")
    if unsupported_rules:
        print_info(
            f"Retained {len(unsupported_rules):,} rules with unsupported opcodes."
        )
    return final


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@memory_safe_operation("Functional Minimization", 85)
def functional_minimization(
    data: List[Tuple[str, int]]
) -> List[Tuple[str, int]]:
    """Eliminate functionally redundant rules using byte-level signature comparison.

    Two rules are considered equivalent when they produce identical output on
    every word in TEST_VECTOR.  When a collision is found, the rule with the
    higher occurrence count is kept and the counts are summed.

    Improvements over the previous RuleEngine-based implementation:
      • latin-1 byte-level engine mirrors hashcat's GPU kernel behaviour.
      • Tuple signatures eliminate false collisions from separator characters.
      • \\xNN hex-escape arguments are handled in rule atoms.
      • Unsupported-opcode rules are kept intact (not silently merged).
      • SQLite-backed path for rulesets > _MIN_SQLITE_THRESHOLD prevents OOM.
    """
    print_section("Functional Minimization")
    print_warning("RAM intensive — may take significant time for large datasets.")

    if not data:
        return data

    if len(data) > 10_000:
        print_warning(f"Large dataset: {len(data):,} rules.")
        est = estimate_memory_usage(len(data))
        print(f"{Colors.CYAN}[MEMORY]{Colors.RESET} Estimated: {format_bytes(est)}")
        if input(
            f"{Colors.YELLOW}Continue? (y/N): {Colors.RESET}"
        ).strip().lower() not in ('y', 'yes'):
            print_info("Functional minimization skipped.")
            return data

    print_info(
        f"Probe-vector size: {len(TEST_VECTOR)} words  "
        "(byte-level engine, tuple signatures)"
    )

    # Very large rulesets → SQLite-backed path
    if len(data) > _MIN_SQLITE_THRESHOLD:
        return _functional_minimization_sqlite(data)

    n_procs   = multiprocessing.cpu_count()
    chunksize = max(1, len(data) // (n_procs * 8))
    print(f"{Colors.CYAN}[MP]{Colors.RESET} {n_procs} processes, chunksize={chunksize}.")

    # sig_tuple → list of (rule_text, count)
    signature_map: Dict[tuple, List[Tuple[str, int]]] = {}

    with multiprocessing.Pool(
        processes=n_procs,
        initializer=_worker_init,
        initargs=(TEST_VECTOR,),
    ) as pool:
        for sig, rule_data in tqdm(
            pool.imap_unordered(_compute_signature, data, chunksize=chunksize),
            total=len(data),
            desc="Simulating rules",
            unit=" rules",
        ):
            signature_map.setdefault(sig, []).append(rule_data)

    # Unsupported-opcode rules: keep all of them (cannot compare functionally)
    unsupported_group = signature_map.pop(_UNSUPPORTED_SIG, [])

    final: List[Tuple[str, int]] = list(unsupported_group)
    for group in signature_map.values():
        group.sort(key=lambda kv: kv[1], reverse=True)
        best_rule = group[0][0]
        total_cnt = sum(cnt for _, cnt in group)
        final.append((best_rule, total_cnt))

    final.sort(key=lambda kv: kv[1], reverse=True)
    removed = len(data) - len(final)
    print_success(f"Removed {removed:,} functionally redundant rules.")
    if unsupported_group:
        print_info(
            f"Retained {len(unsupported_group):,} rules with unsupported opcodes."
        )
    return final


# ==============================================================================
# LEVENSHTEIN FILTERING
# ==============================================================================

def levenshtein_distance(s1: str, s2: str, max_dist: Optional[int] = None) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        cur = [i + 1]
        row_min = i + 1
        for j, c2 in enumerate(s2):
            val = min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (c1 != c2))
            cur.append(val)
            if val < row_min:
                row_min = val
        if max_dist is not None and row_min > max_dist:
            return row_min
        prev = cur

    return prev[-1]


@memory_safe_operation("Levenshtein Filtering", 85)
def levenshtein_filter(
    data:         List[Tuple[str, int]],
    max_distance: int = 2,
) -> List[Tuple[str, int]]:
    print_section("Levenshtein Filtering")
    print_warning("Can be slow for large datasets.")

    if not data:
        return data

    if len(data) > 5_000:
        print_warning(f"Large dataset ({len(data):,} rules). This may take a while.")
        if input(
            f"{Colors.YELLOW}Continue? (y/N): {Colors.RESET}"
        ).strip().lower() not in ('y', 'yes'):
            return data

    while True:
        try:
            raw = input(
                f"{Colors.YELLOW}Enter max Levenshtein distance (1-10) [{max_distance}]: {Colors.RESET}"
            ).strip()
            if not raw:
                break
            v = int(raw)
            if 1 <= v <= 10:
                max_distance = v
                break
            print_error("Enter a value between 1 and 10.")
        except ValueError:
            print_error("Invalid number.")

    unique:  List[Tuple[str, int]] = []
    removed: int                   = 0

    for rule, cnt in tqdm(data, desc="Levenshtein filtering"):
        similar = any(
            levenshtein_distance(rule, existing, max_dist=max_distance) <= max_distance
            for existing, _ in unique
        )
        if similar:
            removed += 1
        else:
            unique.append((rule, cnt))

    print_success(f"Removed {removed:,} similar rules. Remaining: {len(unique):,}")
    return unique


# ==============================================================================
# PARETO ANALYSIS
# ==============================================================================

def display_pareto_curve(data: List[Tuple[str, int]]) -> None:
    if not data:
        print_error("No data to analyse.")
        return
    total_value = sum(c for _, c in data)
    print_header("PARETO ANALYSIS")
    print(f"Total rules:       {colorize(f'{len(data):,}',       Colors.CYAN)}")
    print(f"Total occurrences: {colorize(f'{total_value:,}', Colors.CYAN)}\n")

    targets  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    next_t   = 0
    cum      = 0
    milestones: List[Tuple[int, int, float]] = []

    print(f"{Colors.BOLD}{'Rank':>6} {'Rule':<30} {'Count':>10} {'Cumulative':>12} {'% Total':>8}{Colors.RESET}")
    print(f"{Colors.BOLD}{'-' * 70}{Colors.RESET}")

    for i, (rule, cnt) in enumerate(data):
        cum += cnt
        pct  = cum / total_value * 100
        show = (i < 10) or (next_t < len(targets) and pct >= targets[next_t])
        if show:
            color = Colors.GREEN if i < 10 else (Colors.YELLOW if pct <= 80 else Colors.RED)
            print(f"{color}{i+1:>6} {rule:<30} {cnt:>10,} {cum:>12,} {pct:>7.1f}%{Colors.RESET}")
        if next_t < len(targets) and pct >= targets[next_t]:
            milestones.append((targets[next_t], i + 1, pct))
            next_t += 1
        if i >= 10 and next_t >= len(targets):
            break

    print(f"{Colors.BOLD}{'-' * 70}{Colors.RESET}")
    print(f"\n{Colors.BOLD}PARETO MILESTONES:{Colors.RESET}")
    for target, rules_needed, actual in milestones:
        pct_rules = rules_needed / len(data) * 100
        color     = Colors.GREEN if target <= 50 else (Colors.YELLOW if target <= 80 else Colors.RED)
        print(f"  {color}{target:>2}% of value:{Colors.RESET} "
              f"{rules_needed:>6,} rules ({pct_rules:5.1f}% of total) – actual: {actual:5.1f}%")

    print(f"\n{Colors.BOLD}PARETO CURVE (ASCII):{Colors.RESET}")
    pts  = 20
    step = max(1, len(data) // pts)
    for i in range(pts + 1):
        idx     = min(i * step, len(data) - 1)
        cum_val = sum(c for _, c in data[:idx + 1])
        pct     = cum_val / total_value * 100
        bar     = "█" * int(pct / 5)
        y       = 100 - (i * 5)
        if y % 20 == 0 or i in (0, pts):
            print(f"{y:>4}% ┤ {bar}")
    print("    0% ┼" + "─" * 20)
    print("      0%         50%        100%")
    print("       Cumulative % of rules")


def analyze_cumulative_value(
    sorted_data: List[Tuple[str, int]], total_lines: int
) -> None:
    if not sorted_data:
        print_error("No data to analyse.")
        return
    total_value = sum(c for _, c in sorted_data)
    cum         = 0
    milestones: List[Tuple[int, int]] = []
    targets     = [50, 80, 90, 95]
    next_t      = 0

    for i, (_, cnt) in enumerate(sorted_data):
        cum += cnt
        pct  = cum / total_value * 100
        if next_t < len(targets) and pct >= targets[next_t]:
            milestones.append((targets[next_t], i + 1))
            next_t += 1
        if next_t >= len(targets):
            break

    print_header("CUMULATIVE VALUE ANALYSIS (PARETO) – SUGGESTED CUTOFFS")
    print(f"Total value:  {colorize(f'{total_value:,}', Colors.CYAN)}")
    print(f"Unique rules: {colorize(f'{len(sorted_data):,}', Colors.CYAN)}")
    for target, rules_needed in milestones:
        pct_rules = rules_needed / len(sorted_data) * 100
        color     = Colors.GREEN if target <= 80 else (Colors.YELLOW if target <= 90 else Colors.RED)
        print(f"{color}[{target}% OF VALUE]:{Colors.RESET} "
              f"{colorize(f'{rules_needed:,}', Colors.CYAN)} rules ({pct_rules:.2f}%)")
    print(f"{Colors.BOLD}{'-' * 60}{Colors.RESET}")
    if milestones:
        last = milestones[-1][1]
        print(f"{Colors.GREEN}[SUGGESTION]{Colors.RESET} "
              f"Consider: {colorize(f'{last:,}', Colors.CYAN)} or "
              f"{colorize(f'{int(last * 1.1):,}', Colors.CYAN)} rules.")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")


# ==============================================================================
# FILTERING FUNCTIONS
# ==============================================================================

def filter_by_min_occurrence(data: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    if not data:
        return data
    max_cnt   = data[0][1]
    suggested = max(1, sum(c for _, c in data) // 1000)
    while True:
        try:
            thresh = int(input(
                f"{Colors.YELLOW}Enter MIN occurrence (1–{max_cnt:,}, suggested {suggested:,}): {Colors.RESET}"
            ))
            if 1 <= thresh <= max_cnt:
                filtered = [(r, c) for r, c in data if c >= thresh]
                print_success(f"Kept {len(filtered):,} rules.")
                return filtered
            print_error(f"Value must be between 1 and {max_cnt:,}.")
        except ValueError:
            print_error("Invalid number.")


def filter_by_max_rules(data: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    if not data:
        return data
    maximum = len(data)
    while True:
        try:
            limit = int(input(
                f"{Colors.YELLOW}Enter MAX number of rules to keep (1–{maximum:,}): {Colors.RESET}"
            ))
            if 1 <= limit <= maximum:
                filtered = data[:limit]
                print_success(f"Kept top {len(filtered):,} rules.")
                return filtered
            print_error(f"Value must be between 1 and {maximum:,}.")
        except ValueError:
            print_error("Invalid number.")


def inverse_mode_filter(data: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    if not data:
        return data
    maximum = len(data)
    while True:
        try:
            cutoff = int(input(
                f"{Colors.YELLOW}Enter cutoff rank (rules BELOW this rank kept, 1–{maximum:,}): {Colors.RESET}"
            ))
            if 1 <= cutoff <= maximum:
                filtered = data[cutoff:]
                print_success(f"Kept {len(filtered):,} rules.")
                return filtered
            print_error(f"Value must be between 1 and {maximum:,}.")
        except ValueError:
            print_error("Invalid number.")


# ==============================================================================
# OUTPUT FORMATTING AND SAVING
# ==============================================================================

def expand_rule(rule: str) -> str:
    """Return *rule* with each operator+args token separated by a space."""
    return ' '.join(TOKEN_REGEX.findall(rule))


def save_rules(
    data:      List[Tuple],
    filename:  Optional[str] = None,
    mode_name: str           = 'filtered',
) -> bool:
    """
    Unified rule-save function.

    v3.1: final safety-net pass that strips any rule still containing a
    NEVER_PRODUCE_OP before writing to disk.
    """
    if not data:
        print_error("No rules to save!")
        return False

    def _extract_rule(item) -> str:
        return item[0] if isinstance(item, tuple) else item

    clean_data = [item for item in data if not _has_banned_op(_extract_rule(item))]
    dropped = len(data) - len(clean_data)
    if dropped:
        print_warning(f"save_rules: dropped {dropped:,} rule(s) containing banned operators.")
    if not clean_data:
        print_error("No producible rules to save after banned-op filter!")
        return False

    if filename is None:
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"concentrator_{mode_name}_{len(clean_data)}rules_{ts}.rule"

    try:
        with open(filename, 'w', encoding='utf-8') as fh:
            fh.write(f"# CONCENTRATOR v3.3 – {mode_name.upper()} MODE OUTPUT\n")
            fh.write(f"# Generated:   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.write(f"# Total rules: {len(clean_data):,}\n")
            fh.write(f"# Format:      {STATE.output_format}\n#\n")
            for item in clean_data:
                rule = _extract_rule(item)
                line = expand_rule(rule) if STATE.output_format == 'expanded' else rule
                fh.write(line + '\n')
        print_success(f"Saved {len(clean_data):,} rules → {filename}")
        return True
    except IOError as exc:
        print_error(f"Failed to save: {exc}")
        return False


# ==============================================================================
# HASHCAT RULE CLEANUP — data-driven validator
# ==============================================================================

_OP_VALIDATION_SPEC: Dict[str, List[str]] = {
    ':': [], 'l': [], 'u': [], 'c': [], 'C': [], 't': [], 'r': [], 'd': [], 'f': [],
    '{': [], '}': [], '[': [], ']': [], 'q': [], 'k': [], 'K': [], 'a': [], 'E': [],
    'T': ['N'], 'p': ['N'], 'D': ['N'], 'z': ['N'], 'Z': ['N'], "'": ['N'],
    'L': ['N'], 'R': ['N'], '+': ['N'], '-': ['N'], '.': ['N'], ',': ['N'],
    'y': ['N'], 'Y': ['N'], '_': ['N'],
    'x': ['N', 'N'], 'O': ['N', 'N'], '*': ['N', 'N'],
    '$': ['C'], '^': ['C'], '@': ['C'], 'e': ['C'],
    's': ['C', 'C'],
    'i': ['N', 'C'], 'o': ['N', 'C'], '3': ['N', 'C'],
}


class HashcatRuleCleaner:
    """
    Validates hashcat rules against CPU or GPU compatibility constraints.

    v3.1: NEVER_PRODUCE_OPS operators are always rejected.
    The _OP_VALIDATION_SPEC table no longer has entries for those operators.
    """

    MAX_RULES = 255

    def __init__(self, mode: int = 1) -> None:
        if mode not in (1, 2):
            raise ValueError("mode must be 1 (CPU) or 2 (GPU)")
        self.mode = mode

    @staticmethod
    def _conv_ctoi(c: str) -> int:
        if '0' <= c <= '9':
            return ord(c) - ord('0')
        if 'A' <= c <= 'Z':
            return ord(c) - ord('A') + 10
        return -1

    def validate_rule(self, rule_line: str) -> bool:
        clean = rule_line.replace(' ', '')
        if not clean:
            return False

        cnt  = 0
        pos  = 0
        n    = len(clean)

        while pos < n:
            op = clean[pos]

            if op not in _OP_VALIDATION_SPEC:
                return False

            spec = _OP_VALIDATION_SPEC[op]
            for kind in spec:
                pos += 1
                if pos >= n:
                    return False
                if kind == 'N' and self._conv_ctoi(clean[pos]) == -1:
                    return False

            cnt += 1
            pos += 1

            if cnt > self.MAX_RULES:
                return False

        return True

    def clean_rules(
        self, rules_data: List[Tuple[str, int]]
    ) -> List[Tuple[str, int]]:
        mode_label = 'GPU' if self.mode == 2 else 'CPU'
        print_section(f"Hashcat Rule Validation ({mode_label} mode)")
        print(f"Validating {colorize(f'{len(rules_data):,}', Colors.CYAN)} rules...")
        valid:   List[Tuple[str, int]] = []
        invalid: int                   = 0
        for rule, cnt in tqdm(rules_data, desc="Validating rules"):
            if self.validate_rule(rule):
                valid.append((rule, cnt))
            else:
                invalid += 1
        print_success(f"Removed {invalid:,} invalid rules. {len(valid):,} valid remaining.")
        return valid


def hashcat_rule_cleanup(
    data: List[Tuple[str, int]], mode: int = 1
) -> List[Tuple[str, int]]:
    return HashcatRuleCleaner(mode).clean_rules(data)


def gpu_extract_and_validate_rules(
    full_rule_counts: Dict[str, int],
    top_rules:        int,
    gpu_enabled:      bool,
) -> List[Tuple[str, int]]:
    sorted_rules = sorted(full_rule_counts.items(), key=lambda kv: kv[1], reverse=True)
    candidates = [r for r, _ in sorted_rules[:top_rules * 2] if not _has_banned_op(r)]

    if gpu_enabled:
        gpu_valid = gpu_validate_rules(candidates)
        result: List[Tuple[str, int]] = []
        for rule, is_valid in zip(candidates, gpu_valid):
            if not is_valid:
                continue
            if STATE.gpu_mode_enabled and not HashcatRuleCleaner(2).validate_rule(rule):
                continue
            result.append((rule, full_rule_counts[rule]))
        return result[:top_rules]
    else:
        return [
            (r, full_rule_counts[r]) for r in candidates if is_valid_hashcat_rule(r)
        ][:top_rules]


# ==============================================================================
# ENHANCED INTERACTIVE PROCESSING LOOP
# ==============================================================================

def enhanced_interactive_processing_loop(
    original_data: List[Tuple[str, int]],
    total_lines:   int,
    args:          Any,
    initial_mode:  str = "extracted",
) -> List[Tuple[str, int]]:
    current_data = original_data
    orig_count   = len(current_data)
    print_header("ENHANCED RULE PROCESSING – INTERACTIVE MENU")
    print(f"Initial dataset: {colorize(f'{orig_count:,}', Colors.CYAN)} unique rules")

    try:
        while True:
            print(f"\n{Colors.BOLD}{'-' * 80}{Colors.RESET}")
            print(f"{Colors.BOLD}ADVANCED FILTERING OPTIONS:{Colors.RESET}")
            print(f" {Colors.GREEN}(1){Colors.RESET} Filter by MINIMUM OCCURRENCE")
            print(f" {Colors.GREEN}(2){Colors.RESET} Filter by MAXIMUM NUMBER OF RULES (top N)")
            print(f" {Colors.GREEN}(3){Colors.RESET} Filter by FUNCTIONAL REDUNDANCY [RAM intensive]")
            print(f" {Colors.GREEN}(4){Colors.RESET} INVERSE MODE – keep rules BELOW the cut-off rank")
            print(f" {Colors.GREEN}(5){Colors.RESET} HASHCAT CLEANUP – validate (CPU/GPU modes)")
            print(f" {Colors.GREEN}(6){Colors.RESET} LEVENSHTEIN FILTER – remove similar rules")
            print(f" {Colors.GREEN}(7){Colors.RESET} TOGGLE OUTPUT FORMAT (currently: {STATE.output_format})")
            print(f"\n{Colors.BOLD}ANALYSIS & UTILITIES:{Colors.RESET}")
            print(f" {Colors.BLUE}(p){Colors.RESET} PARETO analysis")
            print(f" {Colors.BLUE}(s){Colors.RESET} SAVE current rules")
            print(f" {Colors.BLUE}(r){Colors.RESET} RESET to original dataset")
            print(f" {Colors.BLUE}(i){Colors.RESET} Dataset information")
            print(f" {Colors.BLUE}(q){Colors.RESET} QUIT")
            print(f"{Colors.BOLD}{'-' * 80}{Colors.RESET}")
            choice = input(f"{Colors.YELLOW}Enter choice: {Colors.RESET}").strip().lower()

            if choice == 'q':
                print_header("THANK YOU FOR USING CONCENTRATOR v3.3!")
                break

            elif choice == 'p':
                display_pareto_curve(current_data)

            elif choice == 's':
                print(f"\n{Colors.CYAN}Save Options:{Colors.RESET}")
                print(f" {Colors.GREEN}(1){Colors.RESET} Auto filename")
                print(f" {Colors.GREEN}(2){Colors.RESET} Custom filename")
                print(f" {Colors.GREEN}(3){Colors.RESET} Cancel")
                sc = input(f"{Colors.YELLOW}Choose: {Colors.RESET}").strip()
                if sc == '1':
                    save_rules(current_data, mode_name=f"{initial_mode}_filtered")
                elif sc == '2':
                    name = input(f"{Colors.YELLOW}Enter filename: {Colors.RESET}").strip()
                    if name:
                        if not name.endswith(('.rule', '.txt')):
                            name += '.rule'
                        save_rules(current_data, filename=name, mode_name=f"{initial_mode}_filtered")

            elif choice == 'r':
                current_data = original_data
                print_success(f"Restored original dataset: {len(current_data):,} rules.")

            elif choice == 'i':
                print_section("DATASET INFORMATION")
                reduction = (orig_count - len(current_data)) / orig_count * 100 if orig_count else 0.0
                print(f"Original: {colorize(f'{orig_count:,}', Colors.CYAN)}")
                print(f"Current:  {colorize(f'{len(current_data):,}', Colors.CYAN)}")
                print(f"Reduction:{colorize(f'{reduction:.1f}%', Colors.GREEN if reduction > 0 else Colors.YELLOW)}")
                if current_data:
                    maxc = current_data[0][1]
                    minc = current_data[-1][1]
                    avgc = sum(c for _, c in current_data) / len(current_data)
                    print(f"Max occ:  {colorize(f'{maxc:,}', Colors.CYAN)}")
                    print(f"Min occ:  {colorize(f'{minc:,}', Colors.CYAN)}")
                    print(f"Avg occ:  {colorize(f'{avgc:.1f}', Colors.CYAN)}")

            elif choice == '1':
                current_data = filter_by_min_occurrence(current_data)
            elif choice == '2':
                current_data = filter_by_max_rules(current_data)
            elif choice == '3':
                result = functional_minimization(current_data)
                if result is not None:
                    current_data = result
            elif choice == '4':
                current_data = inverse_mode_filter(current_data)
            elif choice == '5':
                print(f"\n{Colors.MAGENTA}[HASHCAT CLEANUP]{Colors.RESET} Choose mode:")
                print(f" {Colors.CYAN}(1){Colors.RESET} CPU (transformation rules only — memory/reject ops always excluded)")
                print(f" {Colors.CYAN}(2){Colors.RESET} GPU (same as CPU, no extra restrictions)")
                m    = input(f"{Colors.YELLOW}Mode (1 or 2): {Colors.RESET}").strip()
                mode = 1 if m == '1' else 2
                current_data = hashcat_rule_cleanup(current_data, mode)
            elif choice == '6':
                result = levenshtein_filter(
                    current_data, getattr(args, 'levenshtein_max_dist', 2)
                )
                if result is not None:
                    current_data = result
            elif choice == '7':
                STATE.output_format = 'expanded' if STATE.output_format == 'line' else 'line'
                print_success(f"Output format → {STATE.output_format}")
                continue
            else:
                print_error("Invalid choice.")
                continue

            if choice in ('1', '2', '3', '4', '5', '6'):
                reduction = (
                    (orig_count - len(current_data)) / orig_count * 100
                    if orig_count else 0.0
                )
                print_success(
                    f"Dataset updated: {len(current_data):,} rules ({reduction:.1f}% reduction)"
                )
                if current_data:
                    if input(
                        f"{Colors.YELLOW}Show Pareto analysis? (Y/n): {Colors.RESET}"
                    ).strip().lower() not in ('n', 'no'):
                        display_pareto_curve(current_data)
                if input(
                    f"{Colors.YELLOW}Save current dataset? (y/N): {Colors.RESET}"
                ).strip().lower() in ('y', 'yes'):
                    save_rules(current_data, mode_name=f"{initial_mode}_filtered")

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interactive menu interrupted.{Colors.RESET}")
        if input(
            f"{Colors.YELLOW}Save before exiting? (y/N): {Colors.RESET}"
        ).strip().lower() in ('y', 'yes'):
            save_rules(current_data, mode_name=f"{initial_mode}_filtered")

    return current_data


# ==============================================================================
# MAIN PROCESSING FUNCTIONS
# ==============================================================================

def process_multiple_files_concentrator(args: Any) -> None:
    print_header("PROCESSING MODE – Interactive Rule Minimization")
    all_fps = find_rule_files_recursive(args.paths, max_depth=3)
    if not all_fps:
        print_error("No rule files found.")
        return
    set_global_flags(args.temp_dir, args.in_memory)
    sorted_ops, full_rule_counts, _ = analyze_rule_files_parallel(all_fps, args.max_length)
    if not full_rule_counts:
        print_error("No rules found in files.")
        return
    rules_data = sorted(full_rule_counts.items(), key=lambda kv: kv[1], reverse=True)
    print_success(f"Loaded {len(rules_data):,} unique rules.")
    final = enhanced_interactive_processing_loop(
        rules_data, sum(full_rule_counts.values()), args, "processed"
    )
    if final:
        save_rules(final, filename=args.output_base_name + "_processed.rule", mode_name="processed")


def concentrator_main_processing(args: Any) -> None:
    global STATE

    MODE_META = {
        'extraction': ('extracted', '_extracted.rule', Colors.GREEN),
        'combo':      ('combo',     '_combo.rule',     Colors.BLUE),
        'markov':     ('markov',    '_markov.rule',    Colors.MAGENTA),
    }

    if args.extract_rules:
        active_mode = 'extraction'
    elif args.generate_combo:
        active_mode = 'combo'
    elif args.generate_markov_rules:
        active_mode = 'markov'
    else:
        print_error("No mode selected. Exiting.")
        return

    initial_mode, output_suffix, mode_color = MODE_META[active_mode]
    output_file = args.output_base_name + output_suffix

    STATE.output_format = args.output_format if args.output_format in ('line', 'expanded') else 'line'

    print(f"\n{Colors.CYAN}Active Mode:{Colors.END} {mode_color}{Colors.BOLD}{active_mode.upper()}{Colors.END}")
    print(f"{Colors.CYAN}Output File:{Colors.END} {Colors.WHITE}{output_file}{Colors.END}")
    print(f"{Colors.CYAN}Output Format:{Colors.END} {STATE.output_format}")

    if active_mode == 'markov':
        markov_min = args.markov_length[0]
        markov_max = args.markov_length[-1]
    elif active_mode == 'combo':
        combo_min  = args.combo_length[0]
        combo_max  = args.combo_length[-1]

    gpu_enabled = False
    if not args.no_gpu:
        gpu_enabled = setup_opencl()
        if gpu_enabled:
            STATE.gpu_mode_enabled = True
            print_success("GPU Acceleration: ENABLED")
        else:
            print_warning("GPU Acceleration: Disabled (CPU fallback)")
    else:
        print_warning("GPU Acceleration: manually disabled")

    print_section("Collecting Rule Files (recursive, max depth 3)")
    all_fps = find_rule_files_recursive(args.paths, max_depth=3)
    all_fps = [fp for fp in all_fps if os.path.basename(fp) != os.path.basename(output_file)]
    if not all_fps:
        print_error("No rule files found. Exiting.")
        return
    print_success(f"Found {len(all_fps)} rule files.")
    set_global_flags(args.temp_dir, args.in_memory)

    print_section("Parallel Rule File Analysis")
    sorted_ops, full_rule_counts, _ = analyze_rule_files_parallel(all_fps, args.max_length)
    if not sorted_ops:
        print_error("No operators found. Exiting.")
        return

    markov_probs, markov_totals = None, None
    needs_markov = (
        active_mode == 'markov'
        or (active_mode == 'extraction' and getattr(args, 'statistical_sort', False))
    )
    if needs_markov:
        print_section("Building Markov Model")
        markov_probs, markov_totals = get_markov_model(full_rule_counts)
    else:
        print_info("Skipping Markov model (not needed for this mode).")

    result_data: List[Tuple] = []

    if active_mode == 'extraction':
        print_section("Rule Extraction and Validation")
        if args.statistical_sort:
            print_info("Sort: Statistical (Markov weight)")
            if markov_probs is None:
                print_error("Statistical sort requires the Markov model.")
                return
            sorted_by_weight = get_markov_weighted_rules(full_rule_counts, markov_probs, markov_totals)
            if gpu_enabled and sorted_by_weight:
                candidates = [r for r, _ in sorted_by_weight[:args.top_rules * 2]]
                gpu_valid  = gpu_validate_rules(candidates)
                weight_map = {r: w for r, w in sorted_by_weight}
                validated: List[Tuple[str, float]] = []
                for rule, is_valid in zip(candidates, gpu_valid):
                    if not is_valid:
                        continue
                    if STATE.gpu_mode_enabled and not HashcatRuleCleaner(2).validate_rule(rule):
                        continue
                    validated.append((rule, weight_map[rule]))
                result_data = validated[:args.top_rules]
                print_success(f"GPU validated {len(result_data):,} statistically sorted rules.")
            else:
                result_data = sorted_by_weight[:args.top_rules]
        else:
            print_info("Sort: Frequency (raw count) with GPU validation")
            result_data = gpu_extract_and_validate_rules(full_rule_counts, args.top_rules, gpu_enabled)
        print_success(f"Extracted {len(result_data):,} top unique rules.")

    elif active_mode == 'markov':
        print_section("Markov Rule Generation")
        markov_results = generate_rules_from_markov_model(
            markov_probs, args.generate_target, markov_min, markov_max,
            gpu_mode=STATE.gpu_mode_enabled,
        )
        if gpu_enabled and markov_results:
            rules_only = [r for r, _ in markov_results]
            gpu_valid  = gpu_validate_rules(rules_only, args.max_length)
            w_map      = {r: w for r, w in markov_results}
            valid_m    = [
                (r, w_map[r]) for r, v in zip(rules_only, gpu_valid)
                if v and (not STATE.gpu_mode_enabled or HashcatRuleCleaner(2).validate_rule(r))
            ]
            print_success(f"GPU validated {len(valid_m):,}/{len(markov_results):,} Markov rules.")
            result_data = valid_m[:args.generate_target]
        else:
            result_data = markov_results

    elif active_mode == 'combo':
        print_section("Combinatorial Rule Generation")
        top_ops = find_min_operators_for_target(sorted_ops, args.combo_target, combo_min, combo_max)
        print_info(f"Using {len(top_ops)} operators for ~{args.combo_target:,} target rules.")
        generated_set = generate_rules_parallel(top_ops, combo_min, combo_max, gpu_mode=STATE.gpu_mode_enabled)
        result_data   = [(r, 1) for r in generated_set]
        print_success(f"Generated {len(result_data):,} combinatorial rules.")

    print(f"\n{Colors.CYAN}{Colors.BOLD}" + "=" * 60)
    print("ENHANCED PROCESSING OPTIONS")
    print("=" * 60 + f"{Colors.END}")
    enter_interactive = input(
        f"\n{Colors.YELLOW}Enter enhanced interactive mode? (Y/n): {Colors.RESET}"
    ).strip().lower()

    if enter_interactive not in ('n', 'no'):
        total_lines = sum(full_rule_counts.values())
        final_data  = enhanced_interactive_processing_loop(result_data, total_lines, args, initial_mode)
        if final_data:
            save_rules(final_data, filename=output_file, mode_name=active_mode)
            print_success(f"Final rules saved → {output_file}")
    else:
        if result_data:
            save_rules(result_data, filename=output_file, mode_name=active_mode)
            print_success(f"Rules saved → {output_file}")

    print_success("Processing complete.")
    if gpu_enabled:
        print_success("GPU Acceleration was used.")
    print_memory_status()


# ==============================================================================
# INTERACTIVE MODE
# ==============================================================================

def interactive_mode() -> Optional[Dict]:
    print_header("CONCENTRATOR v3.3 – INTERACTIVE MODE")
    settings: Dict[str, Any] = {}

    print(f"\n{Colors.CYAN}Input Configuration:{Colors.END}")
    while True:
        raw = input(f"{Colors.YELLOW}Enter rule files/directories (space-separated): {Colors.END}").strip()
        if not raw:
            print_error("Please provide at least one path.")
            continue
        paths = raw.split()
        valid = [p for p in paths if os.path.exists(p)]
        for p in paths:
            if p not in valid:
                print_warning(f"Path not found: {p}")
        if valid:
            settings['paths'] = valid
            break
        print_error("No valid paths provided.")

    print(f"\n{Colors.CYAN}Analysing Input Data...{Colors.END}")
    recommended_mode: Optional[str] = None
    try:
        all_fps = find_rule_files_recursive(settings['paths'], max_depth=3)
        if not all_fps:
            print_error("No rule files found.")
            return None
        total_rules   = 0
        unique_rules:  Set[str] = set()
        max_rule_len  = 0
        for fp in all_fps[:10]:
            try:
                with open(fp, 'r', errors='ignore') as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith('#') or len(line) > 100:
                            continue
                        total_rules += 1
                        unique_rules.add(line)
                        max_rule_len = max(max_rule_len, len(line))
            except OSError:
                continue

        est_total = total_rules * max(1, len(all_fps) // 10)
        print(f"{Colors.CYAN}Quick Analysis:{Colors.END}")
        print(f"  Files:          {len(all_fps)}")
        print(f"  Sampled rules:  {total_rules}")
        print(f"  Est. total:     {est_total:,}")
        print(f"  Unique sample:  {len(unique_rules)}")
        print(f"  Max rule len:   {max_rule_len}")

        if est_total < 1000:
            recommended_mode = 'combo'
        elif len(unique_rules) / max(1, total_rules) < 0.3:
            recommended_mode = 'extraction'
        else:
            recommended_mode = 'markov'

        recommendations = {
            'combo':      'Small dataset → Combinatorial Generation',
            'extraction': 'Low uniqueness → Extraction',
            'markov':     'Good diversity → Markov',
        }
        print(f"\n{Colors.CYAN}Recommendation:{Colors.END} {recommendations[recommended_mode]}")
        if max_rule_len > 20:
            print(f"  Long rules detected → consider functional minimization later.")

    except Exception as exc:
        print_warning(f"Quick analysis failed: {exc}")

    print(f"\n{Colors.CYAN}Processing Mode:{Colors.END}")
    print(f"  {Colors.GREEN}1{Colors.END} – Extract top existing rules")
    print(f"  {Colors.GREEN}2{Colors.END} – Generate combinatorial rules")
    print(f"  {Colors.GREEN}3{Colors.END} – Generate Markov rules")
    if recommended_mode:
        rmap = {'extraction': '1', 'combo': '2', 'markov': '3'}
        print(f"{Colors.YELLOW}  Recommended: Mode {rmap[recommended_mode]}{Colors.END}")

    while True:
        choice = input(f"{Colors.YELLOW}Select mode (1-3): {Colors.END}").strip()
        if choice == '1':
            settings['mode'] = 'extraction'; break
        elif choice == '2':
            settings['mode'] = 'combo';      break
        elif choice == '3':
            settings['mode'] = 'markov';     break
        else:
            print_error("Enter 1, 2, or 3.")

    if settings['mode'] == 'extraction':
        while True:
            try:
                n = int(input(f"{Colors.YELLOW}Top rules to extract [10000]: {Colors.END}") or '10000')
                if n > 0:
                    settings['top_rules'] = n; break
                print_error("Positive number required.")
            except ValueError:
                print_error("Invalid number.")
        settings['statistical_sort'] = get_yes_no(
            f"{Colors.YELLOW}Use statistical sort?{Colors.END}", False
        )
    else:
        while True:
            try:
                n = int(input(f"{Colors.YELLOW}Target rules to generate [10000]: {Colors.END}") or '10000')
                if n > 0:
                    settings['target_rules'] = n; break
                print_error("Positive number required.")
            except ValueError:
                print_error("Invalid number.")
        while True:
            try:
                mn = int(input(f"{Colors.YELLOW}Min rule length [1]: {Colors.END}") or '1')
                mx = int(input(f"{Colors.YELLOW}Max rule length [3]: {Colors.END}") or '3')
                if 1 <= mn <= mx:
                    settings['min_len'] = mn
                    settings['max_len'] = mx
                    break
                print_error("min ≤ max and both ≥ 1.")
            except ValueError:
                print_error("Invalid numbers.")

    print(f"\n{Colors.CYAN}Global Settings:{Colors.END}")
    settings['output_base_name'] = (
        input(f"{Colors.YELLOW}Output base name ['concentrator_output']: {Colors.END}").strip()
        or 'concentrator_output'
    )
    while True:
        try:
            n = int(input(f"{Colors.YELLOW}Max rule length to process [31]: {Colors.END}") or '31')
            if n > 0:
                settings['max_length'] = n; break
            print_error("Positive number required.")
        except ValueError:
            print_error("Invalid number.")

    settings['no_gpu']    = not get_yes_no(f"{Colors.YELLOW}Enable GPU acceleration?{Colors.END}", True)
    settings['in_memory'] = get_yes_no(f"{Colors.YELLOW}Process entirely in RAM?{Colors.END}", False)

    print(f"\n{Colors.CYAN}Output Format:{Colors.END}")
    print(f"  {Colors.GREEN}1{Colors.END} – Standard line")
    print(f"  {Colors.GREEN}2{Colors.END} – Expanded (space-separated operators)")
    while True:
        fc = input(f"{Colors.YELLOW}Select (1-2): {Colors.END}").strip()
        if fc == '1':
            settings['output_format'] = 'line';     break
        elif fc == '2':
            settings['output_format'] = 'expanded'; break
        else:
            print_error("Enter 1 or 2.")

    if not settings['in_memory']:
        td = input(f"{Colors.YELLOW}Temp directory [system default]: {Colors.END}").strip()
        settings['temp_dir'] = td or None
    else:
        settings['temp_dir'] = None

    defaults: Dict[str, Any] = {
        'temp_dir': None, 'no_gpu': False, 'in_memory': False,
        'max_length': 31, 'output_base_name': 'concentrator_output', 'output_format': 'line',
    }
    if settings['mode'] == 'extraction':
        defaults.update({'top_rules': 10000, 'statistical_sort': False})
    else:
        defaults.update({'target_rules': 10000, 'min_len': 1, 'max_len': 3})
    for key, val in defaults.items():
        settings.setdefault(key, val)

    print(f"\n{Colors.CYAN}Configuration Summary:{Colors.END}")
    print(f"  Mode:          {settings['mode']}")
    print(f"  Input paths:   {len(settings['paths'])} location(s)")
    print(f"  Output base:   {settings['output_base_name']}")
    print(f"  Max rule len:  {settings['max_length']}")
    print(f"  GPU:           {'Enabled' if not settings['no_gpu'] else 'Disabled'}")
    print(f"  In-memory:     {'Yes' if settings['in_memory'] else 'No'}")
    print(f"  Output format: {settings['output_format']}")
    if settings['mode'] == 'extraction':
        print(f"  Top rules:     {settings['top_rules']}")
        print(f"  Stat sort:     {'Yes' if settings['statistical_sort'] else 'No'}")
    else:
        print(f"  Target rules:  {settings['target_rules']}")
        print(f"  Rule length:   {settings['min_len']}–{settings['max_len']}")

    if get_yes_no(f"\n{Colors.YELLOW}Start processing?{Colors.END}", True):
        return settings
    print_info("Configuration cancelled.")
    return None


# ==============================================================================
# USAGE
# ==============================================================================

def print_usage() -> None:
    C = Colors
    print(f"{C.BOLD}{C.CYAN}USAGE:{C.END}")
    print(f"  {C.WHITE}python concentrator.py [OPTIONS] FILE_OR_DIR ...{C.END}\n")

    sections = [
        ("MODES (choose one)",
         [("-e, --extract-rules",       "Extract top existing rules from input files"),
          ("-g, --generate-combo",      "Generate combinatorial rules from top operators"),
          ("-gm, --generate-markov-rules", "Generate statistically probable Markov rules"),
          ("-p, --process-rules",       "Interactive rule processing and minimization")]),
        ("EXTRACTION (-e)",
         [("-t INT",  "Number of top rules (default: 10000)"),
          ("-s",      "Sort by statistical weight")]),
        ("COMBINATORIAL (-g)",
         [("-n INT",      "Target rules (default: 100000)"),
          ("-l MIN MAX",  "Rule length range (default: 1 3)")]),
        ("MARKOV (-gm)",
         [("-gt INT",     "Target rules (default: 10000)"),
          ("-ml MIN MAX", "Rule length range (default: 1 3)")]),
        ("PROCESSING (-p)",
         [("-d",    "Use disk for large datasets"),
          ("-ld INT", "Max Levenshtein distance (default: 2)")]),
        ("OUTPUT",
         [("-f FORMAT",  "Output format: line or expanded (default: line)"),
          ("-ob NAME",   "Base name for output file")]),
        ("GLOBAL",
         [("-m INT",    "Max rule length (default: 31)"),
          ("--temp-dir DIR", "Temp directory"),
          ("--in-memory",   "Process entirely in RAM"),
          ("--no-gpu",      "Disable GPU acceleration")]),
    ]
    for title, opts in sections:
        print(f"\n{C.BOLD}{C.CYAN}{title}:{C.END}")
        for flag, desc in opts:
            print(f"  {C.YELLOW}{flag:<30}{C.END}{desc}")

    print(f"\n{C.BOLD}{C.CYAN}NOTES (v3.3):{C.END}")
    print(f"  {C.WHITE}Memory operators (M 4 6 X) and reject operators (< > ! / ( ) = % Q){C.END}")
    print(f"  {C.WHITE}are filtered at every pipeline stage and will never appear in output.{C.END}")
    print(f"  {C.WHITE}Combinatorial generation uses full token units ($5, sae, T3) with{C.END}")
    print(f"  {C.WHITE}round-trip re-parse validation to ensure only valid rules are saved.{C.END}")

    print(f"\n{C.BOLD}{C.CYAN}EXAMPLES:{C.END}")
    examples = [
        ("Extract top 5000 rules",            "python concentrator.py -e -t 5000 rules/*.rule"),
        ("Generate 50k combinatorial rules",   "python concentrator.py -g -n 50000 -l 2 4 hashcat/rules/"),
        ("Process with functional minimization", "python concentrator.py -p -d -f expanded rules/"),
        ("Interactive mode",                   "python concentrator.py"),
    ]
    for comment, cmd in examples:
        print(f"  {C.WHITE}# {comment}{C.END}")
        print(f"  {C.WHITE}{cmd}{C.END}\n")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    multiprocessing.freeze_support()

    print_banner()
    print_memory_status()

    mem_info = get_memory_usage()
    if mem_info and mem_info['ram_percent'] > 85:
        print_warning(f"High RAM usage detected ({mem_info['ram_percent']:.1f}%)")
        if mem_info['swap_total'] == 0:
            print_error("CRITICAL: No swap space available.")
            if not get_yes_no(f"{Colors.YELLOW}Continue anyway?{Colors.END}", default=False):
                sys.exit(1)
        else:
            print_warning("System will use swap. Performance may degrade.")

    if len(sys.argv) == 1:
        settings = interactive_mode()
        if not settings:
            sys.exit(0)

        ns = argparse.Namespace(
            paths            = settings['paths'],
            output_base_name = settings['output_base_name'],
            max_length       = settings['max_length'],
            no_gpu           = settings['no_gpu'],
            in_memory        = settings['in_memory'],
            temp_dir         = settings['temp_dir'],
            output_format    = settings['output_format'],
            extract_rules    = (settings['mode'] == 'extraction'),
            generate_combo   = (settings['mode'] == 'combo'),
            generate_markov_rules = (settings['mode'] == 'markov'),
            process_rules    = False,
        )
        if ns.extract_rules:
            ns.top_rules       = settings['top_rules']
            ns.statistical_sort = settings['statistical_sort']
        elif ns.generate_combo:
            ns.combo_target = settings['target_rules']
            ns.combo_length = [settings['min_len'], settings['max_len']]
        elif ns.generate_markov_rules:
            ns.generate_target = settings['target_rules']
            ns.markov_length   = [settings['min_len'], settings['max_len']]

        concentrator_main_processing(ns)

    elif len(sys.argv) == 2 and sys.argv[1] in ('-h', '--help'):
        print_usage()
        sys.exit(0)

    else:
        parser = argparse.ArgumentParser(
            description=f'{Colors.CYAN}Unified Hashcat Rule Processor with OpenCL support.{Colors.END}',
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            'paths', nargs='+',
            help='Paths to rule files or directories (max depth 3)',
        )
        parser.add_argument('-ob', '--output_base_name', default='concentrator_output')
        parser.add_argument('-f',  '--output-format', choices=['line', 'expanded'], default='line')

        mode_group = parser.add_mutually_exclusive_group(required=True)
        mode_group.add_argument('-e',  '--extract_rules',       action='store_true')
        mode_group.add_argument('-g',  '--generate_combo',      action='store_true')
        mode_group.add_argument('-gm', '--generate_markov_rules', action='store_true')
        mode_group.add_argument('-p',  '--process_rules',       action='store_true')

        parser.add_argument('-t',  '--top_rules',       type=int, default=10000)
        parser.add_argument('-s',  '--statistical_sort', action='store_true')
        parser.add_argument('-n',  '--combo_target',    type=int, default=100000)
        parser.add_argument('-l',  '--combo_length',    nargs='+', type=int, default=[1, 3])
        parser.add_argument('-gt', '--generate_target', type=int, default=10000)
        parser.add_argument('-ml', '--markov_length',   nargs='+', type=int, default=None)
        parser.add_argument('-d',  '--use_disk',        action='store_true')
        parser.add_argument('-ld', '--levenshtein_max_dist', type=int, default=2)
        parser.add_argument('-m',  '--max_length',      type=int, default=31)
        parser.add_argument('--temp-dir',  default=None)
        parser.add_argument('--in-memory', action='store_true')
        parser.add_argument('--no-gpu',    action='store_true')

        args = parser.parse_args()

        if args.markov_length is None:
            args.markov_length = [1, 3]
        if args.use_disk:
            args.in_memory = False
            print_info("Disk mode active (--use-disk).")

        if args.process_rules:
            process_multiple_files_concentrator(args)
        else:
            concentrator_main_processing(args)

    cleanup_temp_files()
    sys.exit(0)
