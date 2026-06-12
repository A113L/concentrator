#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations  # Python 3.8+ type-hint compatibility
"""
CONCENTRATOR v3.5 - Unified Hashcat Rule Processor

"""

import sys
import os
import re
import signal
import argparse
import hashlib
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

# -----------------------------------------------------------------------------
# Windows terminal color support (ANSI → WinAPI conversion)
# -----------------------------------------------------------------------------
if sys.platform == 'win32':
    try:
        import colorama
        colorama.init()                     # converts ANSI codes to Windows calls
    except ImportError:
        # Fallback: no conversion – ANSI codes may appear raw in older consoles
        # but modern Windows Terminal / PowerShell 7+ support them natively.
        pass

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

_cleanup_lock              = threading.Lock()
_cleanup_in_progress: bool = False
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


STATE = AppState()


# ==============================================================================
# COLORS (ANSI codes – converted on Windows if colorama is present)
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
    RESET     = END   # alias — both mean "reset all attributes"
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

# ALL_RULE_CHARS: every character that may legally appear in a hashcat rule.
# This includes all printable ASCII except space (space is only a token
# *separator* in expanded-format files; it is never stripped as an argument
# because process_single_file normalises via TOKEN_REGEX which preserves
# embedded-space arguments).
#
# v3.5 fix: the previous hard-coded set was missing the following valid
# operator/argument characters, causing them to be silently stripped from
# every rule during loading and corrupting any rule that used them:
#   {  }   — rotate-left / rotate-right (zero-arg operators)
#   _      — reject-unless-length  (one-num-arg operator, e.g.  _8)
#   '      — truncate-at           (one-num-arg operator, e.g.  '6)
#   \      — valid literal character argument for s, i, o, $, ^ …
#   "      — ditto
#   ;      — ditto
# Using PRINTABLE_ASCII minus space is both correct and forward-compatible:
# any future hashcat operator that uses a printable char will be handled.
ALL_RULE_CHARS: Set[str] = PRINTABLE_ASCII - {' '}

# ---------------------------------------------------------------------------
# NEVER_PRODUCE_OPS  (v3.1)
# ---------------------------------------------------------------------------
NEVER_PRODUCE_OPS: Set[str] = frozenset({'M', '4', '6', 'X', '<', '>', '!', '/', '(', ')', '=', '%', 'Q'})


# ---------------------------------------------------------------------------
# CRACK_FOCUSED_TOKENS  (v3.5)
# ---------------------------------------------------------------------------
# Curated set of tokens empirically known to produce many password cracks.
# Always injected into the combinatorial operator pool and Markov training
# corpus so they fire even when the input rule files are small or homogeneous.
#
# Categories:
#   • Case/structural transforms — the single most effective class of rules
#   • Digit appends/prepends — extremely common real-world password suffixes
#   • Common leet substitutions — sa@, se3, so0, ss5, st7 …
#   • Symbol appends — !, @, #, . (very frequent trailing chars)
#   • Truncation ops — passwords are often stem + short suffix
CRACK_FOCUSED_TOKENS: List[str] = [tok for tok in [
    # Case / word-level transforms
    'l', 'u', 'c', 'C', 't', 'E', 'r', 'd', 'f',
    # Structural
    '[', ']', '{', '}',
    # Digit appends
    '$0','$1','$2','$3','$4','$5','$6','$7','$8','$9',
    # Digit prepends
    '^0','^1','^2','^3','^4','^5','^6','^7','^8','^9',
    # Common leet substitutions (operator token, not banned)
    'sa@','se3','si!','so0','ss5','st7','sb6',
    # Symbol appends
    '$!','$@','$#','$.',
    # Truncation to common password lengths
    "'4","'5","'6","'7","'8",
] if tok[0] not in NEVER_PRODUCE_OPS]


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
    # Single-char operators — length sorting is irrelevant, but keep for safety
    patterns = [re.escape(op) for op in ALL_OPERATORS]
    patterns.sort(key=len, reverse=True)
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
    print("          CONCENTRATOR v3.5 - Unified Hashcat Rule Processor")
    print("=" * 80 + f"{Colors.END}")
    features = [
        "OpenCL GPU Acceleration for validation and generation",
        "Three Processing Modes: Extraction, Combinatorial, Markov",
        "Hashcat Rule Engine Simulation & Functional Minimization",
        "Rule Validation and Cleanup (CPU/GPU compatible)",
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
    for fp in list(_temp_files_to_cleanup):  # snapshot — safe to mutate original during loop
        try:
            if os.path.exists(fp):
                os.remove(fp)
                print_info(f"Cleaned up: {fp}")
        except OSError:
            pass
    _temp_files_to_cleanup.clear()


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
            return resp in ('y', 'yes')
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
# RULE VALIDATION (CPU) – replaced with rulest's logic
# ==============================================================================

def should_exclude_rule(rule: str) -> bool:
    """Return True if *rule* is a trivially-excluded single operator.

    This is a fast pre-check called before the full token-by-token loop in
    is_valid_hashcat_rule().  It only catches degenerate 1- or 2-character
    rules whose single operator is in NEVER_PRODUCE_OPS or is completely
    unknown to hashcat's parser.  All multi-operator rules are handled by
    the main loop.

    v3.5 fix: removed the previous catch-all for 3-char rules starting with
    '?', '=', or 'v'.  '=' and '%' are already in NEVER_PRODUCE_OPS and are
    caught by _has_banned_op(); '?' and 'v' are not recognised operators and
    are correctly rejected by the "any other character" fall-through in the
    main loop.  The over-broad 3-char check was incorrectly blocking any rule
    whose THIRD character happened to be '?', '=', or 'v', because the check
    looked at rule[0], not at the actual operator position.
    """
    if not rule:
        return False
    # A bare single-character excluded operator (M, 4, 6, Q already in
    # NEVER_PRODUCE_OPS; _ is a reject op that concentrator never generates)
    if len(rule) == 1 and rule in ('_', 'M', '4', '6', 'Q'):
        return True
    return False


def is_valid_hashcat_rule(rule: str) -> bool:
    """Return True if *rule* is syntactically valid for hashcat (CPU mode).

    Covers all operators listed in OPERATOR_ARGS.  Spaces inside the rule are
    treated as token separators and skipped (matching hashcat's own parser and
    rulest's validate_rule_for_gpu).

    v3.5 fix: added missing operators that caused mode-5 to incorrectly remove
    valid rules:
      _N  — reject unless password length == N  (one base-36 digit arg)
      {   — rotate left                         (zero args)
      }   — rotate right                        (zero args)
      '   — truncate at position N              (one base-36 digit arg)
    The _ operator is NOT in NEVER_PRODUCE_OPS, so it must be accepted here.
    """
    # Quick rejection for permanently excluded operators
    if should_exclude_rule(rule):
        return False
    # An empty rule (or all-whitespace) is not a valid rule
    if not rule.strip():
        return False
    pos = 0
    cnt = 0
    n = len(rule)

    def is_digit(c: str) -> bool:
        # hashcat uses base-36 positions: 0-9 and A-Z (for positions 10-35)
        return ('0' <= c <= '9') or ('A' <= c <= 'Z')

    while pos < n:
        c = rule[pos]
        if c == ' ':
            pos += 1
            continue
        # p, z, Z: one optional digit
        if c in ('p', 'z', 'Z'):
            cnt += 1
            pos += 1
            if pos < n and is_digit(rule[pos]):
                pos += 1
            continue
        # Zero-argument operators
        if c in (':', 'l', 'u', 'c', 'C', 't', 'r', 'd', 'f', 'q', 'k', 'K',
                 'E', '{', '}', '[', ']'):
            pos += 1
            cnt += 1
            continue
        # Operators with exactly one digit argument
        # v3.5: added '_' (reject-unless-length) and "'" (truncate-at)
        if c in ('T', 'D', 'L', 'R', '+', '-', '.', ',', "'", 'y', 'Y', '_'):
            pos += 1
            if pos >= n or not is_digit(rule[pos]):
                return False
            pos += 1
            cnt += 1
            continue
        # Operators with one digit and then one character (i, o, 3)
        if c in ('i', 'o', '3'):
            pos += 1
            if pos >= n or not is_digit(rule[pos]):
                return False
            pos += 1
            if pos >= n:
                return False
            pos += 1
            cnt += 1
            continue
        # Operators with two digit arguments (x, *, O)
        if c in ('x', '*', 'O'):
            pos += 1
            if pos >= n or not is_digit(rule[pos]):
                return False
            pos += 1
            if pos >= n or not is_digit(rule[pos]):
                return False
            pos += 1
            cnt += 1
            continue
        # s operator: two literal characters
        if c == 's':
            pos += 1
            if pos + 1 >= n:
                return False
            pos += 2
            cnt += 1
            continue
        # Operators with one literal character (^, $, @, e, etc.)
        if c in ('@', 'e', '$', '^'):
            pos += 1
            if pos >= n:
                return False
            pos += 1
            cnt += 1
            continue
        # Any other character is invalid
        return False

    # GPU rules limit (rulest uses MAX_GPU_RULES = 255)
    # Also require at least one valid operator (empty/all-space string already
    # caught above, but guard here too)
    return 1 <= cnt <= 255


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
      (operator_counts, rule_counts, clean_rules_list, temp_filepath_or_None,
       comment_lines)

    v3.1: rules containing any operator from NEVER_PRODUCE_OPS are silently
    dropped at this stage so they never enter the processing pipeline.

    v3.2: operator counting now uses TOKEN_REGEX.findall so that full tokens
    (e.g. '$5', 'sae', 'T3') are counted as atomic units instead of counting
    the operator character and its argument bytes separately.

    v3.5: comment lines (lines starting with '#') are counted separately and
    returned as the fifth element of the tuple.  They are never included in
    rule totals or occurrence statistics.
    """
    operator_counts:  Dict[str, int] = defaultdict(int)
    full_rule_counts: Dict[str, int] = defaultdict(int)
    clean_rules:      List[str]      = []
    tmp_path:         Optional[str]  = None
    comment_lines:    int            = 0

    try:
        with open(filepath, 'r', errors='ignore') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                # v3.5: count comment lines explicitly; never let them enter rule counts
                if line.startswith('#'):
                    comment_lines += 1
                    continue

                # ------------------------------------------------------------------
                # Normalise to compact form.
                #
                # v3.5 fix: the previous approach filtered characters with
                # ALL_RULE_CHARS and then checked len(line).  This had two bugs:
                #
                # 1.  ALL_RULE_CHARS was missing valid operator characters ({, },
                #     _, ', \, …), so those characters were silently stripped,
                #     corrupting any rule that used them.
                #
                # 2.  The length check ran on the *raw* line, so expanded-format
                #     input (tokens separated by spaces, e.g. "sA- sS~ $1 $2")
                #     was incorrectly rejected when the expanded form was longer
                #     than max_rule_length even though the compact form was fine.
                #
                # Fix: use TOKEN_REGEX.findall() which:
                #   • naturally ignores whitespace between tokens (spaces that are
                #     token separators are simply not matched and are skipped)
                #   • preserves whitespace that IS a literal character argument
                #     (e.g. "sP " = replace P with space — the trailing space IS
                #     part of the token and is included in the match)
                #   • drops characters that are not part of any valid token
                #     (truly malformed bytes), producing a clean compact form
                #
                # After findall we join tokens and apply the length limit on the
                # resulting compact string, not on the original raw line.
                # ------------------------------------------------------------------
                tokens = TOKEN_REGEX.findall(line)
                if not tokens:
                    continue
                clean = ''.join(tokens)
                if not clean or len(clean) > max_rule_length:
                    continue
                if _has_banned_op(clean):
                    continue
                # ------------------------------------------------------------------
                full_rule_counts[clean] += 1
                clean_rules.append(clean)
                for token in tokens:
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
            print_success(
                f"Processed: {filepath} → {tmp_path}"
                + (f" ({comment_lines:,} comment lines skipped)" if comment_lines else "")
            )
            return operator_counts, full_rule_counts, [], tmp_path, comment_lines
        else:
            print_success(
                f"Processed (in-memory): {filepath}"
                + (f" ({comment_lines:,} comment lines skipped)" if comment_lines else "")
            )
            return operator_counts, full_rule_counts, clean_rules, None, comment_lines

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
        return defaultdict(int), defaultdict(int), [], None, 0


def analyze_rule_files_parallel(
    filepaths: List[str], max_rule_length: int
) -> Tuple[List, Dict, List]:
    valid_fps = [fp for fp in filepaths if os.path.isfile(fp)]
    if not valid_fps:
        print_warning("No valid rule files to process.")
        return [], defaultdict(int), []

    total_op_counts:    Dict[str, int] = defaultdict(int)
    total_rule_counts:  Dict[str, int] = defaultdict(int)
    total_comment_lines: int           = 0
    temp_files: List[str] = []
    all_rules:  List[str] = []

    n_procs = min(os.cpu_count() or 1, len(valid_fps))
    tasks   = [(fp, max_rule_length) for fp in valid_fps]
    print_info(f"Parallel analysis of {len(valid_fps)} files using {n_procs} processes...")

    with multiprocessing.Pool(processes=n_procs) as pool:
        # v3.5: process_single_file now returns a 5-tuple (adds comment_lines)
        for op_c, rule_c, rules, tmp, comment_lines in pool.starmap(process_single_file, tasks):
            for op, cnt in op_c.items():
                total_op_counts[op] += cnt
            for rule, cnt in rule_c.items():
                total_rule_counts[rule] += cnt
            total_comment_lines += comment_lines
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
    if total_comment_lines:
        print_info(
            f"Comment lines skipped (not counted in totals): "
            f"{colorize(f'{total_comment_lines:,}', Colors.YELLOW)}"
        )
    sorted_op_counts = sorted(total_op_counts.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_op_counts, total_rule_counts, all_rules


# ==============================================================================
# MARKOV MODEL
# ==============================================================================

def get_markov_model(
    unique_rules: Dict[str, int],
    kn_discount:  float = 0.75,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Build a KN-smoothed fourth-order token-level Markov model.

    v3.5: Kneser-Ney with CORRECT += 1 semantics — each unique rule string
    contributes exactly 1 count to every n-gram it generates, regardless of
    corpus frequency.  This is the same equal-vote principle as the MLE model
    (restored after the += freq regression in v3.5 caused degenerate output).

    KN adds two improvements over plain MLE:
      1. Absolute discounting: max(count - D, 0) / total   removes D from
         every observed count, reserving probability mass for unseen pairs.
      2. Continuation-count unigram prior: P_KN(w) ∝ |unique contexts w follows|
         — tokens that are contextually flexible (c, l, sa@) get a higher floor
         than tokens that only ever appear after one specific context.
    """
    if not memory_intensive_operation_warning("Markov model building"):
        return None, None
    print_section("Building KN-Smoothed Token-Level Markov Model (D=%.2f, 4-grams)" % kn_discount)
    START = '^'
    D     = kn_discount

    counts: Dict = defaultdict(lambda: defaultdict(int))
    continuation_contexts: Dict[str, set] = defaultdict(set)
    skipped = 0

    for rule in unique_rules:           # keys only — += 1 per unique rule
        if not rule:
            continue
        tokens = TOKEN_REGEX.findall(rule)
        if not tokens or ''.join(tokens) != rule:
            skipped += 1
            continue
        n = len(tokens)
        counts[START][tokens[0]] += 1
        for i in range(n - 1):
            counts[tokens[i]][tokens[i + 1]] += 1
            continuation_contexts[tokens[i + 1]].add(tokens[i])
        for i in range(n - 2):
            ctx = (tokens[i], tokens[i + 1])
            counts[ctx][tokens[i + 2]] += 1
            continuation_contexts[tokens[i + 2]].add(ctx)
        for i in range(n - 3):
            ctx3 = (tokens[i], tokens[i+1], tokens[i+2])
            counts[ctx3][tokens[i+3]] += 1
            continuation_contexts[tokens[i+3]].add(ctx3)
        for i in range(n - 4):
            ctx4 = (tokens[i], tokens[i+1], tokens[i+2], tokens[i+3])
            counts[ctx4][tokens[i+4]] += 1
            continuation_contexts[tokens[i+4]].add(ctx4)

    if skipped:
        print_warning(f"Markov training: skipped {skipped:,} rules that did not tokenize cleanly.")

    # KN unigram: continuation probability
    cont_counts = {tok: len(ctxs) for tok, ctxs in continuation_contexts.items()}
    total_cont  = sum(cont_counts.values()) or 1
    kn_unigram  = {tok: c / total_cont for tok, c in cont_counts.items()}

    totals = {k: sum(v.values()) for k, v in counts.items()}
    probs: Dict = defaultdict(lambda: defaultdict(float))

    for ctx, next_counts in counts.items():
        total = totals[ctx]
        if total == 0:
            continue
        if ctx == START:
            for nxt, cnt in next_counts.items():
                probs[START][nxt] = cnt / total
            continue
        n_types = len(next_counts)
        lam     = (D * n_types) / total
        for nxt, cnt in next_counts.items():
            lower         = kn_unigram.get(nxt, 1e-9)
            probs[ctx][nxt] = max(cnt - D, 0.0) / total + lam * lower

    probs['__kn_unigram__'] = dict(kn_unigram)

    u_start    = len(probs.get(START, {}))
    u_unigram  = sum(1 for k in probs if isinstance(k, str) and k not in (START, '__kn_unigram__'))
    u_bigram   = sum(1 for k in probs if isinstance(k, tuple) and len(k) == 2)
    u_trigram  = sum(1 for k in probs if isinstance(k, tuple) and len(k) == 3)
    u_fourgram = sum(1 for k in probs if isinstance(k, tuple) and len(k) == 4)
    print_success(
        f"KN model (D={D}): {u_start} starters | "
        f"{u_unigram} unigram  {u_bigram} bigram  {u_trigram} trigram  {u_fourgram} 4-gram"
    )
    return probs, totals

def get_markov_weighted_rules(
    unique_rules:         Dict[str, int],
    markov_probabilities: Dict,
    total_transitions:    Dict,
) -> List[Tuple[str, float]]:
    """Score each rule by its log-probability under the token-level Markov model.

    Scoring strategy — highest available context wins at each position:
      1. P(tokens[0] | START)
      2. For each subsequent token tokens[i] (i >= 1), try in order:
           a. 4-gram context  tuple(tokens[i-4:i])  if i >= 4
           b. Trigram context tuple(tokens[i-3:i])  if i >= 3
           c. Bigram context  (tokens[i-2], tokens[i-1])  if i >= 2
           d. Unigram context tokens[i-1]
           e. None of the above -> rule is skipped (assigned -inf)

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
        skip  = False

        # ── Score first token ─────────────────────────────────────────────
        if tokens[0] not in markov_probabilities.get(START, {}):
            continue
        logp += math.log(markov_probabilities[START][tokens[0]])

        # ── Score subsequent tokens (4-gram -> trigram -> bigram -> unigram)
        for i in range(1, len(tokens)):
            scored = False

            # 4-gram context: (tok[i-4], tok[i-3], tok[i-2], tok[i-1]) -> tok[i]
            if not scored and i >= 4:
                key = (tokens[i - 4], tokens[i - 3], tokens[i - 2], tokens[i - 1])
                if (key in markov_probabilities
                        and tokens[i] in markov_probabilities[key]):
                    logp += math.log(markov_probabilities[key][tokens[i]])
                    scored = True

            # Trigram context: (tok[i-3], tok[i-2], tok[i-1]) -> tok[i]
            if not scored and i >= 3:
                key = (tokens[i - 3], tokens[i - 2], tokens[i - 1])
                if (key in markov_probabilities
                        and tokens[i] in markov_probabilities[key]):
                    logp += math.log(markov_probabilities[key][tokens[i]])
                    scored = True

            # Bigram context: (tok[i-2], tok[i-1]) -> tok[i]
            if not scored and i >= 2:
                key = (tokens[i - 2], tokens[i - 1])
                if (key in markov_probabilities
                        and tokens[i] in markov_probabilities[key]):
                    logp += math.log(markov_probabilities[key][tokens[i]])
                    scored = True

            # Unigram fallback: tok[i-1] -> tok[i]
            if not scored:
                prev = tokens[i - 1]
                if prev in markov_probabilities and tokens[i] in markov_probabilities[prev]:
                    logp += math.log(markov_probabilities[prev][tokens[i]])
                else:
                    skip = True
                    break

        if not skip:
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
    """Generate *target* valid hashcat rules via best-first Priority Queue search.

    v3.5: PQ generation restored with corrected model semantics (KN + +=1).
    Rules are emitted in descending probability order — the first N rules are
    provably the N most likely under the model, unlike the random walk which
    produced an arbitrary ordering.

    Skip-gram backoff: when a context has no observed transition, drops one
    middle position before falling back to shorter context, then KN unigram.
    PQ is capped at target*15 entries to prevent exponential blowup.
    """
    import heapq as _heapq

    if not memory_intensive_operation_warning("Markov rule generation"):
        return []
    if excluded_operators is None:
        excluded_operators = NEVER_PRODUCE_OPS

    print_section(
        f"Generating Token-Level Markov Rules — Best-First PQ"
        f"({min_len}–{max_len} tokens, target: {target:,})"
    )
    print_info(f"Excluding operators: {', '.join(sorted(excluded_operators))}")

    START      = '^'
    KN_UNIGRAM = markov_probabilities.get('__kn_unigram__', {})
    PQ_CAP     = max(target * 15, 50000)

    def _get_transitions(token_seq: List[str]) -> List[Tuple[str, float]]:
        """Transitions for token_seq with skip-gram backoff."""
        def _from_ctx(ctx):
            d = markov_probabilities.get(ctx)
            if not d:
                return []
            return [(t, p) for t, p in d.items() if t[0] not in excluded_operators]

        seq_len = len(token_seq)
        result  = []

        # Standard cascade: 4-gram → trigram → bigram → unigram
        for ctx_len in range(min(seq_len, 4), 0, -1):
            ctx    = tuple(token_seq[-ctx_len:]) if ctx_len > 1 else token_seq[-1]
            result = _from_ctx(ctx)
            if result:
                break

        # Skip-gram: drop one interior position from the best available context
        if not result and seq_len >= 2:
            for skip in range(1, min(seq_len, 4)):
                reduced = token_seq[:-skip-1] + token_seq[-skip:] if skip < seq_len else []
                if not reduced:
                    continue
                ctx    = tuple(reduced[-min(len(reduced), 4):])
                ctx    = ctx if len(ctx) > 1 else ctx[0]
                result = _from_ctx(ctx)
                if result:
                    break

        # Final fallback: KN unigram
        if not result and KN_UNIGRAM:
            result = [(t, p) for t, p in KN_UNIGRAM.items()
                      if t[0] not in excluded_operators]

        return sorted(result, key=lambda x: -x[1])

    # Seed PQ with all starters sorted by probability
    starters = sorted(
        ((t, p) for t, p in markov_probabilities.get(START, {}).items()
         if t[0] not in excluded_operators),
        key=lambda x: -x[1]
    )
    print_info(f"Seeding PQ with {len(starters):,} start tokens.")

    pq:       List = []
    counter:  int  = 0
    generated: Set[str] = set()
    length_counts: Dict[int, int] = {l: 0 for l in range(min_len, max_len + 1)}
    n_lengths = max_len - min_len + 1

    for tok, prob in starters:
        _heapq.heappush(pq, (-prob, counter, [tok], prob))
        counter += 1

    candidates_processed = 0

    while pq and len(generated) < target:
        neg_prob, _, token_seq, prob = _heapq.heappop(pq)
        candidates_processed += 1
        n_tok = len(token_seq)

        if min_len <= n_tok <= max_len:
            rule = ''.join(token_seq)
            if (rule not in generated
                    and not _has_banned_op(rule)
                    and TOKEN_REGEX.findall(rule) == token_seq
                    and is_valid_hashcat_rule(rule)):
                generated.add(rule)
                length_counts[n_tok] += 1
                if len(generated) % 10000 == 0:
                    print_info(f"  {len(generated):,}/{target:,} rules  "
                               f"(queue={len(pq):,}, prob={prob:.6f})")

        if n_tok < max_len and len(pq) < PQ_CAP:
            for nxt, trans_prob in _get_transitions(token_seq):
                new_prob = prob * trans_prob
                _heapq.heappush(pq, (-new_prob, counter, token_seq + [nxt], new_prob))
                counter += 1

    print_success(
        f"PQ generated {len(generated):,} rules "
        f"({candidates_processed:,} candidates, queue remainder {len(pq):,})."
    )
    if n_lengths > 1 and generated:
        dist_parts = [f"len={l}: {length_counts[l]:,}" for l in range(min_len, max_len + 1)]
        print_info("Length distribution: " + "  |  ".join(dist_parts))

    if not generated:
        return []
    dummy = {r: 1 for r in generated}
    return get_markov_weighted_rules(dummy, markov_probabilities, {})[:target]


# ==============================================================================
# COMBINATORIAL GENERATION
# ==============================================================================

# ==============================================================================
# CRACK-FOCUSED SYNTHETIC CORPUS  (v3.5)
# ==============================================================================

def build_crack_synthetic_corpus() -> Dict[str, int]:
    """Return a dict {rule: weight} of high-value synthetic rules.

    These rules represent the most statistically common real-world password
    mutations and are used in two ways:

      1. Pre-seeded into the Markov training corpus so the model learns the
         token transitions they contain (e.g. c → $2 → $0 → $2 → $4).
      2. Directly injected into the combinatorial output so they are
         guaranteed to appear regardless of operator pool selection.

    The synthetic weight is deliberately lower than a high-frequency corpus
    rule so that genuine corpus patterns still dominate the Markov model
    while these act as a reliable enrichment floor.
    """
    WEIGHT_HIGH   = 800   # year patterns, single-digit suffixes — very common
    WEIGHT_MEDIUM = 400   # leet combos, symbol appends
    WEIGHT_LOW    = 150   # longer/rarer synthetic combos

    rules: Dict[str, int] = {}

    def _add(rule: str, weight: int) -> None:
        tokens = TOKEN_REGEX.findall(rule)
        if not tokens or ''.join(tokens) != rule:
            return
        if _has_banned_op(rule):
            return
        rules[rule] = rules.get(rule, 0) + weight

    # ── Year appends: [case_op]$Y$Y$Y$Y ─────────────────────────────────────
    # Years 1970-2029 cover 99 %+ of real "birthyear" passwords.
    years = [str(y) for y in list(range(1970, 2030))]
    for year in years:
        suffix = ''.join(f'${d}' for d in year)
        for case_op in ('l', 'u', 'c', 'C', ':'):
            _add(case_op + suffix if case_op != ':' else suffix, WEIGHT_HIGH)

    # ── Short digit suffix patterns: $N, $NN, $NNN, $NNNN ───────────────────
    for n_digits in range(1, 5):
        for combo in itertools.product('0123456789', repeat=n_digits):
            suffix = ''.join(f'${d}' for d in combo)
            _add(suffix, WEIGHT_HIGH)
            for case_op in ('l', 'u', 'c', 'C'):
                _add(case_op + suffix, WEIGHT_MEDIUM)

    # ── Symbol appends: [case_op]$SYM ────────────────────────────────────────
    for sym in ('!', '@', '#', '.', '?', '_', '-', '*'):
        token = f'${sym}'
        _add(token, WEIGHT_MEDIUM)
        for case_op in ('l', 'u', 'c', 'C'):
            _add(case_op + token, WEIGHT_MEDIUM)

    # ── Leet substitutions (single) ──────────────────────────────────────────
    leet_ops = ['sa@', 'se3', 'si!', 'so0', 'ss5', 'st7', 'sb6']
    for leet in leet_ops:
        _add(leet, WEIGHT_MEDIUM)

    # ── Case + leet: [case_op][leet] ─────────────────────────────────────────
    for case_op in ('l', 'u', 'c', 'C'):
        for leet in leet_ops:
            _add(case_op + leet, WEIGHT_MEDIUM)

    # ── Leet combos (up to 2 substitutions) ─────────────────────────────────
    for l1 in leet_ops:
        for l2 in leet_ops:
            if l2 != l1:
                _add(l1 + l2, WEIGHT_LOW)

    # ── Case + leet + digit suffix ────────────────────────────────────────────
    for case_op in ('c', 'C', 'l'):
        for leet in leet_ops[:4]:                 # top-4 leet ops only
            for digit in ('$1', '$2', '$3', '$!'):
                _add(case_op + leet + digit, WEIGHT_LOW)

    # ── Pure structural transforms ────────────────────────────────────────────
    for op in ('r', 'd', 'f', 'c', 'C', 'l', 'u', 't'):
        _add(op, WEIGHT_HIGH)

    print_info(
        f"[crack-corpus] Built {len(rules):,} synthetic crack-focused rules "
        f"(WEIGHT_HIGH={WEIGHT_HIGH}, MEDIUM={WEIGHT_MEDIUM}, LOW={WEIGHT_LOW})"
    )
    return rules


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
    v3.5: CRACK_FOCUSED_TOKENS are always present in the candidate pool with a
    boosted frequency so they are never crowded out by a sparse corpus.
    """
    # ── Build enriched operator pool ─────────────────────────────────────────
    # Start from corpus operators (banned ops removed)
    corpus_counts: Dict[str, int] = {
        op: cnt for op, cnt in sorted_operators if op not in NEVER_PRODUCE_OPS
    }
    # Inject CRACK_FOCUSED_TOKENS with frequency just above the highest corpus
    # count so they sort to the front of the pool when the corpus is empty, but
    # yield gracefully to genuinely dominant corpus operators.
    boost = (max(corpus_counts.values(), default=0) + 1)
    enriched: Dict[str, int] = dict(corpus_counts)
    for tok in CRACK_FOCUSED_TOKENS:
        if tok not in enriched:
            enriched[tok] = boost   # not in corpus at all → always included
        # tokens already in corpus keep their original (higher or equal) count

    safe_operators = sorted(enriched.items(), key=lambda kv: kv[1], reverse=True)

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

    # ── v3.5: inject synthetic crack-focused rules directly ──────────────────
    # The cartesian product may not reach all high-value patterns (budget limit,
    # length constraints).  Explicitly adding the synthetic corpus guarantees
    # the most effective real-world mutations are always in the output set.
    synthetic = build_crack_synthetic_corpus()
    n_before  = len(generated)
    for rule in synthetic:
        tokens = TOKEN_REGEX.findall(rule)
        if not tokens or ''.join(tokens) != rule:
            continue
        if _has_banned_op(rule):
            continue
        if not is_valid_hashcat_rule(rule):
            continue
        n_toks = len(tokens)
        if min_len <= n_toks <= max_len:
            generated.add(rule)
    n_injected = len(generated) - n_before
    if n_injected:
        print_info(f"[crack-corpus] Injected {n_injected:,} synthetic crack rules into combo output.")

    print_success(f"Generated {len(generated):,} valid rules ({n_injected:,} from crack corpus).")
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
            if not args:
                return word
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
            # xNM — extract M characters starting at position N  (M = count, not end)
            n, m = _i36(args[0]), _i36(args[1])
            if n < 0 or m < 0:
                return word
            return word[n:n + m]
        elif op == 'O':
            # ONM — delete M characters starting at position N
            n, m = _i36(args[0]), _i36(args[1])
            if n < 0 or m < 0 or n >= len(word):
                return word
            return word[:n] + word[n + m:]
        elif op == 'i':
            pos  = min(_i36(args[0]), len(word))
            char = args[1]
            return word[:pos] + char + word[pos:]
        elif op == 'o':
            pos  = _i36(args[0])
            char = args[1]
            return word[:pos] + char + word[pos + 1:] if pos < len(word) else word
        elif op == "'":
            # 'N — keep first N characters
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
            if n <= 0 or not word:
                return word
            return word + word[-n:]
        elif op == 'E':
            # Title-case: lowercase everything, then uppercase after space only.
            # Hashcat's E opcode uses only ASCII space (0x20) as the word separator.
            out = []
            cap = True
            for ch in word:
                if cap and ch.islower():
                    out.append(ch.upper())
                elif not cap and ch.isupper():
                    out.append(ch.lower())
                else:
                    out.append(ch)
                cap = (ch == ' ')
            return ''.join(out)
        elif op == 'e':
            # Title-case with custom separator: lowercase everything, then uppercase after sep.
            sep = args[0]
            out = []
            cap = True
            for ch in word:
                if cap and ch.islower():
                    out.append(ch.upper())
                elif not cap and ch.isupper():
                    out.append(ch.lower())
                else:
                    out.append(ch)
                cap = (ch == sep)
            return ''.join(out)
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
# 3. Unified probe vector (TEST_VECTOR)
#    Now sourced exclusively from minimizer.py's BUILTIN_PROBES so that both
#    tools use an identical word set.  Covers lengths 2–11, mixed-case,
#    embedded-digit, special-char, and repeated-char strings.
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

# Sentinel prefix for rules with unsupported opcodes.
# Each such rule gets a UNIQUE signature ('__UNSUPPORTED__', rule_text) so that
# two different unsupported rules are never mistakenly identified as duplicates.
# The old constant ('__UNSUPPORTED__',) caused all unsupported rules to share
# one bucket — e.g. 200 reject-op rules (<, >, !, /, …) collapsed to 1 kept rule.
# NOTE: _UNSUPPORTED_SIG is kept for reference only; always use _min_compute_signature
# to generate the actual per-rule sentinel ('__UNSUPPORTED__', rule_text).
_UNSUPPORTED_SIG_PREFIX: str = '__UNSUPPORTED__'


def _is_unsupported_sig(sig: tuple) -> bool:
    """Return True if *sig* is an unsupported-opcode sentinel (any variant)."""
    return len(sig) >= 1 and sig[0] == '__UNSUPPORTED__'

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
        if '0' <= c <= '9': return ord(c) - 48
        if 'A' <= c <= 'Z': return ord(c) - 55   # A=10, B=11, …, Z=35
        return -1

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
            # Title-case: lowercase everything, then uppercase after space/hyphen/underscore.
            out = []
            cap = True
            for c in w:
                if cap and 97 <= c <= 122:
                    out.append(c & ~0x20)        # lowercase → uppercase (word start)
                elif not cap and 65 <= c <= 90:
                    out.append(c | 0x20)         # uppercase → lowercase (mid-word)
                else:
                    out.append(c)
                cap = (c == 32)                  # only space triggers capitalisation
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
            # 'N — keep only the first N characters (w[:N])
            p = dg(rule[1])
            if 0 <= p: w = w[:p]
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
            # Title-case with custom separator: lowercase everything, then uppercase after sep.
            sep = _min_arg_ord(rule, 1)
            out = []
            cap = True
            for c in w:
                if cap and 97 <= c <= 122:
                    out.append(c & ~0x20)
                elif not cap and 65 <= c <= 90:
                    out.append(c | 0x20)
                else:
                    out.append(c)
                cap = (c == sep)
            w = out
        elif cmd == 'x' and len(rule) >= 3:
            # xNM — extract M characters starting at position N  (M is a count, not end)
            n, m = dg(rule[1]), dg(rule[2])
            if n >= 0 and m >= 0:
                w = w[n:n + m]
        elif cmd == 'O' and len(rule) >= 3:
            p, m = dg(rule[1]), dg(rule[2])
            if 0 <= p < len(w) and m > 0: w = w[:p] + w[p + m:]
        elif cmd == '*' and len(rule) >= 3:
            a, b = dg(rule[1]), dg(rule[2])
            if 0 <= a < len(w) and 0 <= b < len(w) and a != b:
                w[a], w[b] = w[b], w[a]
        elif cmd == '3' and len(rule) >= 3:
            # 3NX — toggle after the Nth separator X  (N is 0-based: 30X = first sep)
            # Fix: cnt is incremented before compare, so match at cnt == n+1.
            n, sep = dg(rule[1]), _min_arg_ord(rule, 2)
            cnt = 0
            for i, c in enumerate(w):
                if c == sep:
                    cnt += 1
                    if cnt == n + 1 and i + 1 < len(w):
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

    If any opcode in the rule is unsupported, returns a unique sentinel tuple
    that embeds the rule text itself: ('__UNSUPPORTED__', rule).  This ensures
    that two different unsupported rules (e.g. two different reject-op rules)
    are never placed in the same dedup bucket and accidentally collapsed to one.

    The old behaviour — returning the shared constant ('__UNSUPPORTED__',) for
    every unsupported rule — caused false deduplication: 200 rules using reject
    ops (<, >, !, /, …) or memory ops (M, 4, 6, X) would all share one bucket
    and only the first one in file order would be kept.

    Using a tuple (not a joined string) eliminates false collisions from
    output values that contain the separator character.
    """
    outputs = []
    for word in probe_words:
        out = _min_apply_chain(rule, word)
        if out is None:
            return ('__UNSUPPORTED__', rule)   # unique per rule — never false-deduplicates
        outputs.append(out)
    return tuple(outputs)


# ---------------------------------------------------------------------------
# Probe vector (TEST_VECTOR)
# ---------------------------------------------------------------------------
# Sourced exclusively from minimizer.py's BUILTIN_PROBES (v1.4, 50 words).
# Hand-curated to exercise every interesting opcode category:
#
#   len 2–4      → k, K, {, }, [, ] edge cases; x/O/D on short words
#   len 4–6      → T3, i0X, D0, position ops within short words
#   len 7–9      → typical real-world password base word range
#   len 10–11    → truncation and repeat ops ('y','Y','z','Z','p')
#   len 12–36    → high-position ops (B–Z); rules like 'B–'Z, TB–TZ,
#                  DB–DZ, iCX–iZX all get distinct signatures
#   All 95 ASCII → @X / sXY rules are distinguishable from no-ops even
#                  for rare chars like j, x, z and most punctuation
#   Mixed-case   → l, u, c, C, t, E, T, k, K
#   Digits       → @, s, o on numeric chars; pure numeric suffix probing
#   Specials     → @-removal, s-substitution on punctuation chars
#   Repeated     → q (char doubling), z/Z (char extension)

TEST_VECTOR: List[str] = [
    # ── very short — edge cases for k, K, {, }, [, ] ────────────────
    "ab",
    "abc",
    "abcd",
    # ── short alphanumeric (len 4–6) ─────────────────────────────────
    "pass",
    "root",
    "test",
    "admin",
    "login",
    # ── typical password base words (len 7–9) ────────────────────────
    "letmein",          # len 7
    "welcome",          # len 7
    "password",         # len 8  ← THE critical probe word
    "sunshine",         # len 8
    "football",         # len 8
    "baseball",         # len 8
    "princess",         # len 8
    "dragon12",         # len 8, ends with digits
    # ── longer words (len 10–11) — truncation / repeat ops ──────────
    "qwertyuiop",       # len 10
    "iloveyou12",       # len 10, trailing digits
    "monkey12345",      # len 11
    "superman123",      # len 11
    "mustang2024",      # len 11
    # ── extended-length words (len 12–36) — cover positions B(11)–Z(35)
    # Without these, every rule that only touches position 11+ is a no-op
    # on the entire probe set and collapses into the same signature as ":"
    # causing mass false deduplication of 'B–'Z, TB–TZ, DB–DZ, etc.
    "administrator1",                        # len 14
    "iloveyouforever",                       # len 15
    "qwertyuiopasdfgh",                      # len 16
    "correcthorsebattery",                   # len 20
    "averylongpassword1234",                 # len 22
    "averylongpassword12345678",             # len 26
    "averylongpassword1234567890ab",         # len 30
    "averylongpassword1234567890abcdef",     # len 34
    "averylongpassword1234567890abcdefghi",  # len 36 — covers Z(35)
    # ── alphabet coverage — all 95 printable ASCII chars (0x20–0x7E) ─
    # Without full char coverage, rules like @j / @x / @z (purge j/x/z),
    # sja / sxA (replace j/x with something), and rules targeting the 19
    # uppercase letters not present in the basic words (B C D E F G I J K
    # L N O Q R T V X Y Z) are all no-ops on the probe set → falsely merged.
    "abcdefghijklmnopqrstuvwxyz",       # all 26 lowercase
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",       # all 26 uppercase
    "!@#$%^&*()-_=+[]{}|;:,.<>?/~",    # 30 common punctuation chars
    "a`b",    # backtick  (0x60)
    'a"b',    # double-quote (0x22)
    "a'b",    # single-quote (0x27)
    "a\\b",   # backslash (0x5C)
    "a b",    # space (0x20) — completes 95/95 printable ASCII coverage
    # ── mixed-case — l/u/c/C/t/E/T/k/K ─────────────────────────────
    "Password",
    "AdminUser",
    "MySecret",
    "HelloWorld",
    # ── words with embedded digits — s, o, @, T ──────────────────────
    "pass123",
    "admin2024",
    "test1234",
    "user9999",
    # ── words with special chars — @ removal, s substitution ─────────
    "p@ssw0rd",
    "s3cur1ty",
    # ── repeated chars — q (double each), z/Z (extend) ───────────────
    "aaaa",
    "bbbb",
]

# Deduplicate while preserving order (inline — no namespace pollution)
TEST_VECTOR = list(dict.fromkeys(TEST_VECTOR))


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
                sig   TEXT    PRIMARY KEY,
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
            if _is_unsupported_sig(sig):
                unsupported_rules.append((rule_text, count))
                continue
            sig_blob = pickle.dumps(sig, protocol=4)
            sig_key  = hashlib.sha256(sig_blob).hexdigest()   # 64-char TEXT — B-tree indexable
            # INSERT new sig; on collision keep the higher-count rule and sum totals
            cur.execute("""
                INSERT INTO sigs (sig, rule, count) VALUES (?, ?, ?)
                ON CONFLICT(sig) DO UPDATE SET
                    rule  = CASE WHEN excluded.count > sigs.count
                                 THEN excluded.rule ELSE sigs.rule END,
                    count = sigs.count + excluded.count
            """, (sig_key, rule_text, count))
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
    unsupported_group: List[Tuple[str, int]] = []
    supported_map: Dict[tuple, List[Tuple[str, int]]] = {}
    for sig, rule_data in signature_map.items():
        if _is_unsupported_sig(sig):
            unsupported_group.extend(rule_data)
        else:
            supported_map[sig] = rule_data

    final: List[Tuple[str, int]] = list(unsupported_group)
    for group in supported_map.values():
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
    cum_running = 0
    prev_idx    = -1
    for i in range(pts + 1):
        idx = min(i * step, len(data) - 1)
        # Accumulate only the delta since the last checkpoint
        for j in range(prev_idx + 1, idx + 1):
            cum_running += data[j][1]
        prev_idx = idx
        pct  = cum_running / total_value * 100
        bar  = "█" * int(pct / 5)
        y    = 100 - (i * 5)
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

    v3.5: a ':' (no-op / passthrough) rule is automatically inserted as the
    first active rule, immediately after the file header comments.  This
    follows the hashcat convention of including an unmodified-candidate pass
    as the very first rule in every rule set.
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
            # Header comments
            fh.write(f"# CONCENTRATOR v3.5 – {mode_name.upper()} MODE OUTPUT\n")
            fh.write(f"# Generated:   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.write(f"# Total rules: {len(clean_data):,} (+ 1 no-op passthrough prepended)\n")
            fh.write(f"# Format:      {STATE.output_format}\n")
            fh.write(f"#\n")
            # v3.5: ':' passthrough rule as the first active entry
            fh.write(":\n")
            for item in clean_data:
                rule = _extract_rule(item)
                line = expand_rule(rule) if STATE.output_format == 'expanded' else rule
                fh.write(line + '\n')
        print_success(f"Saved {len(clean_data):,} rules (+1 ':' passthrough) → {filename}")
        return True
    except IOError as exc:
        print_error(f"Failed to save: {exc}")
        return False


# ==============================================================================
# HASHCAT RULE CLEANUP — using rulest's validator (replaces old HashcatRuleCleaner)
# ==============================================================================

# The old _OP_VALIDATION_SPEC and HashcatRuleCleaner class are removed.
# Instead, we reuse the is_valid_hashcat_rule() function defined above,
# which exactly mirrors rulest's HashcatRuleValidator.validate_rule_for_gpu.

class HashcatRuleCleaner:
    """
    Validates hashcat rules using the same logic as rulest_v2.py.

    v3.4: replaced the previous ad‑hoc table with a direct call to
    is_valid_hashcat_rule(), which implements the complete rulest validator.
    This ensures that rules produced by rulest are never wrongly discarded.

    The `mode` argument is ignored – the validator is identical for CPU and GPU
    (rulest's validator already enforces GPU compatibility with MAX_GPU_RULES=255).
    """

    MAX_RULES = 255

    def __init__(self, mode: int = 1) -> None:
        if mode not in (1, 2):
            raise ValueError("mode must be 1 (CPU) or 2 (GPU)")
        self.mode = mode   # kept for API compatibility, but not used in validation

    @staticmethod
    def validate_rule(rule_line: str) -> bool:
        """Return True if the rule is syntactically valid according to rulest."""
        # Trim surrounding spaces; rulest's validator does not expect spaces inside.
        rule = rule_line.strip()
        if not rule:
            return False
        # Quick rejection: contains any NEVER_PRODUCE_OPS operator
        if _has_banned_op(rule):
            return False
        # Use the rulest validator
        return is_valid_hashcat_rule(rule)

    def clean_rules(
        self, rules_data: List[Tuple[str, int]]
    ) -> List[Tuple[str, int]]:
        """
        Validate every rule in *rules_data* and return only the passing ones.

        v3.5 fix: is_valid_hashcat_rule now correctly handles all operators,
        including _ (reject-unless-length), { and } (rotate), and ' (truncate),
        which were previously missing and caused valid rules to be rejected.
        Rules that contain banned operators (NEVER_PRODUCE_OPS) are counted
        separately from genuine syntax errors.
        """
        mode_label = 'GPU' if self.mode == 2 else 'CPU'
        print_section(f"Hashcat Rule Validation ({mode_label} mode)")
        print(
            f"Input: {colorize(f'{len(rules_data):,}', Colors.CYAN)} rules  "
            f"│  Mode: {colorize(mode_label, Colors.MAGENTA)}  "
            f"│  Banned-op filter + syntax check"
        )
        valid:        List[Tuple[str, int]] = []
        n_banned:     int = 0
        n_syntax_err: int = 0

        for rule, cnt in tqdm(rules_data, desc="Validating rules"):
            stripped = rule.strip()
            if not stripped:
                n_syntax_err += 1
                continue
            if _has_banned_op(stripped):
                n_banned += 1
                continue
            if is_valid_hashcat_rule(stripped):
                valid.append((rule, cnt))
            else:
                n_syntax_err += 1

        total_removed = n_banned + n_syntax_err
        print(f"\n{Colors.BOLD}Cleanup summary:{Colors.RESET}")
        print(
            f"  {Colors.RED}Banned operators removed : "
            f"{colorize(f'{n_banned:,}', Colors.RED)}{Colors.RESET}"
        )
        print(
            f"  {Colors.YELLOW}Syntax errors removed    : "
            f"{colorize(f'{n_syntax_err:,}', Colors.YELLOW)}{Colors.RESET}"
        )
        print(
            f"  {Colors.GREEN}Total removed            : "
            f"{colorize(f'{total_removed:,}', Colors.RED)}{Colors.RESET}"
        )
        print(
            f"  {Colors.GREEN}Rules retained           : "
            f"{colorize(f'{len(valid):,}', Colors.GREEN)}{Colors.RESET}"
        )
        if rules_data:
            pct_kept = len(valid) / len(rules_data) * 100
            print(
                f"  {Colors.CYAN}Retention rate           : "
                f"{colorize(f'{pct_kept:.1f}%', Colors.CYAN)}{Colors.RESET}"
            )
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
            print(f" {Colors.GREEN}(5){Colors.RESET} HASHCAT CLEANUP – validate rules (CPU/GPU modes)")
            print(f" {Colors.GREEN}(6){Colors.RESET} TOGGLE OUTPUT FORMAT (currently: {STATE.output_format})")
            print(f"\n{Colors.BOLD}ANALYSIS & UTILITIES:{Colors.RESET}")
            print(f" {Colors.BLUE}(p){Colors.RESET} PARETO analysis")
            print(f" {Colors.BLUE}(s){Colors.RESET} SAVE current rules")
            print(f" {Colors.BLUE}(r){Colors.RESET} RESET to original dataset")
            print(f" {Colors.BLUE}(i){Colors.RESET} Dataset information")
            print(f" {Colors.BLUE}(q){Colors.RESET} QUIT")
            print(f"{Colors.BOLD}{'-' * 80}{Colors.RESET}")
            choice = input(f"{Colors.YELLOW}Enter choice: {Colors.RESET}").strip().lower()

            if choice == 'q':
                print_header("THANK YOU FOR USING CONCENTRATOR v3.5!")
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
                print(f"\n{Colors.MAGENTA}[HASHCAT CLEANUP]{Colors.RESET} Choose validation mode:")
                print(f" {Colors.CYAN}(1){Colors.RESET} CPU — transformation rules only")
                print(f"     Memory ops (M 4 6 X) and reject ops (< > ! / ( ) = % Q) are")
                print(f"     always excluded regardless of mode selection.")
                print(f" {Colors.CYAN}(2){Colors.RESET} GPU — same validator, MAX_RULES=255 enforced")
                print(f" {Colors.CYAN}(3){Colors.RESET} Cancel")
                m = input(f"{Colors.YELLOW}Mode (1/2/3): {Colors.RESET}").strip()
                if m in ('1', '2'):
                    mode = int(m)
                    current_data = hashcat_rule_cleanup(current_data, mode)
                else:
                    print_info("Hashcat cleanup cancelled.")
                    continue
            elif choice == '6':
                STATE.output_format = 'expanded' if STATE.output_format == 'line' else 'line'
                print_success(f"Output format → {STATE.output_format}")
                continue
            else:
                print_error("Invalid choice.")
                continue

            if choice in ('1', '2', '3', '4', '5'):
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
        try:
            if input(
                f"{Colors.YELLOW}Save before exiting? (y/N): {Colors.RESET}"
            ).strip().lower() in ('y', 'yes'):
                save_rules(current_data, mode_name=f"{initial_mode}_filtered")
        except EOFError:
            pass
    except EOFError:
        print(f"\n{Colors.YELLOW}Input closed — exiting interactive menu and saving current dataset.{Colors.RESET}")
        save_rules(current_data, mode_name=f"{initial_mode}_filtered")

    return current_data


# ==============================================================================
# MAIN PROCESSING FUNCTIONS
# ==============================================================================

def process_multiple_files_concentrator(args: Any) -> None:
    global STATE
    print_header("PROCESSING MODE – Interactive Rule Minimization")
    all_fps = find_rule_files_recursive(args.paths, max_depth=3)
    if not all_fps:
        print_error("No rule files found.")
        return
    STATE.output_format = args.output_format if args.output_format in ('line', 'expanded') else 'line'
    print(f"{Colors.CYAN}Output Format:{Colors.END} {STATE.output_format}")
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
        # ── v3.5: pre-seed training corpus with synthetic crack rules ────────
        # get_markov_model counts each unique rule ONCE (+=1) regardless of
        # its frequency.  We therefore add synthetic rules as plain presence
        # markers (value=1) — this injects new token transitions without
        # overwhelming corpus-learned patterns the way large weights would.
        # Existing corpus rules are never overwritten.
        synthetic_corpus = build_crack_synthetic_corpus()
        enriched_rule_counts: Dict[str, int] = dict(full_rule_counts)
        n_new = 0
        for rule in synthetic_corpus:
            if rule not in enriched_rule_counts:
                enriched_rule_counts[rule] = 1   # presence only — equal weight to corpus rules
                n_new += 1
        print_info(
            f"[crack-corpus] Pre-seeded Markov training with {n_new:,} new synthetic rules "
            f"({len(synthetic_corpus) - n_new:,} already in corpus, skipped)."
        )
        markov_probs, markov_totals = get_markov_model(enriched_rule_counts)
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
    try:
        enter_interactive = input(
            f"\n{Colors.YELLOW}Enter enhanced interactive mode? (Y/n): {Colors.RESET}"
        ).strip().lower()
    except EOFError:
        enter_interactive = 'n'

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
    print_header("CONCENTRATOR v3.5 – INTERACTIVE MODE")
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
        choice = input(f"{Colors.YELLOW}Select mode (1-3): {Colors.RESET}").strip()
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
        fc = input(f"{Colors.YELLOW}Select (1-2): {Colors.RESET}").strip()
        if fc == '1':
            settings['output_format'] = 'line';     break
        elif fc == '2':
            settings['output_format'] = 'expanded'; break
        else:
            print_error("Enter 1 or 2.")

    if not settings['in_memory']:
        td = input(f"{Colors.YELLOW}Temp directory [system default]: {Colors.RESET}").strip()
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
         [("-d",    "Use disk for large datasets")]),
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

    print(f"\n{C.BOLD}{C.CYAN}NOTES (v3.5):{C.END}")
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
