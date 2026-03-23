#!/usr/bin/env python3
"""
CONCENTRATOR v3.0 - Unified Hashcat Rule Processor
Enhanced with comprehensive rule filtering options and multiple output formats.
Supports expanded format: each rule on its own line with operators and arguments
separated by spaces.

Hashcat compatibility: 6.x and newer.
Since hashcat 6.0, memory operators (M, 4, 6, X, Q) and reject operators
(<, >, !, /, (, ), =, %, Q) are fully supported in both CPU and GPU/OpenCL
execution paths. The former GPU denial for these rule families has been removed.
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
from collections import defaultdict
from typing import List, Tuple, Dict, Callable, Any, Set, Optional

# ==============================================================================
# THIRD-PARTY IMPORTS WITH GRACEFUL FALLBACKS
# ==============================================================================

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

    class tqdm:
        """Minimal tqdm replacement used when the real package is unavailable."""

        def __init__(
            self,
            iterable=None,
            total: Optional[int] = None,
            desc: Optional[str] = None,
            unit: Optional[str] = None,
        ) -> None:
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
# GLOBAL VARIABLES AND CONSTANTS
# ==============================================================================

_temp_files_to_cleanup: List[str] = []
_TEMP_DIR_PATH: Optional[str] = None
_IN_MEMORY_MODE: bool = False
_OPENCL_CONTEXT = None
_OPENCL_QUEUE = None
_OPENCL_PROGRAM = None
_cleanup_lock = threading.Lock()
_cleanup_in_progress: bool = False
_OUTPUT_FORMAT: str = "line"   # "line" or "expanded"
_GPU_MODE_ENABLED: bool = False


class Colors:
    RED       = "\033[91m"
    GREEN     = "\033[92m"
    YELLOW    = "\033[93m"
    BLUE      = "\033[94m"
    MAGENTA   = "\033[95m"
    CYAN      = "\033[96m"
    WHITE     = "\033[97m"
    BOLD      = "\033[1m"
    UNDERLINE = "\033[4m"
    END       = "\033[0m"
    RESET     = "\033[0m"
    BG_RED     = "\033[41m"
    BG_GREEN   = "\033[42m"
    BG_YELLOW  = "\033[43m"
    BG_BLUE    = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN    = "\033[46m"

# ==============================================================================
# OPERATOR DEFINITIONS – COMPREHENSIVE HASHCAT RULE SYNTAX
# ==============================================================================

BASE36_CHARS: Set[str] = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
PRINTABLE_ASCII: Set[str] = set(chr(i) for i in range(32, 127))

# Each key maps to a list describing its argument types.
# 'num'  = one base-36 digit [0-9A-Z]
# 'char' = one printable ASCII character
OPERATOR_ARGS: Dict[str, List[str]] = {
    # No arguments
    ":": [], "l": [], "u": [], "c": [], "C": [], "t": [], "r": [], "d": [],
    "f": [], "{": [], "}": [], "[": [], "]": [], "q": [], "M": [], "4": [],
    "6": [], "k": [], "K": [], "Q": [], "E": [],

    # One base-36 number
    "T": ["num"], "p": ["num"], "D": ["num"], "z": ["num"], "Z": ["num"],
    "'": ["num"], "+": ["num"], "-": ["num"], ".": ["num"], ",": ["num"],
    "L": ["num"], "R": ["num"], "y": ["num"], "Y": ["num"],
    "<": ["num"], ">": ["num"], "_": ["num"],

    # Two base-36 numbers
    "x": ["num", "num"], "O": ["num", "num"], "*": ["num", "num"],

    # One literal character (any printable ASCII)
    "$": ["char"], "^": ["char"], "@": ["char"], "!": ["char"],
    "/": ["char"], "(": ["char"], ")": ["char"],

    # Two literal characters
    "s": ["char", "char"],

    # One base-36 number + one literal character
    "i": ["num", "char"], "o": ["num", "char"],
    "=": ["num", "char"], "%": ["num", "char"], "3": ["num", "char"],

    # Three base-36 numbers
    "X": ["num", "num", "num"],

    # One literal character
    "e": ["char"],
}

ALL_OPERATORS: List[str] = list(OPERATOR_ARGS.keys())


def build_token_regex() -> re.Pattern:
    """Build a compiled regex that matches a single complete hashcat rule token."""
    patterns: List[str] = []
    for op, args in OPERATOR_ARGS.items():
        escaped_op = re.escape(op)
        if not args:
            patterns.append(escaped_op)
        else:
            arg_pattern = ""
            for arg_type in args:
                if arg_type == "num":
                    arg_pattern += "[0-9A-Z]"
                else:
                    arg_pattern += "[ -~]"   # printable ASCII range
            patterns.append(escaped_op + arg_pattern)
    # Longer patterns first to avoid partial matches
    patterns.sort(key=len, reverse=True)
    return re.compile("|".join(patterns))


__ruleregex__: re.Pattern = build_token_regex()


def build_count_regex() -> re.Pattern:
    """Build a compiled regex that matches any operator character (for counting)."""
    patterns = [re.escape(op) for op in ALL_OPERATORS]
    patterns.sort(key=len, reverse=True)
    return re.compile("|".join(patterns))


COMPILED_REGEX: re.Pattern = build_count_regex()

# Mapping: operator -> number of arguments it expects
OPERATORS_REQUIRING_ARGS: Dict[str, int] = {
    op: len(args) for op, args in OPERATOR_ARGS.items() if args
}

ALL_RULE_CHARS: Set[str] = set(
    "0123456789abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ":,.lu.#()=%!?|~+*-^$sStTiIoOcCrRyYzZeEfFxXdDpPbBqQ`[]><@&vV"
)

# ==============================================================================
# UTILITY / PRINT HELPERS
# ==============================================================================

def print_banner() -> None:
    print(f"\n{Colors.CYAN}{Colors.BOLD}" + "=" * 80)
    print("          CONCENTRATOR v3.0 - Unified Hashcat Rule Processor")
    print("=" * 80 + f"{Colors.END}")
    print(f"{Colors.YELLOW}Combined Features:{Colors.END}")
    print(f"  {Colors.GREEN}•{Colors.END} OpenCL GPU Acceleration for validation and generation")
    print(f"  {Colors.GREEN}•{Colors.END} Three Processing Modes: Extraction, Combinatorial, Markov")
    print(f"  {Colors.GREEN}•{Colors.END} Hashcat Rule Engine Simulation & Functional Minimization")
    print(f"  {Colors.GREEN}•{Colors.END} Rule Validation and Cleanup (hashcat 6.x compatible)")
    print(f"  {Colors.GREEN}•{Colors.END} Memory & Reject rules supported on GPU (hashcat 6.x+)")
    print(f"  {Colors.GREEN}•{Colors.END} Levenshtein Distance Filtering")
    print(f"  {Colors.GREEN}•{Colors.END} Smart Processing Selection & Memory Safety")
    print(f"  {Colors.GREEN}•{Colors.END} Interactive & CLI Modes with Colorized Output")
    print(f"  {Colors.GREEN}•{Colors.END} Multiple output formats: line, expanded")
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
    return response in ("y", "yes")

# ==============================================================================
# MEMORY MANAGEMENT
# ==============================================================================

def signal_handler(sig, frame) -> None:
    global _cleanup_in_progress
    with _cleanup_lock:
        if _cleanup_in_progress:
            return
        _cleanup_in_progress = True
    if multiprocessing.current_process().name != "MainProcess":
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


def get_memory_usage() -> Optional[Dict[str, Any]]:
    if not PSUTIL_AVAILABLE:
        return None
    try:
        virtual_mem = psutil.virtual_memory()
        swap_mem    = psutil.swap_memory()
        return {
            "ram_used":      virtual_mem.used,
            "ram_total":     virtual_mem.total,
            "ram_percent":   virtual_mem.percent,
            "swap_used":     swap_mem.used,
            "swap_total":    swap_mem.total,
            "swap_percent":  swap_mem.percent,
            "total_used":    virtual_mem.used + swap_mem.used,
            "total_available": virtual_mem.total + swap_mem.total,
            "total_percent": (
                (virtual_mem.used + swap_mem.used)
                / (virtual_mem.total + swap_mem.total)
                * 100
            ),
        }
    except Exception as e:
        print_error(f"Could not monitor memory usage: {e}")
        return None


def format_bytes(bytes_size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def check_memory_safety(threshold_percent: float = 85.0) -> bool:
    mem_info = get_memory_usage()
    if not mem_info:
        return True
    total_percent = mem_info["total_percent"]
    if total_percent >= threshold_percent:
        print_warning(
            f"System memory usage at {total_percent:.1f}% "
            f"(threshold: {threshold_percent}%)"
        )
        print(
            f"   {Colors.CYAN}RAM:{Colors.RESET} "
            f"{format_bytes(mem_info['ram_used'])} / "
            f"{format_bytes(mem_info['ram_total'])} "
            f"({mem_info['ram_percent']:.1f}%)"
        )
        print(
            f"   {Colors.CYAN}Swap:{Colors.RESET} "
            f"{format_bytes(mem_info['swap_used'])} / "
            f"{format_bytes(mem_info['swap_total'])} "
            f"({mem_info['swap_percent']:.1f}%)"
        )
        return False
    return True


def memory_safe_operation(operation_name: str, threshold_percent: float = 85.0):
    """Decorator: checks available memory before running a memory-intensive function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print_section(f"Memory Check before {operation_name}")
            if not check_memory_safety(threshold_percent):
                print_error(f"{operation_name} requires significant memory.")
                print(
                    f"   Current memory usage exceeds "
                    f"{threshold_percent}% threshold."
                )
                response = (
                    input(
                        f"{Colors.YELLOW}Continue with "
                        f"{operation_name} anyway? (y/N): {Colors.RESET}"
                    )
                    .strip()
                    .lower()
                )
                if response not in ("y", "yes"):
                    print_error(f"{operation_name} cancelled due to memory constraints.")
                    return None
            print_success(f"Starting {operation_name}...")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def estimate_memory_usage(rules_count: int, avg_rule_length: int = 50) -> int:
    """Rough byte estimate for holding *rules_count* rules in memory."""
    return rules_count * (avg_rule_length + 50)


def print_memory_status() -> None:
    mem_info = get_memory_usage()
    if not mem_info:
        return
    ram_color = Colors.GREEN
    if mem_info["ram_percent"] > 85:
        ram_color = Colors.RED
    elif mem_info["ram_percent"] > 70:
        ram_color = Colors.YELLOW
    print(
        f"{Colors.CYAN}Memory Status:{Colors.END} "
        f"{ram_color}RAM {mem_info['ram_percent']:.1f}% "
        f"({format_bytes(mem_info['ram_used'])}/"
        f"{format_bytes(mem_info['ram_total'])}){Colors.END}",
        end="",
    )
    if mem_info["swap_total"] > 0:
        if mem_info["swap_used"] > 0:
            swap_color = (
                Colors.YELLOW if mem_info["swap_percent"] < 50 else Colors.RED
            )
            print(
                f" | {Colors.CYAN}SWAP:{Colors.END} "
                f"{swap_color}{mem_info['swap_percent']:.1f}% "
                f"({format_bytes(mem_info['swap_used'])}/"
                f"{format_bytes(mem_info['swap_total'])}){Colors.END}"
            )
        else:
            print(
                f" | {Colors.CYAN}Swap:{Colors.END} "
                f"{Colors.GREEN}available "
                f"({format_bytes(mem_info['swap_total'])}){Colors.END}"
            )
    else:
        print(f" | {Colors.CYAN}Swap:{Colors.END} {Colors.RED}not available{Colors.END}")


def memory_intensive_operation_warning(operation_name: str) -> bool:
    mem_info = get_memory_usage()
    if not mem_info:
        return True
    if mem_info["ram_percent"] > 85:
        print(
            f"{Colors.RED}{Colors.BOLD}WARNING:{Colors.END} "
            f"{Colors.YELLOW}High RAM usage detected "
            f"({mem_info['ram_percent']:.1f}%) for {operation_name}{Colors.END}"
        )
        print_memory_status()
        if mem_info["swap_total"] == 0:
            print(
                f"{Colors.RED}CRITICAL: No swap space available. "
                f"System may become unstable.{Colors.END}"
            )
            response = (
                input(
                    f"{Colors.YELLOW}Continue with memory-intensive "
                    f"operation? (y/N): {Colors.END}"
                )
                .strip()
                .lower()
            )
            return response in ("y", "yes")
        else:
            print(
                f"{Colors.YELLOW}System will use swap space. "
                f"Performance may be slower.{Colors.END}"
            )
            response = (
                input(
                    f"{Colors.YELLOW}Continue with memory-intensive "
                    f"operation? (Y/n): {Colors.END}"
                )
                .strip()
                .lower()
            )
            return response not in ("n", "no")
    return True

# ==============================================================================
# FILE MANAGEMENT
# ==============================================================================

def find_rule_files_recursive(
    paths: List[str], max_depth: int = 3
) -> List[str]:
    """Collect rule files from a list of file/directory paths (recursive, max_depth)."""
    all_filepaths: List[str] = []
    rule_extensions = {".rule", ".rules", ".hr", ".hashcat", ".txt", ".lst"}
    for path in paths:
        if os.path.isfile(path):
            if os.path.splitext(path.lower())[1] in rule_extensions:
                all_filepaths.append(path)
                print_success(f"Rule file: {path}")
            else:
                print_warning(f"Not a rule file (wrong extension): {path}")
        elif os.path.isdir(path):
            print_info(f"Scanning directory: {path} (max depth: {max_depth})")
            found = 0
            for root, dirs, files in os.walk(path):
                depth = root[len(path):].count(os.sep)
                if depth >= max_depth:
                    dirs.clear()
                    continue
                for file in files:
                    if os.path.splitext(file.lower())[1] in rule_extensions:
                        full = os.path.join(root, file)
                        all_filepaths.append(full)
                        found += 1
                        depth_label = (
                            f" (depth {depth})" if depth else ""
                        )
                        print_success(f"Rule file{depth_label}: {full}")
            if not found:
                print_warning(f"No rule files found in: {path}")
            else:
                print_success(f"Found {found} rule files in: {path}")
        else:
            print_error(f"Path not found: {path}")
    return sorted(list(set(all_filepaths)))


def set_global_flags(
    temp_dir_path: Optional[str], in_memory_mode: bool
) -> None:
    global _TEMP_DIR_PATH, _IN_MEMORY_MODE
    _IN_MEMORY_MODE = in_memory_mode
    if temp_dir_path and not in_memory_mode:
        _TEMP_DIR_PATH = temp_dir_path
        try:
            os.makedirs(_TEMP_DIR_PATH, exist_ok=True)
            print_info(f"Using temporary directory: {_TEMP_DIR_PATH}")
        except OSError as e:
            print_warning(
                f"Could not create temp dir {temp_dir_path}. "
                f"Using system temp. Error: {e}"
            )
            _TEMP_DIR_PATH = None
    elif in_memory_mode:
        print_info("In-Memory Mode activated. Temporary files will be skipped.")


def cleanup_temp_files() -> None:
    if not _temp_files_to_cleanup:
        return
    print_info("Cleaning up temporary files...")
    for temp_file in _temp_files_to_cleanup[:]:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print_info(f"Cleaned up: {temp_file}")
            _temp_files_to_cleanup.remove(temp_file)
        except Exception:
            pass

# ==============================================================================
# HASHCAT RULE ENGINE SIMULATION
# ==============================================================================

def i36(string: str) -> int:
    """Convert a base-36 character to an integer."""
    return int(string, 36)


# Rule-function dispatch table.  Each function signature is (word, args) -> str.
# For reject operators the function raises RuleRejected to signal that the
# word should be skipped (no transformation applied).
class RuleRejected(Exception):
    """Raised by reject-family operators to signal word rejection."""
    pass


FUNCTS: Dict[str, Callable] = {}

FUNCTS[":"] = lambda x, i: x
FUNCTS["l"] = lambda x, i: x.lower()
FUNCTS["u"] = lambda x, i: x.upper()
FUNCTS["c"] = lambda x, i: x.capitalize()
FUNCTS["C"] = lambda x, i: x.capitalize().swapcase()
FUNCTS["t"] = lambda x, i: x.swapcase()


def _toggle_at(x: str, i: str) -> str:
    number = i36(i)
    if number >= len(x):
        return x
    return x[:number] + x[number].swapcase() + x[number + 1:]


FUNCTS["T"] = _toggle_at
FUNCTS["r"] = lambda x, i: x[::-1]
FUNCTS["d"] = lambda x, i: x + x
FUNCTS["p"] = lambda x, i: x * (i36(i) + 1)
FUNCTS["f"] = lambda x, i: x + x[::-1]
FUNCTS["{"] = lambda x, i: x[1:] + x[0] if x else x
FUNCTS["}"] = lambda x, i: x[-1] + x[:-1] if x else x
FUNCTS["$"] = lambda x, i: x + i
FUNCTS["^"] = lambda x, i: i + x
FUNCTS["["] = lambda x, i: x[1:]
FUNCTS["]"] = lambda x, i: x[:-1]


def _delete_at(x: str, i: str) -> str:
    idx = i36(i)
    if idx >= len(x):
        return x
    return x[:idx] + x[idx + 1:]


FUNCTS["D"] = _delete_at


def _extract(x: str, i: str) -> str:
    start = i36(i[0])
    end   = i36(i[1])
    if start < 0 or end < 0 or start > len(x) or end > len(x) or start > end:
        return ""
    return x[start:end]


FUNCTS["x"] = _extract


def _omit(x: str, i: str) -> str:
    start = i36(i[0])
    end   = i36(i[1])
    if start < 0 or end < 0 or start > len(x) or end > len(x):
        return x
    if start > end:
        return x
    return x[:start] + x[end + 1:]


FUNCTS["O"] = _omit


def _insert(x: str, i: str) -> str:
    pos  = i36(i[0])
    char = i[1]
    if pos > len(x):
        pos = len(x)
    return x[:pos] + char + x[pos:]


FUNCTS["i"] = _insert


def _overstrike(x: str, i: str) -> str:
    pos  = i36(i[0])
    char = i[1]
    if pos >= len(x):
        return x
    return x[:pos] + char + x[pos + 1:]


FUNCTS["o"] = _overstrike

FUNCTS["'"] = lambda x, i: x[: i36(i)]
FUNCTS["s"] = lambda x, i: x.replace(i[0], i[1])
FUNCTS["@"] = lambda x, i: x.replace(i, "")


def _dupe_first(x: str, i: str) -> str:
    num = i36(i)
    if x:
        return x[0] * num + x
    return ""


FUNCTS["z"] = _dupe_first


def _dupe_last(x: str, i: str) -> str:
    num = i36(i)
    if x:
        return x + x[-1] * num
    return ""


FUNCTS["Z"] = _dupe_last
FUNCTS["q"] = lambda x, i: "".join(a * 2 for a in x)

# Memory operators (fully supported since hashcat 6.x on both CPU and GPU)
__memorized__: List[str] = [""]


def _extract_memory(string: str, args: str) -> str:
    if not __memorized__[0]:
        return string
    try:
        pos, length, ins = map(i36, args)
        string_list  = list(string)
        mem_segment  = __memorized__[0][pos: pos + length]
        string_list.insert(ins, mem_segment)
        return "".join(string_list)
    except Exception:
        return string


FUNCTS["X"] = _extract_memory
FUNCTS["4"] = lambda x, i: x + __memorized__[0]
FUNCTS["6"] = lambda x, i: __memorized__[0] + x


def _memorize(string: str, _: str) -> str:
    __memorized__[0] = string
    return string


FUNCTS["M"] = _memorize

# Reject operators (fully supported since hashcat 6.x on both CPU and GPU).
# Each raises RuleRejected when the condition is satisfied so that
# RuleEngine.apply() can propagate the rejection to the caller.


def _reject_less(x: str, i: str) -> str:
    """Reject if word length < N."""
    if len(x) < i36(i):
        raise RuleRejected
    return x


def _reject_greater(x: str, i: str) -> str:
    """Reject if word length > N."""
    if len(x) > i36(i):
        raise RuleRejected
    return x


def _reject_contain(x: str, i: str) -> str:
    """Reject if word contains character i."""
    if i in x:
        raise RuleRejected
    return x


def _reject_not_contain(x: str, i: str) -> str:
    """Reject if word does NOT contain character i."""
    if i not in x:
        raise RuleRejected
    return x


def _reject_equal_first(x: str, i: str) -> str:
    """Reject if first character != i."""
    if not x or x[0] != i:
        raise RuleRejected
    return x


def _reject_equal_last(x: str, i: str) -> str:
    """Reject if last character != i."""
    if not x or x[-1] != i:
        raise RuleRejected
    return x


def _reject_equal_at(x: str, i: str) -> str:
    """Reject if character at position i[0] != i[1]."""
    pos  = i36(i[0])
    char = i[1]
    if pos >= len(x) or x[pos] != char:
        raise RuleRejected
    return x


def _reject_contains_n(x: str, i: str) -> str:
    """Reject if word contains character i[1] fewer than i[0] times."""
    n    = i36(i[0])
    char = i[1]
    if x.count(char) < n:
        raise RuleRejected
    return x


def _reject_memory(x: str, _: str) -> str:
    """Reject if the current word differs from the memorized word."""
    if x != __memorized__[0]:
        raise RuleRejected
    return x


FUNCTS["<"] = _reject_less
FUNCTS[">"] = _reject_greater
FUNCTS["!"] = _reject_contain
FUNCTS["/"] = _reject_not_contain
FUNCTS["("] = _reject_equal_first
FUNCTS[")"] = _reject_equal_last
FUNCTS["="] = _reject_equal_at
FUNCTS["%"] = _reject_contains_n
FUNCTS["Q"] = _reject_memory


class RuleEngine:
    """Simulate hashcat rule application on a candidate word."""

    def __init__(self, rules: List[str]) -> None:
        self.rules: Tuple = tuple(map(__ruleregex__.findall, rules))

    def apply(self, string: str) -> Optional[str]:
        """
        Apply the first rule in self.rules to *string*.

        Returns the transformed word, or None if a reject operator fired.
        """
        for rule_functions in self.rules:
            word = string
            __memorized__[0] = ""
            try:
                for function in rule_functions:
                    word = FUNCTS[function[0]](word, function[1:])
            except RuleRejected:
                return None
            except Exception:
                pass
            return word
        return string

# ==============================================================================
# RULE VALIDATION
# ==============================================================================

def is_valid_hashcat_rule(rule: str) -> bool:
    """
    Validate *rule* against the hashcat rule syntax.

    Returns True if every token in the rule is syntactically correct.
    Supports all operators including memory and reject families (hashcat 6.x+).
    """
    tokens = __ruleregex__.findall(rule)
    if "".join(tokens) != rule:
        return False
    for token in tokens:
        op = token[0]
        if op not in OPERATOR_ARGS:
            return False
        args_spec    = OPERATOR_ARGS[op]
        expected_len = 1 + len(args_spec)
        if len(token) != expected_len:
            return False
        for idx, arg_type in enumerate(args_spec):
            arg_char = token[1 + idx]
            if arg_type == "num" and arg_char not in BASE36_CHARS:
                return False
    return True

# ==============================================================================
# OPENCL SETUP
# ==============================================================================

OPENCL_VALIDATION_KERNEL = """
__kernel void validate_rules_batch(
    __global const uchar* rules,
    __global uchar* results,
    const uint rule_stride,
    const uint max_rule_len,
    const uint num_rules)
{
    uint rule_idx = get_global_id(0);
    if (rule_idx >= num_rules) return;
    __global const uchar* rule = rules + rule_idx * rule_stride;
    bool valid = true;
    for (uint i = 0; i < max_rule_len && rule[i] != 0; i++) {
        uchar c = rule[i];
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
              c == '<' || c == '@' || c == '&' || c == 'v' || c == 'V' ||
              c == 'M' || c == '4' || c == '6')) {
            valid = false;
            break;
        }
    }
    results[rule_idx] = valid ? 1 : 0;
}
"""


def setup_opencl() -> bool:
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
        _OPENCL_QUEUE   = cl.CommandQueue(_OPENCL_CONTEXT)
        _OPENCL_PROGRAM = cl.Program(_OPENCL_CONTEXT, OPENCL_VALIDATION_KERNEL).build()
        print_success(f"OpenCL initialized on: {devices[0].name}")
        return True
    except Exception as e:
        print_error(f"OpenCL initialization failed: {e}")
        return False


def gpu_validate_rules(
    rules_list: List[str], max_rule_length: int = 64
) -> List[bool]:
    if not _OPENCL_CONTEXT or not rules_list:
        return [False] * len(rules_list)
    try:
        num_rules   = len(rules_list)
        rule_stride = ((max_rule_length + 15) // 16) * 16
        rules_buffer = np.zeros((num_rules, rule_stride), dtype=np.uint8)
        for idx, rule in enumerate(rules_list):
            rule_bytes = rule.encode("ascii", "ignore")
            length     = min(len(rule_bytes), rule_stride)
            rules_buffer[idx, :length] = np.frombuffer(
                rule_bytes[:length], dtype=np.uint8
            )
        results = np.zeros(num_rules, dtype=np.uint8)
        mf           = cl.mem_flags
        rules_gpu    = cl.Buffer(
            _OPENCL_CONTEXT, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rules_buffer
        )
        results_gpu  = cl.Buffer(
            _OPENCL_CONTEXT, mf.WRITE_ONLY, results.nbytes
        )
        _OPENCL_PROGRAM.validate_rules_batch(
            _OPENCL_QUEUE,
            (num_rules,),
            None,
            rules_gpu,
            results_gpu,
            np.uint32(rule_stride),
            np.uint32(max_rule_length),
            np.uint32(num_rules),
        )
        cl.enqueue_copy(_OPENCL_QUEUE, results, results_gpu)
        _OPENCL_QUEUE.finish()
        return [bool(r) for r in results]
    except Exception as e:
        print_error(f"GPU validation failed: {e}, falling back to CPU")
        return [is_valid_hashcat_rule(rule) for rule in rules_list]

# ==============================================================================
# PARALLEL FILE PROCESSING
# ==============================================================================

def process_single_file(
    filepath: str, max_rule_length: int
) -> Tuple[Dict, Dict, List[str], Optional[str]]:
    """
    Read one rule file, clean lines, count operators and full rules.

    Returns (operator_counts, full_rule_counts, clean_rules_list, temp_filepath).
    When running in disk mode the third element is empty and temp_filepath is set.
    """
    operator_counts: Dict[str, int]   = defaultdict(int)
    full_rule_counts: Dict[str, int]  = defaultdict(int)
    clean_rules_list: List[str]       = []
    temp_rule_filepath: Optional[str] = None
    global _IN_MEMORY_MODE, _TEMP_DIR_PATH, _temp_files_to_cleanup
    try:
        with open(filepath, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or len(line) > max_rule_length:
                    continue
                clean_line = "".join(c for c in line if c in ALL_RULE_CHARS)
                if not clean_line:
                    continue
                full_rule_counts[clean_line] += 1
                clean_rules_list.append(clean_line)
                for op_match in COMPILED_REGEX.finditer(clean_line):
                    operator_counts[op_match.group(0)] += 1
        if not _IN_MEMORY_MODE:
            temp_rule_file = tempfile.NamedTemporaryFile(
                mode="w+",
                delete=False,
                encoding="utf-8",
                dir=_TEMP_DIR_PATH,
                prefix="concentrator_",
                suffix=".tmp",
            )
            temp_rule_filepath = temp_rule_file.name
            for rule in clean_rules_list:
                temp_rule_file.write(rule + "\n")
            temp_rule_file.close()
            with _cleanup_lock:
                _temp_files_to_cleanup.append(temp_rule_filepath)
            print_success(
                f"File analysis complete: {filepath}. "
                f"Temp rules saved to {temp_rule_filepath}"
            )
            return operator_counts, full_rule_counts, [], temp_rule_filepath
        else:
            print_success(
                f"File analysis complete: {filepath}. Rules returned in memory."
            )
            return operator_counts, full_rule_counts, clean_rules_list, None
    except Exception as e:
        print_error(f"An error occurred while processing {filepath}: {e}")
        if temp_rule_filepath and os.path.exists(temp_rule_filepath):
            try:
                os.remove(temp_rule_filepath)
                with _cleanup_lock:
                    if temp_rule_filepath in _temp_files_to_cleanup:
                        _temp_files_to_cleanup.remove(temp_rule_filepath)
            except Exception:
                pass
        return defaultdict(int), defaultdict(int), [], None


def analyze_rule_files_parallel(
    filepaths: List[str], max_rule_length: int
) -> Tuple[List[Tuple[str, int]], Dict[str, int], List[str]]:
    total_op_counts: Dict[str, int]    = defaultdict(int)
    total_rule_counts: Dict[str, int]  = defaultdict(int)
    temp_files: List[str]              = []
    all_rules: List[str]               = []
    global _IN_MEMORY_MODE, _cleanup_lock
    existing = [fp for fp in filepaths if os.path.exists(fp) and os.path.isfile(fp)]
    if not existing:
        print_warning("No valid rule files found to process.")
        return [], defaultdict(int), []
    num_procs = min(os.cpu_count() or 1, len(existing))
    tasks     = [(fp, max_rule_length) for fp in existing]
    print_info(
        f"Starting parallel analysis of {len(existing)} files "
        f"using {num_procs} processes..."
    )
    with multiprocessing.Pool(processes=num_procs) as pool:
        results = pool.starmap(process_single_file, tasks)
    for opc, rulec, rules, tmp in results:
        for op, cnt in opc.items():
            total_op_counts[op] += cnt
        for rule, cnt in rulec.items():
            total_rule_counts[rule] += cnt
        if _IN_MEMORY_MODE:
            all_rules.extend(rules)
        else:
            if tmp:
                temp_files.append(tmp)
    if not _IN_MEMORY_MODE:
        print_info("Merging temporary rule files...")
        for tmp in temp_files:
            try:
                if os.path.exists(tmp):
                    with open(tmp, "r", encoding="utf-8") as f:
                        all_rules.extend(line.strip() for line in f)
                    os.remove(tmp)
                    with _cleanup_lock:
                        if tmp in _temp_files_to_cleanup:
                            _temp_files_to_cleanup.remove(tmp)
            except Exception as e:
                print_error(f"Error merging temp file {tmp}: {e}")
    print_success(f"Total unique rules loaded: {len(total_rule_counts)}")
    sorted_op_counts = sorted(
        total_op_counts.items(), key=lambda x: x[1], reverse=True
    )
    return sorted_op_counts, total_rule_counts, all_rules

# ==============================================================================
# MARKOV MODEL
# ==============================================================================

def get_markov_model(
    unique_rules: Dict[str, int]
) -> Tuple[Optional[Dict], Optional[Dict]]:
    if not memory_intensive_operation_warning("Markov model building"):
        return None, None
    print_section("Building Markov Sequence Probability Model")
    markov_model_counts: Dict = defaultdict(lambda: defaultdict(int))
    START = "^"
    for rule in unique_rules.keys():
        markov_model_counts[START][rule[0]] += 1
        for idx in range(len(rule) - 1):
            markov_model_counts[rule[idx]][rule[idx + 1]] += 1
        for idx in range(len(rule) - 2):
            prefix = rule[idx: idx + 2]
            suffix = rule[idx + 2]
            markov_model_counts[prefix][suffix] += 1
    total_transitions = {
        char: sum(counts.values())
        for char, counts in markov_model_counts.items()
    }
    markov_probabilities: Dict = defaultdict(lambda: defaultdict(float))
    for prefix, next_counts in markov_model_counts.items():
        total = total_transitions[prefix]
        for next_op, count in next_counts.items():
            markov_probabilities[prefix][next_op] = count / total
    return markov_probabilities, total_transitions


def get_markov_weighted_rules(
    unique_rules: Dict[str, int],
    markov_probabilities: Dict,
    total_transitions: Dict,
) -> List[Tuple[str, float]]:
    if not memory_intensive_operation_warning("Markov weighting"):
        return []
    weighted: List[Tuple[str, float]] = []
    for rule in unique_rules.keys():
        logp    = 0.0
        current = "^"
        next_char = rule[0]
        if next_char in markov_probabilities[current]:
            logp += math.log(markov_probabilities[current][next_char])
        else:
            continue
        for idx in range(len(rule) - 1):
            if idx >= 1:
                prefix    = rule[idx - 1: idx + 1]
                next_char = rule[idx + 1]
                if (
                    prefix in markov_probabilities
                    and next_char in markov_probabilities[prefix]
                ):
                    logp += math.log(markov_probabilities[prefix][next_char])
                    continue
            prefix    = rule[idx]
            next_char = rule[idx + 1]
            if (
                prefix in markov_probabilities
                and next_char in markov_probabilities[prefix]
            ):
                logp += math.log(markov_probabilities[prefix][next_char])
            else:
                logp = -float("inf")
                break
        if logp > -float("inf"):
            weighted.append((rule, logp))
    return sorted(weighted, key=lambda x: x[1], reverse=True)


def generate_rules_from_markov_model(
    markov_probabilities: Dict,
    target: int,
    min_len: int,
    max_len: int,
    gpu_mode: bool = False,
) -> List[Tuple[str, float]]:
    if not memory_intensive_operation_warning("Markov rule generation"):
        return []
    print_section(
        f"Generating Markov Rules ({min_len}-{max_len}, Target: {target})"
    )
    generated: Set[str] = set()
    START = "^"

    def get_next(prefix: str) -> Optional[str]:
        if prefix not in markov_probabilities:
            return None
        choices = list(markov_probabilities[prefix].keys())
        weights = list(markov_probabilities[prefix].values())
        if not choices:
            return None
        return random.choices(choices, weights=weights, k=1)[0]

    attempts = target * 5
    for _ in range(attempts):
        if len(generated) >= target:
            break
        rule = get_next(START)
        if not rule:
            continue
        while len(rule) < max_len:
            last     = rule[-1]
            last_two = rule[-2:] if len(rule) >= 2 else None
            nxt: Optional[str] = None
            if last_two and last_two in markov_probabilities:
                nxt = get_next(last_two)
            if not nxt and last in markov_probabilities:
                nxt = get_next(last)
            if not nxt:
                break
            rule += nxt
            if min_len <= len(rule) <= max_len:
                if gpu_mode:
                    cleaner = HashcatRuleCleaner()
                    if cleaner.validate_rule(rule):
                        generated.add(rule)
                else:
                    if is_valid_hashcat_rule(rule):
                        generated.add(rule)
    print_success(f"Generated {len(generated)} valid rules.")
    if generated:
        dummy_counts = {r: 1 for r in generated}
        weighted     = get_markov_weighted_rules(dummy_counts, markov_probabilities, {})
        return weighted[:target]
    return []

# ==============================================================================
# COMBINATORIAL GENERATION
# ==============================================================================

def find_min_operators_for_target(
    sorted_operators: List[Tuple[str, int]],
    target: int,
    min_len: int,
    max_len: int,
) -> List[str]:
    current = 0
    n       = 0
    while current < target and n < len(sorted_operators):
        n       += 1
        top_ops  = [op for op, _ in sorted_operators[:n]]
        current  = 0
        for length in range(min_len, max_len + 1):
            current += len(top_ops) ** length
    return [op for op, _ in sorted_operators[:n]]


def generate_rules_for_length_validated(
    args: Tuple[List[str], int, bool]
) -> Set[str]:
    top_ops, length, gpu_mode = args
    generated: Set[str] = set()
    for combo in itertools.product(top_ops, repeat=length):
        rule = "".join(combo)
        if not is_valid_hashcat_rule(rule):
            continue
        if gpu_mode:
            cleaner = HashcatRuleCleaner()
            if not cleaner.validate_rule(rule):
                continue
        generated.add(rule)
    return generated


def generate_rules_parallel(
    top_operators: List[str],
    min_len: int,
    max_len: int,
    gpu_mode: bool = False,
) -> Set[str]:
    if not memory_intensive_operation_warning("combinatorial generation"):
        return set()
    lengths   = list(range(min_len, max_len + 1))
    tasks     = [(top_operators, length, gpu_mode) for length in lengths]
    num_procs = min(os.cpu_count() or 1, len(lengths))
    print_info(
        f"Generating rules of length {min_len}-{max_len} using "
        f"{len(top_operators)} operators, {num_procs} processes..."
    )
    with multiprocessing.Pool(processes=num_procs) as pool:
        results = pool.map(generate_rules_for_length_validated, tasks)
    generated = set().union(*results)
    print_success(f"Generated {len(generated)} valid rules.")
    return generated

# ==============================================================================
# FUNCTIONAL MINIMIZATION
# ==============================================================================

TEST_VECTOR: List[str] = [
    "Password", "123456", "ADMIN", "1aB", "QWERTY", "longword",
    "spec!", "!spec", "a", "b", "c", "0123", "xYz!", "TEST",
    "tEST", "test", "0", "1", "$^", "lorem", "ipsum",
]


def worker_generate_signature(
    rule_data: Tuple[str, int]
) -> Tuple[str, Tuple[str, int]]:
    rule_text, count = rule_data
    engine    = RuleEngine([rule_text])
    parts     = [engine.apply(w) or "" for w in TEST_VECTOR]
    signature = "|".join(parts)
    return signature, (rule_text, count)


@memory_safe_operation("Functional Minimization", 85)
def functional_minimization(
    data: List[Tuple[str, int]]
) -> List[Tuple[str, int]]:
    print_section("Functional Minimization")
    print_warning(
        "This operation is RAM intensive and may take significant time "
        "for large datasets."
    )
    if not data:
        return data
    if len(data) > 10000:
        print_warning(f"Large dataset detected ({len(data):,} rules).")
        est = estimate_memory_usage(len(data))
        print(f"{Colors.CYAN}[MEMORY]{Colors.RESET} Estimated usage: {format_bytes(est)}")
        if (
            input(f"{Colors.YELLOW}Continue? (y/N): ").strip().lower()
            not in ("y", "yes")
        ):
            print_info("Skipping.")
            return data
    print_info(
        f"Using hashcat rule engine simulation with test vector "
        f"(size {len(TEST_VECTOR)})"
    )
    signature_map: Dict[str, List[Tuple[str, int]]] = {}
    num_procs = multiprocessing.cpu_count()
    print(f"{Colors.CYAN}[MP]{Colors.RESET} Using {num_procs} processes.")
    with multiprocessing.Pool(processes=num_procs) as pool:
        results = list(
            tqdm(
                pool.imap(worker_generate_signature, data),
                total=len(data),
                desc="Simulating rules",
                unit=" rules",
            )
        )
    for sig, rd in results:
        signature_map.setdefault(sig, []).append(rd)
    final: List[Tuple[str, int]] = []
    for rules_list in signature_map.values():
        rules_list.sort(key=lambda x: x[1], reverse=True)
        best  = rules_list[0][0]
        total = sum(c for _, c in rules_list)
        final.append((best, total))
    final.sort(key=lambda x: x[1], reverse=True)
    removed = len(data) - len(final)
    print_success(f"Removed {removed:,} functionally redundant rules.")
    return final

# ==============================================================================
# LEVENSHTEIN FILTERING
# ==============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        cur = [i + 1]
        for j, c2 in enumerate(s2):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (c1 != c2)))
        prev = cur
    return prev[-1]


@memory_safe_operation("Levenshtein Filtering", 85)
def levenshtein_filter(
    data: List[Tuple[str, int]], max_distance: int = 2
) -> List[Tuple[str, int]]:
    print_section("Levenshtein Filtering")
    print_warning("This operation can be slow for large datasets.")
    if not data:
        return data
    if len(data) > 5000:
        print_warning(f"Large dataset ({len(data):,} rules). This may take a while.")
        if (
            input(f"{Colors.YELLOW}Continue? (y/N): ").strip().lower()
            not in ("y", "yes")
        ):
            return data
    while True:
        try:
            d = input(
                f"{Colors.YELLOW}Enter max Levenshtein distance "
                f"(1-10) [{max_distance}]: "
            ).strip()
            if not d:
                break
            max_distance = int(d)
            if 1 <= max_distance <= 10:
                break
            else:
                print_error("Enter 1-10.")
        except ValueError:
            print_error("Invalid number.")
    unique: List[Tuple[str, int]] = []
    removed = 0
    for rule, cnt in tqdm(data, desc="Levenshtein filtering"):
        similar = False
        for existing, _ in unique:
            if levenshtein_distance(rule, existing) <= max_distance:
                similar = True
                removed += 1
                break
        if not similar:
            unique.append((rule, cnt))
    print_success(f"Removed {removed} similar rules. Final count: {len(unique)}")
    return unique

# ==============================================================================
# PARETO ANALYSIS
# ==============================================================================

def display_pareto_curve(data: List[Tuple[str, int]]) -> None:
    if not data:
        print_error("No data to analyze.")
        return
    total_value = sum(c for _, c in data)
    print_header("PARETO ANALYSIS")
    print(f"Total rules: {colorize(f'{len(data):,}', Colors.CYAN)}")
    print(f"Total occurrences: {colorize(f'{total_value:,}', Colors.CYAN)}\n")
    milestones: List[Tuple[int, int, float]] = []
    targets    = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    next_target = 0
    cum         = 0
    print(
        f"{Colors.BOLD}{'Rank':>6} {'Rule':<30} {'Count':>10} "
        f"{'Cumulative':>12} {'% Total':>8}{Colors.RESET}"
    )
    print(f"{Colors.BOLD}{'-' * 70}{Colors.RESET}")
    for i, (rule, cnt) in enumerate(data):
        cum += cnt
        pct  = (cum / total_value) * 100
        if i < 10 or (
            next_target < len(targets) and pct >= targets[next_target]
        ):
            color = (
                Colors.GREEN
                if i < 10
                else Colors.YELLOW if pct <= 80
                else Colors.RED
            )
            print(
                f"{color}{i + 1:>6} {rule:<30} {cnt:>10,} "
                f"{cum:>12,} {pct:>7.1f}%{Colors.RESET}"
            )
            if next_target < len(targets) and pct >= targets[next_target]:
                milestones.append((targets[next_target], i + 1, pct))
                next_target += 1
        if i >= 10 and next_target >= len(targets):
            break
    print(f"{Colors.BOLD}{'-' * 70}{Colors.RESET}")
    print(f"\n{Colors.BOLD}PARETO MILESTONES:{Colors.RESET}")
    for target, rules_needed, actual in milestones:
        pct_rules = (rules_needed / len(data)) * 100
        color = (
            Colors.GREEN
            if target <= 50
            else Colors.YELLOW if target <= 80
            else Colors.RED
        )
        print(
            f"  {color}{target:>2}% of value:{Colors.RESET} "
            f"{rules_needed:>6,} rules ({pct_rules:5.1f}% of total) "
            f"- Actual: {actual:5.1f}%"
        )
    print(f"\n{Colors.BOLD}PARETO CURVE (ASCII):{Colors.RESET}")
    print("  100% ┤")
    points = 20
    step   = len(data) // points
    for i in range(points + 1):
        idx      = min(i * step, len(data) - 1)
        cum_val  = sum(c for _, c in data[: idx + 1])
        pct      = (cum_val / total_value) * 100
        bar_len  = int(pct / 5)
        bar      = "█" * bar_len
        y        = 100 - (i * 5)
        if y % 20 == 0 or i == 0 or i == points:
            print(f"{y:>4}% ┤ {bar}")
    print("    0% ┼" + "─" * 20)
    print("      0%         50%        100%")
    print("       Cumulative % of rules")


def analyze_cumulative_value(
    sorted_data: List[Tuple[str, int]], total_lines: int
) -> None:
    if not sorted_data:
        print_error("No data to analyze.")
        return
    total_value = sum(c for _, c in sorted_data)
    cum         = 0
    milestones: List[Tuple[int, int]] = []
    targets     = [50, 80, 90, 95]
    next_target = 0
    for i, (_, cnt) in enumerate(sorted_data):
        cum += cnt
        pct  = (cum / total_value) * 100
        if next_target < len(targets) and pct >= targets[next_target]:
            milestones.append((targets[next_target], i + 1))
            next_target += 1
        if next_target >= len(targets):
            break
    print_header("CUMULATIVE VALUE ANALYSIS (PARETO) - SUGGESTED CUTOFFS")
    print(f"Total value: {colorize(f'{total_value:,}', Colors.CYAN)}")
    print(f"Unique rules: {colorize(f'{len(sorted_data):,}', Colors.CYAN)}")
    for target, rules_needed in milestones:
        pct_rules = (rules_needed / len(sorted_data)) * 100
        color     = (
            Colors.GREEN
            if target <= 80
            else Colors.YELLOW if target <= 90
            else Colors.RED
        )
        print(
            f"{color}[{target}% OF VALUE]:{Colors.RESET} "
            f"Reached with {colorize(f'{rules_needed:,}', Colors.CYAN)} rules "
            f"({pct_rules:.2f}% of unique rules)"
        )
    print(f"{Colors.BOLD}{'-' * 60}{Colors.RESET}")
    if milestones:
        last = milestones[-1][1]
        print(
            f"{Colors.GREEN}[SUGGESTION]{Colors.RESET} Consider a limit of: "
            f"{colorize(f'{last:,}', Colors.CYAN)} or "
            f"{colorize(f'{int(last * 1.1):,}', Colors.CYAN)}."
        )
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")

# ==============================================================================
# FILTERING FUNCTIONS
# ==============================================================================

def filter_by_min_occurrence(
    data: List[Tuple[str, int]]
) -> List[Tuple[str, int]]:
    if not data:
        return data
    max_count = data[0][1]
    suggested = max(1, sum(c for _, c in data) // 1000)
    while True:
        try:
            thresh = int(
                input(
                    f"{Colors.YELLOW}Enter MIN occurrence "
                    f"(1-{max_count:,}, suggested: {suggested:,}): {Colors.RESET}"
                )
            )
            if 1 <= thresh <= max_count:
                filtered = [(r, c) for r, c in data if c >= thresh]
                print_success(f"Kept {len(filtered):,} rules.")
                return filtered
            else:
                print_error(f"Value must be between 1 and {max_count:,}.")
        except ValueError:
            print_error("Invalid number.")


def filter_by_max_rules(
    data: List[Tuple[str, int]]
) -> List[Tuple[str, int]]:
    if not data:
        return data
    max_possible = len(data)
    while True:
        try:
            limit = int(
                input(
                    f"{Colors.YELLOW}Enter MAX number of rules to keep "
                    f"(1-{max_possible:,}): {Colors.RESET}"
                )
            )
            if 1 <= limit <= max_possible:
                filtered = data[:limit]
                print_success(f"Kept top {len(filtered):,} rules.")
                return filtered
            else:
                print_error(f"Value must be between 1 and {max_possible:,}.")
        except ValueError:
            print_error("Invalid number.")


def inverse_mode_filter(
    data: List[Tuple[str, int]]
) -> List[Tuple[str, int]]:
    if not data:
        return data
    max_possible = len(data)
    while True:
        try:
            cutoff = int(
                input(
                    f"{Colors.YELLOW}Enter cutoff rank "
                    f"(rules BELOW this rank will be kept, "
                    f"1-{max_possible:,}): {Colors.RESET}"
                )
            )
            if 1 <= cutoff <= max_possible:
                filtered = data[cutoff:]
                print_success(f"Kept {len(filtered):,} rules.")
                return filtered
            else:
                print_error(f"Value must be between 1 and {max_possible:,}.")
        except ValueError:
            print_error("Invalid number.")

# ==============================================================================
# OUTPUT FORMATTING AND SAVING
# ==============================================================================

def expand_rule(rule: str) -> str:
    """Return an expanded representation: operator tokens separated by spaces."""
    tokens = __ruleregex__.findall(rule)
    return " ".join(tokens)


def _write_rules_to_file(
    f,
    rules_data: List[Tuple[str, int]],
    output_format: str,
) -> None:
    """Write rule strings to an already-open file handle, respecting output_format."""
    rule_strings = [
        item[0] if isinstance(item, tuple) else item
        for item in rules_data
    ]
    if output_format == "expanded":
        for r in rule_strings:
            f.write(expand_rule(r) + "\n")
    else:
        for r in rule_strings:
            f.write(r + "\n")


def save_rules_to_file(
    data: List[Tuple[str, int]],
    filename: Optional[str] = None,
    mode: str = "filtered",
) -> bool:
    global _OUTPUT_FORMAT
    if not data:
        print_error("No rules to save!")
        return False
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"concentrator_{mode}_{len(data)}rules_{timestamp}.rule"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# CONCENTRATOR v3.0 - {mode.upper()} Rules\n")
            f.write(
                f"# Generated: "
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"# Total rules: {len(data):,}\n")
            f.write(f"# Mode: {mode}\n")
            f.write(f"# Output format: {_OUTPUT_FORMAT}\n#\n")
            _write_rules_to_file(f, data, _OUTPUT_FORMAT)
        print_success(f"Saved {len(data):,} rules to {filename}")
        return True
    except IOError as e:
        print_error(f"Failed to save file: {e}")
        return False


def save_concentrator_rules(
    rules_data: List,
    output_filename: str,
    mode_name: str,
) -> bool:
    global _OUTPUT_FORMAT
    if not rules_data:
        print_error("No rules to save!")
        return False
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(
                f"# CONCENTRATOR v3.0 - {mode_name.upper()} MODE OUTPUT\n"
            )
            f.write(
                f"# Generated: "
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"# Total rules: {len(rules_data)}\n")
            f.write(f"# Output format: {_OUTPUT_FORMAT}\n#\n")
            _write_rules_to_file(f, rules_data, _OUTPUT_FORMAT)
        print_success(f"Saved {len(rules_data)} rules to {output_filename}")
        return True
    except Exception as e:
        print_error(f"Failed to save rules: {e}")
        return False


def gpu_extract_and_validate_rules(
    full_rule_counts: Dict[str, int],
    top_rules: int,
    gpu_enabled: bool,
) -> List[Tuple[str, int]]:
    sorted_rules = sorted(
        full_rule_counts.items(), key=lambda x: x[1], reverse=True
    )
    to_validate = [r for r, _ in sorted_rules[: top_rules * 2]]
    if gpu_enabled:
        valid  = gpu_validate_rules(to_validate)
        result: List[Tuple[str, int]] = []
        for rule, is_valid in zip(to_validate, valid):
            if is_valid:
                if _GPU_MODE_ENABLED:
                    cleaner = HashcatRuleCleaner()
                    if not cleaner.validate_rule(rule):
                        continue
                result.append((rule, full_rule_counts[rule]))
        return result[:top_rules]
    else:
        result = []
        for rule in to_validate:
            if is_valid_hashcat_rule(rule):
                result.append((rule, full_rule_counts[rule]))
        return result[:top_rules]

# ==============================================================================
# HASHCAT RULE CLEANUP — HASHCAT 6.x COMPATIBLE
#
# Since hashcat 6.0 the following operator families work identically on both
# the CPU and the OpenCL/GPU execution paths:
#
#   • Memory operators  : M  4  6  X  Q
#   • Reject operators  : <  >  !  /  (  )  =  %  Q
#
# The former GPU-denial logic (rc = -1 when mode == 2) has been removed.
# HashcatRuleCleaner now validates rules uniformly regardless of the mode
# argument, which is retained only for backward-compatibility of call sites.
# ==============================================================================

class HashcatRuleCleaner:
    """
    Implements hashcat rule validation logic.

    Based on the official cleanup-rules.c from hashcat, updated for 6.x:
    memory and reject operators are accepted in both CPU and GPU modes.

    Parameters
    ----------
    mode : int
        1 = CPU path (historical), 2 = GPU/OpenCL path (historical).
        Since hashcat 6.x this distinction no longer affects which operators
        are considered valid; the parameter is kept for API compatibility only.
    """

    # ------------------------------------------------------------------ #
    # Rule operation constants (mirror hashcat source naming)             #
    # ------------------------------------------------------------------ #
    RULE_OP_MANGLE_NOOP             = ":"
    RULE_OP_MANGLE_LREST            = "l"
    RULE_OP_MANGLE_UREST            = "u"
    RULE_OP_MANGLE_LREST_UFIRST     = "c"
    RULE_OP_MANGLE_UREST_LFIRST     = "C"
    RULE_OP_MANGLE_TREST            = "t"
    RULE_OP_MANGLE_TOGGLE_AT        = "T"
    RULE_OP_MANGLE_REVERSE          = "r"
    RULE_OP_MANGLE_DUPEWORD         = "d"
    RULE_OP_MANGLE_DUPEWORD_TIMES   = "p"
    RULE_OP_MANGLE_REFLECT          = "f"
    RULE_OP_MANGLE_ROTATE_LEFT      = "{"
    RULE_OP_MANGLE_ROTATE_RIGHT     = "}"
    RULE_OP_MANGLE_APPEND           = "$"
    RULE_OP_MANGLE_PREPEND          = "^"
    RULE_OP_MANGLE_DELETE_FIRST     = "["
    RULE_OP_MANGLE_DELETE_LAST      = "]"
    RULE_OP_MANGLE_DELETE_AT        = "D"
    RULE_OP_MANGLE_EXTRACT          = "x"
    RULE_OP_MANGLE_INSERT           = "i"
    RULE_OP_MANGLE_OVERSTRIKE       = "o"
    RULE_OP_MANGLE_TRUNCATE_AT      = "'"
    RULE_OP_MANGLE_REPLACE          = "s"
    RULE_OP_MANGLE_PURGECHAR        = "@"
    RULE_OP_MANGLE_TOGGLECASE_REC   = "a"
    RULE_OP_MANGLE_DUPECHAR_FIRST   = "z"
    RULE_OP_MANGLE_DUPECHAR_LAST    = "Z"
    RULE_OP_MANGLE_DUPECHAR_ALL     = "q"
    RULE_OP_MANGLE_EXTRACT_MEMORY   = "X"   # memory – supported on GPU since 6.x
    RULE_OP_MANGLE_APPEND_MEMORY    = "4"   # memory – supported on GPU since 6.x
    RULE_OP_MANGLE_PREPEND_MEMORY   = "6"   # memory – supported on GPU since 6.x
    RULE_OP_MEMORIZE_WORD           = "M"   # memory – supported on GPU since 6.x
    RULE_OP_REJECT_LESS             = "<"   # reject – supported on GPU since 6.x
    RULE_OP_REJECT_GREATER          = ">"   # reject – supported on GPU since 6.x
    RULE_OP_REJECT_CONTAIN          = "!"   # reject – supported on GPU since 6.x
    RULE_OP_REJECT_NOT_CONTAIN      = "/"   # reject – supported on GPU since 6.x
    RULE_OP_REJECT_EQUAL_FIRST      = "("   # reject – supported on GPU since 6.x
    RULE_OP_REJECT_EQUAL_LAST       = ")"   # reject – supported on GPU since 6.x
    RULE_OP_REJECT_EQUAL_AT         = "="   # reject – supported on GPU since 6.x
    RULE_OP_REJECT_CONTAINS         = "%"   # reject – supported on GPU since 6.x
    RULE_OP_REJECT_MEMORY           = "Q"   # reject – supported on GPU since 6.x
    RULE_OP_MANGLE_SWITCH_FIRST     = "k"
    RULE_OP_MANGLE_SWITCH_LAST      = "K"
    RULE_OP_MANGLE_SWITCH_AT        = "*"
    RULE_OP_MANGLE_CHR_SHIFTL       = "L"
    RULE_OP_MANGLE_CHR_SHIFTR       = "R"
    RULE_OP_MANGLE_CHR_INCR         = "+"
    RULE_OP_MANGLE_CHR_DECR         = "-"
    RULE_OP_MANGLE_REPLACE_NP1      = "."
    RULE_OP_MANGLE_REPLACE_NM1      = ","
    RULE_OP_MANGLE_DUPEBLOCK_FIRST  = "y"
    RULE_OP_MANGLE_DUPEBLOCK_LAST   = "Y"
    RULE_OP_MANGLE_TITLE            = "E"
    RULE_OP_MANGLE_TITLE_SEP        = "e"

    MAX_RULES = 255   # hashcat enforces the same limit for both CPU and GPU

    def __init__(self, mode: int = 1) -> None:
        """
        Parameters
        ----------
        mode : int
            Accepted values: 1 (CPU) or 2 (GPU).  Since hashcat 6.x the value
            no longer influences which operators pass validation.  Retained for
            call-site compatibility.
        """
        if mode not in (1, 2):
            raise ValueError("mode must be 1 (CPU) or 2 (GPU)")
        self.mode = mode

    # ------------------------------------------------------------------ #
    # Character-class helpers                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def class_num(c: str) -> bool:
        return "0" <= c <= "9"

    @staticmethod
    def class_upper(c: str) -> bool:
        return "A" <= c <= "Z"

    @staticmethod
    def conv_ctoi(c: str) -> int:
        """Convert a base-36 digit character to its integer value (-1 on error)."""
        if HashcatRuleCleaner.class_num(c):
            return ord(c) - ord("0")
        if HashcatRuleCleaner.class_upper(c):
            return ord(c) - ord("A") + 10
        return -1

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def validate_rule(self, rule_line: str) -> bool:
        """
        Return True if *rule_line* is a syntactically valid hashcat rule.

        Validates all operator families including memory (M, 4, 6, X) and
        reject (<, >, !, /, (, ), =, %, Q) operators, which are supported on
        both CPU and GPU execution paths since hashcat 6.0.
        """
        clean_line = rule_line.replace(" ", "")
        if not clean_line:
            return False

        rc       = 0
        cnt      = 0
        pos      = 0
        line_len = len(clean_line)

        while pos < line_len:
            op = clean_line[pos]

            if op == " ":
                pos += 1
                continue

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
                elif op == self.RULE_OP_MANGLE_TITLE_SEP:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                # ----------------------------------------------------------
                # Memory operators – valid on both CPU and GPU since hashcat 6.x
                # ----------------------------------------------------------
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
                elif op == self.RULE_OP_MANGLE_APPEND_MEMORY:
                    pass   # no additional arguments
                elif op == self.RULE_OP_MANGLE_PREPEND_MEMORY:
                    pass   # no additional arguments
                elif op == self.RULE_OP_MEMORIZE_WORD:
                    pass   # no additional arguments
                # ----------------------------------------------------------
                # Reject operators – valid on both CPU and GPU since hashcat 6.x
                # ----------------------------------------------------------
                elif op == self.RULE_OP_REJECT_LESS:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_REJECT_GREATER:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                elif op == self.RULE_OP_REJECT_CONTAIN:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_REJECT_NOT_CONTAIN:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_REJECT_EQUAL_FIRST:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_REJECT_EQUAL_LAST:
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_REJECT_EQUAL_AT:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_REJECT_CONTAINS:
                    pos += 1
                    if pos >= line_len or self.conv_ctoi(clean_line[pos]) == -1:
                        rc = -1
                    pos += 1
                    if pos >= line_len:
                        rc = -1
                elif op == self.RULE_OP_REJECT_MEMORY:
                    pass   # no additional arguments
                else:
                    rc = -1
            except IndexError:
                rc = -1

            if rc == -1:
                break

            cnt += 1
            pos += 1

            if cnt > self.MAX_RULES:
                rc = -1
                break

        return rc == 0

    def clean_rules(
        self, rules_data: List[Tuple[str, int]]
    ) -> List[Tuple[str, int]]:
        """Validate each rule and return only those that pass."""
        print_section("Hashcat Rule Validation (hashcat 6.x compatible)")
        print(
            f"Validating {colorize(f'{len(rules_data):,}', Colors.CYAN)} rules..."
        )
        valid: List[Tuple[str, int]] = []
        invalid = 0
        for rule, cnt in tqdm(rules_data, desc="Validating rules"):
            if self.validate_rule(rule):
                valid.append((rule, cnt))
            else:
                invalid += 1
        print_success(
            f"Removed {invalid:,} invalid rules. "
            f"{len(valid):,} valid remaining."
        )
        return valid


def hashcat_rule_cleanup(
    data: List[Tuple[str, int]], mode: int = 1
) -> List[Tuple[str, int]]:
    """Convenience wrapper around HashcatRuleCleaner.clean_rules()."""
    cleaner = HashcatRuleCleaner(mode)
    return cleaner.clean_rules(data)

# ==============================================================================
# ENHANCED INTERACTIVE PROCESSING LOOP
# ==============================================================================

def enhanced_interactive_processing_loop(
    original_data: List[Tuple[str, int]],
    total_lines: int,
    args: Any,
    initial_mode: str = "extracted",
) -> List[Tuple[str, int]]:
    global _OUTPUT_FORMAT
    current_data = original_data
    orig_count   = len(current_data)
    print_header("ENHANCED RULE PROCESSING - INTERACTIVE MENU")
    print(
        f"Initial dataset: {colorize(f'{orig_count:,}', Colors.CYAN)} unique rules"
    )
    try:
        while True:
            print(f"\n{Colors.BOLD}{'-' * 80}{Colors.RESET}")
            print(f"{Colors.BOLD}ADVANCED FILTERING OPTIONS:{Colors.RESET}")
            print(f" {Colors.GREEN}(1){Colors.RESET} Filter by MINIMUM OCCURRENCE")
            print(f" {Colors.GREEN}(2){Colors.RESET} Filter by MAXIMUM NUMBER OF RULES (TOP N)")
            print(
                f" {Colors.GREEN}(3){Colors.RESET} Filter by FUNCTIONAL REDUNDANCY "
                f"(Logic Minimization) [RAM INTENSIVE]"
            )
            print(
                f" {Colors.GREEN}(4){Colors.RESET} INVERSE MODE - "
                f"Save rules BELOW the MAX_COUNT limit"
            )
            print(
                f" {Colors.GREEN}(5){Colors.RESET} HASHCAT CLEANUP - "
                f"Validate rules (hashcat 6.x, all operators accepted)"
            )
            print(
                f" {Colors.GREEN}(6){Colors.RESET} LEVENSHTEIN FILTER - "
                f"Remove similar rules"
            )
            print(
                f" {Colors.GREEN}(7){Colors.RESET} TOGGLE OUTPUT FORMAT - "
                f"Switch between line and expanded"
            )
            print(f"\n{Colors.BOLD}ANALYSIS & UTILITIES:{Colors.RESET}")
            print(f" {Colors.BLUE}(p){Colors.RESET} Show PARETO analysis with detailed curve")
            print(f" {Colors.BLUE}(s){Colors.RESET} SAVE current rules to file")
            print(f" {Colors.BLUE}(r){Colors.RESET} RESET to original dataset")
            print(f" {Colors.BLUE}(i){Colors.RESET} Show dataset information")
            print(f" {Colors.BLUE}(q){Colors.RESET} QUIT program")
            print(f"{Colors.BOLD}{'-' * 80}{Colors.RESET}")
            choice = input(
                f"{Colors.YELLOW}Enter your choice: {Colors.RESET}"
            ).strip().lower()
            if choice == "q":
                print_header("THANK YOU FOR USING CONCENTRATOR v3.0!")
                break
            elif choice == "p":
                display_pareto_curve(current_data)
                continue
            elif choice == "s":
                print(f"\n{Colors.CYAN}Save Options:{Colors.RESET}")
                print(f" {Colors.GREEN}(1){Colors.RESET} Auto-generated filename")
                print(f" {Colors.GREEN}(2){Colors.RESET} Custom filename")
                print(f" {Colors.GREEN}(3){Colors.RESET} Cancel")
                sc = input(f"{Colors.YELLOW}Choose: ").strip()
                if sc == "1":
                    save_rules_to_file(current_data, mode=f"{initial_mode}_filtered")
                elif sc == "2":
                    name = input(f"{Colors.YELLOW}Enter filename: ").strip()
                    if name:
                        if not name.endswith((".rule", ".txt")):
                            name += ".rule"
                        save_rules_to_file(
                            current_data, name, f"{initial_mode}_filtered"
                        )
                continue
            elif choice == "r":
                current_data = original_data
                print_success(
                    f"Restored original dataset: {len(current_data):,} rules"
                )
                continue
            elif choice == "i":
                print_section("DATASET INFORMATION")
                print(
                    f"Original rules: {colorize(f'{orig_count:,}', Colors.CYAN)}"
                )
                print(
                    f"Current rules: "
                    f"{colorize(f'{len(current_data):,}', Colors.CYAN)}"
                )
                reduction = (
                    (orig_count - len(current_data)) / orig_count * 100
                ) if orig_count else 0
                print(
                    f"Reduction: "
                    f"{colorize(f'{reduction:.1f}%', Colors.GREEN if reduction > 0 else Colors.YELLOW)}"
                )
                if current_data:
                    maxc = current_data[0][1]
                    minc = current_data[-1][1]
                    avgc = sum(c for _, c in current_data) / len(current_data)
                    print(f"Max occurrence: {colorize(f'{maxc:,}', Colors.CYAN)}")
                    print(f"Min occurrence: {colorize(f'{minc:,}', Colors.CYAN)}")
                    print(f"Avg occurrence: {colorize(f'{avgc:.1f}', Colors.CYAN)}")
                continue
            elif choice == "1":
                current_data = filter_by_min_occurrence(current_data)
            elif choice == "2":
                current_data = filter_by_max_rules(current_data)
            elif choice == "3":
                current_data = functional_minimization(current_data)
            elif choice == "4":
                current_data = inverse_mode_filter(current_data)
            elif choice == "5":
                # Since hashcat 6.x both modes validate identically; mode
                # selection is kept for user awareness / historical reference.
                print(f"\n{Colors.MAGENTA}[HASHCAT CLEANUP]{Colors.RESET} Choose mode:")
                print(
                    f" {Colors.CYAN}(1){Colors.RESET} CPU path "
                    f"(all operators, hashcat 6.x default)"
                )
                print(
                    f" {Colors.CYAN}(2){Colors.RESET} GPU/OpenCL path "
                    f"(all operators accepted since hashcat 6.x)"
                )
                m    = input(f"{Colors.YELLOW}Enter mode (1 or 2): ").strip()
                mode = 1 if m == "1" else 2
                current_data = hashcat_rule_cleanup(current_data, mode)
            elif choice == "6":
                current_data = levenshtein_filter(
                    current_data, getattr(args, "levenshtein_max_dist", 2)
                )
            elif choice == "7":
                _OUTPUT_FORMAT = (
                    "expanded" if _OUTPUT_FORMAT == "line" else "line"
                )
                print_success(f"Output format switched to {_OUTPUT_FORMAT}")
                continue
            else:
                print_error("Invalid choice. Please try again.")
                continue
            if choice in ("1", "2", "3", "4", "5", "6"):
                reduction = (
                    (orig_count - len(current_data)) / orig_count * 100
                ) if orig_count else 0
                print_success(
                    f"Dataset updated: {len(current_data):,} rules "
                    f"({reduction:.1f}% reduction)"
                )
                if len(current_data) > 0:
                    if (
                        input(
                            f"{Colors.YELLOW}Show Pareto analysis? (Y/n): "
                        ).strip().lower()
                        not in ("n", "no")
                    ):
                        display_pareto_curve(current_data)
                if (
                    input(
                        f"{Colors.YELLOW}Save current dataset? (y/N): "
                    ).strip().lower()
                    in ("y", "yes")
                ):
                    save_rules_to_file(
                        current_data, mode=f"{initial_mode}_filtered"
                    )
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interactive menu interrupted.{Colors.RESET}")
        if (
            input(
                f"{Colors.YELLOW}Save current dataset before exiting? "
                f"(y/N): "
            ).strip().lower()
            in ("y", "yes")
        ):
            save_rules_to_file(current_data, mode=f"{initial_mode}_filtered")
        print(f"{Colors.CYAN}Returning to main program...{Colors.RESET}")
    return current_data

# ==============================================================================
# MAIN PROCESSING FUNCTIONS
# ==============================================================================

def process_multiple_files_concentrator(args: Any) -> None:
    print_header("PROCESSING MODE - Interactive Rule Minimization")
    all_filepaths = find_rule_files_recursive(args.paths, max_depth=3)
    if not all_filepaths:
        print_error("No rule files found to process.")
        return
    set_global_flags(args.temp_dir, args.in_memory)
    sorted_op_counts, full_rule_counts, _ = analyze_rule_files_parallel(
        all_filepaths, args.max_length
    )
    if not full_rule_counts:
        print_error("No rules found in files.")
        return
    rules_data: List[Tuple[str, int]] = sorted(
        full_rule_counts.items(), key=lambda x: x[1], reverse=True
    )
    print_success(f"Loaded {len(rules_data):,} unique rules for processing.")
    final_data = enhanced_interactive_processing_loop(
        rules_data, sum(full_rule_counts.values()), args, "processed"
    )
    if final_data:
        output_file = args.output_base_name + "_processed.rule"
        save_rules_to_file(final_data, output_file, "processed")


def concentrator_main_processing(args: Any) -> None:
    global _OUTPUT_FORMAT, _GPU_MODE_ENABLED

    if args.extract_rules:
        active_mode    = "extraction"
        output_suffix  = "_extracted.rule"
        mode_color     = Colors.GREEN
        initial_mode   = "extracted"
    elif args.generate_combo:
        active_mode    = "combo"
        output_suffix  = "_combo.rule"
        mode_color     = Colors.BLUE
        initial_mode   = "combo"
    elif args.generate_markov_rules:
        active_mode    = "markov"
        output_suffix  = "_markov.rule"
        mode_color     = Colors.MAGENTA
        initial_mode   = "markov"
    else:
        print_error("No mode selected. Exiting.")
        return

    output_file_name = args.output_base_name + output_suffix
    _OUTPUT_FORMAT   = (
        args.output_format
        if args.output_format in ("line", "expanded")
        else "line"
    )

    print(
        f"\n{Colors.CYAN}Active Mode:{Colors.END} "
        f"{mode_color}{Colors.BOLD}{active_mode.upper()}{Colors.END}"
    )
    print(
        f"{Colors.CYAN}Output File:{Colors.END} "
        f"{Colors.WHITE}{output_file_name}{Colors.END}"
    )
    print(f"{Colors.CYAN}Output Format:{Colors.END} {_OUTPUT_FORMAT}")

    if active_mode == "markov":
        markov_min_len = args.markov_length[0]
        markov_max_len = args.markov_length[-1]
    elif active_mode == "combo":
        combo_min_len  = args.combo_length[0]
        combo_max_len  = args.combo_length[-1]

    gpu_enabled = False
    if not args.no_gpu:
        gpu_enabled = setup_opencl()
        if gpu_enabled:
            _GPU_MODE_ENABLED = True
            print_success("GPU Acceleration: ENABLED")
        else:
            print_warning("GPU Acceleration: Disabled (falling back to CPU)")
    else:
        print_warning("GPU Acceleration: Manually disabled")

    print_section("Collecting Rule Files (Recursive Search, Max Depth 3)")
    all_filepaths = find_rule_files_recursive(args.paths, max_depth=3)
    output_files_to_exclude = {os.path.basename(output_file_name)}
    all_filepaths = [
        fp
        for fp in all_filepaths
        if os.path.basename(fp) not in output_files_to_exclude
    ]
    if not all_filepaths:
        print_error("No rule files found to process. Exiting.")
        return
    print_success(f"Found {len(all_filepaths)} rule files to analyze.")
    set_global_flags(args.temp_dir, args.in_memory)

    print_section("Starting Parallel Rule File Analysis")
    sorted_op_counts, full_rule_counts, all_clean_rules = (
        analyze_rule_files_parallel(all_filepaths, args.max_length)
    )
    if not sorted_op_counts:
        print_error("No operators found in files. Exiting.")
        return

    markov_probabilities, total_transitions = None, None
    build_markov = False
    if (
        active_mode == "extraction"
        and hasattr(args, "statistical_sort")
        and args.statistical_sort
    ):
        build_markov = True
        print_section("Building Markov Model for Statistical Sort")
    elif active_mode == "markov":
        build_markov = True
        print_section("Building Markov Model for Rule Generation")
    else:
        print_info("Skipping Markov Model Build (Not needed for current mode)")
    if build_markov:
        markov_probabilities, total_transitions = get_markov_model(full_rule_counts)

    result_data: Optional[List] = None
    if active_mode == "extraction":
        print_section("GPU-Accelerated Rule Extraction and Validation")
        if args.statistical_sort:
            print_info("Sort Mode: Statistical Sort (Markov Weight)")
            if markov_probabilities is None:
                print_error(
                    "Statistical sort (-s) requires the Markov model, "
                    "but it was skipped."
                )
                return
            sorted_rule_data = get_markov_weighted_rules(
                full_rule_counts, markov_probabilities, total_transitions
            )
            if gpu_enabled and sorted_rule_data:
                to_validate = [r for r, _ in sorted_rule_data[: args.top_rules * 2]]
                valid        = gpu_validate_rules(to_validate)
                validated: List[Tuple[str, Any]] = []
                for rule, is_valid in zip(to_validate, valid):
                    if is_valid:
                        if _GPU_MODE_ENABLED:
                            cleaner = HashcatRuleCleaner()
                            if not cleaner.validate_rule(rule):
                                continue
                        for orig_rule, weight in sorted_rule_data:
                            if rule == orig_rule:
                                validated.append((rule, weight))
                                break
                result_data = validated[: args.top_rules]
                print_success(
                    f"GPU validated {len(result_data)} statistically sorted rules"
                )
            else:
                result_data = sorted_rule_data[: args.top_rules]
        else:
            print_info("Sort Mode: Frequency Sort (Raw Count) with GPU Validation")
            result_data = gpu_extract_and_validate_rules(
                full_rule_counts, args.top_rules, gpu_enabled
            )
        print_success(
            f"Extracted {len(result_data)} top unique rules "
            f"(max length: {args.max_length} characters)."
        )
    elif active_mode == "markov":
        print_section("Starting STATISTICAL Markov Rule Generation (Validated)")
        markov_rules_data = generate_rules_from_markov_model(
            markov_probabilities,
            args.generate_target,
            markov_min_len,
            markov_max_len,
            gpu_mode=_GPU_MODE_ENABLED,
        )
        if gpu_enabled and markov_rules_data:
            markov_rules = [r for r, _ in markov_rules_data]
            valid         = gpu_validate_rules(markov_rules, args.max_length)
            valid_markov: List[Tuple[str, Any]] = []
            for (rule, weight), is_valid in zip(markov_rules_data, valid):
                if is_valid:
                    if _GPU_MODE_ENABLED:
                        cleaner = HashcatRuleCleaner()
                        if not cleaner.validate_rule(rule):
                            continue
                    valid_markov.append((rule, weight))
            print_success(
                f"GPU validated {len(valid_markov)}/{len(markov_rules_data)} "
                f"Markov rules"
            )
            result_data = valid_markov[: args.generate_target]
        else:
            result_data = markov_rules_data
    elif active_mode == "combo":
        print_section("Starting COMBINATORIAL Rule Generation (Validated)")
        top_operators_needed = find_min_operators_for_target(
            sorted_op_counts, args.combo_target, combo_min_len, combo_max_len
        )
        print_info(
            f"Using top {len(top_operators_needed)} operators to "
            f"approximate {args.combo_target} rules."
        )
        generated_rules_set = generate_rules_parallel(
            top_operators_needed,
            combo_min_len,
            combo_max_len,
            gpu_mode=_GPU_MODE_ENABLED,
        )
        result_data = [(rule, 1) for rule in generated_rules_set]
        print_success(f"Generated {len(result_data)} combinatorial rules.")

    print(f"\n{Colors.CYAN}{Colors.BOLD}" + "=" * 60)
    print("ENHANCED PROCESSING OPTIONS")
    print("=" * 60 + f"{Colors.END}")
    print(
        f"{Colors.YELLOW}Would you like to apply additional "
        f"filtering and optimization?{Colors.END}"
    )
    print(f"{Colors.CYAN}Available options:{Colors.END}")
    print(f"  {Colors.GREEN}•{Colors.END} Filter by occurrence count")
    print(f"  {Colors.GREEN}•{Colors.END} Remove functionally redundant rules")
    print(f"  {Colors.GREEN}•{Colors.END} Apply Levenshtein distance filtering")
    print(
        f"  {Colors.GREEN}•{Colors.END} Hashcat rule validation "
        f"(all operators, hashcat 6.x)"
    )
    print(f"  {Colors.GREEN}•{Colors.END} Pareto analysis for optimal cutoff selection")
    print(f"  {Colors.GREEN}•{Colors.END} Toggle output format (line/expanded)")

    enter_interactive = (
        input(
            f"\n{Colors.YELLOW}Enter enhanced interactive mode? "
            f"(Y/n): {Colors.RESET}"
        )
        .strip()
        .lower()
    )
    if enter_interactive not in ("n", "no"):
        total_lines_estimate = sum(full_rule_counts.values())
        final_data = enhanced_interactive_processing_loop(
            result_data, total_lines_estimate, args, initial_mode
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
    print(
        f"{Colors.YELLOW}This tool can significantly reduce rule file size while\n"
        f"maintaining or even improving cracking effectiveness.\n"
        f"For even better results, it is recommended to debug rules "
        f"obtained by using Concentrator.{Colors.END}"
    )
    if gpu_enabled:
        print_success("GPU Acceleration was used for improved performance")
    print_memory_status()

# ==============================================================================
# INTERACTIVE MODE
# ==============================================================================

def interactive_mode() -> Optional[Dict[str, Any]]:
    print_header("CONCENTRATOR v3.0 - INTERACTIVE MODE")
    settings: Dict[str, Any] = {}
    print(f"\n{Colors.CYAN}Input Configuration:{Colors.END}")

    # ---- path collection -------------------------------------------------- #
    while True:
        paths_input = input(
            f"{Colors.YELLOW}Enter rule files/directories "
            f"(space separated): {Colors.END}"
        ).strip()
        if paths_input:
            paths       = paths_input.split()
            valid_paths = [p for p in paths if os.path.exists(p)]
            for p in paths:
                if not os.path.exists(p):
                    print_warning(f"Path not found: {p}")
            if valid_paths:
                settings["paths"] = valid_paths
                break
            else:
                print_error("No valid paths provided. Please try again.")
        else:
            print_error("Please provide at least one path.")

    # ---- quick dataset analysis ------------------------------------------- #
    print(f"\n{Colors.CYAN}Analyzing Input Data...{Colors.END}")
    recommended_mode: Optional[str] = None
    try:
        all_filepaths = find_rule_files_recursive(settings["paths"], max_depth=3)
        if not all_filepaths:
            print_error("No rule files found in the provided paths.")
            return None
        print_success(f"Found {len(all_filepaths)} rule files.")
        total_rules  = 0
        unique_rules: Set[str] = set()
        max_rule_len = 0
        for fp in all_filepaths[:10]:
            try:
                with open(fp, "r", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or len(line) > 100:
                            continue
                        total_rules += 1
                        unique_rules.add(line)
                        max_rule_len = max(max_rule_len, len(line))
            except Exception:
                continue
        estimated_total_rules = total_rules * max(1, len(all_filepaths) // 10)
        print(f"{Colors.CYAN}Quick Analysis Results:{Colors.END}")
        print(f"  Files found: {len(all_filepaths)}")
        print(f"  Sampled rules: {total_rules}")
        print(f"  Estimated total rules: {estimated_total_rules:,}")
        print(f"  Unique rules in sample: {len(unique_rules)}")
        print(f"  Max rule length observed: {max_rule_len}")
        print(f"\n{Colors.CYAN}Mode Recommendations:{Colors.END}")
        if estimated_total_rules < 1000:
            print(
                f"  {Colors.GREEN}•{Colors.END} Small dataset: Consider "
                f"{Colors.YELLOW}Combinatorial Generation{Colors.END} mode"
            )
            recommended_mode = "combo"
        elif len(unique_rules) / max(1, total_rules) < 0.3:
            print(
                f"  {Colors.GREEN}•{Colors.END} Low uniqueness: Consider "
                f"{Colors.YELLOW}Extraction{Colors.END} mode"
            )
            recommended_mode = "extraction"
        else:
            print(
                f"  {Colors.GREEN}•{Colors.END} Good dataset diversity: Consider "
                f"{Colors.YELLOW}Markov{Colors.END} mode"
            )
            recommended_mode = "markov"
        if max_rule_len > 20:
            print(
                f"  {Colors.GREEN}•{Colors.END} Long rules detected: Enable "
                f"{Colors.YELLOW}functional minimization{Colors.END}"
            )
    except Exception as e:
        print_warning(f"Quick analysis failed: {e}")

    # ---- mode selection ---------------------------------------------------- #
    print(f"\n{Colors.CYAN}Processing Mode:{Colors.END}")
    print(f"  {Colors.GREEN}1{Colors.END} - Extract top existing rules")
    print(f"  {Colors.GREEN}2{Colors.END} - Generate combinatorial rules")
    print(f"  {Colors.GREEN}3{Colors.END} - Generate Markov rules")
    if recommended_mode:
        mode_display = {"extraction": "1", "combo": "2", "markov": "3"}
        print(
            f"{Colors.YELLOW}  Recommended based on analysis: "
            f"Mode {mode_display[recommended_mode]}{Colors.END}"
        )
    while True:
        mode_choice = input(
            f"{Colors.YELLOW}Select mode (1-3): {Colors.END}"
        ).strip()
        if mode_choice == "1":
            settings["mode"] = "extraction"
            break
        elif mode_choice == "2":
            settings["mode"] = "combo"
            break
        elif mode_choice == "3":
            settings["mode"] = "markov"
            break
        else:
            print_error("Invalid choice. Please enter 1, 2, or 3.")

    # ---- mode-specific settings ------------------------------------------- #
    if settings["mode"] == "extraction":
        while True:
            try:
                top_rules = int(
                    input(
                        f"{Colors.YELLOW}Number of top rules to extract "
                        f"(default 10000): {Colors.END}"
                    ) or "10000"
                )
                if top_rules > 0:
                    settings["top_rules"] = top_rules
                    break
                else:
                    print_error("Please enter a positive number.")
            except ValueError:
                print_error("Please enter a valid number.")
        settings["statistical_sort"] = get_yes_no(
            f"{Colors.YELLOW}Use statistical sort instead of frequency?{Colors.END}",
            False,
        )
    else:
        while True:
            try:
                target_rules = int(
                    input(
                        f"{Colors.YELLOW}Target number of rules to generate "
                        f"(default 10000): {Colors.END}"
                    ) or "10000"
                )
                if target_rules > 0:
                    settings["target_rules"] = target_rules
                    break
                else:
                    print_error("Please enter a positive number.")
            except ValueError:
                print_error("Please enter a valid number.")
        while True:
            try:
                min_len = int(
                    input(
                        f"{Colors.YELLOW}Minimum rule length (default 1): "
                        f"{Colors.END}"
                    ) or "1"
                )
                max_len = int(
                    input(
                        f"{Colors.YELLOW}Maximum rule length (default 3): "
                        f"{Colors.END}"
                    ) or "3"
                )
                if 1 <= min_len <= max_len:
                    settings["min_len"] = min_len
                    settings["max_len"] = max_len
                    break
                else:
                    print_error("Minimum must be <= maximum and both >= 1.")
            except ValueError:
                print_error("Please enter valid numbers.")

    # ---- global settings --------------------------------------------------- #
    print(f"\n{Colors.CYAN}Global Settings:{Colors.END}")
    settings["output_base_name"] = (
        input(
            f"{Colors.YELLOW}Output base name "
            f"(default 'concentrator_output'): {Colors.END}"
        ) or "concentrator_output"
    )
    while True:
        try:
            max_length = int(
                input(
                    f"{Colors.YELLOW}Maximum rule length to process "
                    f"(default 31): {Colors.END}"
                ) or "31"
            )
            if max_length > 0:
                settings["max_length"] = max_length
                break
            else:
                print_error("Please enter a positive number.")
        except ValueError:
            print_error("Please enter a valid number.")
    settings["no_gpu"]    = not get_yes_no(
        f"{Colors.YELLOW}Enable GPU acceleration?{Colors.END}", True
    )
    settings["in_memory"] = get_yes_no(
        f"{Colors.YELLOW}Process entirely in RAM?{Colors.END}", False
    )

    # ---- output format ----------------------------------------------------- #
    print(f"\n{Colors.CYAN}Output Format:{Colors.END}")
    print(
        f"  {Colors.GREEN}1{Colors.END} - Standard "
        f"(each rule on its own line, concatenated)"
    )
    print(
        f"  {Colors.GREEN}2{Colors.END} - Expanded "
        f"(operators/arguments separated by spaces)"
    )
    while True:
        fmt_choice = input(
            f"{Colors.YELLOW}Select format (1-2): {Colors.END}"
        ).strip()
        if fmt_choice == "1":
            settings["output_format"] = "line"
            break
        elif fmt_choice == "2":
            settings["output_format"] = "expanded"
            break
        else:
            print_error("Invalid choice. Please enter 1 or 2.")

    if not settings["in_memory"]:
        temp_dir = input(
            f"{Colors.YELLOW}Temporary directory (default: system temp): "
            f"{Colors.END}"
        ).strip()
        settings["temp_dir"] = temp_dir if temp_dir else None
    else:
        settings["temp_dir"] = None

    # ---- fill in any missing keys with defaults ---------------------------- #
    required_keys: Dict[str, Any] = {
        "temp_dir":          None,
        "no_gpu":            False,
        "in_memory":         False,
        "max_length":        31,
        "output_base_name":  "concentrator_output",
        "output_format":     "line",
    }
    if settings["mode"] == "extraction":
        required_keys.update({"top_rules": 10000, "statistical_sort": False})
    else:
        required_keys.update({"target_rules": 10000, "min_len": 1, "max_len": 3})
    for key, default_value in required_keys.items():
        settings.setdefault(key, default_value)

    # ---- confirmation summary --------------------------------------------- #
    print(f"\n{Colors.CYAN}Configuration Summary:{Colors.END}")
    print(f"  Mode: {settings['mode']}")
    print(f"  Input paths: {len(settings['paths'])} locations")
    print(f"  Output: {settings['output_base_name']}")
    print(f"  Max rule length: {settings['max_length']}")
    print(f"  GPU: {'Enabled' if not settings['no_gpu'] else 'Disabled'}")
    print(f"  In-memory: {'Yes' if settings['in_memory'] else 'No'}")
    print(f"  Temp directory: {settings['temp_dir'] or 'System default'}")
    print(f"  Output format: {settings['output_format']}")
    if settings["mode"] == "extraction":
        print(f"  Top rules: {settings['top_rules']}")
        print(
            f"  Statistical sort: "
            f"{'Yes' if settings['statistical_sort'] else 'No'}"
        )
    else:
        print(f"  Target rules: {settings['target_rules']}")
        print(f"  Rule length: {settings['min_len']}-{settings['max_len']}")

    proceed = get_yes_no(
        f"\n{Colors.YELLOW}Start processing with these settings?{Colors.END}", True
    )
    if proceed:
        return settings
    else:
        print_info("Configuration cancelled.")
        return None

# ==============================================================================
# MAIN
# ==============================================================================

def print_usage() -> None:
    print(f"{Colors.BOLD}{Colors.CYAN}USAGE:{Colors.END}")
    print(
        f"  {Colors.WHITE}python concentrator.py [OPTIONS] "
        f"FILE_OR_DIRECTORY [FILE_OR_DIRECTORY...]{Colors.END}"
    )
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}MODES (choose one):{Colors.END}")
    print(
        f"  {Colors.GREEN}-e, --extract-rules{Colors.END}     "
        f"Extract top existing rules from input files"
    )
    print(
        f"  {Colors.GREEN}-g, --generate-combo{Colors.END}    "
        f"Generate combinatorial rules from top operators"
    )
    print(
        f"  {Colors.GREEN}-gm, --generate-markov-rules{Colors.END} "
        f"Generate statistically probable Markov rules"
    )
    print(
        f"  {Colors.GREEN}-p, --process-rules{Colors.END}     "
        f"Interactive rule processing and minimization"
    )
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}EXTRACTION MODE (-e):{Colors.END}")
    print(
        f"  {Colors.YELLOW}-t, --top-rules INT{Colors.END}     "
        f"Number of top rules to extract (default: 10000)"
    )
    print(
        f"  {Colors.YELLOW}-s, --statistical-sort{Colors.END}  "
        f"Sort by statistical weight instead of frequency"
    )
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}COMBINATORIAL MODE (-g):{Colors.END}")
    print(
        f"  {Colors.YELLOW}-n, --combo-target INT{Colors.END}  "
        f"Target number of rules (default: 100000)"
    )
    print(
        f"  {Colors.YELLOW}-l, --combo-length MIN MAX{Colors.END} "
        f"Rule length range (default: 1 3)"
    )
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}MARKOV MODE (-gm):{Colors.END}")
    print(
        f"  {Colors.YELLOW}-gt, --generate-target INT{Colors.END} "
        f"Target rules (default: 10000)"
    )
    print(
        f"  {Colors.YELLOW}-ml, --markov-length MIN MAX{Colors.END} "
        f"Rule length range (default: 1 3)"
    )
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}PROCESSING MODE (-p):{Colors.END}")
    print(
        f"  {Colors.YELLOW}-d, --use-disk{Colors.END}         "
        f"Use disk for large datasets to save RAM"
    )
    print(
        f"  {Colors.YELLOW}-ld, --levenshtein-max-dist INT{Colors.END} "
        f"Max Levenshtein distance (default: 2)"
    )
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}OUTPUT OPTIONS:{Colors.END}")
    print(
        f"  {Colors.MAGENTA}-f, --output-format FORMAT{Colors.END} "
        f"Output format: line, expanded (default: line)"
    )
    print(
        f"  {Colors.MAGENTA}-ob, --output-base-name NAME{Colors.END} "
        f"Base name for output file"
    )
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}GLOBAL OPTIONS:{Colors.END}")
    print(
        f"  {Colors.MAGENTA}-m, --max-length INT{Colors.END}    "
        f"Maximum rule length to process (default: 31)"
    )
    print(
        f"  {Colors.MAGENTA}--temp-dir DIR{Colors.END}        "
        f"Temporary directory for file mode"
    )
    print(
        f"  {Colors.MAGENTA}--in-memory{Colors.END}           "
        f"Process entirely in RAM"
    )
    print(
        f"  {Colors.MAGENTA}--no-gpu{Colors.END}             "
        f"Disable GPU acceleration"
    )
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}COMPATIBILITY NOTE:{Colors.END}")
    print(
        f"  {Colors.WHITE}Memory (M, 4, 6, X, Q) and reject (<, >, !, /, (, ), =, %, Q){Colors.END}"
    )
    print(
        f"  {Colors.WHITE}rules are fully supported on both CPU and GPU paths "
        f"since hashcat 6.x.{Colors.END}"
    )
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}INTERACTIVE MODE:{Colors.END}")
    print(
        f"  {Colors.WHITE}python concentrator.py{Colors.END}   "
        f"(run without arguments for interactive mode)"
    )
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}EXAMPLES:{Colors.END}")
    print(f"  {Colors.WHITE}# Extract top 5000 rules with GPU acceleration{Colors.END}")
    print(f"  {Colors.WHITE}python concentrator.py -e -t 5000 rules/*.rule{Colors.END}")
    print()
    print(f"  {Colors.WHITE}# Generate 50k combinatorial rules{Colors.END}")
    print(
        f"  {Colors.WHITE}python concentrator.py -g -n 50000 "
        f"-l 2 4 hashcat/rules/{Colors.END}"
    )
    print()
    print(
        f"  {Colors.WHITE}# Process rules interactively with "
        f"functional minimization and expanded output{Colors.END}"
    )
    print(
        f"  {Colors.WHITE}python concentrator.py -p -d "
        f"-f expanded rules/{Colors.END}"
    )
    print()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    multiprocessing.freeze_support()
    print_banner()
    print_memory_status()
    mem_info = get_memory_usage()
    if mem_info and mem_info["ram_percent"] > 85:
        print_warning(f"High RAM usage detected ({mem_info['ram_percent']:.1f}%)")
        if mem_info["swap_total"] == 0:
            print_error(
                "CRITICAL: No swap space available. System may become unstable."
            )
            proceed = get_yes_no(
                f"{Colors.YELLOW}Continue anyway? (y/N): {Colors.END}",
                default=False,
            )
            if not proceed:
                sys.exit(1)
        else:
            print_warning("System will use swap space. Performance may be slower.")

    if len(sys.argv) == 1:
        # ---- interactive mode --------------------------------------------- #
        settings = interactive_mode()
        if not settings:
            sys.exit(0)

        class Args:
            def __init__(self, s: Dict[str, Any]) -> None:
                self.paths            = s["paths"]
                self.output_base_name = s["output_base_name"]
                self.max_length       = s["max_length"]
                self.no_gpu           = s["no_gpu"]
                self.in_memory        = s["in_memory"]
                self.temp_dir         = s["temp_dir"]
                self.output_format    = s["output_format"]
                self.extract_rules          = s["mode"] == "extraction"
                self.generate_combo         = s["mode"] == "combo"
                self.generate_markov_rules  = s["mode"] == "markov"
                self.process_rules          = False
                if self.extract_rules:
                    self.top_rules        = s["top_rules"]
                    self.statistical_sort = s["statistical_sort"]
                elif self.generate_combo:
                    self.combo_target  = s["target_rules"]
                    self.combo_length  = [s["min_len"], s["max_len"]]
                elif self.generate_markov_rules:
                    self.generate_target = s["target_rules"]
                    self.markov_length   = [s["min_len"], s["max_len"]]

        args = Args(settings)
        concentrator_main_processing(args)

    elif len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
        print_usage()
        sys.exit(0)

    else:
        # ---- CLI mode -------------------------------------------------------- #
        parser = argparse.ArgumentParser(
            description=(
                f"{Colors.CYAN}Unified Hashcat Rule Processor "
                f"with OpenCL support (hashcat 6.x compatible).{Colors.END}"
            ),
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
""",
        )
        parser.add_argument(
            "paths",
            nargs="+",
            help="Paths to rule files or directories (recursive, max depth 3)",
        )
        parser.add_argument(
            "-ob", "--output_base_name",
            type=str,
            default="concentrator_output",
            help="Base name for the output file.",
        )
        parser.add_argument(
            "-f", "--output-format",
            type=str,
            choices=["line", "expanded"],
            default="line",
            help="Output format: line or expanded.",
        )
        output_group = parser.add_mutually_exclusive_group(required=True)
        output_group.add_argument(
            "-e", "--extract_rules",
            action="store_true",
            help="Extract top existing rules from input files.",
        )
        parser.add_argument(
            "-t", "--top_rules",
            type=int,
            default=10000,
            help="Number of top rules to extract (used with -e).",
        )
        parser.add_argument(
            "-s", "--statistical_sort",
            action="store_true",
            help="Sort extracted rules by Markov sequence probability.",
        )
        output_group.add_argument(
            "-g", "--generate_combo",
            action="store_true",
            help="Generate combinatorial rules.",
        )
        parser.add_argument(
            "-n", "--combo_target",
            type=int,
            default=100000,
            help="Target number of rules in combinatorial mode (used with -g).",
        )
        parser.add_argument(
            "-l", "--combo_length",
            nargs="+",
            type=int,
            default=[1, 3],
            help="Rule length range for combinatorial mode (e.g., 1 3).",
        )
        output_group.add_argument(
            "-gm", "--generate_markov_rules",
            action="store_true",
            help="Generate statistically probable Markov rules.",
        )
        parser.add_argument(
            "-gt", "--generate_target",
            type=int,
            default=10000,
            help="Target number of rules in Markov mode (used with -gm).",
        )
        parser.add_argument(
            "-ml", "--markov_length",
            nargs="+",
            type=int,
            default=None,
            help="Rule length range for Markov mode (e.g., 1 5). Defaults to [1,3].",
        )
        output_group.add_argument(
            "-p", "--process_rules",
            action="store_true",
            help="Interactive rule processing and minimization.",
        )
        parser.add_argument(
            "-d", "--use_disk",
            action="store_true",
            help="Use disk for large datasets to save RAM.",
        )
        parser.add_argument(
            "-ld", "--levenshtein_max_dist",
            type=int,
            default=2,
            help="Maximum Levenshtein distance for similarity filtering.",
        )
        parser.add_argument(
            "-m", "--max_length",
            type=int,
            default=31,
            help="Maximum rule length to consider.",
        )
        parser.add_argument(
            "--temp-dir",
            type=str,
            default=None,
            help="Specify a directory for temporary files.",
        )
        parser.add_argument(
            "--in-memory",
            action="store_true",
            help="Process all rules entirely in RAM.",
        )
        parser.add_argument(
            "--no-gpu",
            action="store_true",
            help="Disable GPU acceleration.",
        )
        args = parser.parse_args()

        if args.markov_length is None:
            args.markov_length = [1, 3]

        if args.use_disk:
            args.in_memory = False
            print_info("Using disk mode for large datasets (--use-disk).")

        if args.process_rules:
            process_multiple_files_concentrator(args)
        else:
            concentrator_main_processing(args)

    cleanup_temp_files()
    sys.exit(0)

