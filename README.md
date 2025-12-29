**Concentrator v3.0: Unified Hashcat Rule Processor**

Concentrator v3.0 is an advanced, high-performance tool written in Python 3 designed to unify the processes of extracting, validating, cleaning, and generating highly effective Hashcat rulesets. It features multi-processing for parallel file ingestion and optional OpenCL (GPU) acceleration for massive-scale rule validation and filtering.

✨ **Key Features**

- Three Processing Modes: Extract top-performing rules, generate combinatorial rule sets, or generate Markov-chain based rules.

- OpenCL Acceleration: Optional GPU-backed processing for rule validation, providing significant speed improvements over CPU-only methods for large datasets.

- Hashcat Engine Simulation: Includes a built-in Python simulation of the Hashcat rule engine for functional testing and minimization (preventing functionally duplicate rules).

- Memory Safety: Features proactive memory usage monitoring (psutil) to warn users before performing memory-intensive operations, preventing system instability.

- Advanced Filtering: Supports complex cleanup and deduplication strategies post-generation, including Levenshtein distance filtering.

- Interactive and CLI Modes: Supports full command-line arguments as well as a user-friendly, colorized interactive setup mode.

🚀 **Getting Started**

Prerequisites

Concentrator v3.0 requires Python 3.8 or higher. For full functionality, including GPU acceleration and advanced monitoring, the following dependencies are required.

We recommend installing them within a virtual environment:

```# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install tqdm psutil numpy

# Install OpenCL dependencies (Note: pyopencl installation may require system-level OpenCL drivers)
pip install pyopencl
```


***Installation***

Clone the repository and run the script directly:

```
git clone https://github.com/A113L/concentrator.git
cd concentrator
python3 concentrator-v3.py --help
```

⚙️ **Usage**

```
python concentrator-v3.py -h

================================================================================
          CONCENTRATOR v3.0 - Unified Hashcat Rule Processor
================================================================================
Combined Features:
  • OpenCL GPU Acceleration for validation and generation
  • Three Processing Modes: Extraction, Combinatorial, Markov
  • Hashcat Rule Engine Simulation & Functional Minimization
  • Rule Validation and Cleanup (CPU/GPU compatible)
  • Levenshtein Distance Filtering
  • Smart Processing Selection & Memory Safety
  • Interactive & CLI Modes with Colorized Output
================================================================================

Memory Status: RAM 42.4% (6.49 GB/15.42 GB) | SWAP: 6.4% (1.39 GB/21.91 GB)
USAGE:
  python concentrator.py [OPTIONS] FILE_OR_DIRECTORY [FILE_OR_DIRECTORY...]

MODES (choose one):
  -e, --extract-rules     Extract top existing rules from input files
  -g, --generate-combo    Generate combinatorial rules from top operators
  -gm, --generate-markov-rules Generate statistically probable Markov rules
  -p, --process-rules     Interactive rule processing and minimization

EXTRACTION MODE (-e):
  -t, --top-rules INT     Number of top rules to extract (default: 10000)
  -s, --statistical-sort  Sort by statistical weight instead of frequency

COMBINATORIAL MODE (-g):
  -n, --combo-target INT  Target number of rules (default: 100000)
  -l, --combo-length MIN MAX Rule length range (default: 1 3)

MARKOV MODE (-gm):
  -gt, --generate-target INT Target rules (default: 10000)
  -ml, --markov-length MIN MAX Rule length range (default: 1 3)

PROCESSING MODE (-p):
  -d, --use-disk         Use disk for large datasets to save RAM
  -ld, --levenshtein-max-dist INT Max Levenshtein distance (default: 2)

GLOBAL OPTIONS:
  -ob, --output-base-name NAME Base name for output file
  -m, --max-length INT    Maximum rule length to process (default: 31)
  --temp-dir DIR        Temporary directory for file mode
  --in-memory           Process entirely in RAM
  --no-gpu             Disable GPU acceleration

INTERACTIVE MODE:
  python concentrator.py   (run without arguments for interactive mode)

EXAMPLES:
  # Extract top 5000 rules with GPU acceleration
  python concentrator.py -e -t 5000 --no-gpu rules/*.rule

  # Generate 50k combinatorial rules
  python concentrator.py -g -n 50000 -l 2 4 hashcat/rules/

  # Process rules interactively with functional minimization
  python concentrator.py -p -d rules/
```

🧠 **Architecture Overview**

The Concentrator system operates in several key phases:

**File Ingestion:** Input paths are recursively searched, and files are streamed or read into memory (depending on the --in-memory flag).

**Preprocessing & Validation:** Initial rules are filtered for basic Hashcat syntax using multi-processing.

**Generation/Extraction:** The selected mode (Extraction, Combo, or Markov) generates a massive candidate set of rules.

**Functional Minimization:** Candidates are passed through the Python-implemented RuleEngine to ensure they produce unique output for common test strings, reducing redundancy.

**GPU Validation (Optional):** If OpenCL is enabled, the final candidate rules are batched and sent to the GPU for highly optimized validation against character set constraints and length limits.

**Final Output:** Cleaned, unique, and validated rules are written to the output file.

⚠️ **Memory and Safety**

The tool is designed to handle very large rule files (billions of rules). It includes a robust memory monitoring system (check_memory_safety and memory_intensive_operation_warning) that leverages psutil. If RAM + Swap usage exceeds a safety threshold (default 85%) before a major operation (like functional minimization), the user is warned and asked to confirm continuation to prevent system lockups.

🛠️**OpenCL Integration**

The OpenCL portion requires pyopencl and is currently implemented for the final-stage validation (validate_rules_batch kernel). This offloads basic rule integrity checks to the GPU, making the overall process highly scalable. Ensure your system has the correct vendor drivers (NVIDIA, AMD, or Intel) installed for OpenCL support.

📝 License

Distributed under the MIT License. See LICENSE for more information.

**Website**

https://hcrt.pages.dev/concentrator.static_workflow

[![concentrator1.png](https://i.postimg.cc/dtmTPxmJ/concentrator1.png)](https://postimg.cc/nCrLqTPW)

**Credits**

- Penguinkeeper for testing rules
- https://github.com/0xVavaldi/ruleprocessorY
- https://github.com/synacktiv/rulesfinder
- https://github.com/mkb2091/PyRuleEngine/blob/master/PyRuleEngine.py
- https://github.com/hashcat/hashcat-utils/blob/master/src/cleanup-rules.c
