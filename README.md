# CONCENTRATOR v3.4

> **Unified Hashcat Rule Processor** — Extract, generate, and process hashcat password rules with GPU acceleration, Markov chain modeling, and functional minimization.

---

## Features

- **OpenCL GPU Acceleration** — Batch rule validation offloaded to GPU for high throughput
- **Three Processing Modes** — Extraction, Combinatorial generation, and Markov-based generation
- **Hashcat Rule Engine Simulation** — Full CPU-side implementation of hashcat's rule operators
- **Functional Minimization** — Deduplicate rules that produce identical outputs across a shared probe-word vector
- **Memory Safety** — Monitors RAM/swap usage with configurable thresholds and disk-spill mode
- **Multiple Output Formats** — `line` (compact) or `expanded` (operator + args separated by spaces)
- **Interactive & CLI Modes** — Guided wizard or full argument-driven usage

---

## Requirements

### Python
Python 3.8+

### Core (standard library — no install needed)
`sys`, `os`, `re`, `argparse`, `math`, `itertools`, `multiprocessing`, `tempfile`, `random`, `datetime`, `threading`, `collections`, `typing`

### Optional (install for full functionality)

| Package | Purpose | Install |
|---|---|---|
| `pyopencl` | GPU-accelerated rule validation | `pip install pyopencl` |
| `numpy` | Array operations for GPU buffers | `pip install numpy` |
| `tqdm` | Progress bars | `pip install tqdm` |
| `psutil` | RAM/swap monitoring | `pip install psutil` |

> All optional packages degrade gracefully — the tool runs on pure Python if none are installed.

---

## Installation

```bash
git clone https://github.com/youruser/concentrator.git
cd concentrator
pip install pyopencl numpy tqdm psutil   # optional but recommended
```

---

## Usage

### Interactive Mode
Run without arguments to launch the guided wizard:
```bash
python concentrator.py
```

### CLI Mode

```
python concentrator.py [OPTIONS] FILE_OR_DIRECTORY [FILE_OR_DIRECTORY ...]
```

One mode flag is **required**.

---

## Modes

### `-e` / `--extract-rules` — Extraction Mode
Extract the most frequent (or statistically weighted) rules from existing rule files.

```bash
# Extract top 5000 rules by frequency
python concentrator.py -e -t 5000 rules/

# Extract top 10000 rules sorted by Markov sequence probability
python concentrator.py -e -t 10000 -s rules/*.rule
```

| Flag | Default | Description |
|---|---|---|
| `-t`, `--top-rules` | `10000` | Number of top rules to extract |
| `-s`, `--statistical-sort` | off | Sort by Markov probability instead of raw frequency |

---

### `-g` / `--generate-combo` — Combinatorial Mode
Generate rules by exhaustively combining the most common operators up to a target count.

```bash
# Generate 50k rules using operator combinations of length 2–4
python concentrator.py -g -n 50000 -l 2 4 hashcat/rules/
```

| Flag | Default | Description |
|---|---|---|
| `-n`, `--combo-target` | `100000` | Target number of rules to generate |
| `-l`, `--combo-length` | `1 3` | Min and max operator-chain length |

---

### `-gm` / `--generate-markov-rules` — Markov Mode
Generate statistically probable rules using a second-order token-level Markov model trained on the input rule files.

Each walk samples a target length **uniformly** from `[min, max]`, producing an even distribution of short and long rules across the full range. The length breakdown is printed after generation.

```bash
# Generate 10k Markov rules of length 1–5
python concentrator.py -gm -gt 10000 -ml 1 5 hashcat/rules/

# Generate 25k rules covering lengths 1–6
python concentrator.py -gm -gt 25000 -ml 1 6 hashcat/rules/
```

| Flag | Default | Description |
|---|---|---|
| `-gt`, `--generate-target` | `10000` | Target number of rules to generate |
| `-ml`, `--markov-length` | `1 3` | Min and max rule length (in tokens/operators) |

---

### `-p` / `--process-rules` — Processing Mode
Load, validate, deduplicate, and functionally minimize existing rule sets interactively.

```bash
# Process rules using disk mode to avoid RAM exhaustion
python concentrator.py -p -d rules/
```

| Flag | Default | Description |
|---|---|---|
| `-d`, `--use-disk` | off | Spill to disk instead of keeping everything in RAM |

---

## Interactive Processing Menu

After loading or generating rules, an interactive menu is available with these options:

| Key | Action |
|---|---|
| `1` | Filter by minimum occurrence count |
| `2` | Keep top N rules |
| `3` | Functional redundancy filter (RAM-intensive) |
| `4` | Inverse mode — keep rules *below* a rank cutoff |
| `5` | Hashcat cleanup — validate against CPU or GPU rule syntax |
| `6` | Toggle output format (`line` ↔ `expanded`) |
| `p` | Pareto analysis |
| `s` | Save current ruleset |
| `r` | Reset to original dataset |
| `i` | Dataset information |
| `q` | Quit |

---

## Global Options

| Flag | Default | Description |
|---|---|---|
| `-ob`, `--output-base-name` | `concentrator_output` | Base filename for output (no extension) |
| `-f`, `--output-format` | `line` | Output format: `line` or `expanded` |
| `-m`, `--max-length` | `31` | Maximum rule token length to process |
| `--temp-dir` | system default | Directory to write temporary files |
| `--in-memory` | off | Process entirely in RAM (overrides disk mode) |
| `--no-gpu` | off | Disable OpenCL GPU acceleration |

---

## Output Formats

**`line`** — Standard hashcat rule format, one rule per line:
```
i3li4ei5y
li6po7io8e
D2i41o50
D1$9$6$0
o6g$1$5
D1i3hi4o
```

**`expanded`** — Each operator and its arguments separated by spaces, one rule per line:
```
i3l i4e i5y
l i6p o7i o8e
D2 i41 o50
D1 $9 $6 $0
o6g $1 $5
D1 i3h i4o
```

---

## Input Files

Concentrator recursively scans directories up to **3 levels deep** for files with these extensions:

`.rule` `.rules` `.hr` `.hashcat` `.txt` `.lst`

You can pass individual files, directories, or a mix of both.

---

## Filtered Operators

The following operators are filtered at **every pipeline stage** and will never appear in output:

| Category | Operators |
|---|---|
| Memory | `M` `4` `6` `X` |
| Reject / Conditional | `<` `>` `!` `/` `(` `)` `=` `%` `Q` |

---

## Supported Hashcat Rule Operators

Concentrator validates and simulates the full hashcat rule operator set, including:

| Category | Operators |
|---|---|
| Case | `l` `u` `c` `C` `t` `T` `E` `e` |
| Reverse / Duplicate | `r` `d` `f` `p` `q` |
| Rotation | `{` `}` |
| Trim | `[` `]` `D` `x` `O` `'` |
| Insert / Overwrite | `i` `o` `^` `$` |
| Substitute / Delete | `s` `@` `.` `,` |
| Extend | `z` `Z` `y` `Y` |
| Arithmetic | `+` `-` `L` `R` |
| Swap | `k` `K` `*` |
| Leet / Separator | `3` |
| Misc | `:` `_` |

---

## Examples

```bash
# Extract top 5000 rules (GPU off) from a glob
python concentrator.py -e -t 5000 --no-gpu rules/*.rule

# Generate 100k combinatorial rules, output in expanded format
python concentrator.py -g -n 100000 -l 1 3 -f expanded hashcat/rules/ -ob my_rules

# Markov generation across a wide length range
python concentrator.py -gm -gt 25000 -ml 1 6 hashcat/rules/

# Process and minimize rules, writing temp files to /tmp/scratch
python concentrator.py -p -d --temp-dir /tmp/scratch rules/

# Interactive mode
python concentrator.py
```

---

## Memory Considerations

- At startup, Concentrator prints current RAM and swap usage.
- If RAM usage exceeds **85%**, a warning is raised and you are prompted before continuing.
- Use `--use-disk` / `-d` with `-p` mode to spill intermediate data to disk.
- Use `--in-memory` to force full in-RAM processing (fastest, but watch your available memory).
- Functional minimization uses a SQLite-backed path automatically for rulesets exceeding 1 million rules to prevent OOM.
- Install `psutil` to enable memory monitoring.

---

## License

MIT License. See `LICENSE` for details.

## Credits

- Penguinkeeper for testing rules
- https://github.com/0xVavaldi/ruleprocessorY
- https://github.com/synacktiv/rulesfinder
- https://github.com/mkb2091/PyRuleEngine/blob/master/PyRuleEngine.py
- https://github.com/hashcat/hashcat-utils/blob/master/src/cleanup-rules.c
