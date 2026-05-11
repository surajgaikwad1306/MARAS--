"""
main.py
-------
Entry point for the MARAS project.

Run with:
    python main.py

Options you can toggle at the top:
  VERBOSE_SIMS  — print round-by-round trace for each strategy
  SHOW_TRACES   — print detailed allocation trace after comparison table
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from comparison import run_comparison, print_allocation_trace

# ── Config ────────────────────────────────────
VERBOSE_SIMS = True    # set False for cleaner output (just comparison table)
SHOW_TRACES  = False   # set True for per-round trace after the summary table
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "█" * 58)
    print("  MARAS — Multi-Agent Resource Allocation System")
    print("  Adversarial Search + AI Planning")
    print("█" * 58)

    print("""
  Problem Instance
  ─────────────────────────────────────────────────
  Tasks : T1(v=10,cpu)  T2(v=7,balanced)  T3(v=4,memory)
          T4(v=9,cpu)   T5(v=3,balanced)

  Agents & Goal Plans (planning layer):
    A1  [CPU-heavy]   → goal plan: T1, T4, T2
    A2  [Mem-heavy]   → goal plan: T4, T1, T3
    A3  [Balanced]    → goal plan: T2, T3, T5
  ─────────────────────────────────────────────────
""")

    results = run_comparison(verbose_sims=VERBOSE_SIMS)

    if SHOW_TRACES:
        for r in results:
            print_allocation_trace(r)

    print("\n  Done. See comparison table above for full metrics.\n")
