#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from dynport.data import load_returns, save_wide_returns

def main():
    p = argparse.ArgumentParser(description="Convert long price CSV or wide returns CSV into canonical wide returns format.")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--start_date", default=None); p.add_argument("--end_date", default=None)
    args = p.parse_args()
    dates, assets, returns = load_returns(args.input, start_date=args.start_date, end_date=args.end_date)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_wide_returns(args.output, dates, assets, returns)
    print(f"Wrote {args.output}: {len(dates)} dates, {len(assets)} assets")
if __name__ == "__main__": main()
