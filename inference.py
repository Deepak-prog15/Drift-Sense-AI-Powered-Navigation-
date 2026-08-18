import argparse
import sys
import os

try:
    from code.localize_dram import localize as localize_dram
    from code.localize_secondary import localize as localize_secondary
except ImportError:
    print("Error: Could not import localize_dram and localize_secondary from code/ folder.", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Unified Localizer")
    parser.add_argument("--ref", required=True, help="Path to reference image")
    parser.add_argument("--search", required=True, help="Path to search image")
    
    args = parser.parse_args()
    
    res_dram = None
    res_secondary = None

    try:
        res_dram = localize_dram(args.ref, args.search)
    except Exception as e:
        pass
        
    try:
        res_secondary = localize_secondary(args.ref, args.search)
    except Exception as e:
        pass
        
    if res_dram is None and res_secondary is None:
        print("0.0, 0.0")
        sys.exit(1)
        
    # Decision Logic
    if res_dram and not res_secondary:
        best_res = res_dram
    elif res_secondary and not res_dram:
        best_res = res_secondary
    else:
        if res_dram.ambiguous and not res_secondary.ambiguous:
            best_res = res_secondary
        elif res_secondary.ambiguous and not res_dram.ambiguous:
            best_res = res_dram
        else:
            if res_dram.score >= res_secondary.score:
                best_res = res_dram
            else:
                best_res = res_secondary
                
    # Output exact coordinates
    print(f"{best_res.x:.4f}, {best_res.y:.4f}")

if __name__ == "__main__":
    main()
