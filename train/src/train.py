# Copyright 2025 the LlamaFactory team.
# ... (License header kept as is)

import os
import sys
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from llamafactory.train.tuner import run_exp

def main():
    
    cfg_path = None
    remaining_argv = []
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ("--cfg", "--config") and i + 1 < len(sys.argv):
            cfg_path = sys.argv[i + 1]
            i += 2
        else:
            remaining_argv.append(arg)
            i += 1

    if cfg_path is None:
        print("⚠️  No --cfg provided, assuming standard CLI usage.")
        run_exp()
        return

    
    print(f"✅ Loading config from: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    
    
    flat_args = []
    for key, value in cfg.items():
        
        if value is None:
            continue
            
        
        if isinstance(value, (list, tuple)):
            value = ",".join(map(str, value))
        
        
        if isinstance(value, bool):
            value = str(value)
            
        flat_args.append(f"--{key}={value}")

    print(f"✅ Injected {len(flat_args)} args from config.")

    
    
    
    sys.argv = [sys.argv[0]] + flat_args + remaining_argv
    
    
    print(f"🚀 Launching with sys.argv constructed.")
    
    run_exp()

if __name__ == "__main__":
    main()