#!/bin/bash
# Job name:
#SBATCH --job-name=recover_iter_2_unload
#
# Account:
#SBATCH --account=fc_humnet
#
# Partition:
#SBATCH --partition=savio3
#
# Request one node:
#SBATCH --nodes=1
# Number of tasks per node:
#SBATCH --ntasks-per-node=1
# Cpus prt task
#SBATCH --cpus-per-task=20
#
#SBATCH --mem=200G
# Wall clock limit:
#SBATCH --time=12:00:00
#
# Email notification:
#SBATCH --mail-type=END,
#SBATCH --mail-user=aamaduzzi@berkeley.edu
## Command(s) to run (example):
# In your SLURM script
module load gcc/11.4.0 python/3.11.6-gcc-11.4.0

# Create and activate virtual environment
if [ ! -d "$HOME/myenv" ]; then
    python -m venv $HOME/myenv
fi
source $HOME/myenv/bin/activate

# Install compatible polars version
pip uninstall -y polars
pip install polars-lts-cpu
pip install numpy
pip install pandas
pip install datetime
pip install pyarrow

export ARROW_DEFAULT_MEMORY_POOL=system
export POLARS_MAX_THREADS=4
export POLARS_MEMORY_POOL_SIZE=100000000000  # 100GB

cat > memory_wrapper.py << 'EOF'
#!/usr/bin/env python3
import os
import gc
import sys

# Patch memory allocation in polars/pyarrow
os.environ["ARROW_DEFAULT_MEMORY_POOL"] = "system"
os.environ["POLARS_MAX_THREADS"] = "4"

# Set up garbage collection
gc.enable()
gc.set_threshold(100, 5, 5)

# Import and run the main script
try:
    import compute_unload_curve_savio
    print("Script completed successfully")
except MemoryError:
    print("Memory error encountered")
    gc.collect()
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(2)
EOF

# Run your script
python compute_unload_curve_savio.py