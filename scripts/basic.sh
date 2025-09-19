#!/bin/bash

# Define benchmark list to run (note array syntax)
benchmarks=("Livermore" "Nguyen" "Jin" "NguyenC")

# Define output directory and config file
log_dir="basic_logs"

# Create log directory if not exists
mkdir -p "${log_dir}"

for benchmark in "${benchmarks[@]}"; do
    # Dynamically select config file
    case $benchmark in
        "Jin"|"NguyenC")
            config_file="constant_basic.yaml"
            ;;
        *)
            config_file="basic.yaml"
            ;;
    esac

    # Generate job submission content
    sbatch <<EOT
#!/bin/bash
#SBATCH -J ${benchmark}_Job
#SBATCH -o ${log_dir}/${benchmark}_%j.out  # Output to specified directory
#SBATCH --cpus-per-task=20
#SBATCH -N 1
#SBATCH -t 120:00:00

module purge
module load conda
source activate gomoku

python benchmark_runner.py --benchmark ${benchmark} --config configs/${config_file}

EOT

    echo "Submitted ${benchmark} job using config file ${config_file}"
    sleep 0.5
done