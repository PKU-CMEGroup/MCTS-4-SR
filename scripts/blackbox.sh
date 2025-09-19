#!/bin/bash

# Define log directory and create it
LOG_DIR="blackbox_logs"
mkdir -p "${LOG_DIR}"

# Job parameter configuration
CASES_PER_JOB=10
TOTAL_CASES=122
TOTAL_JOBS=$(( (TOTAL_CASES + CASES_PER_JOB - 1) / CASES_PER_JOB ))

for (( job_id=1; job_id<=TOTAL_JOBS; job_id++ )); do
    # Calculate case range
    start_case=$(( (job_id - 1) * CASES_PER_JOB + 1 ))
    end_case=$(( start_case + CASES_PER_JOB - 1 ))
    
    # Handle last job overflow
    (( end_case > TOTAL_CASES )) && end_case=$TOTAL_CASES

    # Generate temporary config
    config_file="configs/blackbox_${start_case}_${end_case}.yaml"
    cp configs/blackbox.yaml "$config_file"
    sed -i "s/start_case:.*/start_case: $start_case/; s/end_case:.*/end_case: $end_case/" "$config_file"

    # Calculate CPU requirements
    run_num=$(grep 'run_num:' "$config_file" | awk '{print $2}')
    cpus=$run_num  # Adjust according to actual formula

    # Submit job with EOT aligned
    sbatch <<EOT
#!/bin/bash
#SBATCH -J BLACKBOX_${job_id}
#SBATCH -o ${LOG_DIR}/BLACKBOX_${job_id}_%j.out
#SBATCH --cpus-per-task=20
#SBATCH -N 1
#SBATCH -t 120:00:00

module purge
module load conda
source activate gomoku

# Execution with error handling
if ! python benchmark_runner.py --benchmark BlackBox --config "$config_file"; then
    echo "[ERROR] Job ${job_id} failed | Config file retained: $config_file" >&2
    exit 1
fi

# Cleanup temporary files
rm "$config_file"
EOT

    echo "Submitted job ${job_id}: Cases ${start_case}-${end_case} | CPU=${cpus}"
    sleep 1 # Optimize submission interval
done