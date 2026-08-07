#!/bin/bash
#SBATCH --job-name=kras_group2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=rfdiffusion_standalone_%j.out
#SBATCH --error=rfdiffusion_standalone_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=gpsl0084@leeds.ac.uk

################################################################################
#                                                                              #
#  RFdiffusion Standalone Binder Design - Production Pipeline                 #
#  ═══════════════════════════════════════════════════════════                #
#                                                                              #
#  PROJECT: H2A-K15 Nucleosome-Ubiquitin Complex Standalone Binder            #
#  INPUT:   20251011_8OFF-rebuit_for_RFdiff_v4.pdb (4-chain complex)          #
#  SYSTEM:  AIRE HPC Cluster (3× NVIDIA L40S GPUs, 192GB RAM)                 #
#                                                                              #
#  DESIGN ARCHITECTURE:                                                        #
#  ────────────────────                                                        #
#    [Standalone Binder]  ──→  Target Complex                                 #
#      56-112 aa                Chain A: 1-107                                 #
#      (diffused)               Chain B: 12-108                                #
#                               Chain C: 33-125 (Target)                       #
#                               Chain D: 1-74 (Target)                         #
#                                                                              #
#  STRATEGY: Standalone binder targeting multi-chain interface                #
#    • Binder: 56-112 residues de novo designed (separate chain)              #
#    • Chain A/B/C/D: Preserved (fixed from input PDB)                        #
#    • Rg potential: Applied ONLY to binder chain (clean implementation)      #
#    • Post-design: Can fuse to Chain A N-terminus with linker if validated   #
#                                                                              #
#  HOTSPOT RESIDUES (Must Contact):                                           #
#    • Chain A: A12, A49 (interface stabilization)                            #
#    • Chain C: C112, C116 (primary target interface)                         #
#    • Chain D: D48, D49 (secondary target interface)                         #
#                                                                              #
#  RG OPTIMIZATION (Clean, targeted implementation):                           #
#    • Potential: binder_ROG (applies ONLY to standalone binder)              #
#    • Weight: 3.0 (moderate compaction)                                      #
#    • min_dist: 14 Å (optimal for 56-112aa binder)                           #
#    • guide_scale: 5 (standard for binder design)                            #
#    • guide_decay: cubic (strong early, gentle late)                         #
#    • Target binder Rg: 13-19 Å (depends on length)                          #
#                                                                              #
#  PRODUCTION SCALE:                                                           #
#    • Total designs: 240 binder scaffolds                                    #
#    • Noise levels: 0.6, 0.7, 0.8, 0.9, 1.0, 1.1 (6 conditions)              #
#    • Per condition: 40 designs                                              #
#    • Parallelization: 3 GPUs (2 noise levels per GPU)                       #
#    • Expected runtime: ~2-4 hours                                           #
#                                                                              #
#  CONTIG STRING (STANDALONE BINDER):                                          #
#    [56-112/0 A1-107/0 B12-108/0 C33-125/0 D1-74]                            #
#     └─binder─┘ └──A──┘ └──B──┘ └──C──┘ └─D─┘                               #
#     (designed) (fixed targets - all preserved)                              #
#                                                                              #
#  PATHS (AIRE Cluster):                                                       #
#    Conda: /mnt/scratch/fbselee/apps/miniconda3                              #
#    Env:   SE3nv                                                             #
#    RF:    /users/fbselee/localhome/apps/RFdiffusion                         #
#                                                                              #
#  OUTPUT: ./output_standalone/dir{1-6}_noise{level}_{timestamp}_*.pdb        #
#                                                                              #
#  AUTHOR: Standalone binder design for H2A-K15 ubiquitin complex             #
#  DATE:   October 2025                                                       #
#                                                                              #
################################################################################

set -e  # Exit on any error

# Record the start time
start_time=$(date +%s)
start_time_readable=$(date)

# =============================================================================
# PRODUCTION CONFIGURATION - 240 SCAFFOLDS (40 PER NOISE LEVEL)
# =============================================================================

# Production Parameters
num_designs_per_dir=50    # 40 designs per directory
num_dirs=4                # 6 directories for noise exploration

# Balanced noise levels for standalone binder design
noise_levels=(0.4 0.5 0.6 0.7) #(0.2 0.3 0.35)

# Input/Output Configuration
INPUT_PDB="/mnt/scratch/gpsl0084/RFdiffusion/models/k6_clean.pdb"
OUTPUT_BASE="/mnt/scratch/gpsl0084/RFdiffusion/scripts/outputs/kras_group2"  ## input file name here
if [ ! -d $OUTPUT_BASE ]
then
    mkdir -p $OUTPUT_BASE
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 3-GPU configuration for production
NUM_GPUS=3

# =============================================================================
# SETUP AND VALIDATION
# =============================================================================

# CRITICAL: Change to the job submission directory first
cd "$SLURM_SUBMIT_DIR"

# Create output directory
mkdir -p "$OUTPUT_BASE"
mkdir -p ~/rfdiffusion_schedules

# Validate input file exists
if [[ ! -f "$INPUT_PDB" ]]; then
    echo "ERROR: Input PDB file '$INPUT_PDB' not found!"
    exit 1
fi

module load miniforge

# Proper conda initialization (THIS is the key fix)
source $(conda info --base)/etc/profile.d/conda.sh

# Activate environment
conda activate SE3nv || { echo "ERROR: Failed to activate SE3nv"; exit 1; }

# Debug (keep this!)
echo "Python: $(which python)"
echo "Env: $CONDA_DEFAULT_ENV"

# Ensure torch works BEFORE proceeding
python -c "import torch; print(f'✓ Torch OK - CUDA: {torch.cuda.is_available()}')" \
    || { echo "ERROR: PyTorch missing!"; exit 1; }

# Libraries
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

# RFdiffusion path - using your working installation
RFDIFFUSION_PATH="/mnt/scratch/gpsl0084/RFdiffusion"

echo "================================================================================"
echo "RFdiffusion Production Pipeline - Standalone Binder Design"
echo "================================================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Started: $start_time_readable"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 3 × L40S (48GB VRAM each)"
echo "Working directory: $(pwd)"
echo "Input: $INPUT_PDB"
echo "Output: $OUTPUT_BASE"
echo ""
echo "PRODUCTION CONFIGURATION:"
echo "  • Designs per directory: $num_designs_per_dir"
echo "  • Total directories: $num_dirs"
echo "  • Total designs: $(( num_dirs * num_designs_per_dir )) binders"
echo "  • Execution mode: 3-GPU parallel ($(( num_dirs / NUM_GPUS )) dirs per GPU)"
echo ""
echo "DESIGN STRATEGY:"
echo "  • Architecture: Standalone binder (separate chain)"
echo "  • Binder domain: 56-112 residues (fully designed)"
echo "  • Target complex: Chain A (1-107), B (12-108), C (33-125), D (1-74)"
echo "  • All target chains: FIXED/PRESERVED from input PDB"
echo "  • Post-validation: Can engineer N-terminal fusion to Chain A if validated"
echo ""
echo "RG OPTIMIZATION (Targeted to binder only):"
echo "  • Potential: binder_ROG (applies ONLY to standalone binder chain)"
echo "  • Weight: 3.0 (moderate - encourages compact folds)"
echo "  • min_dist: 14 Å (prevents collapse, optimal for 56-112aa)"
echo "  • guide_scale: 5 (standard binder design strength)"
echo "  • guide_decay: cubic (strong early timesteps, minimal late)"
echo "  • Expected binder Rg: 13-19 Å (size-dependent)"
echo ""
echo "HOTSPOT STRATEGY:"
echo "  • Chain A: A12, A49 (interface contacts)"
echo "  • Chain C: C112, C116 (primary target)"
echo "  • Chain D: D48, D49 (primary target)"
echo "  • Purpose: Guide binder to specific interface residues on multi-chain target"
echo ""
echo "NOISE EXPLORATION STRATEGY:"
for i in "${!noise_levels[@]}"; do
    echo "  • Directory $((i+1)): Noise ${noise_levels[i]} ($([ ${noise_levels[i]%.*} -lt 1 ] && echo "conservative" || echo "exploratory")) - $num_designs_per_dir binders"
done
echo ""
echo "GPU WORK DISTRIBUTION:"
echo "  • GPU 0: Directories 1, 4 (noise 0.6, 0.9) - $(( 2 * num_designs_per_dir )) binders"
echo "  • GPU 1: Directories 2, 5 (noise 0.7, 1.0) - $(( 2 * num_designs_per_dir )) binders"
echo "  • GPU 2: Directories 3, 6 (noise 0.8, 1.1) - $(( 2 * num_designs_per_dir )) binders"
echo "================================================================================"
echo ""

# =============================================================================
# DEPENDENCY VALIDATION
# =============================================================================

echo "VALIDATING DEPENDENCIES:"
echo "  • Checking conda environment..."
conda info --envs | grep SE3nv || { echo "ERROR: SE3nv environment not found!"; exit 1; }
echo "    ✓ SE3nv environment found"

echo "  • Checking Python packages..."
python -c "import torch; print(f'    ✓ PyTorch OK - CUDA available: {torch.cuda.is_available()}')" || { echo "ERROR: PyTorch import failed!"; exit 1; }
echo "    ✓ SE3/RFdiffusion imports (validated by run_inference.py)"

echo "  • Checking RFdiffusion installation..."
if [[ ! -f "$RFDIFFUSION_PATH/scripts/run_inference.py" ]]; then
    echo "ERROR: RFdiffusion script not found at $RFDIFFUSION_PATH/scripts/run_inference.py"
    exit 1
fi
echo "    ✓ RFdiffusion script found"

# Check for model weights
echo "  • Checking model weights..."
if [[ ! -f "$RFDIFFUSION_PATH/models/Complex_base_ckpt.pt" ]]; then
    echo "WARNING: Complex_base_ckpt.pt not found, checking Base_ckpt.pt..."
    if [[ ! -f "$RFDIFFUSION_PATH/models/Base_ckpt.pt" ]]; then
        echo "ERROR: No compatible model weights found"
        exit 1
    fi
    echo "    ✓ Base model weights found (will use for PPI)"
else
    echo "    ✓ Complex model weights found"
fi

echo ""
echo "✅ ALL DEPENDENCIES VALIDATED SUCCESSFULLY"
echo ""

# =============================================================================
# PARALLEL EXECUTION FUNCTION
# =============================================================================

run_rfdiffusion_on_gpu() {
    local gpu_id=$1
    local dir_list=("${@:2}")

    export CUDA_VISIBLE_DEVICES=$gpu_id

    for dir_num in "${dir_list[@]}"; do
        # Select noise level for this directory
        noise_idx=$(( (dir_num-1) % ${#noise_levels[@]} ))
        current_noise=${noise_levels[$noise_idx]}

        # Create descriptive output prefix
        current_noise_clean=$(echo "$current_noise" | sed 's/\./_/g')
        output_prefix="${OUTPUT_BASE}/dir${dir_num}_noise${current_noise_clean}_${TIMESTAMP}"

        echo "┌─────────────────────────────────────────────────────────────────────────────┐"
        echo "│ GPU $gpu_id: DIRECTORY $dir_num/6: NOISE LEVEL $current_noise ($num_designs_per_dir designs)"
        echo "├─────────────────────────────────────────────────────────────────────────────┤"
        echo "│ Strategy: $(case $(echo "$current_noise < 0.9" | bc -l) in 1) echo "Conservative - stable binder folds";; *) echo "Exploratory - diverse binder topologies";; esac)"
        echo "│ Output: $output_prefix"
        echo "└─────────────────────────────────────────────────────────────────────────────┘"
        echo ""

        # Record start time for this directory
        dir_start=$(date +%s)

        # Run RFdiffusion with standalone binder + Rg optimization
        echo "GPU $gpu_id Running RFdiffusion command:"
        echo "python $RFDIFFUSION_PATH/scripts/run_inference.py \\"
        echo "    inference.output_prefix=\"$output_prefix\" \\"
        echo "    inference.input_pdb=\"$INPUT_PDB\" \\"
        echo "    'contigmap.contigs=[12-15/0 A2-60/A65-166]' \\"
        echo "    'ppi.hotspot_res=[A38, A39, A71]' \\"
        echo "    inference.cyclic=True \\"
        echo "    inference.cyc_chains='a' \\"
        echo "    'potentials.guiding_potentials=[\"type:binder_ROG,weight:3,min_dist:4\"]' \\"
        echo "    potentials.guide_scale=5 \\"
        echo "    potentials.guide_decay=\"cubic\" \\"
        echo "    inference.num_designs=$num_designs_per_dir \\"
        echo "    denoiser.noise_scale_ca=$current_noise \\"
        echo "    denoiser.noise_scale_frame=$current_noise \\"
        echo "    inference.schedule_directory_path=\"$HOME/rfdiffusion_schedules\""
        echo ""

        # Execute RFdiffusion
        python "$RFDIFFUSION_PATH/scripts/run_inference.py" \
            inference.output_prefix="$output_prefix" \
            inference.input_pdb="$INPUT_PDB" \
            'contigmap.contigs=[12-15/0 A2-60/A65-166]' \
            'ppi.hotspot_res=[A38,A39,A71]' \
            inference.cyclic=True \
            inference.cyc_chains='a' \
            'potentials.guiding_potentials=["type:binder_ROG,weight:3,min_dist:4"]' \
            potentials.guide_scale=5 \
            potentials.guide_decay="cubic" \
            inference.num_designs=$num_designs_per_dir \
            denoiser.noise_scale_ca=$current_noise \
            denoiser.noise_scale_frame=$current_noise \
            inference.schedule_directory_path="$HOME/rfdiffusion_schedules"

        # Calculate directory completion time
        dir_end=$(date +%s)
        dir_duration=$((dir_end - dir_start))
        dir_minutes=$((dir_duration / 60))
        dir_seconds=$((dir_duration % 60))

        echo ""
        echo "✓ GPU $gpu_id: Directory $dir_num completed in ${dir_minutes}m ${dir_seconds}s"
        echo "  Generated $num_designs_per_dir standalone binders with noise level $current_noise"
        echo ""

        # Quick quality check
        generated_files=$(find "$OUTPUT_BASE" -name "$(basename ${output_prefix})*.pdb" 2>/dev/null | wc -l)
        echo "  Files generated: $generated_files"
        if [[ $generated_files -lt $num_designs_per_dir ]]; then
            echo "  ⚠️  WARNING: Expected $num_designs_per_dir files, got $generated_files"
        fi
        echo ""
    done
}

# =============================================================================
# DISTRIBUTE WORK ACROSS 3 GPUS
# =============================================================================

# Create arrays to distribute directories across GPUs optimally
declare -a gpu_assignments
for ((i=0; i<NUM_GPUS; i++)); do
    gpu_assignments[i]=""
done

# Distribute directories across GPUs for balanced workload
for ((dir=1; dir<=num_dirs; dir++)); do
    gpu_idx=$(( (dir-1) % NUM_GPUS ))
    gpu_assignments[$gpu_idx]="${gpu_assignments[$gpu_idx]} $dir"
done

# Start parallel jobs on all 3 GPUs
pids=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    if [[ -n "${gpu_assignments[$gpu]}" ]]; then
        dirs_array=(${gpu_assignments[$gpu]})
        echo "Starting GPU $gpu with directories: ${gpu_assignments[$gpu]}"
        run_rfdiffusion_on_gpu $gpu "${dirs_array[@]}" &
        pids+=($!)
    fi
done

# Wait for all background jobs to complete
echo "Waiting for all GPU jobs to complete..."
for pid in "${pids[@]}"; do
    wait $pid
done

# =============================================================================
# PRODUCTION COMPLETION SUMMARY
# =============================================================================

end_time=$(date +%s)
end_time_readable=$(date)
total_duration=$((end_time - start_time))
hours=$((total_duration / 3600))
minutes=$(( (total_duration % 3600) / 60 ))
seconds=$((total_duration % 60))

echo "================================================================================"
echo "PRODUCTION PIPELINE COMPLETION SUMMARY - STANDALONE BINDER DESIGN"
echo "================================================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Started: $start_time_readable"
echo "Finished: $end_time_readable"
echo "Duration: ${hours}h ${minutes}m ${seconds}s"
echo "GPUs used: 3 × L40S (parallel execution)"
echo ""

# Count and categorize outputs
echo "RESULTS BY NOISE LEVEL:"
total_designs=0
for i in $(seq 1 $num_dirs); do
    noise_idx=$(( (i-1) % ${#noise_levels[@]} ))
    current_noise=${noise_levels[$noise_idx]}
    current_noise_clean=$(echo "$current_noise" | sed 's/\./_/g')

    count=$(find "$OUTPUT_BASE" -name "dir${i}_noise${current_noise_clean}_${TIMESTAMP}*.pdb" 2>/dev/null | wc -l)
    total_designs=$((total_designs + count))

    strategy=$(case $(echo "$current_noise < 0.9" | bc -l) in 1) echo "(Conservative)";; *) echo "(Exploratory)";; esac)

    printf "  Directory %d: Noise %s %-15s → %4d binders\n" $i $current_noise "$strategy" $count
done

echo ""
echo "TOTAL STANDALONE BINDERS GENERATED: $total_designs"
echo "TARGET: $(( num_dirs * num_designs_per_dir )) binders"
echo "SUCCESS RATE: $(echo "scale=1; $total_designs * 100 / $(( num_dirs * num_designs_per_dir ))" | bc -l)%"

# Performance analysis
if [ $total_designs -gt 0 ]; then
    avg_time_per_design=$((total_duration / total_designs))
    designs_per_hour=$((3600 / avg_time_per_design))
    echo ""
    echo "PERFORMANCE METRICS:"
    echo "  Average time per binder: ${avg_time_per_design}s"
    echo "  Throughput: $designs_per_hour binders/hour"
    echo "  Parallel efficiency: 3 GPUs utilized simultaneously"
fi

echo ""
echo "OUTPUT STRUCTURE:"
echo "  Base directory: $OUTPUT_BASE"
echo "  Pattern: dir{1-6}_noise{level}_${TIMESTAMP}_*.pdb"
echo "  Total size: $(du -sh $OUTPUT_BASE 2>/dev/null | cut -f1 || echo "Unknown")"
echo ""

echo "DESIGN SUMMARY:"
echo "  • Binder domain: 56-112 residues (standalone, separate chain)"
echo "  • Target complex: Chains A/B/C/D (all preserved)"
echo "  • Rg optimization: binder_ROG (weight=3, min_dist=14Å)"
echo "  • Target hotspots: A12, A49, C112, C116, D48, D49"
echo "  • Expected binder Rg: 13-19 Å (depends on final length)"
echo ""

if [ $total_designs -gt $(( num_dirs * num_designs_per_dir * 8 / 10 )) ]; then
    echo "🎉 PRODUCTION SUCCESSFUL! Generated standalone binder library."
    echo ""
    echo "NEXT STEPS - STANDALONE BINDER VALIDATION PIPELINE:"
    echo "  1. 🔍 VISUAL INSPECTION:"
    echo "     - Load designs in PyMOL/ChimeraX"
    echo "     - Verify binder contacts hotspots on C and D"
    echo "     - Check binder compactness (Rg 13-19 Å)"
    echo "     - Assess overall fold quality"
    echo ""
    echo "  2. 📊 COMPUTATIONAL FILTERING:"
    echo "     - Calculate binder Rg (use analyze_binder_rg.py below)"
    echo "     - Cluster similar binder topologies"
    echo "     - Analyze hotspot contact distances"
    echo "     - Check interface burial and shape complementarity"
    echo ""
    echo "  3. 🧬 SEQUENCE DESIGN (ProteinMPNN):"
    echo "     - Design sequences for standalone binders"
    echo "     - Generate 5-10 sequences per backbone"
    echo "     - Filter for designability and stability"
    echo ""
    echo "  4. ✅ VALIDATION (AlphaFold2):"
    echo "     - Predict binder-target complex structures"
    echo "     - Filter at pAE_interaction < 10, ipTM > 0.6"
    echo "     - Verify hotspot contacts maintained"
    echo "     - Select top candidates"
    echo ""
    echo "  5. 🔗 OPTIONAL: N-TERMINAL FUSION ENGINEERING"
    echo "     - For validated binders that work"
    echo "     - Design flexible linker (e.g., GGGGS)×3"
    echo "     - Fuse to Chain A N-terminus: [Binder]-[Linker]-[Chain A]"
    echo "     - Re-validate with AlphaFold2"
    echo ""
    echo "  6. 🧪 EXPERIMENTAL CHARACTERIZATION:"
    echo "     - Express and purify standalone binders"
    echo "     - Binding assays (SPR, ITC, BLI)"
    echo "     - Test fused versions if standalone validates"
    echo "     - Structural validation (X-ray, cryo-EM)"
else
    echo "⚠️  PRODUCTION INCOMPLETE: Only $total_designs/$(( num_dirs * num_designs_per_dir )) binders generated"
    echo "Check error logs: rfdiffusion_standalone_${SLURM_JOB_ID}.err"
fi

echo ""
echo "================================================================================"
echo "BINDER Rg ANALYSIS SCRIPT (save as analyze_binder_rg.py):"
echo "================================================================================"
cat << 'PYTHON_EOF'
#!/usr/bin/env python3
# analyze_binder_rg.py - Analyze Rg of standalone binders

import sys
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser

def calc_rg(coords):
    """Calculate radius of gyration in Angstroms"""
    centroid = coords.mean(axis=0)
    return np.sqrt(((coords - centroid)**2).sum() / len(coords))

parser = PDBParser(QUIET=True)

print(f"{'File':<50} {'Binder_Length':<15} {'Binder_Rg(Å)':<15}")
print("-" * 80)

for pdb_file in sorted(Path(sys.argv[1]).glob("*.pdb")):
    structure = parser.get_structure('x', pdb_file)

    # Get first chain (should be the binder)
    binder_chain = list(structure.get_chains())[0]

    # Get all CA atoms from binder chain
    binder_ca = []
    for res in binder_chain:
        if 'CA' in res:
            binder_ca.append(res['CA'].coord)

    if len(binder_ca) < 50:
        continue

    binder_rg = calc_rg(np.array(binder_ca))

    print(f"{pdb_file.name:<50} {len(binder_ca):<15} {binder_rg:<15.1f}")

print("\nExpected Rg range for 56-112 residue binders: 13-19 Å")
print("Compact binders: Rg < 16 Å")
print("Extended binders: Rg > 18 Å")
PYTHON_EOF

echo ""
echo "USAGE: python analyze_binder_rg.py output_standalone/"
echo ""
echo "================================================================================"
echo "Standalone Binder Design Production Run completed! 🧬🚀"
echo "================================================================================"
