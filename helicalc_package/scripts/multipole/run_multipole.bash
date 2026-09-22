#!/bin/bash
# Calculate the multipole assembly field across all 4 GPUs, splitting by bar.
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate helicalc

helicalc_data=$(python ../get_data_dir.py)
outdir="${helicalc_data}/Bmaps/multipole/"
mkdir -p "${outdir}logs"

region="${1:-body}"
geom="${2:-Multipole_HLLHC_V1}"
ngpu=4

echo "region=${region} geom=${geom} across ${ngpu} GPUs"
for d in $(seq 0 $((ngpu-1))); do
    python calculate_multipole_grid.py -r "${region}" -g "${geom}" \
        -D "${d}" -N "${ngpu}" --log &
done
wait

python sum_multipole_partials.py "${geom}.${region}_region.GPU*_of_${ngpu}.pkl"
