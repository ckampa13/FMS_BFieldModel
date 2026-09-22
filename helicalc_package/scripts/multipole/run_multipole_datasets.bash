#!/bin/bash
# Regenerate every dataset for one multipole geometry, on all 4 GPUs.
#
#   ./run_multipole_datasets.bash [geom] [dz] [dxy]
#   ./run_multipole_datasets.bash Multipole_HLLHC_V1_saddle 0.020 0.020
#
# Produces, under $helicalc_data/Bmaps/multipole/ :
#   <geom>.measurement_region.summed.pkl   propeller sampling -- what the fit uses
#   <geom>.map_region.summed.pkl           cartesian test sample (z offset by dz/2)
#   <geom>.rings_centres_r50mm.pkl         harmonic rings at each element centre
#   <geom>.rings_centres_r50mm.harmonics.csv
#
# Field is in TESLA and cartesian only; the downstream repo converts to gauss
# and adds R/Phi/Br/Bphi.
#
# The map grid is offset by half a z step on purpose: without it the propeller's
# phi = 0/90/180/270 arms at r = 0/20/40/60 mm coincide with a 20 mm cartesian
# grid at the same z, and 45% of the test sample is locations the fit trained on.
set -euo pipefail

source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate helicalc

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEOM="${1:-Multipole_HLLHC_V1_saddle}"
DZ="${2:-0.020}"
DXY="${3:-0.020}"
NGPU=4

helicalc_data=$(python "$HERE/../get_data_dir.py")
outdir="${helicalc_data}/Bmaps/multipole/"
logdir="${outdir}logs/"
mkdir -p "$logdir"
stamp=$(date +"%Y-%m-%d_%H%M%S")

run_region () {           # $1 = region name, $2... = extra flags
    local region="$1"; shift
    echo "=== ${region} : ${NGPU} GPUs ==="
    for d in $(seq 0 $((NGPU-1))); do
        python "$HERE/calculate_multipole_grid.py" -r "${region}" -g "${GEOM}" \
            -D "${d}" -N "${NGPU}" --dxy "${DXY}" --dz "${DZ}" "$@" \
            > "${logdir}${stamp}_${region}_GPU${d}.log" 2>&1 &
    done
    wait
    python "$HERE/sum_multipole_partials.py" \
        "${GEOM}.${region}_region.GPU*_of_${NGPU}.pkl"
}

run_region measurement
run_region map --offset-map

echo "=== harmonic rings (single GPU; 12 rings x 256 pts is cheap) ==="
python "$HERE/calculate_harmonic_rings.py" -g "${GEOM}" -D 0 --per-element

echo
echo "=== done; datasets in ${outdir} ==="
ls -la "${outdir}${GEOM}".*summed.pkl "${outdir}${GEOM}".rings_* 2>/dev/null || true
