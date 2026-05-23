PY="${PWD}/.venv/bin/python3"; [ -x "$PY" ] || PY=python3
R=results_big

"$PY" gpu_heatmap.py "$R" -o "$R/heatmaps"
# "$PY" gpu_plot.py "$R" -o "$R/plots" --src 'cn.rank' --dst 'cn.rank' --metric med_us
"$PY" gpu_netcdf.py "$R" -o "$R/nc"
# ncview "$R/nc/cn_12_16_rep1_auto_one_to_one.nc"
