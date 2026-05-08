include ../../config

GPU_BIN = gpu
GPU_SHARE_DIR = $(INSTALL_DIR)/share/gpu_tests
GPU_SCRIPTS = \
	build.sh \
	build_ucx.sh \
	py.sh \
	fixed.sbatch \
	gpu_heatmap.py \
	gpu_plot.py \
	gpu_diff.py \
	sort_raw.py \
	requirements.txt \
	cluster.md

all: $(GPU_BIN)

$(GPU_BIN):
	sh ./build.sh build

clean:
	rm -f $(GPU_BIN) ./*.o

install: all
	mkdir -p $(INSTALL_DIR)/bin $(GPU_SHARE_DIR)
	cp -f $(GPU_BIN) $(INSTALL_DIR)/bin/
	cp -f $(GPU_SCRIPTS) $(GPU_SHARE_DIR)/
