#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct NetcdfBundle {
	bool enabled = false;
	int nproc = 0;
	int avg_file_id = -1;
	int avg_data_id = -1;
	int med_file_id = -1;
	int med_data_id = -1;
	int min_file_id = -1;
	int min_data_id = -1;
	int std_file_id = -1;
	int std_data_id = -1;
	std::vector<double> avg;
	std::vector<double> med;
	std::vector<double> min;
	std::vector<double> stddev;
};

NetcdfBundle netcdf_open_bundle(const std::string &out_path, size_t nbytes,
								int iters, int nproc);
void netcdf_store_pair(NetcdfBundle &nc, int src_rank, int dst_rank,
					   const std::vector<double> &metric);
void netcdf_flush_and_close(NetcdfBundle &nc);
