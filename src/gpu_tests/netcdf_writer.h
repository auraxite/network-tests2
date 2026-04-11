#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct NetcdfBundle {
	bool enabled = false;
	int nproc = 0;
	int avg_file_id = -1;
	int avg_data_id = -1;
	int var_file_id = -1;
	int var_data_id = -1;
	int min_file_id = -1;
	int min_data_id = -1;
	int max_file_id = -1;
	int max_data_id = -1;
	int med_file_id = -1;
	int med_data_id = -1;
	int std_file_id = -1;
	int std_data_id = -1;
	std::vector<double> avg;
	std::vector<double> var;
	std::vector<double> min;
	std::vector<double> max;
	std::vector<double> med;
	std::vector<double> stddev;
};

NetcdfBundle netcdf_open_bundle(const std::string &out_path, size_t nbytes,
								size_t end_nbytes, size_t step_nbytes, int iters, int nproc);
void netcdf_reset_matrix(NetcdfBundle &nc);
void netcdf_store_pair(NetcdfBundle &nc, int src_rank, int dst_rank,
					   const std::vector<double> &metric);
void netcdf_write_matrix_slice(NetcdfBundle &nc, int matrix_idx);
void netcdf_flush_and_close(NetcdfBundle &nc);
