#include "netcdf_writer.h"

#include <algorithm>
#include <limits>
#include <iostream>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif
#include "../core/data_write_operations.h"
#include "../core/string_id_converters.h"
#include "../core/types.h"
#ifdef __cplusplus
}
#endif

namespace {

int clamp_size_to_int_or_abort(size_t v, const char *name) {
	if (v <= static_cast<size_t>(std::numeric_limits<int>::max()))
		return static_cast<int>(v);
	std::cerr << "gpu_benchmark: value for " << name
			  << " does not fit into int: " << v << "\n";
	MPI_Abort(MPI_COMM_WORLD, 1);
	return 0;
}

/* Значения data_type внутри NetCDF для gpu_benchmark (суффиксы файлов — отдельно). */
enum : int {
	GPU_NC_AVG = 1,
	GPU_NC_VAR = 2,
	GPU_NC_MIN = 3,
	GPU_NC_MAX = 4,
	GPU_NC_MED = 5,
	GPU_NC_STD = 6,
};

} // namespace

NetcdfBundle netcdf_open_bundle(const std::string &out_path, size_t nbytes,
								size_t end_nbytes, size_t step_nbytes, int iters,
								int nproc) {
	NetcdfBundle nc{};
	if (out_path.empty())
		return nc;

	std::string prefix = out_path;
	const std::string txt_suffix = ".txt";
	if (prefix.size() >= txt_suffix.size() &&
		prefix.compare(prefix.size() - txt_suffix.size(), txt_suffix.size(), txt_suffix) == 0) {
		prefix.erase(prefix.size() - txt_suffix.size());
	}
	if (prefix.empty())
		return nc;

	nc.enabled = true;
	nc.nproc = nproc;
	const size_t total = static_cast<size_t>(nproc) * static_cast<size_t>(nproc);
	nc.avg.assign(total, 0.0);
	nc.var.assign(total, 0.0);
	nc.min.assign(total, 0.0);
	nc.max.assign(total, 0.0);
	nc.med.assign(total, 0.0);
	nc.stddev.assign(total, 0.0);

	network_test_parameters_struct p{};
	p.num_procs = nproc;
	p.test_type = ONE_TO_ONE_TEST_TYPE;
	p.begin_message_length = clamp_size_to_int_or_abort(nbytes, "--bytes");
	p.end_message_length = clamp_size_to_int_or_abort(end_nbytes, "--bytes-end");
	p.step_length = clamp_size_to_int_or_abort(step_nbytes, "--bytes-step");
	p.num_repeats = iters;
	p.noise_message_length = 0;
	p.num_noise_messages = 0;
	p.num_noise_procs = 0;
	p.file_name_prefix = prefix.c_str();

	auto create_one = [&](int datatype, int &file_id, int &data_id,
						  const char *suffix, const char *label) {
		const int rc = create_netcdf_header_with_suffix(datatype, &p, suffix,
														&file_id, &data_id);
		if (rc != 0) {
			std::cerr << "gpu_benchmark: failed to create NetCDF for " << label
					  << " with prefix '" << prefix << "', rc=" << rc << "\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	};
	create_one(GPU_NC_AVG, nc.avg_file_id, nc.avg_data_id, "avg", "avg");
	create_one(GPU_NC_VAR, nc.var_file_id, nc.var_data_id, "var", "var");
	create_one(GPU_NC_MIN, nc.min_file_id, nc.min_data_id, "min", "min");
	create_one(GPU_NC_MAX, nc.max_file_id, nc.max_data_id, "max", "max");
	create_one(GPU_NC_MED, nc.med_file_id, nc.med_data_id, "med", "med");
	create_one(GPU_NC_STD, nc.std_file_id, nc.std_data_id, "std", "std");

	return nc;
}

void netcdf_reset_matrix(NetcdfBundle &nc) {
	if (!nc.enabled)
		return;
	std::fill(nc.avg.begin(), nc.avg.end(), 0.0);
	std::fill(nc.var.begin(), nc.var.end(), 0.0);
	std::fill(nc.min.begin(), nc.min.end(), 0.0);
	std::fill(nc.max.begin(), nc.max.end(), 0.0);
	std::fill(nc.med.begin(), nc.med.end(), 0.0);
	std::fill(nc.stddev.begin(), nc.stddev.end(), 0.0);
}

void netcdf_store_pair(NetcdfBundle &nc, int src_rank, int dst_rank,
					   const std::vector<double> &metric) {
	if (!nc.enabled)
		return;
	const size_t idx = static_cast<size_t>(src_rank) * static_cast<size_t>(nc.nproc) +
					   static_cast<size_t>(dst_rank);
	/* metric[0..5]: mean, median, min, max, var, std (как в fill_stats6). */
	nc.avg[idx] = metric[0];
	nc.med[idx] = metric[1];
	nc.min[idx] = metric[2];
	nc.max[idx] = metric[3];
	nc.var[idx] = metric[4];
	nc.stddev[idx] = metric[5];
}

void netcdf_write_matrix_slice(NetcdfBundle &nc, int matrix_idx) {
	if (!nc.enabled)
		return;

	auto write_one = [&](int file_id, int data_id, const std::vector<double> &matrix,
						 const char *label) {
		const int rc = netcdf_write_matrix(file_id, data_id, matrix_idx, nc.nproc,
										   nc.nproc, matrix.data());
		if (rc != 0) {
			std::cerr << "gpu_benchmark: failed to write NetCDF matrix " << label
					  << ", rc=" << rc << "\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	};
	write_one(nc.avg_file_id, nc.avg_data_id, nc.avg, "avg");
	write_one(nc.var_file_id, nc.var_data_id, nc.var, "var");
	write_one(nc.min_file_id, nc.min_data_id, nc.min, "min");
	write_one(nc.max_file_id, nc.max_data_id, nc.max, "max");
	write_one(nc.med_file_id, nc.med_data_id, nc.med, "med");
	write_one(nc.std_file_id, nc.std_data_id, nc.stddev, "std");
}

void netcdf_flush_and_close(NetcdfBundle &nc) {
	if (!nc.enabled)
		return;

	netcdf_close_file(nc.avg_file_id);
	netcdf_close_file(nc.var_file_id);
	netcdf_close_file(nc.min_file_id);
	netcdf_close_file(nc.max_file_id);
	netcdf_close_file(nc.med_file_id);
	netcdf_close_file(nc.std_file_id);
}
