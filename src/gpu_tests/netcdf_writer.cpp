#include "netcdf_writer.h"

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
	std::cerr << "gpu_one_to_one: value for " << name
			  << " does not fit into int: " << v << "\n";
	MPI_Abort(MPI_COMM_WORLD, 1);
	return 0;
}

} // namespace

NetcdfBundle netcdf_open_bundle(const std::string &out_path, size_t nbytes,
								int iters, int nproc) {
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
	nc.med.assign(total, 0.0);
	nc.min.assign(total, 0.0);
	nc.stddev.assign(total, 0.0);

	network_test_parameters_struct p{};
	p.num_procs = nproc;
	p.test_type = ONE_TO_ONE_TEST_TYPE;
	p.begin_message_length = clamp_size_to_int_or_abort(nbytes, "--bytes");
	p.end_message_length = clamp_size_to_int_or_abort(nbytes, "--bytes");
	p.step_length = 1;
	p.num_repeats = iters;
	p.noise_message_length = 0;
	p.num_noise_messages = 0;
	p.num_noise_procs = 0;
	p.file_name_prefix = prefix.c_str();

	auto create_one = [&](int datatype, int &file_id, int &data_id, const char *label) {
		const int rc = create_netcdf_header(datatype, &p, &file_id, &data_id);
		if (rc != 0) {
			std::cerr << "gpu_one_to_one: failed to create NetCDF for " << label
					  << " with prefix '" << prefix << "', rc=" << rc << "\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	};
	create_one(AVERAGE_NETWORK_TEST_DATATYPE, nc.avg_file_id, nc.avg_data_id, "avg");
	create_one(MEDIAN_NETWORK_TEST_DATATYPE, nc.med_file_id, nc.med_data_id, "median");
	create_one(MIN_NETWORK_TEST_DATATYPE, nc.min_file_id, nc.min_data_id, "min");
	create_one(DEVIATION_NETWORK_TEST_DATATYPE, nc.std_file_id, nc.std_data_id, "std");

	return nc;
}

void netcdf_store_pair(NetcdfBundle &nc, int src_rank, int dst_rank,
					   const std::vector<double> &metric) {
	if (!nc.enabled)
		return;
	const size_t idx = static_cast<size_t>(src_rank) * static_cast<size_t>(nc.nproc) +
					   static_cast<size_t>(dst_rank);
	nc.avg[idx] = metric[0];
	nc.med[idx] = metric[1];
	nc.min[idx] = metric[2];
	nc.stddev[idx] = metric[5];
}

void netcdf_flush_and_close(NetcdfBundle &nc) {
	if (!nc.enabled)
		return;

	auto write_one = [&](int file_id, int data_id, const std::vector<double> &matrix,
						 const char *label) {
		const int rc = netcdf_write_matrix(file_id, data_id, 0, nc.nproc, nc.nproc,
										   matrix.data());
		if (rc != 0) {
			std::cerr << "gpu_one_to_one: failed to write NetCDF matrix " << label
					  << ", rc=" << rc << "\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	};
	write_one(nc.avg_file_id, nc.avg_data_id, nc.avg, "avg");
	write_one(nc.med_file_id, nc.med_data_id, nc.med, "median");
	write_one(nc.min_file_id, nc.min_data_id, nc.min, "min");
	write_one(nc.std_file_id, nc.std_data_id, nc.stddev, "std");

	netcdf_close_file(nc.avg_file_id);
	netcdf_close_file(nc.med_file_id);
	netcdf_close_file(nc.min_file_id);
	netcdf_close_file(nc.std_file_id);
}
