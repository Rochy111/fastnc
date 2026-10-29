# Baseline: today's readers on the corpus

Files by format:
- HDF5 (netCDF4): 15
- HDF5: 8
- netCDF3 64-bit offset: 3
- netCDF3 CDF-5 (64-bit data): 1
- netCDF3 classic: 1

Variable layouts: record 286, chunked 280, contiguous 40
Filters seen: deflate 192, shuffle 180, fletcher32 6, szip 6, lzf 3, bzip2 3

| file | format | raw MB | reader | first s | best s | MB/s |
|---|---|---|---|---|---|---|
| kalman_2h / wrfout_d02_2024-11-06_01_30_00 | netCDF3 64-bit offset | 801.9 | netCDF4-python raw | 0.352 | 0.352 | 2276 |
| kalman_2h / wrfout_d02_2024-11-06_01_30_00 | netCDF3 64-bit offset | 801.9 | netCDF4-python mask+scale | 0.492 | 0.490 | 1637 |
| kalman_2h / wrfout_d02_2024-11-06_01_30_00 | netCDF3 64-bit offset | 801.9 | scipy.io.netcdf | 0.653 | 0.501 | 1601 |
| kalman_2h / wrfout_d02_2024-11-06_01_30_00 | netCDF3 64-bit offset | 801.9 | xarray scipy | 1.693 | 0.366 | 2194 |
| kalman_2h / wrfout_d02_2024-11-06_01_30_00 | netCDF3 64-bit offset | 801.9 | xarray netcdf4 | 0.384 | 0.384 | 2086 |
| prepared / wrfbdy_d01 | netCDF3 64-bit offset | 312.2 | netCDF4-python raw | 0.155 | 0.155 | 2012 |
| prepared / wrfbdy_d01 | netCDF3 64-bit offset | 312.2 | netCDF4-python mask+scale | 0.193 | 0.193 | 1622 |
| prepared / wrfbdy_d01 | netCDF3 64-bit offset | 312.2 | scipy.io.netcdf | 0.234 | 0.213 | 1466 |
| prepared / wrfbdy_d01 | netCDF3 64-bit offset | 312.2 | xarray scipy | 0.143 | 0.143 | 2182 |
| prepared / wrfbdy_d01 | netCDF3 64-bit offset | 312.2 | xarray netcdf4 | 0.178 | 0.178 | 1751 |
| WRF_OUT / wrfout_d01_2025-10-26_06_20_00.nc | HDF5 | 202.2 | netCDF4-python raw | 0.557 | 0.549 | 368 |
| WRF_OUT / wrfout_d01_2025-10-26_06_20_00.nc | HDF5 | 202.2 | netCDF4-python mask+scale | 0.603 | 0.603 | 335 |
| WRF_OUT / wrfout_d01_2025-10-26_06_20_00.nc | HDF5 | 202.2 | h5py | 0.499 | 0.499 | 405 |
| WRF_OUT / wrfout_d01_2025-10-26_06_20_00.nc | HDF5 | 202.2 | xarray h5netcdf | 10.521 | 10.521 | 19 |
| WRF_OUT / wrfout_d01_2025-10-26_06_20_00.nc | HDF5 | 202.2 | xarray netcdf4 | 6.135 | 5.979 | 34 |
| h5_chunk_index_fixed_array.h5 | HDF5 | 6.5 | netCDF4-python raw | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_chunk_index_fixed_array.h5 | HDF5 | 6.5 | netCDF4-python mask+scale | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_chunk_index_fixed_array.h5 | HDF5 | 6.5 | h5py | 0.023 | 0.022 | 295 |
| h5_chunk_index_fixed_array.h5 | HDF5 | 6.5 | xarray h5netcdf | 0.026 | 0.025 | 258 |
| h5_chunk_index_fixed_array.h5 | HDF5 | 6.5 | xarray netcdf4 | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_chunk_index_latest.h5 | HDF5 | 6.5 | netCDF4-python raw | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_chunk_index_latest.h5 | HDF5 | 6.5 | netCDF4-python mask+scale | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_chunk_index_latest.h5 | HDF5 | 6.5 | h5py | 0.022 | 0.021 | 302 |
| h5_chunk_index_latest.h5 | HDF5 | 6.5 | xarray h5netcdf | 0.025 | 0.025 | 255 |
| h5_chunk_index_latest.h5 | HDF5 | 6.5 | xarray netcdf4 | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_chunked_gzip.h5 | HDF5 | 6.5 | netCDF4-python raw | 0.024 | 0.023 | 277 |
| h5_chunked_gzip.h5 | HDF5 | 6.5 | netCDF4-python mask+scale | 0.025 | 0.025 | 262 |
| h5_chunked_gzip.h5 | HDF5 | 6.5 | h5py | 0.021 | 0.021 | 305 |
| h5_chunked_gzip.h5 | HDF5 | 6.5 | xarray h5netcdf | 0.025 | 0.025 | 255 |
| h5_chunked_gzip.h5 | HDF5 | 6.5 | xarray netcdf4 | 0.112 | 0.105 | 62 |
| h5_contiguous.h5 | HDF5 | 6.5 | netCDF4-python raw | 0.004 | 0.004 | 1591 |
| h5_contiguous.h5 | HDF5 | 6.5 | netCDF4-python mask+scale | 0.005 | 0.005 | 1326 |
| h5_contiguous.h5 | HDF5 | 6.5 | h5py | 0.002 | 0.002 | 2893 |
| h5_contiguous.h5 | HDF5 | 6.5 | xarray h5netcdf | 0.005 | 0.005 | 1196 |
| h5_contiguous.h5 | HDF5 | 6.5 | xarray netcdf4 | 0.080 | 0.080 | 81 |
| h5_gzip_shuffle_fletcher.h5 | HDF5 | 6.5 | netCDF4-python raw | 0.013 | 0.012 | 549 |
| h5_gzip_shuffle_fletcher.h5 | HDF5 | 6.5 | netCDF4-python mask+scale | 0.013 | 0.013 | 498 |
| h5_gzip_shuffle_fletcher.h5 | HDF5 | 6.5 | h5py | 0.010 | 0.010 | 631 |
| h5_gzip_shuffle_fletcher.h5 | HDF5 | 6.5 | xarray h5netcdf | 0.015 | 0.015 | 445 |
| h5_gzip_shuffle_fletcher.h5 | HDF5 | 6.5 | xarray netcdf4 | 0.093 | 0.090 | 72 |
| h5_lzf.h5 | HDF5 | 6.5 | netCDF4-python raw | error: RuntimeError: NetCDF: Filter error: undefined filter encount | | |
| h5_lzf.h5 | HDF5 | 6.5 | netCDF4-python mask+scale | error: RuntimeError: NetCDF: Filter error: undefined filter encount | | |
| h5_lzf.h5 | HDF5 | 6.5 | h5py | 0.009 | 0.007 | 879 |
| h5_lzf.h5 | HDF5 | 6.5 | xarray h5netcdf | 0.011 | 0.011 | 577 |
| h5_lzf.h5 | HDF5 | 6.5 | xarray netcdf4 | error: RuntimeError: NetCDF: Filter error: undefined filter encount | | |
| h5_single_chunk.h5 | HDF5 | 6.5 | netCDF4-python raw | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_single_chunk.h5 | HDF5 | 6.5 | netCDF4-python mask+scale | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_single_chunk.h5 | HDF5 | 6.5 | h5py | 0.023 | 0.022 | 287 |
| h5_single_chunk.h5 | HDF5 | 6.5 | xarray h5netcdf | 0.026 | 0.026 | 249 |
| h5_single_chunk.h5 | HDF5 | 6.5 | xarray netcdf4 | error: OSError: [Errno -101] NetCDF: HDF error: 'C:\\Users\\Roch\\D | | |
| h5_szip.h5 | HDF5 | 6.5 | netCDF4-python raw | 0.032 | 0.031 | 206 |
| h5_szip.h5 | HDF5 | 6.5 | netCDF4-python mask+scale | 0.033 | 0.032 | 200 |
| h5_szip.h5 | HDF5 | 6.5 | h5py | 0.031 | 0.030 | 218 |
| h5_szip.h5 | HDF5 | 6.5 | xarray h5netcdf | 0.034 | 0.034 | 191 |
| h5_szip.h5 | HDF5 | 6.5 | xarray netcdf4 | 0.114 | 0.112 | 58 |
| nc3_64bit_offset.nc | netCDF3 64-bit offset | 8.8 | netCDF4-python raw | 0.007 | 0.007 | 1306 |
| nc3_64bit_offset.nc | netCDF3 64-bit offset | 8.8 | netCDF4-python mask+scale | 0.009 | 0.009 | 938 |
| nc3_64bit_offset.nc | netCDF3 64-bit offset | 8.8 | scipy.io.netcdf | 0.006 | 0.006 | 1447 |
| nc3_64bit_offset.nc | netCDF3 64-bit offset | 8.8 | xarray scipy | 0.011 | 0.007 | 1237 |
| nc3_64bit_offset.nc | netCDF3 64-bit offset | 8.8 | xarray netcdf4 | 0.010 | 0.010 | 908 |
| nc3_cdf5.nc | netCDF3 CDF-5 (64-bit data) | 8.8 | netCDF4-python raw | 0.007 | 0.007 | 1326 |
| nc3_cdf5.nc | netCDF3 CDF-5 (64-bit data) | 8.8 | netCDF4-python mask+scale | 0.009 | 0.009 | 938 |
| nc3_cdf5.nc | netCDF3 CDF-5 (64-bit data) | 8.8 | scipy.io.netcdf | error: ValueError: Unexpected header. | | |
| nc3_cdf5.nc | netCDF3 CDF-5 (64-bit data) | 8.8 | xarray scipy | error: ValueError: Unexpected header. | | |
| nc3_cdf5.nc | netCDF3 CDF-5 (64-bit data) | 8.8 | xarray netcdf4 | 0.009 | 0.009 | 959 |
| nc3_classic.nc | netCDF3 classic | 8.8 | netCDF4-python raw | 0.007 | 0.006 | 1355 |
| nc3_classic.nc | netCDF3 classic | 8.8 | netCDF4-python mask+scale | 0.010 | 0.010 | 916 |
| nc3_classic.nc | netCDF3 classic | 8.8 | scipy.io.netcdf | 0.006 | 0.006 | 1427 |
| nc3_classic.nc | netCDF3 classic | 8.8 | xarray scipy | 0.011 | 0.007 | 1257 |
| nc3_classic.nc | netCDF3 classic | 8.8 | xarray netcdf4 | 0.009 | 0.009 | 948 |
| nc4_bzip2.nc | HDF5 | 7.4 | netCDF4-python raw | 0.218 | 0.214 | 34 |
| nc4_bzip2.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.217 | 0.217 | 34 |
| nc4_bzip2.nc | HDF5 | 7.4 | h5py | 0.218 | 0.214 | 34 |
| nc4_bzip2.nc | HDF5 | 7.4 | xarray h5netcdf | 0.226 | 0.225 | 33 |
| nc4_bzip2.nc | HDF5 | 7.4 | xarray netcdf4 | 0.300 | 0.288 | 26 |
| nc4_chunked_raw.nc | HDF5 | 7.4 | netCDF4-python raw | 0.007 | 0.005 | 1371 |
| nc4_chunked_raw.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.006 | 0.006 | 1260 |
| nc4_chunked_raw.nc | HDF5 | 7.4 | h5py | 0.004 | 0.003 | 2309 |
| nc4_chunked_raw.nc | HDF5 | 7.4 | xarray h5netcdf | 0.015 | 0.015 | 504 |
| nc4_chunked_raw.nc | HDF5 | 7.4 | xarray netcdf4 | 0.083 | 0.083 | 89 |
| nc4_classic_zlib.nc | HDF5 | 8.8 | netCDF4-python raw | 0.017 | 0.015 | 599 |
| nc4_classic_zlib.nc | HDF5 | 8.8 | netCDF4-python mask+scale | 0.018 | 0.018 | 482 |
| nc4_classic_zlib.nc | HDF5 | 8.8 | h5py | 0.013 | 0.013 | 658 |
| nc4_classic_zlib.nc | HDF5 | 8.8 | xarray h5netcdf | 0.036 | 0.035 | 249 |
| nc4_classic_zlib.nc | HDF5 | 8.8 | xarray netcdf4 | 0.147 | 0.147 | 60 |
| nc4_contiguous.nc | HDF5 | 7.4 | netCDF4-python raw | 0.006 | 0.006 | 1284 |
| nc4_contiguous.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.006 | 0.006 | 1311 |
| nc4_contiguous.nc | HDF5 | 7.4 | h5py | 0.003 | 0.003 | 2358 |
| nc4_contiguous.nc | HDF5 | 7.4 | xarray h5netcdf | 0.015 | 0.014 | 520 |
| nc4_contiguous.nc | HDF5 | 7.4 | xarray netcdf4 | 0.082 | 0.082 | 90 |
| nc4_quantize_BitGroom.nc | HDF5 | 7.4 | netCDF4-python raw | 0.016 | 0.015 | 506 |
| nc4_quantize_BitGroom.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.015 | 0.015 | 479 |
| nc4_quantize_BitGroom.nc | HDF5 | 7.4 | h5py | 0.014 | 0.013 | 560 |
| nc4_quantize_BitGroom.nc | HDF5 | 7.4 | xarray h5netcdf | 0.025 | 0.025 | 299 |
| nc4_quantize_BitGroom.nc | HDF5 | 7.4 | xarray netcdf4 | 0.092 | 0.088 | 84 |
| nc4_quantize_BitRound.nc | HDF5 | 7.4 | netCDF4-python raw | 0.011 | 0.009 | 797 |
| nc4_quantize_BitRound.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.011 | 0.010 | 708 |
| nc4_quantize_BitRound.nc | HDF5 | 7.4 | h5py | 0.010 | 0.008 | 895 |
| nc4_quantize_BitRound.nc | HDF5 | 7.4 | xarray h5netcdf | 0.024 | 0.021 | 359 |
| nc4_quantize_BitRound.nc | HDF5 | 7.4 | xarray netcdf4 | 0.086 | 0.085 | 87 |
| nc4_quantize_GranularBitRound.nc | HDF5 | 7.4 | netCDF4-python raw | 0.014 | 0.014 | 530 |
| nc4_quantize_GranularBitRound.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.015 | 0.015 | 494 |
| nc4_quantize_GranularBitRound.nc | HDF5 | 7.4 | h5py | 0.013 | 0.013 | 583 |
| nc4_quantize_GranularBitRound.nc | HDF5 | 7.4 | xarray h5netcdf | 0.028 | 0.025 | 292 |
| nc4_quantize_GranularBitRound.nc | HDF5 | 7.4 | xarray netcdf4 | 0.107 | 0.107 | 69 |
| nc4_szip.nc | HDF5 | 7.4 | netCDF4-python raw | 0.038 | 0.038 | 195 |
| nc4_szip.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.040 | 0.039 | 187 |
| nc4_szip.nc | HDF5 | 7.4 | h5py | 0.037 | 0.036 | 202 |
| nc4_szip.nc | HDF5 | 7.4 | xarray h5netcdf | 0.050 | 0.049 | 149 |
| nc4_szip.nc | HDF5 | 7.4 | xarray netcdf4 | 0.136 | 0.135 | 55 |
| nc4_tiny_chunks.nc | HDF5 | 7.4 | netCDF4-python raw | 0.058 | 0.054 | 137 |
| nc4_tiny_chunks.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.056 | 0.054 | 136 |
| nc4_tiny_chunks.nc | HDF5 | 7.4 | h5py | 0.049 | 0.047 | 158 |
| nc4_tiny_chunks.nc | HDF5 | 7.4 | xarray h5netcdf | 0.067 | 0.063 | 117 |
| nc4_tiny_chunks.nc | HDF5 | 7.4 | xarray netcdf4 | 0.139 | 0.133 | 56 |
| nc4_whole_field_chunks.nc | HDF5 | 7.4 | netCDF4-python raw | 0.014 | 0.012 | 598 |
| nc4_whole_field_chunks.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.016 | 0.013 | 570 |
| nc4_whole_field_chunks.nc | HDF5 | 7.4 | h5py | 0.012 | 0.010 | 751 |
| nc4_whole_field_chunks.nc | HDF5 | 7.4 | xarray h5netcdf | 0.021 | 0.021 | 356 |
| nc4_whole_field_chunks.nc | HDF5 | 7.4 | xarray netcdf4 | 0.091 | 0.091 | 81 |
| nc4_zlib1.nc | HDF5 | 7.4 | netCDF4-python raw | 0.013 | 0.012 | 600 |
| nc4_zlib1.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.013 | 0.013 | 576 |
| nc4_zlib1.nc | HDF5 | 7.4 | h5py | 0.011 | 0.011 | 687 |
| nc4_zlib1.nc | HDF5 | 7.4 | xarray h5netcdf | 0.023 | 0.023 | 326 |
| nc4_zlib1.nc | HDF5 | 7.4 | xarray netcdf4 | 0.093 | 0.090 | 82 |
| nc4_zlib9_shuffle.nc | HDF5 | 7.4 | netCDF4-python raw | 0.012 | 0.011 | 652 |
| nc4_zlib9_shuffle.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.012 | 0.012 | 604 |
| nc4_zlib9_shuffle.nc | HDF5 | 7.4 | h5py | 0.010 | 0.010 | 743 |
| nc4_zlib9_shuffle.nc | HDF5 | 7.4 | xarray h5netcdf | 0.025 | 0.023 | 318 |
| nc4_zlib9_shuffle.nc | HDF5 | 7.4 | xarray netcdf4 | 0.093 | 0.089 | 83 |
| nc4_zlib_bigendian.nc | HDF5 | 7.4 | netCDF4-python raw | 0.015 | 0.014 | 509 |
| nc4_zlib_bigendian.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.016 | 0.016 | 474 |
| nc4_zlib_bigendian.nc | HDF5 | 7.4 | h5py | 0.011 | 0.010 | 732 |
| nc4_zlib_bigendian.nc | HDF5 | 7.4 | xarray h5netcdf | 0.024 | 0.024 | 310 |
| nc4_zlib_bigendian.nc | HDF5 | 7.4 | xarray netcdf4 | 0.099 | 0.099 | 75 |
| nc4_zlib_shuffle_fletcher.nc | HDF5 | 7.4 | netCDF4-python raw | 0.013 | 0.013 | 589 |
| nc4_zlib_shuffle_fletcher.nc | HDF5 | 7.4 | netCDF4-python mask+scale | 0.013 | 0.013 | 548 |
| nc4_zlib_shuffle_fletcher.nc | HDF5 | 7.4 | h5py | 0.011 | 0.011 | 677 |
| nc4_zlib_shuffle_fletcher.nc | HDF5 | 7.4 | xarray h5netcdf | 0.023 | 0.022 | 329 |
| nc4_zlib_shuffle_fletcher.nc | HDF5 | 7.4 | xarray netcdf4 | 0.125 | 0.112 | 66 |
