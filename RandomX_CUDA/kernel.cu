/*
Copyright (c) 2019 SChernykh

This file is part of RandomX CUDA.

RandomX CUDA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RandomX CUDA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RandomX CUDA.  If not, see<http://www.gnu.org/licenses/>.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <chrono>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include "../RandomX/src/blake2/blake2.h"
#include "../RandomX/src/aes_hash.hpp"
#include "../RandomX/src/randomx.h"
#include "../RandomX/src/configuration.h"
#include "../RandomX/src/common.hpp"

#include "blake2b_cuda.hpp"
#include "aes_cuda.hpp"
#include "randomx_cuda.hpp"

bool test_mining(bool validate, int bfactor, int workers_per_hash, bool fast_fp, uint32_t start_nonce, uint32_t intensity);
void tests();

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		printf("Usage: %s --mine device_id [--validate] [--bfactor N] [--workers N] [--fast_fp] [--nonce N] [--intensity N]\n\n", argv[0]);
		printf("device_id    0 if you have only 1 GPU\n");
		printf("bfactor      0-10, default is 0. Increase it if you get CUDA errors/driver crashes/screen lags.\n");
		printf("workers      2,4,8,16, default is 8. Choose the value that gives you the best hashrate (it's usually 8 or 16).\n");
		printf("fast_fp      use faster, but sometimes incorrect code for floating point instructions");
		printf("nonce        any integer >= 0, default is 0. Mining will start from this nonce.\n");
		printf("intensity    number of scratchpads to allocate, if it's not set then as many as possible will be allocated.\n\n");
		printf("Examples:\n%s --test 0\n%s --mine 0 --validate --bfactor 3 --workers 8 --intensity 1280\n", argv[0], argv[0]);
		return 0;
	}

	const int device_id = atoi(argv[2]);

	cudaError_t cudaStatus = cudaSetDevice(device_id);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
		return 1;
	}

	uint32_t flags;
	if (cudaGetDeviceFlags(&flags) == cudaSuccess)
		cudaSetDeviceFlags(flags | cudaDeviceScheduleBlockingSync);

	bool validate = false;
	int bfactor = 0;
	int workers_per_hash = 8;
	uint32_t start_nonce = 0;
	uint32_t intensity = 0;
	bool fast_fp = false;
	for (int i = 0; i < argc; ++i)
	{
		if (strcmp(argv[i], "--validate") == 0)
		{
			validate = true;
		}

		if ((strcmp(argv[i], "--bfactor") == 0) && (i + 1 < argc))
		{
			bfactor = atoi(argv[i + 1]);
			if (bfactor < 0) bfactor = 0;
			if (bfactor > 10) bfactor = 10;
		}

		if ((strcmp(argv[i], "--workers") == 0) && (i + 1 < argc))
		{
			workers_per_hash = atoi(argv[i + 1]);
			switch (workers_per_hash)
			{
			case 2:
			case 4:
			case 8:
			case 16:
				break;

			default:
				workers_per_hash = 8;
			}
		}

		if ((strcmp(argv[i], "--nonce") == 0) && (i + 1 < argc))
		{
			start_nonce = atoi(argv[i + 1]);
		}

		if ((strcmp(argv[i], "--intensity") == 0) && (i + 1 < argc))
		{
			intensity = atoi(argv[i + 1]);
		}

		if ((strcmp(argv[i], "--fast_fp") == 0))
		{
			fast_fp = true;
		}
	}

	if (strcmp(argv[1], "--mine") == 0)
		test_mining(validate, bfactor, workers_per_hash, fast_fp, start_nonce, intensity);
	else if (strcmp(argv[1], "--test") == 0)
		tests();

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}

	return 0;
}

using namespace std::chrono;

static uint8_t blockTemplate[] = {
		0x07, 0x07, 0xf7, 0xa4, 0xf0, 0xd6, 0x05, 0xb3, 0x03, 0x26, 0x08, 0x16, 0xba, 0x3f, 0x10, 0x90, 0x2e, 0x1a, 0x14,
		0x5a, 0xc5, 0xfa, 0xd3, 0xaa, 0x3a, 0xf6, 0xea, 0x44, 0xc1, 0x18, 0x69, 0xdc, 0x4f, 0x85, 0x3f, 0x00, 0x2b, 0x2e,
		0xea, 0x00, 0x00, 0x00, 0x00, 0x77, 0xb2, 0x06, 0xa0, 0x2c, 0xa5, 0xb1, 0xd4, 0xce, 0x6b, 0xbf, 0xdf, 0x0a, 0xca,
		0xc3, 0x8b, 0xde, 0xd3, 0x4d, 0x2d, 0xcd, 0xee, 0xf9, 0x5c, 0xd2, 0x0c, 0xef, 0xc1, 0x2f, 0x61, 0xd5, 0x61, 0x09
};

struct GPUPtr
{
	explicit GPUPtr(size_t size)
	{
		if (cudaMalloc((void**) &p, size) != cudaSuccess)
			p = nullptr;
	}

	~GPUPtr()
	{
		if (p)
			cudaFree(p);
	}

	operator void*() const { return p; }

private:
	void* p;
};

bool test_mining(bool validate, int bfactor, int workers_per_hash, bool fast_fp, uint32_t start_nonce, uint32_t intensity)
{
	printf("Testing mining: CPU validation is %s, bfactor is %d, %d workers per hash%s, start nonce %u, intensity %u\n", validate ? "ON" : "OFF", bfactor, workers_per_hash, fast_fp ? ", fast FP" : "", start_nonce, intensity);

	cudaError_t cudaStatus;

	size_t free_mem, total_mem;
	{
		GPUPtr tmp(256);
		cudaStatus = cudaMemGetInfo(&free_mem, &total_mem);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "Failed to get free memory info!\n");
			return false;
		}
	}

	printf("%zu MB GPU memory free\n", free_mem >> 20);
	printf("%zu MB GPU memory total\n", total_mem >> 20);

	// There should be enough GPU memory for the 2080 MB dataset, 32 scratchpads and 64 MB for everything else
	const size_t dataset_size = randomx_dataset_item_count() * RANDOMX_DATASET_ITEM_SIZE;
	if (free_mem <= dataset_size + (32U * (RANDOMX_SCRATCHPAD_L3 + 64)) + (64U << 20))
	{
		fprintf(stderr, "Not enough free GPU memory!\n");
		return false;
	}

	uint32_t batch_size = (intensity >= 32) ? intensity : ((free_mem - dataset_size - (64U << 20)) / (RANDOMX_SCRATCHPAD_L3 + 64));
	batch_size = (batch_size / 32) * 32;

	GPUPtr dataset_gpu(dataset_size);
	if (!dataset_gpu)
	{
		fprintf(stderr, "Failed to allocate GPU memory for dataset!\n");
		return false;
	}

	printf("Allocated %.0f MB dataset\n", dataset_size / 1048576.0);

	printf("Initializing dataset...");

	randomx_dataset *myDataset;
	bool large_pages_available = true;
	{
		const char mySeed[] = "RandomX example seed";

		randomx_cache *myCache = randomx_alloc_cache((randomx_flags)(RANDOMX_FLAG_JIT));
		randomx_init_cache(myCache, mySeed, sizeof mySeed);
		myDataset = randomx_alloc_dataset(RANDOMX_FLAG_LARGE_PAGES);
		if (!myDataset)
		{
			printf("\nCouldn't allocate dataset using large pages\n");
			myDataset = randomx_alloc_dataset(RANDOMX_FLAG_DEFAULT);
			large_pages_available = false;
		}

		auto t1 = high_resolution_clock::now();

		std::vector<std::thread> threads;
		for (uint32_t i = 0, n = std::thread::hardware_concurrency(); i < n; ++i)
			threads.emplace_back([myDataset, myCache, i, n](){ randomx_init_dataset(myDataset, myCache, (i * randomx_dataset_item_count()) / n, ((i + 1) * randomx_dataset_item_count()) / n - (i * randomx_dataset_item_count()) / n); });

		for (auto& t : threads)
			t.join();

		randomx_release_cache(myCache);

		cudaStatus = cudaMemcpy(dataset_gpu, randomx_get_dataset_memory(myDataset), dataset_size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Failed to copy dataset to GPU: %s\n", cudaGetErrorString(cudaStatus));
			return false;
		}

		printf("done in %.3f seconds\n", duration_cast<nanoseconds>(high_resolution_clock::now() - t1).count() / 1e9);
	}

	GPUPtr scratchpads_gpu(batch_size * static_cast<uint64_t>(RANDOMX_SCRATCHPAD_L3 + 64));
	if (!scratchpads_gpu)
	{
		fprintf(stderr, "Failed to allocate GPU memory for scratchpads!\n");
		return false;
	}

	printf("Allocated %u scratchpads\n", batch_size);

	GPUPtr hashes_gpu(batch_size * HASH_SIZE);
	if (!hashes_gpu)
	{
		fprintf(stderr, "Failed to allocate GPU memory for hashes!\n");
		return false;
	}

	GPUPtr entropy_gpu(batch_size * ENTROPY_SIZE);
	if (!entropy_gpu)
	{
		fprintf(stderr, "Failed to allocate GPU memory for programs!\n");
		return false;
	}

	GPUPtr vm_states_gpu(batch_size * VM_STATE_SIZE);
	if (!vm_states_gpu)
	{
		fprintf(stderr, "Failed to allocate GPU memory for VM states!\n");
		return false;
	}

	GPUPtr rounding_gpu(batch_size * sizeof(uint32_t));
	if (!rounding_gpu)
	{
		fprintf(stderr, "Failed to allocate GPU memory for VM rounding data!\n");
		return false;
	}

	GPUPtr num_vm_cycles_gpu(sizeof(uint64_t));
	if (!num_vm_cycles_gpu)
	{
		fprintf(stderr, "Failed to allocate GPU memory for VM num cyles data!\n");
		return false;
	}

	cudaMemset(num_vm_cycles_gpu, 0, sizeof(uint64_t));

	GPUPtr blockTemplate_gpu(sizeof(blockTemplate));
	if (!blockTemplate_gpu)
	{
		fprintf(stderr, "Failed to allocate GPU memory for block template!\n");
		return false;
	}

	cudaStatus = cudaMemcpy(blockTemplate_gpu, blockTemplate, sizeof(blockTemplate), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Failed to copy block template to GPU: %s\n", cudaGetErrorString(cudaStatus));
		return false;
	}

	cudaStatus = cudaMemGetInfo(&free_mem, &total_mem);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Failed to get free memory info!\n");
		return false;
	}

	printf("%zu MB free GPU memory left\n", free_mem >> 20);

	for (const void* p : { init_vm<2>, init_vm<4>, init_vm<8>, init_vm<16> })
	{
		cudaStatus = cudaFuncSetCacheConfig(p, cudaFuncCachePreferShared);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "Failed to set cache config for init_vm!\n");
			return false;
		}
	}

	for (const void* p : { execute_vm<2, false>, execute_vm<4, false>, execute_vm<8, false>, execute_vm<16, false>, execute_vm<2, true>, execute_vm<4, true>, execute_vm<8, true>, execute_vm<16, true> })
	{
		cudaStatus = cudaFuncSetCacheConfig(p, cudaFuncCachePreferShared);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "Failed to set cache config for execute_vm!\n");
			return false;
		}
	}

	auto prev_time = high_resolution_clock::now();

	std::vector<uint8_t> hashes, hashes_check;
	hashes.resize(batch_size * 32);
	hashes_check.resize(batch_size * 32);

	std::vector<std::thread> threads;
	std::atomic<uint32_t> nonce_counter;
	bool cpu_limited = false;

	uint32_t failed_nonces = 0;

	for (uint32_t nonce = start_nonce, k = 0; nonce < 0xFFFFFFFFUL; nonce += batch_size, ++k)
	{
		auto validation_thread = [&nonce_counter, myDataset, &hashes_check, batch_size, nonce, large_pages_available]() {
			randomx_vm *myMachine = randomx_create_vm((randomx_flags)(RANDOMX_FLAG_FULL_MEM | RANDOMX_FLAG_JIT | RANDOMX_FLAG_HARD_AES | (large_pages_available ? RANDOMX_FLAG_LARGE_PAGES : 0)), nullptr, myDataset);

			uint8_t buf[sizeof(blockTemplate)];
			memcpy(buf, blockTemplate, sizeof(buf));

			for (;;)
			{
				const uint32_t i = nonce_counter.fetch_add(1);
				if (i >= batch_size)
					break;

				*(uint32_t*)(buf + 39) = nonce + i;

				randomx_calculate_hash(myMachine, buf, sizeof(buf), (hashes_check.data() + i * 32));
			}
			randomx_destroy_vm(myMachine);
		};

		if (validate)
		{
			nonce_counter = 0;

			const uint32_t n = std::max(std::thread::hardware_concurrency() / 2, 1U);

			threads.clear();
			for (uint32_t i = 0; i < n; ++i)
				threads.emplace_back(validation_thread);
		}

		auto cur_time = high_resolution_clock::now();
		if (k > 0)
		{
			const double dt = duration_cast<nanoseconds>(cur_time - prev_time).count() / 1e9;
			uint64_t data = uint64_t(-1);
			cudaMemcpy(&data, num_vm_cycles_gpu, sizeof(uint64_t), cudaMemcpyDeviceToHost);

			const double num_vm_cycles = static_cast<uint32_t>(data);
			const double num_slots_used = static_cast<uint32_t>(data >> 32);
			if (validate)
			{
				const uint32_t n = nonce - start_nonce;
				printf("%u (%.3f%%) hashes validated successfully, %u (%.3f%%) hashes failed, IPC %.4f, WPC %.4f, %.0f h/s%s\n",
					n - failed_nonces,
					static_cast<double>(n - failed_nonces) / n * 100.0,
					failed_nonces,
					static_cast<double>(failed_nonces) / n * 100.0,
					n * (RANDOMX_PROGRAM_SIZE - RANDOMX_FREQ_NOP) * RANDOMX_PROGRAM_COUNT / num_vm_cycles,
					num_slots_used / num_vm_cycles, batch_size / dt,
					cpu_limited ? ", limited by CPU" : "                "
				);
			}
			else
			{
				printf("%.0f h/s\t\r", batch_size / dt);
			}
		}
		prev_time = cur_time;

		blake2b_initial_hash<sizeof(blockTemplate)><<<batch_size / 32, 32>>>(hashes_gpu, blockTemplate_gpu, nonce);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "blake2b_initial_hash launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return false;
		}

		fillAes1Rx4<RANDOMX_SCRATCHPAD_L3, false, 64><<<batch_size / 32, 32 * 4>>>(hashes_gpu, scratchpads_gpu, batch_size);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "fillAes1Rx4 launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return false;
		}

		cudaStatus = cudaMemset(rounding_gpu, 0, batch_size * sizeof(uint32_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemset failed!\n");
			return false;
		}

		for (size_t i = 0; i < RANDOMX_PROGRAM_COUNT; ++i)
		{
			fillAes4Rx4<ENTROPY_SIZE, false><<<batch_size / 32, 32 * 4>>>(hashes_gpu, entropy_gpu, batch_size);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "fillAes4Rx4 launch failed: %s\n", cudaGetErrorString(cudaStatus));
				return false;
			}

			switch (workers_per_hash)
			{
			case 2:
				init_vm<2><<<batch_size / 4, 4 * 8>>>(entropy_gpu, vm_states_gpu, num_vm_cycles_gpu);
				for (int j = 0, n = 1 << bfactor; j < n; ++j)
				{
					if (fast_fp)
						execute_vm<2, false><<<batch_size / 2, 2 * 8>>>(vm_states_gpu, rounding_gpu, scratchpads_gpu, dataset_gpu, batch_size, RANDOMX_PROGRAM_ITERATIONS >> bfactor, j == 0, j == n - 1);
					else
						execute_vm<2, true><<<batch_size / 2, 2 * 8>>>(vm_states_gpu, rounding_gpu, scratchpads_gpu, dataset_gpu, batch_size, RANDOMX_PROGRAM_ITERATIONS >> bfactor, j == 0, j == n - 1);
				}
				break;

			case 4:
				init_vm<4><<<batch_size / 4, 4 * 8>>>(entropy_gpu, vm_states_gpu, num_vm_cycles_gpu);
				for (int j = 0, n = 1 << bfactor; j < n; ++j)
				{
					if (fast_fp)
						execute_vm<4, false><<<batch_size / 2, 2 * 8>>>(vm_states_gpu, rounding_gpu, scratchpads_gpu, dataset_gpu, batch_size, RANDOMX_PROGRAM_ITERATIONS >> bfactor, j == 0, j == n - 1);
					else
						execute_vm<4, true><<<batch_size / 2, 2 * 8>>>(vm_states_gpu, rounding_gpu, scratchpads_gpu, dataset_gpu, batch_size, RANDOMX_PROGRAM_ITERATIONS >> bfactor, j == 0, j == n - 1);
				}
				break;

			case 8:
				init_vm<8><<<batch_size / 4, 4 * 8>>>(entropy_gpu, vm_states_gpu, num_vm_cycles_gpu);
				for (int j = 0, n = 1 << bfactor; j < n; ++j)
				{
					if (fast_fp)
						execute_vm<8, false><<<batch_size / 2, 2 * 8>>>(vm_states_gpu, rounding_gpu, scratchpads_gpu, dataset_gpu, batch_size, RANDOMX_PROGRAM_ITERATIONS >> bfactor, j == 0, j == n - 1);
					else
						execute_vm<8, true><<<batch_size / 2, 2 * 8>>>(vm_states_gpu, rounding_gpu, scratchpads_gpu, dataset_gpu, batch_size, RANDOMX_PROGRAM_ITERATIONS >> bfactor, j == 0, j == n - 1);
				}
				break;

			case 16:
				init_vm<16><<<batch_size / 4, 4 * 8>>>(entropy_gpu, vm_states_gpu, num_vm_cycles_gpu);
				for (int j = 0, n = 1 << bfactor; j < n; ++j)
				{
					if (fast_fp)
						execute_vm<16, false><<<batch_size / 2, 2 * 16>>>(vm_states_gpu, rounding_gpu, scratchpads_gpu, dataset_gpu, batch_size, RANDOMX_PROGRAM_ITERATIONS >> bfactor, j == 0, j == n - 1);
					else
						execute_vm<16, true><<<batch_size / 2, 2 * 16>>>(vm_states_gpu, rounding_gpu, scratchpads_gpu, dataset_gpu, batch_size, RANDOMX_PROGRAM_ITERATIONS >> bfactor, j == 0, j == n - 1);
				}
				break;
			}

			if (i == RANDOMX_PROGRAM_COUNT - 1)
			{
				hashAes1Rx4<RANDOMX_SCRATCHPAD_L3, 192, VM_STATE_SIZE, 64><<<batch_size / 32, 32 * 4>>>(scratchpads_gpu, vm_states_gpu, batch_size);
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "hashAes1Rx4 launch failed: %s\n", cudaGetErrorString(cudaStatus));
					return false;
				}

				blake2b_hash_registers<REGISTERS_SIZE, VM_STATE_SIZE, 32><<<batch_size / 32, 32>>>(hashes_gpu, vm_states_gpu);
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "blake2b_hash_registers launch failed: %s\n", cudaGetErrorString(cudaStatus));
					return false;
				}
			}
			else
			{
				blake2b_hash_registers<REGISTERS_SIZE, VM_STATE_SIZE, 64><<<batch_size / 32, 32>>>(hashes_gpu, vm_states_gpu);
				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "blake2b_hash_registers launch failed: %s\n", cudaGetErrorString(cudaStatus));
					return false;
				}
			}
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize failed (%d): %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
			return false;
		}

		if (validate)
		{
			cudaMemcpy(hashes.data(), hashes_gpu, batch_size * 32, cudaMemcpyDeviceToHost);

			cpu_limited = nonce_counter.load() < batch_size;

			for (auto& thread : threads)
				thread.join();

			if (memcmp(hashes.data(), hashes_check.data(), batch_size * 32) != 0)
			{
				for (uint32_t i = 0; i < batch_size * 32; i += 32)
				{
					if (memcmp(hashes.data() + i, hashes_check.data() + i, 32))
					{
						fprintf(stderr, "CPU validation error, failing nonce = %u\n", nonce + i / 32);
						++failed_nonces;
					}
				}
			}
		}
	}

	return true;
}

void tests()
{
	constexpr size_t NUM_SCRATCHPADS_TEST = 128;
	constexpr size_t NUM_SCRATCHPADS_BENCH = 2048;
	constexpr size_t BLAKE2B_STEP = 1 << 28;

	std::vector<uint8_t> scratchpads((RANDOMX_SCRATCHPAD_L3 + 64) * NUM_SCRATCHPADS_TEST * 2);
	std::vector<uint8_t> programs(ENTROPY_SIZE * NUM_SCRATCHPADS_TEST * 2);

	uint64_t hash[NUM_SCRATCHPADS_TEST * 8] = {};
	uint64_t hash2[NUM_SCRATCHPADS_TEST * 8] = {};

	uint8_t registers[NUM_SCRATCHPADS_TEST * REGISTERS_SIZE] = {};
	uint8_t registers2[NUM_SCRATCHPADS_TEST * REGISTERS_SIZE] = {};

	GPUPtr hash_gpu(sizeof(hash));
	if (!hash_gpu) {
		fprintf(stderr, "cudaMalloc failed!\n");
		return;
	}

	GPUPtr block_template_gpu(sizeof(blockTemplate));
	if (!block_template_gpu) {
		fprintf(stderr, "cudaMalloc failed!\n");
		return;
	}

	GPUPtr nonce_gpu(sizeof(uint64_t));
	if (!nonce_gpu) {
		fprintf(stderr, "cudaMalloc failed!\n");
		return;
	}

	GPUPtr states_gpu(sizeof(hash) * NUM_SCRATCHPADS_BENCH);
	if (!states_gpu) {
		return;
	}

	GPUPtr scratchpads_gpu((RANDOMX_SCRATCHPAD_L3 + 64) * NUM_SCRATCHPADS_BENCH);
	if (!scratchpads_gpu) {
		fprintf(stderr, "cudaMalloc failed!\n");
		return;
	}

	GPUPtr programs_gpu(ENTROPY_SIZE * NUM_SCRATCHPADS_TEST);
	if (!programs_gpu) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

	GPUPtr registers_gpu(REGISTERS_SIZE * NUM_SCRATCHPADS_TEST);
	if (!registers_gpu) {
		fprintf(stderr, "cudaMalloc failed!\n");
		return;
	}

	cudaError_t cudaStatus = cudaMemcpy(block_template_gpu, blockTemplate, sizeof(blockTemplate), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		return;
	}

	{
		blake2b_initial_hash<sizeof(blockTemplate)><<<NUM_SCRATCHPADS_TEST / 32, 32>>>(hash_gpu, block_template_gpu, 0);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching blake2b_512_double_block_test!\n", cudaStatus);
			return;
		}

		cudaStatus = cudaMemcpy(&hash, hash_gpu, sizeof(hash), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			return;
		}

		for (uint32_t i = 0; i < NUM_SCRATCHPADS_TEST; ++i)
		{
			*(uint32_t*)(blockTemplate + 39) = i;
			blake2b(hash2 + i * 8, 64, blockTemplate, sizeof(blockTemplate), nullptr, 0);
		}

		if (memcmp(hash, hash2, sizeof(hash)) != 0)
		{
			fprintf(stderr, "blake2b_initial_hash test failed!\n");
			return;
		}

		printf("blake2b_initial_hash test passed\n");
	}

	{
		fillAes1Rx4<RANDOMX_SCRATCHPAD_L3, false, 64><<<NUM_SCRATCHPADS_TEST / 32, 32 * 4>>>(hash_gpu, scratchpads_gpu, NUM_SCRATCHPADS_TEST);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching blake2b_512_double_block_test!\n", cudaStatus);
			return;
		}

		cudaStatus = cudaMemcpy(&hash, hash_gpu, sizeof(hash), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			return;
		}

		cudaStatus = cudaMemcpy(scratchpads.data(), scratchpads_gpu, (RANDOMX_SCRATCHPAD_L3 + 64) * NUM_SCRATCHPADS_TEST, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			return;
		}

		for (int i = 0; i < NUM_SCRATCHPADS_TEST; ++i)
		{
			uint8_t* p = scratchpads.data() + (RANDOMX_SCRATCHPAD_L3 + 64) * (NUM_SCRATCHPADS_TEST + i);
			fillAes1Rx4<false>(hash2 + i * 8, RANDOMX_SCRATCHPAD_L3, p);

			if (memcmp(hash + i * 8, hash2 + i * 8, 64) != 0)
			{
				fprintf(stderr, "fillAes1Rx4 test (hash) failed!\n");
				return;
			}

			if (memcmp(scratchpads.data() + (RANDOMX_SCRATCHPAD_L3 + 64) * i, p, RANDOMX_SCRATCHPAD_L3) != 0)
			{
				fprintf(stderr, "fillAes1Rx4 test (scratchpad) failed!\n");
				return;
			}
		}

		printf("fillAes1Rx4 (scratchpads) test passed\n");
	}

	{
		fillAes4Rx4<ENTROPY_SIZE, false><<<NUM_SCRATCHPADS_TEST / 32, 32 * 4>>>(hash_gpu, programs_gpu, NUM_SCRATCHPADS_TEST);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching blake2b_512_double_block_test!\n", cudaStatus);
			return;
		}

		cudaStatus = cudaMemcpy(hash, hash_gpu, sizeof(hash), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			return;
		}

		cudaStatus = cudaMemcpy(programs.data(), programs_gpu, ENTROPY_SIZE * NUM_SCRATCHPADS_TEST, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			return;
		}

		for (int i = 0; i < NUM_SCRATCHPADS_TEST; ++i)
		{
			fillAes4Rx4<false>(hash2 + i * 8, ENTROPY_SIZE, programs.data() + ENTROPY_SIZE * (NUM_SCRATCHPADS_TEST + i));
			if (memcmp(programs.data() + i * ENTROPY_SIZE, programs.data() + (NUM_SCRATCHPADS_TEST + i) * ENTROPY_SIZE, ENTROPY_SIZE) != 0)
			{
				fprintf(stderr, "fillAes4Rx4 test (programs) failed!\n");
				return;
			}
		}

		printf("fillAes4Rx4 (programs) test passed\n");
	}
	
	{
		hashAes1Rx4<RANDOMX_SCRATCHPAD_L3, 192, REGISTERS_SIZE, 64><<<NUM_SCRATCHPADS_TEST / 32, 32 * 4>>>(scratchpads_gpu, registers_gpu, NUM_SCRATCHPADS_TEST);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching blake2b_512_double_block_test!\n", cudaStatus);
			return;
		}

		cudaStatus = cudaMemcpy(registers, registers_gpu, sizeof(registers), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			return;
		}

		for (int i = 0; i < NUM_SCRATCHPADS_TEST; ++i)
		{
			hashAes1Rx4<false>(scratchpads.data() + (RANDOMX_SCRATCHPAD_L3 + 64) * (NUM_SCRATCHPADS_TEST + i), RANDOMX_SCRATCHPAD_L3, registers2 + REGISTERS_SIZE * i + 192);

			if (memcmp(registers + i * REGISTERS_SIZE, registers2 + i * REGISTERS_SIZE, REGISTERS_SIZE) != 0)
			{
				fprintf(stderr, "hashAes1Rx4 test failed!\n");
				return;
			}
		}

		printf("hashAes1Rx4 test passed\n");
	}

	{
		blake2b_hash_registers<REGISTERS_SIZE, REGISTERS_SIZE, 32><<<NUM_SCRATCHPADS_TEST / 32, 32>>>(hash_gpu, registers_gpu);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "blake2b_hash_registers launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return;
		}

		cudaStatus = cudaMemcpy(&hash, hash_gpu, sizeof(hash), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			return;
		}

		for (uint32_t i = 0; i < NUM_SCRATCHPADS_TEST; ++i)
		{
			blake2b(hash2 + i * 4, 32, registers2 + i * REGISTERS_SIZE, REGISTERS_SIZE, nullptr, 0);
		}

		if (memcmp(hash, hash2, NUM_SCRATCHPADS_TEST * 32) != 0)
		{
			fprintf(stderr, "blake2b_hash_registers (32 byte hash) test failed!\n");
			return;
		}

		printf("blake2b_hash_registers (32 byte hash) test passed\n");
	}

	{
		blake2b_hash_registers<REGISTERS_SIZE, REGISTERS_SIZE, 64><<<NUM_SCRATCHPADS_TEST / 32, 32>>>(hash_gpu, registers_gpu);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "blake2b_hash_registers launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return;
		}

		cudaStatus = cudaMemcpy(&hash, hash_gpu, sizeof(hash), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			return;
		}

		for (uint32_t i = 0; i < NUM_SCRATCHPADS_TEST; ++i)
		{
			blake2b(hash2 + i * 8, 64, registers2 + i * REGISTERS_SIZE, REGISTERS_SIZE, nullptr, 0);
		}

		if (memcmp(hash, hash2, NUM_SCRATCHPADS_TEST * 64) != 0)
		{
			fprintf(stderr, "blake2b_hash_registers (64 byte hash) test failed!\n");
			return;
		}

		printf("blake2b_hash_registers (64 byte hash) test passed\n");
	}

	auto start_time = high_resolution_clock::now();

	for (int i = 0; i < 100; ++i)
	{
		printf("Benchmarking fillAes1Rx4 %d/100", i + 1);
		if (i > 0)
		{
			const double dt = duration_cast<nanoseconds>(high_resolution_clock::now() - start_time).count() / 1e9;
			printf(", %.0f scratchpads/s", (i * NUM_SCRATCHPADS_BENCH * 10) / dt);
		}
		printf("\r");

		for (int j = 0; j < 10; ++j)
		{
			fillAes1Rx4<RANDOMX_SCRATCHPAD_L3, false, 64><<<NUM_SCRATCHPADS_BENCH / 32, 32 * 4>>>(states_gpu, scratchpads_gpu, NUM_SCRATCHPADS_BENCH);

			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "fillAes1Rx4 launch failed: %s\n", cudaGetErrorString(cudaStatus));
				return;
			}
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching fillAes1Rx4!\n", cudaStatus);
			return;
		}
	}
	printf("\n");

	start_time = high_resolution_clock::now();

	for (int i = 0; i < 100; ++i)
	{
		printf("Benchmarking hashAes1Rx4 %d/100", i + 1);
		if (i > 0)
		{
			const double dt = duration_cast<nanoseconds>(high_resolution_clock::now() - start_time).count() / 1e9;
			printf(", %.0f scratchpads/s", (i * NUM_SCRATCHPADS_BENCH * 10) / dt);
		}
		printf("\r");

		for (int j = 0; j < 10; ++j)
		{
			hashAes1Rx4<RANDOMX_SCRATCHPAD_L3, 0, 64, 64><<<NUM_SCRATCHPADS_BENCH / 32, 32 * 4>>>(scratchpads_gpu, states_gpu, NUM_SCRATCHPADS_BENCH);

			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "hashAes1Rx4 launch failed: %s\n", cudaGetErrorString(cudaStatus));
				return;
			}
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching hashAes1Rx4!\n", cudaStatus);
			return;
		}
	}
	printf("\n");

	cudaStatus = cudaMemcpy(block_template_gpu, blockTemplate, sizeof(blockTemplate), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		return;
	}

	start_time = high_resolution_clock::now();

	for (uint64_t start_nonce = 0; start_nonce < BLAKE2B_STEP * 100; start_nonce += BLAKE2B_STEP)
	{
		printf("Benchmarking blake2b_512_single_block %zu/100", (start_nonce + BLAKE2B_STEP) / BLAKE2B_STEP);
		if (start_nonce > 0)
		{
			const double dt = duration_cast<nanoseconds>(high_resolution_clock::now() - start_time).count() / 1e9;
			printf(", %.2f MH/s", start_nonce / dt / 1e6);
		}
		printf("\r");

		cudaStatus = cudaMemset(nonce_gpu, 0, sizeof(uint64_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemset failed!\n");
			return;
		}

		void* out = nonce_gpu;
		blake2b_512_single_block_bench<sizeof(blockTemplate)><<<BLAKE2B_STEP / 256, 256>>>((uint64_t*) out, block_template_gpu, start_nonce);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "blake2b_512_single_block_bench launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching blake2b_512_single_block_bench!\n", cudaStatus);
			return;
		}

		uint64_t nonce;
		cudaStatus = cudaMemcpy(&nonce, nonce_gpu, sizeof(nonce), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			return;
		}

		if (nonce)
		{
			*(uint64_t*)(blockTemplate) = nonce;
			blake2b(hash, 64, blockTemplate, sizeof(blockTemplate), nullptr, 0);
			printf("nonce = %zu, hash[7] = %016zx                  \n", nonce, hash[7]);
		}
	}
	printf("\n");

	start_time = high_resolution_clock::now();

	for (uint64_t start_nonce = 0; start_nonce < BLAKE2B_STEP * 100; start_nonce += BLAKE2B_STEP)
	{
		printf("Benchmarking blake2b_512_double_block %zu/100", (start_nonce + BLAKE2B_STEP) / BLAKE2B_STEP);
		if (start_nonce > 0)
		{
			const double dt = duration_cast<nanoseconds>(high_resolution_clock::now() - start_time).count() / 1e9;
			printf(", %.2f MH/s", start_nonce / dt / 1e6);
		}
		printf("\r");

		const uint64_t zero = 0;
		cudaStatus = cudaMemcpy(nonce_gpu, &zero, sizeof(zero), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			return;
		}

		void* out = nonce_gpu;
		blake2b_512_double_block_bench<REGISTERS_SIZE><<<BLAKE2B_STEP / 256, 256>>>((uint64_t*) out, registers_gpu, start_nonce);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "blake2b_512_double_block_bench launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching blake2b_512_double_block_bench!\n", cudaStatus);
			return;
		}

		uint64_t nonce;
		cudaStatus = cudaMemcpy(&nonce, nonce_gpu, sizeof(nonce), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			return;
		}

		if (nonce)
		{
			*(uint64_t*)(registers) = nonce;
			blake2b(hash, 64, registers, REGISTERS_SIZE, nullptr, 0);
			printf("nonce = %zu, hash[7] = %016zx                  \n", nonce, hash[7]);
		}
	}
	printf("\n");
}
