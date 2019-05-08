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

bool test_mining(bool validate, int bfactor, int workers_per_hash);
void tests();

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		printf("Usage: RandomX_CUDA.exe --mine device_id [--validate] [--bfactor N] [--workers N]\n\n");
		printf("device_id is 0 if you only have 1 GPU\n");
		printf("bfactor can be 0-10, default is 0. Increase it if you get CUDA errors/driver crashes/screen lags.\n");
		printf("workers can be 2,4,8, default is 8. Choose the value that gives you the best hashrate (it's usually 4 or 8).\n\n");
		printf("Examples:\nRandomX_CUDA.exe --test 0\nRandomX_CUDA.exe --mine 0 --validate --bfactor 3 --workers 4\n");
		return 0;
	}

	const int device_id = atoi(argv[2]);

	cudaError_t cudaStatus = cudaSetDevice(device_id);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		return 1;
	}

	uint32_t flags;
	if (cudaGetDeviceFlags(&flags) == cudaSuccess)
		cudaSetDeviceFlags(flags | cudaDeviceScheduleBlockingSync);

	bool validate = false;
	int bfactor = 0;
	int workers_per_hash = 8;
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
				break;

			default:
				workers_per_hash = 4;
			}
		}
	}

	if (strcmp(argv[1], "--mine") == 0)
		test_mining(validate, bfactor, workers_per_hash);
	else if (strcmp(argv[1], "--test") == 0)
		tests();

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
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

bool test_mining(bool validate, int bfactor, int workers_per_hash)
{
	printf("Testing mining: CPU validation is %s, bfactor is %d, %d workers per hash\n", validate ? "ON" : "OFF", bfactor, workers_per_hash);

	cudaError_t cudaStatus;

	size_t free_mem, total_mem;
	{
		GPUPtr tmp(256);
		cudaStatus = cudaMemGetInfo(&free_mem, &total_mem);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "Failed to get free memory info!");
			return false;
		}
	}

	printf("%zu MB GPU memory free\n", free_mem >> 20);
	printf("%zu MB GPU memory total\n", total_mem >> 20);

	// There should be enough GPU memory for the 2080 MB dataset, 32 scratchpads and 64 MB for everything else
	const size_t dataset_size = randomx_dataset_item_count() * RANDOMX_DATASET_ITEM_SIZE;
	if (free_mem <= dataset_size + (32U * SCRATCHPAD_SIZE) + (64U << 20))
	{
		fprintf(stderr, "Not enough free GPU memory!");
		return false;
	}

	const uint32_t batch_size = static_cast<uint32_t>((((free_mem - dataset_size - (64U << 20)) / SCRATCHPAD_SIZE) / 32) * 32);

	GPUPtr dataset_gpu(dataset_size);
	if (!dataset_gpu)
	{
		fprintf(stderr, "Failed to allocate GPU memory for dataset!");
		return false;
	}

	printf("Allocated %.0f MB dataset\n", dataset_size / 1048576.0);

	printf("Initializing dataset...");

	randomx_dataset *myDataset;
	{
		const char mySeed[] = "RandomX example seed";

		randomx_cache *myCache = randomx_alloc_cache((randomx_flags)(RANDOMX_FLAG_JIT));
		randomx_init_cache(myCache, mySeed, sizeof mySeed);
		myDataset = randomx_alloc_dataset(RANDOMX_FLAG_LARGE_PAGES);

		time_point<steady_clock> t1 = high_resolution_clock::now();

		std::vector<std::thread> threads;
		for (uint32_t i = 0, n = std::thread::hardware_concurrency(); i < n; ++i)
			threads.emplace_back(randomx_init_dataset, myDataset, myCache, (i * randomx_dataset_item_count()) / n, ((i + 1) * randomx_dataset_item_count()) / n - (i * randomx_dataset_item_count()) / n);

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

	GPUPtr scratchpads_gpu(batch_size * SCRATCHPAD_SIZE);
	if (!scratchpads_gpu)
	{
		fprintf(stderr, "Failed to allocate GPU memory for scratchpads!");
		return false;
	}

	printf("Allocated %u scratchpads\n", batch_size);

	GPUPtr hashes_gpu(batch_size * HASH_SIZE);
	if (!hashes_gpu)
	{
		fprintf(stderr, "Failed to allocate GPU memory for hashes!");
		return false;
	}

	GPUPtr entropy_gpu(batch_size * ENTROPY_SIZE);
	if (!entropy_gpu)
	{
		fprintf(stderr, "Failed to allocate GPU memory for programs!");
		return false;
	}

	GPUPtr vm_states_gpu(batch_size * VM_STATE_SIZE);
	if (!vm_states_gpu)
	{
		fprintf(stderr, "Failed to allocate GPU memory for VM states!");
		return false;
	}

	GPUPtr rounding_gpu(batch_size * sizeof(uint32_t));
	if (!rounding_gpu)
	{
		fprintf(stderr, "Failed to allocate GPU memory for VM rounding data!");
		return false;
	}

	GPUPtr num_vm_cycles_gpu(sizeof(uint64_t));
	if (!num_vm_cycles_gpu)
	{
		fprintf(stderr, "Failed to allocate GPU memory for VM num cyles data!");
		return false;
	}

	cudaMemset(num_vm_cycles_gpu, 0, sizeof(uint64_t));

	GPUPtr blockTemplate_gpu(sizeof(blockTemplate));
	if (!blockTemplate_gpu)
	{
		fprintf(stderr, "Failed to allocate GPU memory for block template!");
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
		fprintf(stderr, "Failed to get free memory info!");
		return false;
	}

	printf("%zu MB free GPU memory left\n", free_mem >> 20);

	const void* init_vm_list[] = { init_vm<2>, init_vm<4>, init_vm<8> };
	const void* execute_vm_list[] = { execute_vm<2>, execute_vm<4>, execute_vm<8> };

	for (int i = 0; i < 3; ++i)
	{
		cudaStatus = cudaFuncSetCacheConfig(init_vm_list[i], cudaFuncCachePreferShared);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "Failed to set cache config for init_vm<%d>!", 1 << i);
			return false;
		}

		cudaStatus = cudaFuncSetCacheConfig(execute_vm_list[i], cudaFuncCachePreferShared);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "Failed to set cache config for execute_vm<%d>!", 1 << i);
			return false;
		}
	}

	time_point<steady_clock> prev_time;

	std::vector<uint8_t> hashes, hashes_check;
	hashes.resize(batch_size * 32);
	hashes_check.resize(batch_size * 32);

	std::vector<std::thread> threads;
	std::atomic<uint32_t> nonce_counter;
	bool cpu_limited = false;

	for (uint32_t nonce = 0, k = 0; nonce < 0xFFFFFFFFUL; nonce += batch_size, ++k)
	{
		auto validation_thread = [&nonce_counter, myDataset, &hashes_check, batch_size, nonce]() {
			randomx_vm *myMachine = randomx_create_vm((randomx_flags)(RANDOMX_FLAG_FULL_MEM | RANDOMX_FLAG_JIT | RANDOMX_FLAG_HARD_AES | RANDOMX_FLAG_LARGE_PAGES), nullptr, myDataset);

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

		time_point<steady_clock> cur_time = high_resolution_clock::now();
		if (k > 0)
		{
			const double dt = duration_cast<nanoseconds>(cur_time - prev_time).count() / 1e9;
			uint64_t data = uint64_t(-1);
			cudaMemcpy(&data, num_vm_cycles_gpu, sizeof(uint64_t), cudaMemcpyDeviceToHost);

			const double num_vm_cycles = static_cast<uint32_t>(data);
			const double num_slots_used = static_cast<uint32_t>(data >> 32);
			if (validate)
				printf("%u hashes validated successfully, IPC %.4f, WPC %.4f, %.0f h/s%s    \r", nonce, nonce * RANDOMX_PROGRAM_SIZE * RANDOMX_PROGRAM_COUNT / num_vm_cycles, num_slots_used / num_vm_cycles, batch_size / dt, cpu_limited ? ", limited by CPU" : "                ");
			else
				printf("%.0f h/s\t\r", batch_size / dt);
		}
		prev_time = cur_time;

		blake2b_initial_hash<sizeof(blockTemplate)><<<batch_size / 32, 32>>>(hashes_gpu, blockTemplate_gpu, nonce);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "blake2b_initial_hash launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return false;
		}

		fillAes1Rx4<SCRATCHPAD_SIZE, true><<<batch_size / 32, 32 * 4>>>(hashes_gpu, scratchpads_gpu, batch_size);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "fillAes1Rx4 launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return false;
		}

		cudaStatus = cudaMemset(rounding_gpu, 0, batch_size * sizeof(uint32_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemset failed!");
			return false;
		}

		for (size_t i = 0; i < RANDOMX_PROGRAM_COUNT; ++i)
		{
			fillAes1Rx4<ENTROPY_SIZE, false><<<batch_size / 32, 32 * 4>>>(hashes_gpu, entropy_gpu, batch_size);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "fillAes1Rx4 launch failed: %s\n", cudaGetErrorString(cudaStatus));
				return false;
			}

			switch (workers_per_hash)
			{
			case 2:
				init_vm<2><<<batch_size / 4, 4 * 8>>>(entropy_gpu, vm_states_gpu, num_vm_cycles_gpu);
				for (int j = 0, n = 1 << bfactor; j < n; ++j)
				{
					execute_vm<2><<<batch_size / 2, 2 * 8>>>(vm_states_gpu, rounding_gpu, scratchpads_gpu, dataset_gpu, batch_size, RANDOMX_PROGRAM_ITERATIONS >> bfactor, j == 0, j == n - 1);
				}
				break;

			case 4:
				init_vm<4><<<batch_size / 4, 4 * 8>>>(entropy_gpu, vm_states_gpu, num_vm_cycles_gpu);
				for (int j = 0, n = 1 << bfactor; j < n; ++j)
				{
					execute_vm<4><<<batch_size / 2, 2 * 8>>>(vm_states_gpu, rounding_gpu, scratchpads_gpu, dataset_gpu, batch_size, RANDOMX_PROGRAM_ITERATIONS >> bfactor, j == 0, j == n - 1);
				}
				break;

			case 8:
				init_vm<8><<<batch_size / 4, 4 * 8>>>(entropy_gpu, vm_states_gpu, num_vm_cycles_gpu);
				for (int j = 0, n = 1 << bfactor; j < n; ++j)
				{
					execute_vm<8><<<batch_size / 2, 2 * 8>>>(vm_states_gpu, rounding_gpu, scratchpads_gpu, dataset_gpu, batch_size, RANDOMX_PROGRAM_ITERATIONS >> bfactor, j == 0, j == n - 1);
				}
				break;
			}

			if (i == RANDOMX_PROGRAM_COUNT - 1)
			{
				hashAes1Rx4<SCRATCHPAD_SIZE, 192, VM_STATE_SIZE><<<batch_size / 32, 32 * 4>>>(scratchpads_gpu, vm_states_gpu, batch_size);
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
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d!\n", cudaStatus);
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
				fprintf(stderr, "\nCPU validation error, ");
				for (uint32_t i = 0; i < batch_size * 32; i += 32)
				{
					if (memcmp(hashes.data() + i, hashes_check.data() + i, 32))
					{
						fprintf(stderr, "failing nonce = %u\n", nonce + i / 32);
						break;
					}
				}
				return false;
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

	std::vector<uint8_t> scratchpads(SCRATCHPAD_SIZE * NUM_SCRATCHPADS_TEST * 2);
	std::vector<uint8_t> programs(ENTROPY_SIZE * NUM_SCRATCHPADS_TEST * 2);

	uint64_t hash[NUM_SCRATCHPADS_TEST * 8] = {};
	uint64_t hash2[NUM_SCRATCHPADS_TEST * 8] = {};

	uint8_t registers[NUM_SCRATCHPADS_TEST * REGISTERS_SIZE] = {};
	uint8_t registers2[NUM_SCRATCHPADS_TEST * REGISTERS_SIZE] = {};

	GPUPtr hash_gpu(sizeof(hash));
	if (!hash_gpu) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

	GPUPtr block_template_gpu(sizeof(blockTemplate));
	if (!block_template_gpu) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

	GPUPtr nonce_gpu(sizeof(uint64_t));
	if (!nonce_gpu) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

	GPUPtr states_gpu(sizeof(hash) * NUM_SCRATCHPADS_BENCH);
	if (!states_gpu) {
		return;
	}

	GPUPtr scratchpads_gpu(SCRATCHPAD_SIZE * NUM_SCRATCHPADS_BENCH);
	if (!scratchpads_gpu) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

	GPUPtr programs_gpu(SCRATCHPAD_SIZE * NUM_SCRATCHPADS_TEST);
	if (!programs_gpu) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

	GPUPtr registers_gpu(REGISTERS_SIZE * NUM_SCRATCHPADS_TEST);
	if (!registers_gpu) {
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

	cudaError_t cudaStatus = cudaMemcpy(block_template_gpu, blockTemplate, sizeof(blockTemplate), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
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
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}

		for (uint32_t i = 0; i < NUM_SCRATCHPADS_TEST; ++i)
		{
			*(uint32_t*)(blockTemplate + 39) = i;
			blake2b(hash2 + i * 8, 64, blockTemplate, sizeof(blockTemplate), nullptr, 0);
		}

		if (memcmp(hash, hash2, sizeof(hash)) != 0)
		{
			fprintf(stderr, "blake2b_initial_hash test failed!");
			return;
		}

		printf("blake2b_initial_hash test passed\n");
	}

	{
		fillAes1Rx4<SCRATCHPAD_SIZE, true><<<NUM_SCRATCHPADS_TEST / 32, 32 * 4>>>(hash_gpu, scratchpads_gpu, NUM_SCRATCHPADS_TEST);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching blake2b_512_double_block_test!\n", cudaStatus);
			return;
		}

		cudaStatus = cudaMemcpy(&hash, hash_gpu, sizeof(hash), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}

		cudaStatus = cudaMemcpy(scratchpads.data(), scratchpads_gpu, SCRATCHPAD_SIZE * NUM_SCRATCHPADS_TEST, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}

		for (int i = 0; i < NUM_SCRATCHPADS_TEST; ++i)
		{
			fillAes1Rx4<false>(hash2 + i * 8, SCRATCHPAD_SIZE, scratchpads.data() + SCRATCHPAD_SIZE * (NUM_SCRATCHPADS_TEST + i));

			if (memcmp(hash + i * 8, hash2 + i * 8, 64) != 0)
			{
				fprintf(stderr, "fillAes1Rx4 test (hash) failed!");
				return;
			}

			const uint8_t* p1 = scratchpads.data() + i * 64;
			const uint8_t* p2 = scratchpads.data() + SCRATCHPAD_SIZE * (NUM_SCRATCHPADS_TEST + i);
			for (int j = 0; j < SCRATCHPAD_SIZE; j += 64)
			{
				if (memcmp(p1 + j * NUM_SCRATCHPADS_TEST, p2 + j, 64) != 0)
				{
					fprintf(stderr, "fillAes1Rx4 test (scratchpad) failed!");
					return;
				}
			}
		}

		printf("fillAes1Rx4 (scratchpads) test passed\n");
	}

	{
		fillAes1Rx4<ENTROPY_SIZE, false><<<NUM_SCRATCHPADS_TEST / 32, 32 * 4 >>>(hash_gpu, programs_gpu, NUM_SCRATCHPADS_TEST);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching blake2b_512_double_block_test!\n", cudaStatus);
			return;
		}

		cudaStatus = cudaMemcpy(hash, hash_gpu, sizeof(hash), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}

		cudaStatus = cudaMemcpy(programs.data(), programs_gpu, ENTROPY_SIZE * NUM_SCRATCHPADS_TEST, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}

		for (int i = 0; i < NUM_SCRATCHPADS_TEST; ++i)
		{
			fillAes1Rx4<false>(hash2 + i * 8, ENTROPY_SIZE, programs.data() + ENTROPY_SIZE * (NUM_SCRATCHPADS_TEST + i));

			if (memcmp(hash + i * 8, hash2 + i * 8, 64) != 0)
			{
				fprintf(stderr, "fillAes1Rx4 test (hash) failed!");
				return;
			}

			if (memcmp(programs.data() + i * ENTROPY_SIZE, programs.data() + (NUM_SCRATCHPADS_TEST + i) * ENTROPY_SIZE, ENTROPY_SIZE) != 0)
			{
				fprintf(stderr, "fillAes1Rx4 test (program) failed!");
				return;
			}
		}

		printf("fillAes1Rx4 (programs) test passed\n");
	}
	
	{
		hashAes1Rx4<SCRATCHPAD_SIZE, 192, REGISTERS_SIZE><<<NUM_SCRATCHPADS_TEST / 32, 32 * 4>>>(scratchpads_gpu, registers_gpu, NUM_SCRATCHPADS_TEST);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching blake2b_512_double_block_test!\n", cudaStatus);
			return;
		}

		cudaStatus = cudaMemcpy(registers, registers_gpu, sizeof(registers), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}

		for (int i = 0; i < NUM_SCRATCHPADS_TEST; ++i)
		{
			hashAes1Rx4<false>(scratchpads.data() + SCRATCHPAD_SIZE * (NUM_SCRATCHPADS_TEST + i), SCRATCHPAD_SIZE, registers2 + REGISTERS_SIZE * i + 192);

			if (memcmp(registers + i * REGISTERS_SIZE, registers2 + i * REGISTERS_SIZE, REGISTERS_SIZE) != 0)
			{
				fprintf(stderr, "hashAes1Rx4 test failed!");
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
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}

		for (uint32_t i = 0; i < NUM_SCRATCHPADS_TEST; ++i)
		{
			blake2b(hash2 + i * 4, 32, registers2 + i * REGISTERS_SIZE, REGISTERS_SIZE, nullptr, 0);
		}

		if (memcmp(hash, hash2, NUM_SCRATCHPADS_TEST * 32) != 0)
		{
			fprintf(stderr, "blake2b_hash_registers (32 byte hash) test failed!");
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
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}

		for (uint32_t i = 0; i < NUM_SCRATCHPADS_TEST; ++i)
		{
			blake2b(hash2 + i * 8, 64, registers2 + i * REGISTERS_SIZE, REGISTERS_SIZE, nullptr, 0);
		}

		if (memcmp(hash, hash2, NUM_SCRATCHPADS_TEST * 64) != 0)
		{
			fprintf(stderr, "blake2b_hash_registers (64 byte hash) test failed!");
			return;
		}

		printf("blake2b_hash_registers (64 byte hash) test passed\n");
	}

	time_point<steady_clock> start_time = high_resolution_clock::now();

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
			fillAes1Rx4<SCRATCHPAD_SIZE, true><<<NUM_SCRATCHPADS_BENCH / 32, 32 * 4>>>(states_gpu, scratchpads_gpu, NUM_SCRATCHPADS_BENCH);

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
			hashAes1Rx4<SCRATCHPAD_SIZE, 0, 64><<<NUM_SCRATCHPADS_BENCH / 32, 32 * 4>>>(scratchpads_gpu, states_gpu, NUM_SCRATCHPADS_BENCH);

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
		fprintf(stderr, "cudaMemcpy failed!");
		return;
	}

	start_time = high_resolution_clock::now();

	for (uint64_t start_nonce = 0; start_nonce < BLAKE2B_STEP * 100; start_nonce += BLAKE2B_STEP)
	{
		printf("Benchmarking blake2b_512_single_block %llu/100", (start_nonce + BLAKE2B_STEP) / BLAKE2B_STEP);
		if (start_nonce > 0)
		{
			const double dt = duration_cast<nanoseconds>(high_resolution_clock::now() - start_time).count() / 1e9;
			printf(", %.2f MH/s", start_nonce / dt / 1e6);
		}
		printf("\r");

		cudaStatus = cudaMemset(nonce_gpu, 0, sizeof(uint64_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemset failed!");
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
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}

		if (nonce)
		{
			*(uint64_t*)(blockTemplate) = nonce;
			blake2b(hash, 64, blockTemplate, sizeof(blockTemplate), nullptr, 0);
			printf("nonce = %llu, hash[7] = %016llx                  \n", nonce, hash[7]);
		}
	}
	printf("\n");

	start_time = high_resolution_clock::now();

	for (uint64_t start_nonce = 0; start_nonce < BLAKE2B_STEP * 100; start_nonce += BLAKE2B_STEP)
	{
		printf("Benchmarking blake2b_512_double_block %llu/100", (start_nonce + BLAKE2B_STEP) / BLAKE2B_STEP);
		if (start_nonce > 0)
		{
			const double dt = duration_cast<nanoseconds>(high_resolution_clock::now() - start_time).count() / 1e9;
			printf(", %.2f MH/s", start_nonce / dt / 1e6);
		}
		printf("\r");

		const uint64_t zero = 0;
		cudaStatus = cudaMemcpy(nonce_gpu, &zero, sizeof(zero), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
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
			fprintf(stderr, "cudaMemcpy failed!");
			return;
		}

		if (nonce)
		{
			*(uint64_t*)(registers) = nonce;
			blake2b(hash, 64, registers, REGISTERS_SIZE, nullptr, 0);
			printf("nonce = %llu, hash[7] = %016llx                  \n", nonce, hash[7]);
		}
	}
	printf("\n");
}
