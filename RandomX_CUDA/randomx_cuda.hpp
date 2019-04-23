#pragma once

/*
Copyright (c) 2019 SChernykh
Portions Copyright (c) 2018-2019 tevador

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

constexpr size_t DATASET_SIZE = 1U << 31;
constexpr size_t SCRATCHPAD_SIZE = 1U << 21;
constexpr size_t HASH_SIZE = 64;
constexpr size_t PROGRAM_SIZE = 128 + 2048;
constexpr size_t PROGRAM_COUNT = 8;
constexpr size_t REGISTERS_SIZE = 256;

constexpr int PROGRAM_ITERATIONS = 2048;

constexpr int ScratchpadL3Mask64 = (1 << 21) - 64;

constexpr uint32_t CacheLineSize = 64;
constexpr uint32_t CacheLineAlignMask = (DATASET_SIZE - 1) & ~(CacheLineSize - 1);

__device__ uint64_t getSmallPositiveFloatBits(uint64_t entropy)
{
	constexpr int mantissaSize = 52;
	constexpr int exponentSize = 11;
	constexpr uint64_t mantissaMask = (1ULL << mantissaSize) - 1;
	constexpr uint64_t exponentMask = (1ULL << exponentSize) - 1;
	constexpr int exponentBias = 1023;

	auto exponent = entropy >> 59; //0..31
	auto mantissa = entropy & mantissaMask;
	exponent += exponentBias;
	exponent &= exponentMask;
	exponent <<= mantissaSize;
	return exponent | mantissa;
}

template<typename T>
__device__ T bit_cast(double value)
{
	return static_cast<T>(__double_as_longlong(value));
}

__device__ double load_E_group(int value, uint64_t eMask)
{
	uint64_t x = bit_cast<uint64_t>(__int2double_rn(value));
	x &= (1ULL << 52) - 1;
	x |= eMask;
	return __longlong_as_double(static_cast<int64_t>(x));
}

__global__ void __launch_bounds__(32) execute_vm(const void* entropy_data, void* registers, void* scratchpads, const void* dataset, uint32_t batch_size)
{
	__shared__ uint64_t registers_buf[32 * 16];

	const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t idx = global_index / 2;
	const uint32_t sub = global_index % 2;

	uint64_t* R = registers_buf + (threadIdx.x / 2) * 32;
	double* F = (double*)(R + 8);
	double* E = (double*)(R + 16);
	double* A = (double*)(R + 24);

	R[sub + 0] = 0;
	R[sub + 2] = 0;
	R[sub + 4] = 0;
	R[sub + 6] = 0;

	const uint64_t* e = ((const uint64_t*) entropy_data) + idx * (PROGRAM_SIZE / sizeof(uint64_t));

	A[sub + 0] = getSmallPositiveFloatBits(e[sub + 0]);
	A[sub + 2] = getSmallPositiveFloatBits(e[sub + 2]);
	A[sub + 4] = getSmallPositiveFloatBits(e[sub + 4]);
	A[sub + 6] = getSmallPositiveFloatBits(e[sub + 6]);

	__syncthreads();

	uint32_t ma = static_cast<uint32_t>(e[8]) & CacheLineAlignMask;
	uint32_t mx = static_cast<uint32_t>(e[10]);

	const uint32_t addressRegisters = static_cast<uint32_t>(e[12]);
	const uint32_t readReg0 = (addressRegisters & 1);
	const uint32_t readReg1 = (addressRegisters & 2) ? 3 : 2;
	const uint32_t readReg2 = (addressRegisters & 4) ? 5 : 4;
	const uint32_t readReg3 = (addressRegisters & 8) ? 7 : 6;

	const uint64_t eMask = (e[sub + 14] & ((1ULL << 22) - 1)) | ((1023ULL - 240) << 52);

	uint32_t spAddr0 = mx;
	uint32_t spAddr1 = ma;

	uint8_t* scratchpad = ((uint8_t*) scratchpads) + idx * 64;

	for (int ic = 0; ic < PROGRAM_ITERATIONS; ++ic)
	{
		const uint64_t spMix = R[readReg0] ^ R[readReg1];
		spAddr0 ^= ((const uint32_t*) &spMix)[0];
		spAddr1 ^= ((const uint32_t*) &spMix)[1];
		spAddr0 &= ScratchpadL3Mask64;
		spAddr1 &= ScratchpadL3Mask64;

		uint64_t* p = (uint64_t*)(scratchpad + spAddr0 * batch_size);
		R[sub + 0] ^= p[sub + 0];
		R[sub + 2] ^= p[sub + 2];
		R[sub + 4] ^= p[sub + 4];
		R[sub + 6] ^= p[sub + 6];

		int32_t* q = (int32_t*)(scratchpad + spAddr1 * batch_size);
		F[sub + 0] = __int2double_rn(q[sub + 0]);
		F[sub + 2] = __int2double_rn(q[sub + 2]);
		F[sub + 4] = __int2double_rn(q[sub + 4]);
		F[sub + 6] = __int2double_rn(q[sub + 6]);

		E[sub + 0] = load_E_group(q[sub +  8], eMask);
		E[sub + 2] = load_E_group(q[sub + 10], eMask);
		E[sub + 4] = load_E_group(q[sub + 12], eMask);
		E[sub + 6] = load_E_group(q[sub + 14], eMask);

		__syncthreads();

		// TODO: execute byte code

		mx ^= R[readReg2] ^ R[readReg3];
		mx &= CacheLineAlignMask;

		const uint64_t* datasetLine = (const uint64_t*)(((const uint8_t*) dataset) + ma);
		R[sub + 0] ^= datasetLine[sub + 0];
		R[sub + 2] ^= datasetLine[sub + 2];
		R[sub + 4] ^= datasetLine[sub + 4];
		R[sub + 6] ^= datasetLine[sub + 6];

		const uint32_t tmp = ma;
		ma = mx;
		mx = tmp;

		p = (uint64_t*)(scratchpad + spAddr1 * batch_size);
		p[sub + 0] = R[sub + 0];
		p[sub + 2] = R[sub + 2];
		p[sub + 4] = R[sub + 4];
		p[sub + 6] = R[sub + 6];

		p = (uint64_t*)(scratchpad + spAddr0 * batch_size);
		p[sub + 0] = bit_cast<uint64_t>(F[sub + 0]) ^ bit_cast<uint64_t>(E[sub + 0]);
		p[sub + 2] = bit_cast<uint64_t>(F[sub + 2]) ^ bit_cast<uint64_t>(E[sub + 2]);
		p[sub + 4] = bit_cast<uint64_t>(F[sub + 4]) ^ bit_cast<uint64_t>(E[sub + 4]);
		p[sub + 6] = bit_cast<uint64_t>(F[sub + 6]) ^ bit_cast<uint64_t>(E[sub + 6]);

		spAddr0 = 0;
		spAddr1 = 0;
	}

	uint64_t* p = ((uint64_t*) registers) + idx * (REGISTERS_SIZE / sizeof(uint64_t));

	p[sub + 0] = R[sub + 0];
	p[sub + 2] = R[sub + 2];
	p[sub + 4] = R[sub + 4];
	p[sub + 6] = R[sub + 6];

	p[sub +  8] = bit_cast<uint64_t>(F[sub + 0]);
	p[sub + 10] = bit_cast<uint64_t>(F[sub + 2]);
	p[sub + 12] = bit_cast<uint64_t>(F[sub + 4]);
	p[sub + 14] = bit_cast<uint64_t>(F[sub + 6]);

	p[sub + 16] = bit_cast<uint64_t>(E[sub + 0]);
	p[sub + 18] = bit_cast<uint64_t>(E[sub + 2]);
	p[sub + 20] = bit_cast<uint64_t>(E[sub + 4]);
	p[sub + 22] = bit_cast<uint64_t>(E[sub + 6]);

	p[sub + 24] = bit_cast<uint64_t>(A[sub + 0]);
	p[sub + 26] = bit_cast<uint64_t>(A[sub + 2]);
	p[sub + 28] = bit_cast<uint64_t>(A[sub + 4]);
	p[sub + 30] = bit_cast<uint64_t>(A[sub + 6]);
}
