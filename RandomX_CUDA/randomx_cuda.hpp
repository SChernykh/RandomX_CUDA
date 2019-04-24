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

__device__ double getSmallPositiveFloatBits(uint64_t entropy)
{
	constexpr int mantissaSize = 52;
	constexpr int exponentSize = 11;
	constexpr uint64_t mantissaMask = (1ULL << mantissaSize) - 1;
	constexpr uint64_t exponentMask = (1ULL << exponentSize) - 1;
	constexpr int exponentBias = 1023;

	uint64_t exponent = entropy >> 59; //0..31
	uint64_t mantissa = entropy & mantissaMask;
	exponent += exponentBias;
	exponent &= exponentMask;
	exponent <<= mantissaSize;
	return __longlong_as_double(exponent | mantissa);
}

template<typename T>
__device__ T bit_cast(double value)
{
	return static_cast<T>(__double_as_longlong(value));
}

__device__ double load_F_E_groups(int value, uint64_t andMask, uint64_t orMask)
{
	uint64_t x = bit_cast<uint64_t>(__int2double_rn(value));
	x &= andMask;
	x |= orMask;
	return __longlong_as_double(static_cast<int64_t>(x));
}

__global__ void __launch_bounds__(32) execute_vm(const void* entropy_data, void* registers, void* scratchpads, const void* dataset, uint32_t batch_size)
{
	__shared__ uint64_t registers_buf[32 * 8];

	const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t idx = global_index / 4;
	const uint32_t sub = global_index % 4;

	uint64_t* R = registers_buf + (threadIdx.x / 4) * 32;
	double* F = (double*)(R + 8);
	double* E = (double*)(R + 16);
	double* A = (double*)(R + 24);

	R[sub + 0] = 0;
	R[sub + 4] = 0;

	const uint64_t* entropy = ((const uint64_t*)entropy_data) + idx * (PROGRAM_SIZE / sizeof(uint64_t));
	A[sub + 0] = getSmallPositiveFloatBits(entropy[sub + 0]);
	A[sub + 4] = getSmallPositiveFloatBits(entropy[sub + 4]);

	__syncthreads();

	uint32_t ma = static_cast<uint32_t>(entropy[8]) & CacheLineAlignMask;
	uint32_t mx = static_cast<uint32_t>(entropy[10]);

	const uint32_t addressRegisters = static_cast<uint32_t>(entropy[12]);
	const uint64_t* readReg0 = R + (addressRegisters & 1);
	const uint64_t* readReg1 = R + ((addressRegisters & 2) ? 3 : 2);
	const uint64_t* readReg2 = R + ((addressRegisters & 4) ? 5 : 4);
	const uint64_t* readReg3 = R + ((addressRegisters & 8) ? 7 : 6);

	ulonglong2 eMask = *(ulonglong2*)(entropy + 14);
	eMask.x = (eMask.x & ((1ULL << 22) - 1)) | ((1023ULL - 240) << 52);
	eMask.y = (eMask.y & ((1ULL << 22) - 1)) | ((1023ULL - 240) << 52);

	uint32_t spAddr0 = mx;
	uint32_t spAddr1 = ma;

	uint8_t* scratchpad = ((uint8_t*) scratchpads) + idx * 64;

	const bool f_group = (sub < 2);

	double* fe = f_group ? (F + sub * 4) : (E + (sub - 2) * 4);
	double* f = F + sub * 2;
	double* e = E + sub * 2;

	const uint64_t andMask = f_group ? uint64_t(-1) : ((1ULL << 52) - 1);
	const uint64_t orMask1 = f_group ? 0 : eMask.x;
	const uint64_t orMask2 = f_group ? 0 : eMask.y;

	for (int ic = 0; ic < PROGRAM_ITERATIONS; ++ic)
	{
		const uint64_t spMix = *readReg0 ^ *readReg1;
		spAddr0 ^= ((const uint32_t*) &spMix)[0];
		spAddr1 ^= ((const uint32_t*) &spMix)[1];
		spAddr0 &= ScratchpadL3Mask64;
		spAddr1 &= ScratchpadL3Mask64;

		uint64_t offset1, offset2;
		asm("mul.wide.u32 %0,%2,%4;\n\tmul.wide.u32 %1,%3,%4;" : "=l"(offset1), "=l"(offset2) : "r"(spAddr0), "r"(spAddr1), "r"(batch_size));

		ulonglong2* p0 = (ulonglong2*)(scratchpad + offset1 + sub * 16);
		ulonglong2* p1 = (ulonglong2*)(scratchpad + offset2 + sub * 16);

		ulonglong2 global_mem_data = *p0;

		uint64_t* r = R + sub * 2;
		r[0] ^= global_mem_data.x;
		r[1] ^= global_mem_data.y;

		global_mem_data = *p1;
		int32_t* q = (int32_t*) &global_mem_data;

		fe[0] = load_F_E_groups(q[0], andMask, orMask1);
		fe[1] = load_F_E_groups(q[1], andMask, orMask2);
		fe[2] = load_F_E_groups(q[2], andMask, orMask1);
		fe[3] = load_F_E_groups(q[3], andMask, orMask2);

		__syncthreads();

		// TODO: execute byte code

		mx ^= *readReg2 ^ *readReg3;
		mx &= CacheLineAlignMask;

		global_mem_data = *(const ulonglong2*)(((const uint8_t*) dataset) + ma + sub * 16);
		r[0] ^= global_mem_data.x;
		r[1] ^= global_mem_data.y;

		const uint32_t tmp = ma;
		ma = mx;
		mx = tmp;

		global_mem_data.x = r[0];
		global_mem_data.y = r[1];
		*p1 = global_mem_data;

		global_mem_data.x = bit_cast<uint64_t>(f[0]) ^ bit_cast<uint64_t>(e[0]);
		global_mem_data.y = bit_cast<uint64_t>(f[1]) ^ bit_cast<uint64_t>(e[1]);
		*p0 = global_mem_data;

		spAddr0 = 0;
		spAddr1 = 0;
	}

	uint64_t* p = ((uint64_t*) registers) + idx * (REGISTERS_SIZE / sizeof(uint64_t));

	p[sub + 0] = R[sub + 0];
	p[sub + 4] = R[sub + 4];

	p[sub +  8] = bit_cast<uint64_t>(F[sub + 0]) ^ bit_cast<uint64_t>(E[sub + 0]);
	p[sub + 12] = bit_cast<uint64_t>(F[sub + 4]) ^ bit_cast<uint64_t>(E[sub + 4]);

	p[sub + 16] = bit_cast<uint64_t>(E[sub + 0]);
	p[sub + 20] = bit_cast<uint64_t>(E[sub + 4]);

	p[sub + 24] = bit_cast<uint64_t>(A[sub + 0]);
	p[sub + 28] = bit_cast<uint64_t>(A[sub + 4]);
}
