#pragma once

/*
Copyright (c) 2019 SChernykh
Portions Copyright (c) 2018-2019 tevador

This file is part of RandomX CUDA.

RandomX CUDA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RandomX is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RandomX.  If not, see<http://www.gnu.org/licenses/>.
*/

constexpr size_t DATASET_SIZE = 1U << 31;
constexpr size_t SCRATCHPAD_SIZE = 1U << 21;
constexpr size_t HASH_SIZE = 64;
constexpr size_t PROGRAM_SIZE = 128 + 2048;
constexpr size_t PROGRAM_COUNT = 8;
constexpr size_t REGISTERS_SIZE = 256;

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

__global__ void initGroupA_registers(const void* entropy_data, void* registers)
{
	const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t* r = ((uint64_t*) registers) + global_index * (REGISTERS_SIZE / sizeof(uint64_t));
	const uint64_t* e = ((const uint64_t*) entropy_data) + global_index * (PROGRAM_SIZE / sizeof(uint64_t));

	#pragma unroll(8)
	for (int i = 0; i < 8; ++i)
		r[i + 24] = getSmallPositiveFloatBits(e[i]);
}
