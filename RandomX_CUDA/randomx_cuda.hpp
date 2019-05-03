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
constexpr size_t ENTROPY_SIZE = 128 + 2048;
constexpr size_t VM_STATE_SIZE = 2048;
constexpr size_t PROGRAM_COUNT = 8;
constexpr size_t REGISTERS_SIZE = 256;
constexpr size_t IMM_BUF_SIZE = 768;

constexpr int ScratchpadL3Mask64 = (1 << 21) - 64;

constexpr uint32_t CacheLineSize = 64;
constexpr uint32_t CacheLineAlignMask = (DATASET_SIZE - 1) & ~(CacheLineSize - 1);

__device__ double getSmallPositiveFloatBits(uint64_t entropy)
{
	auto exponent = entropy >> 59; //0..31
	auto mantissa = entropy & randomx::mantissaMask;
	exponent += randomx::exponentBias;
	exponent &= randomx::exponentMask;
	exponent <<= randomx::mantissaSize;
	return __longlong_as_double(exponent | mantissa);
}

__device__ uint64_t getStaticExponent(uint64_t entropy)
{
	auto exponent = randomx::constExponentBits;
	exponent |= (entropy >> (64 - randomx::staticExponentBits)) << randomx::dynamicExponentBits;
	exponent <<= randomx::mantissaSize;
	return exponent;
}

__device__ uint64_t getFloatMask(uint64_t entropy)
{
	constexpr uint64_t mask22bit = (1ULL << 22) - 1;
	return (entropy & mask22bit) | getStaticExponent(entropy);
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

__device__ void test_memory_access(uint64_t* r, uint8_t* scratchpad, uint32_t batch_size)
{
	uint32_t x = static_cast<uint32_t>(r[0]);
	uint64_t y = r[1];

	#pragma unroll
	for (int i = 0; i < 55; ++i)
	{
		x = x * 0x08088405U + 1;

		uint32_t mask = 16320;
		if (x < 0x58000000U) mask = 262080;
		if (x < 0x20000000U) mask = 2097088;

		uint32_t addr = x & mask;
		uint64_t offset;
		asm("mul.wide.u32 %0,%1,%2;" : "=l"(offset) : "r"(addr), "r"(batch_size));

		x = x * 0x08088405U + 1;
		uint64_t* p = (uint64_t*)(scratchpad + offset + (x & 56));
		if (x <= 3045522264U)
			y ^= *p; // 39/55
		else
			*p = y; // 16/55
	}

	r[1] = y;
}

//
// VM state:
//
// Bytes 0-255: registers
// Bytes 256-1023: imm32 values (up to 192 values can be stored). IMUL_RCP and CBRANCH use 2 consecutive imm32 values.
// Bytes 1024-2047: up to 256 instructions
//
// Instruction encoding:
//
// Bits 0-2: dst (0-7)
// Bits 3-5: src (0-7)
// Bits 6-13: imm32/64 offset (in DWORDs, 0-191)
// Bits 14-18: src location mask shift (register, L1, L2, L3)
// Bits 19-20: src shift (0-3)
// Bit 21: src=imm32
// Bit 22: src=imm64
// Bit 23: src = -src
// Bits 24-28: opcode (add_rs, add, mul, umul_hi, imul_hi, neg, xor, ror, swap, cbranch, store)
// Bits 29-31: reserved
//

#define DST_OFFSET			0
#define SRC_OFFSET			3
#define IMM_OFFSET			6
#define LOC_OFFSET			14
#define SHIFT_OFFSET		19
#define SRC_IS_IMM32_OFFSET	21
#define SRC_IS_IMM64_OFFSET	22
#define NEGATIVE_SRC_OFFSET	23
#define OPCODE_OFFSET		24

// ISWAP r0, r0
#define INST_NOP			(8 << OPCODE_OFFSET)

#define LOC_L1 (32 - 14)
#define LOC_L2 (32 - 18)
#define LOC_L3 (32 - 21)

#define WORKERS_PER_HASH 4

__device__ uint64_t imul_rcp_value(uint32_t divisor)
{
	if (divisor == 0)
	{
		return 1ULL;
	}

	const uint64_t p2exp63 = 1ULL << 63;

	uint64_t quotient = p2exp63 / divisor;
	uint64_t remainder = p2exp63 % divisor;

	uint32_t bsr;
	asm("bfind.u32 %0,%1;" : "=r"(bsr) : "r"(divisor));

	for (uint32_t shift = 0; shift <= bsr; ++shift)
	{
		const bool b = (remainder >= divisor - remainder);
		quotient = (quotient << 1) | (b ? 1 : 0);
		remainder = (remainder << 1) - (b ? divisor : 0);
	}

	return quotient;
}

__device__ void set_byte(uint64_t& a, uint32_t position, uint64_t value)
{
	asm("bfi.b64 %0,%1,%2,%3,8;" : "=l"(a) : "l"(value), "l"(a), "r"(position << 3));
}

template<typename T, typename U, size_t N>
__device__ void set_buffer(T (&dst_buf)[N], const U value)
{
	uint32_t i = threadIdx.x * sizeof(T);
	const uint32_t step = blockDim.x * sizeof(T);
	uint8_t* dst = ((uint8_t*) dst_buf) + i;
	while (i < sizeof(T) * N)
	{
		*(T*)(dst) = static_cast<T>(value);
		dst += step;
		i += step;
	}
}

__device__ uint32_t get_condition_register(uint64_t registerLastChanged, uint64_t registerWasChanged, uint64_t& registerUsageCount, int32_t& lastChanged)
{
	lastChanged = INT_MAX;
	uint32_t minCount = 0xFFFFFFFFU;
	uint32_t creg = 0;

	for (uint32_t j = 0; j < 8; ++j)
	{
		uint64_t change, count;
		asm("bfe.u64 %0,%1,%2,8;" : "=l"(change) : "l"(registerLastChanged), "r"(j * 8));
		asm("bfe.u64 %0,%1,%2,8;" : "=l"(count) : "l"(registerUsageCount), "r"(j * 8));

		const int32_t k = (((registerWasChanged >> (j * 8)) & 1) == 0) ? -1 : static_cast<int32_t>(change);
		if ((k < lastChanged) || ((change == lastChanged) && (count < minCount)))
		{
			lastChanged = k;
			minCount = count;
			creg = j;
		}
	}

	registerUsageCount += (1ULL << (creg * 8));
	return creg;
}

__global__ void __launch_bounds__(32, 16) init_vm(void* entropy_data, void* vm_states)
{
	__shared__ uint32_t execution_plan_buf[RANDOMX_PROGRAM_SIZE * WORKERS_PER_HASH * (32 / 8) / sizeof(uint32_t)];

	set_buffer(execution_plan_buf, 0);
	__syncwarp();

	const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t idx = global_index / 8;
	const uint32_t sub = global_index % 8;

	uint8_t* execution_plan = (uint8_t*)(execution_plan_buf + (threadIdx.x / 8) * RANDOMX_PROGRAM_SIZE * WORKERS_PER_HASH / sizeof(uint32_t));

	uint64_t* R = ((uint64_t*) vm_states) + idx * VM_STATE_SIZE / sizeof(uint64_t);
	R[sub] = 0;

	const uint64_t* entropy = ((const uint64_t*) entropy_data) + idx * ENTROPY_SIZE / sizeof(uint64_t);

	double* A = (double*)(R + 24);
	A[sub] = getSmallPositiveFloatBits(entropy[sub]);

	if (sub == 0)
	{
		uint2* src_program = (uint2*)(entropy + 128 / sizeof(uint64_t));

		uint64_t registerLastChanged = 0;
		uint64_t registerWasChanged = 0;
		uint64_t registerUsageCount = 0;

		// Initialize CBRANCH instructions
		for (uint32_t i = 0; i < RANDOMX_PROGRAM_SIZE; ++i)
		{
			const uint2 src_inst = src_program[i];
			uint2 inst = src_inst;

			uint32_t opcode = inst.x & 0xff;
			const uint32_t dst = (inst.x >> 8) & 7;
			const uint32_t src = (inst.x >> 16) & 7;

			if (opcode < RANDOMX_FREQ_IADD_RS + RANDOMX_FREQ_IADD_M + RANDOMX_FREQ_ISUB_R + RANDOMX_FREQ_ISUB_M + RANDOMX_FREQ_IMUL_R + RANDOMX_FREQ_IMUL_M + RANDOMX_FREQ_IMULH_R + RANDOMX_FREQ_IMULH_M + RANDOMX_FREQ_ISMULH_R + RANDOMX_FREQ_ISMULH_M)
			{
				set_byte(registerLastChanged, dst, i);
				set_byte(registerWasChanged, dst, 1);
				continue;
			}
			opcode -= RANDOMX_FREQ_IADD_RS + RANDOMX_FREQ_IADD_M + RANDOMX_FREQ_ISUB_R + RANDOMX_FREQ_ISUB_M + RANDOMX_FREQ_IMUL_R + RANDOMX_FREQ_IMUL_M + RANDOMX_FREQ_IMULH_R + RANDOMX_FREQ_IMULH_M + RANDOMX_FREQ_ISMULH_R + RANDOMX_FREQ_ISMULH_M;

			if (opcode < RANDOMX_FREQ_IMUL_RCP)
			{
				if (inst.y)
				{
					set_byte(registerLastChanged, dst, i);
					set_byte(registerWasChanged, dst, 1);
				}
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_RCP;

			if (opcode < RANDOMX_FREQ_INEG_R + RANDOMX_FREQ_IXOR_R + RANDOMX_FREQ_IXOR_M + RANDOMX_FREQ_IROR_R)
			{
				set_byte(registerLastChanged, dst, i);
				set_byte(registerWasChanged, dst, 1);
				continue;
			}
			opcode -= RANDOMX_FREQ_INEG_R + RANDOMX_FREQ_IXOR_R + RANDOMX_FREQ_IXOR_M + RANDOMX_FREQ_IROR_R;

			if (opcode < RANDOMX_FREQ_ISWAP_R)
			{
				if (src != dst)
				{
					set_byte(registerLastChanged, dst, i);
					set_byte(registerWasChanged, dst, 1);
					set_byte(registerLastChanged, src, i);
					set_byte(registerWasChanged, src, 1);
				}
				continue;
			}
			opcode -= RANDOMX_FREQ_ISWAP_R;

			if (opcode < RANDOMX_FREQ_CBRANCH)
			{
				int32_t lastChanged;
				uint32_t creg = get_condition_register(registerLastChanged, registerWasChanged, registerUsageCount, lastChanged);

				// Store condition register and branch target in CBRANCH instruction
				*(uint32_t*)(src_program + i) = (src_inst.x & 0xFF0000FFU) | ((creg | ((lastChanged == -1) ? 0x80 : 0)) << 8) | ((static_cast<uint32_t>(lastChanged) & 0xFF) << 16);

				// Mark branch target instruction (src |= 0x40)
				*(uint32_t*)(src_program + lastChanged + 1) |= 0x40 << 8;

				uint32_t tmp = i | (i << 8);
				registerLastChanged = tmp | (tmp << 16);
				registerLastChanged = registerLastChanged | (registerLastChanged << 32);

				registerWasChanged = 0x0101010101010101ULL;
			}
		}

		uint32_t ma = static_cast<uint32_t>(entropy[8]) & CacheLineAlignMask;
		uint32_t mx = static_cast<uint32_t>(entropy[10]) & CacheLineAlignMask;

		uint32_t addressRegisters = static_cast<uint32_t>(entropy[12]);
		addressRegisters = ((addressRegisters & 1) | (((addressRegisters & 2) ? 3U : 2U) << 8) | (((addressRegisters & 4) ? 5U : 4U) << 16) | (((addressRegisters & 8) ? 7U : 6U) << 24)) * sizeof(uint64_t);

		uint32_t datasetOffset = (entropy[13] & randomx::DatasetExtraItems) * randomx::CacheLineSize;

		ulonglong2 eMask = *(ulonglong2*)(entropy + 14);
		eMask.x = getFloatMask(eMask.x);
		eMask.y = getFloatMask(eMask.y);

		((uint32_t*)(R + 16))[0] = ma;
		((uint32_t*)(R + 16))[1] = mx;
		((uint32_t*)(R + 16))[2] = addressRegisters;
		((uint32_t*)(R + 16))[3] = datasetOffset;
		((ulonglong2*)(R + 18))[0] = eMask;

		uint32_t* imm_buf = (uint32_t*)(R + REGISTERS_SIZE / sizeof(uint64_t));
		uint32_t imm_index = 0;
		uint32_t* compiled_program = (uint32_t*)(R + (REGISTERS_SIZE + IMM_BUF_SIZE) / sizeof(uint64_t));

		// Generate opcodes for execute_vm
		for (uint32_t i = 0; i < RANDOMX_PROGRAM_SIZE; ++i)
		{
			const uint2 src_inst = src_program[i];
			uint2 inst = src_inst;

			uint32_t opcode = inst.x & 0xff;
			const uint32_t dst = (inst.x >> 8) & 7;
			const uint32_t src = (inst.x >> 16) & 7;
			const uint32_t mod = (inst.x >> 24);

			inst.x = INST_NOP;

			if (opcode < RANDOMX_FREQ_IADD_RS)
			{
				const uint32_t shift = (mod >> 2) % 4;

				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (shift << SHIFT_OFFSET);

				if (dst != randomx::RegisterNeedsDisplacement)
				{
					// Encode regular ADD (opcode 1)
					inst.x |= (1 << OPCODE_OFFSET);
				}
				else
				{
					// Encode ADD with src and imm32 (opcode 0)
					inst.x |= imm_index << IMM_OFFSET;
					imm_buf[imm_index++] = inst.y;
				}

				*(compiled_program++) = inst.x;
				continue;
			}
			opcode -= RANDOMX_FREQ_IADD_RS;

			if (opcode < RANDOMX_FREQ_IADD_M)
			{
				const uint32_t location = (src == dst) ? LOC_L3 : ((mod % 4) ? LOC_L1 : LOC_L2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (location << LOC_OFFSET) | (1 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				imm_buf[imm_index++] = inst.y;

				*(compiled_program++) = inst.x;
				continue;
			}
			opcode -= RANDOMX_FREQ_IADD_M;

			if (opcode < RANDOMX_FREQ_ISUB_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (1 << OPCODE_OFFSET) | (1 << NEGATIVE_SRC_OFFSET);
				if (src == dst)
				{
					inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM32_OFFSET);
					imm_buf[imm_index++] = inst.y;
				}

				*(compiled_program++) = inst.x;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISUB_R;

			if (opcode < RANDOMX_FREQ_ISUB_M)
			{
				const uint32_t location = (src == dst) ? LOC_L3 : ((mod % 4) ? LOC_L1 : LOC_L2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (location << LOC_OFFSET) | (1 << OPCODE_OFFSET) | (1 << NEGATIVE_SRC_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				imm_buf[imm_index++] = inst.y;

				*(compiled_program++) = inst.x;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISUB_M;

			if (opcode < RANDOMX_FREQ_IMUL_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (2 << OPCODE_OFFSET);
				if (src == dst)
				{
					inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM32_OFFSET);
					imm_buf[imm_index++] = inst.y;
				}

				*(compiled_program++) = inst.x;
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_R;

			if (opcode < RANDOMX_FREQ_IMUL_M)
			{
				const uint32_t location = (src == dst) ? LOC_L3 : ((mod % 4) ? LOC_L1 : LOC_L2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (location << LOC_OFFSET) | (2 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				imm_buf[imm_index++] = inst.y;

				*(compiled_program++) = inst.x;
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_M;

			if (opcode < RANDOMX_FREQ_IMULH_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (3 << OPCODE_OFFSET);

				*(compiled_program++) = inst.x;
				continue;
			}
			opcode -= RANDOMX_FREQ_IMULH_R;

			if (opcode < RANDOMX_FREQ_IMULH_M)
			{
				const uint32_t location = (src == dst) ? LOC_L3 : ((mod % 4) ? LOC_L1 : LOC_L2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (location << LOC_OFFSET) | (3 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				imm_buf[imm_index++] = inst.y;

				*(compiled_program++) = inst.x;
				continue;
			}
			opcode -= RANDOMX_FREQ_IMULH_M;

			if (opcode < RANDOMX_FREQ_ISMULH_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (4 << OPCODE_OFFSET);

				*(compiled_program++) = inst.x;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISMULH_R;

			if (opcode < RANDOMX_FREQ_ISMULH_M)
			{
				const uint32_t location = (src == dst) ? LOC_L3 : ((mod % 4) ? LOC_L1 : LOC_L2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (location << LOC_OFFSET) | (4 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				imm_buf[imm_index++] = inst.y;

				*(compiled_program++) = inst.x;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISMULH_M;

			if (opcode < RANDOMX_FREQ_IMUL_RCP)
			{
				const uint64_t r = imul_rcp_value(inst.y);
				if (r == 1)
				{
					*(compiled_program++) = INST_NOP;
					continue;
				}

				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (2 << OPCODE_OFFSET);
				inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM64_OFFSET);

				imm_buf[imm_index] = ((const uint32_t*) &r)[0];
				imm_buf[imm_index + 1] = ((const uint32_t*) &r)[1];
				imm_index += 2;

				*(compiled_program++) = inst.x;
				continue;
			}
			opcode -= RANDOMX_FREQ_IMUL_RCP;

			if (opcode < RANDOMX_FREQ_INEG_R)
			{
				inst.x = (dst << DST_OFFSET) | (5 << OPCODE_OFFSET);

				*(compiled_program++) = inst.x;
				continue;
			}
			opcode -= RANDOMX_FREQ_INEG_R;

			if (opcode < RANDOMX_FREQ_IXOR_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (6 << OPCODE_OFFSET);
				if (src == dst)
				{
					inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM32_OFFSET);
					imm_buf[imm_index++] = inst.y;
				}

				*(compiled_program++) = inst.x;
				continue;
			}
			opcode -= RANDOMX_FREQ_IXOR_R;

			if (opcode < RANDOMX_FREQ_IXOR_M)
			{
				const uint32_t location = (src == dst) ? LOC_L3 : ((mod % 4) ? LOC_L1 : LOC_L2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (location << LOC_OFFSET) | (6 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				imm_buf[imm_index++] = inst.y;

				*(compiled_program++) = inst.x;
				continue;
			}
			opcode -= RANDOMX_FREQ_IXOR_M;

			if (opcode < RANDOMX_FREQ_IROR_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (7 << OPCODE_OFFSET);
				if (src == dst)
				{
					inst.x |= (imm_index << IMM_OFFSET) | (1 << SRC_IS_IMM32_OFFSET);
					imm_buf[imm_index++] = inst.y;
				}

				*(compiled_program++) = inst.x;
				continue;
			}
			opcode -= RANDOMX_FREQ_IROR_R;

			if (opcode < RANDOMX_FREQ_ISWAP_R)
			{
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (8 << OPCODE_OFFSET);

				*(compiled_program++) = (src != dst) ? inst.x : INST_NOP;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISWAP_R;

			if (opcode < RANDOMX_FREQ_CBRANCH)
			{
				const int32_t lastChanged = (src_inst.x & (0x80 << 8)) ? -1 : static_cast<int32_t>((src_inst.x >> 16) & 0xFF);

				inst.x = (dst << DST_OFFSET) | (9 << OPCODE_OFFSET);
				inst.x |= (imm_index << IMM_OFFSET);

				const uint32_t cshift = (mod >> 4) + randomx::ConditionOffset;

				uint32_t imm = inst.y | (1U << cshift);
				if (cshift > 0)
					imm &= ~(1U << (cshift - 1));

				imm_buf[imm_index] = imm;
				imm_buf[imm_index + 1] = cshift | (static_cast<uint32_t>(lastChanged) << 5);
				imm_index += 2;

				*(compiled_program++) = inst.x;
				continue;
			}
			opcode -= RANDOMX_FREQ_CBRANCH;

			if (opcode < RANDOMX_FREQ_ISTORE)
			{
				const uint32_t location = ((mod >> 4) >= randomx::StoreL3Condition) ? LOC_L3 : ((mod % 4) ? LOC_L1 : LOC_L2);
				inst.x = (dst << DST_OFFSET) | (src << SRC_OFFSET) | (location << LOC_OFFSET) | (10 << OPCODE_OFFSET);
				inst.x |= imm_index << IMM_OFFSET;
				imm_buf[imm_index++] = inst.y;
				*(compiled_program++) = inst.x;
				continue;
			}
			opcode -= RANDOMX_FREQ_ISTORE;

			*(compiled_program++) = inst.x;
		}
	}
}

template<typename T, size_t N>
__device__ void load_buffer(T (&dst_buf)[N], const void* src_buf)
{
	uint32_t i = threadIdx.x * sizeof(T);
	const uint32_t step = blockDim.x * sizeof(T);
	const uint8_t* src = ((const uint8_t*) src_buf) + blockIdx.x * sizeof(T) * N + i;
	uint8_t* dst = ((uint8_t*) dst_buf) + i;
	while (i < sizeof(T) * N)
	{
		*(T*)(dst) = *(T*)(src);
		src += step;
		dst += step;
		i += step;
	}
}

__global__ void __launch_bounds__(16, 16) execute_vm(void* vm_states, void* scratchpads, const void* dataset_ptr, uint32_t batch_size, uint32_t num_iterations, bool first, bool last)
{
	// 2 hashes per warp, 4 KB shared memory for VM states
	__shared__ uint64_t vm_states_local[(VM_STATE_SIZE * 2) / sizeof(uint64_t)];

	load_buffer(vm_states_local, vm_states);

	__syncwarp();

	uint64_t* R = vm_states_local + (threadIdx.x / 8) * VM_STATE_SIZE / sizeof(uint64_t);
	double* F = (double*)(R + 8);
	double* E = (double*)(R + 16);

	const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t idx = global_index / 8;
	const uint32_t sub = global_index % 8;

	uint32_t ma = ((uint32_t*)(R + 16))[0];
	uint32_t mx = ((uint32_t*)(R + 16))[1];

	const uint32_t addressRegisters = ((uint32_t*)(R + 16))[2];
	const uint64_t* readReg0 = (uint64_t*)(((uint8_t*) R) + (addressRegisters & 0xff));
	const uint64_t* readReg1 = (uint64_t*)(((uint8_t*) R) + ((addressRegisters >> 8) & 0xff));
	const uint32_t* readReg2 = (uint32_t*)(((uint8_t*) R) + ((addressRegisters >> 16) & 0xff));
	const uint32_t* readReg3 = (uint32_t*)(((uint8_t*) R) + (addressRegisters >> 24));

	const uint32_t datasetOffset = ((uint32_t*)(R + 16))[3];
	const uint8_t* dataset = ((const uint8_t*) dataset_ptr) + datasetOffset;

	ulonglong2 eMask = ((ulonglong2*)(R + 18))[0];

	uint32_t spAddr0 = first ? mx : 0;
	uint32_t spAddr1 = first ? ma : 0;

	uint8_t* scratchpad = ((uint8_t*) scratchpads) + idx * 64;

	const bool f_group = (sub < 4);

	double* fe = f_group ? (F + sub * 2) : (E + (sub - 4) * 2);
	double* f = F + sub;
	double* e = E + sub;

	const uint64_t andMask = f_group ? uint64_t(-1) : randomx::dynamicMantissaMask;
	const uint64_t orMask1 = f_group ? 0 : eMask.x;
	const uint64_t orMask2 = f_group ? 0 : eMask.y;

	uint32_t* imm_buf = (uint32_t*)(R + REGISTERS_SIZE / sizeof(uint64_t));
	uint32_t* compiled_program = (uint32_t*)(R + (REGISTERS_SIZE + IMM_BUF_SIZE) / sizeof(uint64_t));

	#pragma unroll(1)
	for (int ic = 0; ic < num_iterations; ++ic)
	{
		const uint64_t spMix = *readReg0 ^ *readReg1;
		spAddr0 ^= ((const uint32_t*) &spMix)[0];
		spAddr1 ^= ((const uint32_t*) &spMix)[1];
		spAddr0 &= ScratchpadL3Mask64;
		spAddr1 &= ScratchpadL3Mask64;

		uint64_t offset1, offset2;
		asm("mad.wide.u32 %0,%2,%4,%5;\n\tmad.wide.u32 %1,%3,%4,%5;" : "=l"(offset1), "=l"(offset2) : "r"(spAddr0), "r"(spAddr1), "r"(batch_size), "l"(static_cast<uint64_t>(sub * 8)));

		uint64_t* p0 = (uint64_t*)(scratchpad + offset1);
		uint64_t* p1 = (uint64_t*)(scratchpad + offset2);

		uint64_t* r = R + sub;
		*r ^= *p0;

		uint64_t global_mem_data = *p1;
		int32_t* q = (int32_t*) &global_mem_data;

		fe[0] = load_F_E_groups(q[0], andMask, orMask1);
		fe[1] = load_F_E_groups(q[1], andMask, orMask2);

		__syncwarp();

		if (sub == 0)
		{
			#pragma unroll(1)
			for (int32_t ip = 0; ip < RANDOMX_PROGRAM_SIZE; ++ip)
			{
				uint32_t inst = compiled_program[ip];

				asm("// INSTRUCTION DECODING BEGIN");

				if (inst == INST_NOP)
					continue;

				uint64_t* dst_ptr = (uint64_t*)((uint8_t*)(R) + ((inst >> DST_OFFSET) & 7) * 8);
				uint64_t* src_ptr = (uint64_t*)((uint8_t*)(R) + ((inst >> SRC_OFFSET) & 7) * 8);
				uint32_t* imm_ptr = imm_buf + ((inst >> IMM_OFFSET) & 255);

				uint64_t dst = *dst_ptr;
				uint64_t src = *src_ptr;
				uint2 imm;
				imm.x = imm_ptr[0];
				imm.y = imm_ptr[1];

				const uint32_t opcode = (inst >> OPCODE_OFFSET) & 15;
				const uint32_t location = (inst >> LOC_OFFSET) & 31;

				asm("// INSTRUCTION DECODING END");

				if (location)
				{
					asm("// SCRATCHPAD ACCESS BEGIN");

					const uint32_t mask = 0xFFFFFFFFU >> location;

					const bool is_read = (opcode != 10);
					uint32_t addr = is_read ? ((location == LOC_L3) ? 0 : static_cast<uint32_t>(src)) : static_cast<uint32_t>(dst);
					addr += static_cast<int32_t>(imm.x);
					addr &= mask;

					uint64_t offset;
					asm("mad.wide.u32 %0,%1,%2,%3;" : "=l"(offset) : "r"(addr & 0xFFFFFFC0U), "r"(batch_size), "l"(static_cast<uint64_t>(addr & 0x38)));

					if (is_read)
						src = *(uint64_t*)(scratchpad + offset);
					else
						*(uint64_t*)(scratchpad + offset) = src;

					asm("// SCRATCHPAD ACCESS END");
				}

				{
					asm("// EXECUTION BEGIN");

					if (inst & (1 << SRC_IS_IMM32_OFFSET)) src = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(imm.x)));

					// Check instruction opcodes (most frequent instructions come first)
					if (opcode < 2)
					{
						if (inst & (1 << NEGATIVE_SRC_OFFSET)) src = static_cast<uint64_t>(-static_cast<int64_t>(src));
						if (opcode == 0) dst += static_cast<int32_t>(imm.x);
						const uint32_t shift = (inst >> SHIFT_OFFSET) & 3;
						dst += src << shift;
					}
					else if (opcode == 2)
					{
						if (inst & (1 << SRC_IS_IMM64_OFFSET)) src = *((uint64_t*)&imm);
						dst *= src;
					}
					else if (opcode == 6)
					{
						dst ^= src;
					}
					else if (opcode == 9)
					{
						dst += static_cast<int32_t>(imm.x);
						if ((static_cast<uint32_t>(dst) & (randomx::ConditionMask << (imm.y & 31))) == 0)
							ip = static_cast<int32_t>(imm.y) >> 5;
					}
					else if (opcode == 7)
					{
						const uint32_t shift = src & 63;
						dst = (dst >> shift) | (dst << (64 - shift));
					}
					else if (opcode == 3)
					{
						dst = __umul64hi(dst, src);
					}
					else if (opcode == 4)
					{
						dst = static_cast<uint64_t>(__mul64hi(static_cast<int64_t>(dst), static_cast<int64_t>(src)));
					}
					else if (opcode == 8)
					{
						*src_ptr = dst;
						dst = src;
					}
					else if (opcode == 5)
					{
						dst = static_cast<uint64_t>(-static_cast<int64_t>(dst));
					}

					*dst_ptr = dst;

					asm("// EXECUTION END");
				}

				//__syncwarp((1U << 0) | (1U << 8));
			}
		}

		mx ^= *readReg2 ^ *readReg3;
		mx &= CacheLineAlignMask;

		const uint64_t next_r = *r ^ *(const uint64_t*)(dataset + ma + sub * 8);
		*r = next_r;

		uint32_t tmp = ma;
		ma = mx;
		mx = tmp;

		*p1 = next_r;
		*p0 = bit_cast<uint64_t>(f[0]) ^ bit_cast<uint64_t>(e[0]);

		spAddr0 = 0;
		spAddr1 = 0;
	}

	uint64_t* p = ((uint64_t*) vm_states) + idx * (VM_STATE_SIZE / sizeof(uint64_t));
	p[sub] = R[sub];

	if (last)
	{
		p[sub +  8] = bit_cast<uint64_t>(F[sub]) ^ bit_cast<uint64_t>(E[sub]);
		p[sub + 16] = bit_cast<uint64_t>(E[sub]);
	}
	else if (sub == 0)
	{
		((uint32_t*)(p + 16))[0] = ma;
		((uint32_t*)(p + 16))[1] = mx;
	}
}
