#include "exaone.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <iomanip>
#include <chrono>


#include <immintrin.h>

inline float BF16_to_FP32(uint16_t x)
{
	uint32_t bits = static_cast<uint32_t>(x) << 16;

	float value;
	std::memcpy(&value, &bits, sizeof(float));
	return value;
}

inline void FP8_to_BF16_SIMD(const uint8_t* src, uint16_t* dst)
{
	__m128i uint_8bit = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src)); // read 16 FP8 data to 128-bit SIMD register
	__m256i uint_16bit = _mm256_cvtepu8_epi16(uint_8bit);						// zero-extension 8bit to 16bit (fill MSB)
	__m256i sign = _mm256_and_si256(uint_16bit, _mm256_set1_epi16(0x0080));		// extract sign bit (bit[7]) to bit[15] position for BF16
	sign = _mm256_slli_epi16(sign, 8);											// move sign bit to bit[15] position for BF16
	__m256i body = _mm256_and_si256(uint_16bit, _mm256_set1_epi16(0x007F));		// extract exponent/mantissa bits (bit[6:0])
	body = _mm256_slli_epi16(body, 4);											// move body to bit[10:4] position for 
	__m256i base = _mm256_set1_epi16(0x3800);									// correction bits for exponent bias (0x3800)
	__m256i out = _mm256_or_si256(base, _mm256_or_si256(sign, body));			// combine base, sign, and body to form the final BF16 bit-pattern
	_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), out);					// BF16 format unsigned 16-bit integer stored to destination
}


//std::cout << std::scientific << std::setprecision(8);

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace {

void readExact(std::ifstream& file, char* dst, std::streamsize size, const char* what)
{
	file.read(dst, size);
	if (file.gcount() != size) {
		throw std::runtime_error(std::string("failed to read ") + what);
	}
}

uint32_t readU32LE(std::ifstream& file, const char* what)
{
	unsigned char bytes[4];
	readExact(file, reinterpret_cast<char*>(bytes), 4, what);

	return static_cast<uint32_t>(bytes[0])
		| (static_cast<uint32_t>(bytes[1]) << 8)
		| (static_cast<uint32_t>(bytes[2]) << 16)
		| (static_cast<uint32_t>(bytes[3]) << 24);
}

uint64_t readU64LE(std::ifstream& file, const char* what)
{
	unsigned char bytes[8];
	readExact(file, reinterpret_cast<char*>(bytes), 8, what);

	uint64_t value = 0;
	for (int i = 0; i < 8; ++i) {
		value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
	}
	return value;
}

uint64_t getFileSize(std::ifstream& file)
{
	file.seekg(0, std::ios::end);

	const std::streamoff size = file.tellg();
	if (size <= 0) {
		throw std::runtime_error("invalid model.bin size");
	}

	file.seekg(0, std::ios::beg);
	return static_cast<uint64_t>(size);
}

} // namespace

EXAONE::EXAONE(const std::string& path, int tile_size)
{
	this->tile_size = tile_size;
	loadWeights(path);
}

EXAONE::~EXAONE()
{
#if defined(_MSC_VER)
	_aligned_free(weight);
#else
	std::free(weight);
#endif

	weight = nullptr;
	weight_bytes = 0;
}

void EXAONE::generate()
{
	prefill();
}

void EXAONE::loadWeights(const std::string& path)
{
	std::ifstream file(path, std::ios::binary);
	if (!file) {
		throw std::runtime_error("failed to open weight file: " + path);
	}

	const uint64_t file_size = getFileSize(file);

	const uint32_t count = readU32LE(file, "weight count");
	if (count != NUM_WEIGHT) {
		throw std::runtime_error("unexpected weight count in model.bin");
	}

	std::vector<uint64_t> file_offsets(NUM_WEIGHT);

	for (uint32_t expected = 0; expected < count; ++expected) {
		const uint32_t index = readU32LE(file, "weight index");
		const uint64_t offset = readU64LE(file, "weight offset");

		if (index != expected) {
			throw std::runtime_error("non-sequential weight index in model.bin");
		}

		if (offset % WEIGHT_ALIGNMENT != 0) {
			throw std::runtime_error("weight payload offset is not 64-byte aligned");
		}

		if (offset >= file_size) {
			throw std::runtime_error("weight payload offset is outside model.bin");
		}

		if (expected > 0 && offset < file_offsets[expected - 1]) {
			throw std::runtime_error("weight payload offsets are not sorted");
		}

		file_offsets[expected] = offset;
	}

	const uint64_t first_payload_offset = file_offsets[0];

	weight_bytes = file_size - first_payload_offset;
	if (weight_bytes == 0) {
		throw std::runtime_error("model.bin has no payload bytes");
	}

	if (weight_bytes > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
		throw std::runtime_error("weight payload is too large for size_t");
	}

	if (weight_bytes > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
		throw std::runtime_error("weight payload is too large for streamsize");
	}

	void* allocated = nullptr;

#if defined(_MSC_VER)
	allocated = _aligned_malloc(
		static_cast<std::size_t>(weight_bytes),
		static_cast<std::size_t>(WEIGHT_ALIGNMENT)
	);

	if (allocated == nullptr) {
		throw std::runtime_error("failed to allocate aligned weight buffer");
	}
#else
	if (posix_memalign(
		&allocated,
		static_cast<std::size_t>(WEIGHT_ALIGNMENT),
		static_cast<std::size_t>(weight_bytes)
	) != 0) {
		throw std::runtime_error("failed to allocate aligned weight buffer");
	}
#endif

	uint8_t* buffer = static_cast<uint8_t*>(allocated);

	try {
		file.seekg(static_cast<std::streamoff>(first_payload_offset), std::ios::beg);

		readExact(
			file,
			reinterpret_cast<char*>(buffer),
			static_cast<std::streamsize>(weight_bytes),
			"weight payloads"
		);
	}
	catch (...) {
#if defined(_MSC_VER)
		_aligned_free(buffer);
#else
		std::free(buffer);
#endif
		throw;
	}

#if defined(_MSC_VER)
	_aligned_free(weight);
#else
	std::free(weight);
#endif

	weight = buffer;

	for (uint32_t i = 0; i < NUM_WEIGHT; ++i) {
		header[i] = file_offsets[i] - first_payload_offset;
	}

	std::cout << "loaded weights: " << path
		<< " payload_bytes=" << weight_bytes
		<< " alignment=" << WEIGHT_ALIGNMENT
		<< std::endl;
}

void EXAONE::prefill()
{
	auto t0 = std::chrono::high_resolution_clock::now();
	// prefill memory allocation
	h0 = new uint16_t[max_seq_len * DIM];
	h1 = new uint16_t[max_seq_len * FDIM];
	k_cache = new uint16_t[NUM_LAYERS * max_seq_len * KVDIM];
	v_cache = new uint16_t[NUM_LAYERS * max_seq_len * KVDIM];
	online_tile_0 = new float[tile_size * tile_size];
	online_tile_1 = new float[tile_size * HEAD_DIM];
	gemm_buffer = new float[tile_size * DIM];


	int tokens[128] = {};
	for (int i = 0; i < 128; ++i) {
		tokens[i] = 2 * i;
	} //dummy token list for prefill, the actual token list should be generated by tokenizer according to the input prompt.
	seq_len = 128;

	// token 
	const uint8_t* te_base_idx = weight + header[0];
	for (int i = 0; i < seq_len; ++i) {
		const int token = tokens[i];

		const uint8_t* src = te_base_idx + static_cast<std::size_t>(token) * DIM;
		uint16_t* dst = h0 + static_cast<std::size_t>(i) * DIM;

		
		for (int j = 0; j + 16 <= DIM; j += 16) { // convert 2048 FP8 values to BF16, 16 values per SIMD call
			
			FP8_to_BF16_SIMD(src + j, dst + j);

		}


	}

	for (int layer_id = 0;layer_id < NUM_LAYERS;++layer_id) {
		
		// Query projection
		const uint8_t* q_base_idx = weight + header[1 + layer_id * 11 + 0];		// query prefix
		
		// 일단 백앤드 구현하기 전에 미리 구현해서 그냥 SIMD 기반 GEMV로 여러번 호출.
		// reference 속도 측정용으로 일단 구현. 실제로는 타일링 기반 SIMD + OpenMP GEMM으로 구현할 예정

	}

	auto t1 = std::chrono::high_resolution_clock::now();

	auto elapsed_us =
		std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

	std::cout << "FP8_to_BF16_SIMD time: " << elapsed_us << " ms\n";

	std::cout << std::scientific << std::setprecision(8);

	int a = 56;

	std::cout << "temp: tensor([";
	for (int i = 0; i < 4; ++i) {
		float v = BF16_to_FP32(h0[i+a]);

		std::cout << v;
		if (i != 3) {
			std::cout << ", ";
		}
	}
	std::cout << "])\n";
}