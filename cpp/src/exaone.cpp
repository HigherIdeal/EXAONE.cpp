#include "exaone.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
	std::cout << std::scientific << std::setprecision(8);
	std::cout << "embedding weight num:" << header[0] << "\n";
	int offset = header[1]; // FP8 E4M3
	for (int i = 0; i < 4; ++i) {
		uint8_t x = weight[offset + i];
		uint32_t sign = static_cast<uint32_t>(x & 0x80) << 24;
		uint32_t body = static_cast<uint32_t>(x & 0x7F) << 20;
		uint32_t float_bits = sign | 0x38000000u | body;
		float value = *reinterpret_cast<float*>(&float_bits);
		std::cout << value << ", ";
	}
	std::cout << "\n";
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
