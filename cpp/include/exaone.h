#pragma once
#include "parameters.h"

#include <string>

class EXAONE
{
public:
	int tile_size = 0;              // maximum 64 for edge devices.

    bool is_prefill = false;

    int seq_len = 0;
    int cur_pos = 0;
    int start_pos = 0;
    int max_seq_len = 16384;        // small conversation senario for edge devices.

    uint64_t header[NUM_WEIGHT] = {};    // byte offsets relative to weight
	uint8_t* weight = nullptr;           // 64 byte aligned payload buffer
    uint64_t weight_bytes = 0;
	uint16_t* h0;                   // if prefill mode, h0 will be  64 MiB(max_seq_len *  DIM) else 4 KiB(1 *  DIM) size memory allocation.
	uint16_t* h1;                   // if prefill mode, h1 will be 128 MiB(max_seq_len * FDIM) else 8 KiB(1 * FDIM) size memory allocation.
	uint16_t* k_cache;              // k_cache will be 480 MiB(NUM_LAYERS * max_seq_len * KV_DIM) size memory allocation.
	uint16_t* v_cache;              // k_cache will be 480 MiB(NUM_LAYERS * max_seq_len * KV_DIM) size memory allocation.
	float* online_tile_0;           // if prefill mode, online_tile_0 will be 16 KiB(tile_size * tile_size) else 256 Byte(1 *  tile_size) size memory allocation.
	float* online_tile_1;           // if prefill mode, online_tile_1 will be 16 KiB(tile_size * 64) else 256 Byte(1 *  64) size memory allocation.
	float* gemm_buffer;             // if prefill mode, gemm_buffer will be 512 KiB(tile_size * DIM) else 8 KiB(1 *  DIM) size memory allocation.

	EXAONE(const std::string& path, int tile_size);
	~EXAONE();


    void generate();
private:
	void loadWeights(const std::string& path);
    void prefill();
    void decoding();

	uint16_t unpack(const uint8_t);
	//float unpack(const uint8_t);
	//float unpack(const uint16_t);

};
