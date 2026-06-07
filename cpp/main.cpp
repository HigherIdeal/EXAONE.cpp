#include "parameters.h"
#include "exaone.h"

#include <cmath>
#include <exception>

int main(int argc, char* argv[]) {

	int tile_size = 64;
    std::string path = "C:/Users/Jungwoo_Kim/source/repos/EXAONE/EXAONE.cpp/weight/EXAONE-4.0-1.2B/model.bin";

	try {
		EXAONE exaone(path, tile_size);
	} catch (const std::exception& error) {
		std::cerr << "failed to initialize EXAONE: " << error.what() << std::endl;
		return 1;
	}
	
	return 0;
}
