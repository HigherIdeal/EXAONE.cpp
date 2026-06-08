#include "parameters.h"
#include "exaone.h"

#include <cmath>
#include <exception>

int main(int argc, char* argv[]) {

	int tile_size = 64;
	std::string path;
	std::cout << "Enter the path to the model file: ";
	std::cin >> path;

	try {
		EXAONE exaone(path, tile_size);


		exaone.generate();
	} catch (const std::exception& error) {
		std::cerr << "failed to initialize EXAONE: " << error.what() << std::endl;
		return 1;
	}
	


	return 0;
}
