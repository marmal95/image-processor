#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>
#include <SFML/Graphics/Image.hpp>
#include "CudaAlgorithm.hpp"
#include "FiltersProvider.hpp"
#include "HelperFunctions.hpp"

const std::string IMAGE_NAME = "mountain";
const std::string INPUT_IMAGE_NAME = IMAGE_NAME + ".jpg";
const std::string OUTPUT_IMAGE_NAME = IMAGE_NAME + "_out.jpg";

auto loadImage()
{
	sf::Image image{}, dest{};
	image.loadFromFile("../images/" + INPUT_IMAGE_NAME);
	return image;
}

void saveImage(sf::Image& image)
{
	image.saveToFile("../images/" + OUTPUT_IMAGE_NAME);
}

int main()
{
	auto image = loadImage();
	auto filter = Filter::blurKernel();
	auto duration = runWithTimeMeasurementCpu([&]() {
		Cuda::applyFilter(image, filter);
	});
	std::cout << "Duration [ms]: " << duration << std::endl;
	saveImage(image);
	return EXIT_SUCCESS;
}