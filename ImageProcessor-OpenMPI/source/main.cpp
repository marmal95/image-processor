#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>
#include <mpi.h>
#include <SFML/Graphics/Image.hpp>
#include "FiltersProvider.hpp"
#include "MpiHelpers.hpp"

const std::string IMAGE_NAME = "mountain";
const std::string INPUT_IMAGE_NAME = IMAGE_NAME + ".jpg";
const std::string OUTPUT_IMAGE_NAME = IMAGE_NAME + "_out.jpg";

using KernelRow = std::vector<float>;
using Kernel = std::vector<KernelRow>;

template <typename Callable, typename... Args>
auto runWithTimeMeasurementCpu(Callable&& function, Args&&... params)
{
	const auto start = std::chrono::high_resolution_clock::now();
	std::forward<Callable>(function)(std::forward<Args>(params)...);
	const auto stop = std::chrono::high_resolution_clock::now();
	const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
	return duration;
}

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

void alignChannel(int& channelValue)
{
	channelValue = (channelValue > 255) ? 255 : channelValue;
	channelValue = (channelValue < 0) ? 0 : channelValue;
}

void applyFilter(sf::Image& image, const Kernel& filter)
{
	const int kernelSize = filter.size();
	const int kernelMargin = kernelSize / 2;
	auto outputImage = image;

	const auto rowsPerProcess = image.getSize().y / MPI::getWorldSize();
	auto processBeginRow = MPI::getRank() * rowsPerProcess;
	auto processEndRow = MPI::getRank() * rowsPerProcess + rowsPerProcess;

	if (processBeginRow == 0) processBeginRow += kernelMargin;
	if (processEndRow == image.getSize().y) processEndRow -= kernelMargin;
	
	for (int y = processBeginRow; y < processEndRow; ++y)
	{
		for (int x = kernelMargin; x < image.getSize().x - kernelMargin; ++x)
		{
			int newRedChannel{}, newGreenChannel{}, newBlueChannel{};
			for (int kernelX = -kernelMargin; kernelX <= kernelMargin; ++kernelX)
			{
				for (int kernelY = -kernelMargin; kernelY <= kernelMargin; ++kernelY)
				{
					const auto kernelValue = filter[kernelX + kernelMargin][kernelY + kernelMargin];
					const auto pixel = image.getPixel(x + kernelX, y + kernelY);
					newRedChannel += pixel.r * kernelValue;
					newGreenChannel += pixel.g * kernelValue;
					newBlueChannel += pixel.b * kernelValue;
				}
			}

			alignChannel(newRedChannel);
			alignChannel(newGreenChannel);
			alignChannel(newBlueChannel);
			outputImage.setPixel(x, y, sf::Color(newRedChannel, newGreenChannel, newBlueChannel));
		}
	}

	image = std::move(outputImage);
}

void convertToBlackAndWhite(sf::Image& image)
{
	int x, y;
	for (x = 0; x < image.getSize().x; ++x)
	{
		for (y = 0; y < image.getSize().y; ++y)
		{
			const auto pixel = image.getPixel(x, y);
			const auto sumChannels = pixel.r + pixel.g + pixel.b;
			const auto newChannelValue = sumChannels / 3;
			image.setPixel(x, y, sf::Color(newChannelValue, newChannelValue, newChannelValue));
		}
	}
}

auto distributeImage(const sf::Image& image)
{
	sf::Image processImage;
	std::size_t dataSize, imageWidth, imageHeight;
	std::vector<sf::Uint8> buffer{};

	if (MPI::isMasterProcess())
	{
		dataSize = image.getSize().x * image.getSize().y * 4;
		imageWidth = image.getSize().x;
		imageHeight = image.getSize().y;
		buffer = std::vector<sf::Uint8>(image.getPixelsPtr(), image.getPixelsPtr() + dataSize);
	}

	MPI_Bcast(&dataSize, 1, MPI_UNSIGNED_LONG_LONG, MPI::MASTER_PROCESS, MPI_COMM_WORLD);
	MPI_Bcast(&imageWidth, 1, MPI_UNSIGNED_LONG_LONG, MPI::MASTER_PROCESS, MPI_COMM_WORLD);
	MPI_Bcast(&imageHeight, 1, MPI_UNSIGNED_LONG_LONG, MPI::MASTER_PROCESS, MPI_COMM_WORLD);

	buffer.resize(dataSize);
	MPI_Bcast(buffer.data(), buffer.size(), MPI_UNSIGNED_CHAR, MPI::MASTER_PROCESS, MPI_COMM_WORLD);

	processImage.create(imageWidth, imageHeight, buffer.data());
	return processImage;
}

void collectImage(sf::Image& image, std::vector<sf::Uint8>& buffer)
{
	const auto dataSizePerProcess = buffer.size() / MPI::getWorldSize();
	MPI_Gather(image.getPixelsPtr() + MPI::getRank() * dataSizePerProcess, dataSizePerProcess, MPI_UNSIGNED_CHAR,
		buffer.data(), dataSizePerProcess, MPI_UNSIGNED_CHAR, MPI::MASTER_PROCESS, MPI_COMM_WORLD);
}

auto loadImageForMaster()
{
	sf::Image image{};
	if (MPI::isMasterProcess())
	{
		image = loadImage();
	}
	return image;
}

void restoreImage(sf::Image& image, const std::vector<sf::Uint8>& buffer)
{
	if (MPI::isMasterProcess())
	{
		image.create(image.getSize().x, image.getSize().y, buffer.data());
		saveImage(image);
	}
}

auto calculateImageSize(const sf::Image& image)
{
	return image.getSize().x * image.getSize().y * 4;
}

void logDuration(const uint64_t duration)
{
	if (MPI::isMasterProcess())
	{
		std::cout << "Duration [ms]: " << duration << std::endl;
	}
}

int main()
{
	MPI_Init(nullptr, nullptr);

	sf::Image image = loadImageForMaster();
	std::vector<sf::Uint8> buffer{};

	const auto duration = runWithTimeMeasurementCpu([&]() {
		image = distributeImage(image);
		buffer = std::vector<sf::Uint8>(calculateImageSize(image));
		auto filter = Filter::blurKernel();
		applyFilter(image, filter);
		collectImage(image, buffer);
	});

	restoreImage(image, buffer);
	logDuration(duration);

	MPI_Finalize();
	return EXIT_SUCCESS;
}