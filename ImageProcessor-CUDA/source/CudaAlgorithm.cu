#include "CudaAlgorithm.hpp"
#include "HelperFunctions.hpp"
#include <cstdlib>
#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__device__ void alignChannel(int& channelValue)
{
	channelValue = (channelValue > 255) ? 255 : channelValue;
	channelValue = (channelValue < 0) ? 0 : channelValue;
}

__global__ void applyFilterOnCuda(
	const sf::Uint8* inputImageData, sf::Uint8* outputImageData,
	const std::size_t width, const std::size_t height,
	const float* filter, const int kernelSize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int kernelMargin = kernelSize / 2;


	if (((x - kernelMargin) > 0 && (x + kernelMargin) < width) &&
		((y - kernelMargin) > 0 && (y + kernelMargin) < height))
	{
		int newRedChannel{}, newGreenChannel{}, newBlueChannel{};
		for (int kernelX = -kernelMargin; kernelX <= kernelMargin; ++kernelX)
		{
			for (int kernelY = -kernelMargin; kernelY <= kernelMargin; ++kernelY)
			{
				const auto kernelXIndex = kernelX + kernelMargin;
				const auto kernelYIndex = kernelY + kernelMargin;
				const auto kernelIndex = kernelXIndex * kernelSize + kernelYIndex;
				const auto kernelValue = filter[kernelIndex];
				
				const auto inPixel = &inputImageData[((x + kernelX) + (y + kernelY) * width) * 4];
				newRedChannel += inPixel[0] * kernelValue;
				newGreenChannel += inPixel[1] * kernelValue;
				newBlueChannel += inPixel[2] * kernelValue;
			}
		}

		alignChannel(newRedChannel);
		alignChannel(newGreenChannel);
		alignChannel(newBlueChannel);

		auto outPixel = &outputImageData[(x + y * width) * 4];
		outPixel[0] = newRedChannel;
		outPixel[1] = newGreenChannel;
		outPixel[2] = newBlueChannel;
	}
}

void Cuda::applyFilter(sf::Image& image, const Filter::Kernel& filter)
{
	thrust::host_vector<sf::Uint8> hostImageData{ image.getPixelsPtr(), image.getPixelsPtr() + calculateImageSize(image) };
	thrust::device_vector<sf::Uint8> devImageData(calculateImageSize(image));
	thrust::device_vector<sf::Uint8> devOutputImageData(calculateImageSize(image));
	thrust::copy(hostImageData.begin(), hostImageData.end(), devImageData.begin());

	thrust::device_vector<float> devKernel{};
	for (const auto& filterRow : filter)
	{
		devKernel.insert(devKernel.end(), filterRow.begin(), filterRow.end());
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(ceil((float)image.getSize().x / threadsPerBlock.x), ceil((float)image.getSize().y / threadsPerBlock.y));
	applyFilterOnCuda<<<numBlocks, threadsPerBlock>>>(
		devImageData.data().get(), devOutputImageData.data().get(),
		image.getSize().x, image.getSize().y,
		devKernel.data().get(), filter.size());

	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);

	float timeMs{};
	cudaEventElapsedTime(&timeMs, start, stop);
	std::cout << "[CUDA] (only CUDA calculations): " << timeMs << " ms" << std::endl;

	thrust::copy(devOutputImageData.begin(), devOutputImageData.end(), hostImageData.begin());
	image.create(image.getSize().x, image.getSize().y, hostImageData.data());
}