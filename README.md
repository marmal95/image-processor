# Image-Processor 

The project contains sequential and different parallel implementations of **image processing** algorithm.<br />
The project was created for the needs of master's thesis of Computer Science studies.<br/>
The aim of the project was to compare performance of sequential and parallel implementations, depending on threads and processes number used for computing, with the use of different approaches and technologies.

## Functionality 
The selected image is processed by applying a chosen filter e.g. blur/sharpen/edge detection/emboss/outline or any other by providing appropriate kernel matrix.<br>
The input image is loaded from **images** directory by entering image name in code.<br>
Modified image is saved in the same directory.

Filter can be chosen in **main** function by modifying:
```
auto filter = Filter::blurKernel();
```

A few kernels are available to choose from but new can be easily added.
The result may be poorly visibly on high resolution images as the project was created to analyze performance, not to focus on functional image processing aspects.

## Technologies

* C++
* [OpenMP](https://en.wikipedia.org/wiki/OpenMP)
* [OpenMPI](https://en.wikipedia.org/wiki/Open_MPI)
* [CUDA](https://en.wikipedia.org/wiki/CUDA)
    * [Thrust](https://github.com/NVIDIA/thrust)

## Implementations

The project contains a few **image processing** implementations:
* Sequential implementation - using pure C++
* Multithreaded implementation with the use of **OpenMP**
* Parallel implementation with the use of **OpenMPI**
* Massive parallel implementation with use of **CUDA**

## Installation

### Dependencies
* Visual Studio
* OpenMP support enabled in Visual Studio (for ImageProcessor-OpenMP project)
* OpenMPI implementation installed e.g. Microsoft MPI
* CUDA installed (+ CUDA capable graphic card)
* Thrust

### Repository

```sh
$ git clone https://github.com/marmal95/image-processor.git
```

### Build

The Visual Studio solution contains four projects inside - responding four implementations mentioned in [Implementations](#Implementations) section.
<br/>
Build whole solution by choosing:
```
Build > Build Solution
```
from top menu, or right-click specific project in **Solution Explorer** and choose:
```
Build
```

### Run
Right-click on chosen project in **Solution Explorer** view and click **Set as Startup Project**.<br/>
Click **F5** or choose **Debug >> Start Debugging** from top menu. 


## Customization

### OpenMP

Preferred number of threads in OpenMP implementation used for computing may be changed with function call:
```
omp_set_num_threads(NUM_THREADS)
```
which is called at the beginning of **main()** function.


### OpenMPI

Preferred number of processes is passed as parameter to **mpiexec** command.<br>
The value may be changed in **Visual Studio** in: **Project > Properties > Configuration Properties > Debugging**.<br>
```
Command             mpiexec.exe
Command Arguments   -n 4 "$(TargetPath)"
```


### CUDA

Preferred size of grid used for computing is specified inside **CudaAlgorithm.cu** file.<br>
```
dim3 threadsPerBlock(16, 16);
dim3 numBlocks(..., ...); // currently calculated based on image size
applyFilterOnCuda<<<numBlocks, threadsPerBlock>>>(...);
```
Inside that call the number of blocks and number of threads per each block is specified.
