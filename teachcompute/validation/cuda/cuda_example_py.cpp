#include "cuda_example.cuh"
#include "cuda_example_reduce.cuh"

#include "cuda_runtime.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace cuda_example;

#define py_array_float                                                         \
  py::array_t<float, py::array::c_style | py::array::forcecast>

#define py_array_uint8_t                                                       \
  py::array_t<uint8_t, py::array::c_style | py::array::forcecast>

PYBIND11_MODULE(cuda_example_py, m) {
  m.doc() =
#if defined(__APPLE__)
      "C++ experimental implementations with CUDA."
#else
      R"pbdoc(C++ experimental implementations with CUDA.)pbdoc"
#endif
      ;

#if defined(CUDA_VERSION)
  m.def(
      "cuda_version", []() -> int { return CUDA_VERSION; },
      "Returns the CUDA version the project was compiled with.");
#else
  m.def(
      "cuda_version", []() -> int { return 0; },
      "CUDA was not enabled during the compilation.");
#endif

  m.def(
      "cuda_device_count",
      []() -> int {
        int devices;
        auto status = cudaGetDeviceCount(&devices);
        if (status != cudaSuccess)
          throw std::runtime_error(
              std::string("Unable to retrieve the number of devices ") +
              std::string(cudaGetErrorString(status)));
        return devices;
      },
      "Returns the number of cuda devices.");

  m.def(
      "cuda_device_memory",
      [](int device) -> py::tuple {
        cudaSetDevice(device);
        size_t free_memory, total_memory;
        auto status = cudaMemGetInfo(&free_memory, &total_memory);
        if (status != cudaSuccess)
          throw std::runtime_error(
              std::string("Unable to retrieve the memory for a device ") +
              std::string(cudaGetErrorString(status)));
        return py::make_tuple(free_memory, total_memory);
      },
      py::arg("device") = 0,
      "Returns the free and total memory for a particular device.");

  m.def(
      "cuda_devices_memory",
      []() -> py::list {
        int devices;
        auto status = cudaGetDeviceCount(&devices);
        if (status != cudaSuccess)
          throw std::runtime_error(
              std::string("Unable to retrieve the number of devices ") +
              std::string(cudaGetErrorString(status)));
        py::list res;
        size_t free_memory, total_memory;
        for (int i = 0; i < devices; ++i) {
          cudaSetDevice(i);
          status = cudaMemGetInfo(&free_memory, &total_memory);
          if (status != cudaSuccess)
            throw std::runtime_error(
                std::string("Unable to retrieve the memory for a device ") +
                std::string(cudaGetErrorString(status)));
          res.append(py::make_tuple(free_memory, total_memory));
        }
        return res;
      },
      "Returns the free and total memory for all devices.");

  m.def(
      "get_device_prop",
      [](int device_id) -> py::dict {
        cudaDeviceProp prop;
        auto status = cudaGetDeviceProperties(&prop, device_id);
        if (status != cudaSuccess)
          throw std::runtime_error(
              std::string("Unable to retrieve the device property ") +
              std::string(cudaGetErrorString(status)));
        py::dict res;
        res["name"] = py::str(prop.name);
        res["totalGlobalMem"] = prop.totalGlobalMem;
        res["maxThreadsPerBlock"] = prop.maxThreadsPerBlock;
        res["computeMode"] = prop.computeMode;
        res["major"] = prop.major;
        res["minor"] = prop.minor;
        res["isMultiGpuBoard"] = prop.isMultiGpuBoard;
        res["concurrentKernels"] = prop.concurrentKernels;
        res["totalConstMem"] = prop.totalConstMem;
        res["clockRate"] = prop.clockRate;
        res["sharedMemPerBlock"] = prop.sharedMemPerBlock;
        res["multiProcessorCount"] = prop.multiProcessorCount;
        return res;
      },
      py::arg("device_id") = 0, "Returns the device properties.");

  m.def(
      "vector_add",
      [](const py_array_float &v1, const py_array_float &v2, int cuda_device,
         int repeat) -> py_array_float {
        if (v1.size() != v2.size()) {
          throw std::runtime_error(
              "Vectors v1 and v2 have different number of elements.");
        }
        auto ha1 = v1.request();
        float *ptr1 = reinterpret_cast<float *>(ha1.ptr);
        auto ha2 = v2.request();
        float *ptr2 = reinterpret_cast<float *>(ha2.ptr);

        std::vector<int64_t> shape(v1.ndim());
        for (int i = 0; i < v1.ndim(); ++i) {
          shape[i] = v1.shape(i);
        }
        py_array_float result = py::array_t<float>(shape);
        py::buffer_info br = result.request();

        float *pr = static_cast<float *>(br.ptr); // pointer on result data
        if (ptr1 == nullptr || ptr2 == nullptr || pr == nullptr) {
          throw std::runtime_error("One vector is empty.");
        }
        vector_add(v1.size(), ptr1, ptr2, pr, cuda_device, repeat);
        return result;
      },
      py::arg("v1"), py::arg("v2"), py::arg("cuda_device") = 0,
      py::arg("repeat") = 1,
      R"pbdoc(Computes the additions of two vectors
of the same size with CUDA.

:param v1: array
:param v2: array
:param repeat: number of times to repeat the addition
:param cuda_device: device id (if mulitple one)
:return: addition of the two arrays
)pbdoc");

  m.def(
      "vector_add_stream",
      [](const py_array_float &v1, const py_array_float &v2, int cuda_device,
         int repeat) -> py_array_float {
        if (v1.size() != v2.size()) {
          throw std::runtime_error(
              "Vectors v1 and v2 have different number of elements.");
        }
        auto ha1 = v1.request();
        float *ptr1 = reinterpret_cast<float *>(ha1.ptr);
        auto ha2 = v2.request();
        float *ptr2 = reinterpret_cast<float *>(ha2.ptr);

        std::vector<int64_t> shape(v1.ndim());
        for (int i = 0; i < v1.ndim(); ++i) {
          shape[i] = v1.shape(i);
        }
        py_array_float result = py::array_t<float>(shape);
        py::buffer_info br = result.request();

        float *pr = static_cast<float *>(br.ptr); // pointer on result data
        if (ptr1 == nullptr || ptr2 == nullptr || pr == nullptr) {
          throw std::runtime_error("One vector is empty.");
        }
        vector_add_stream(v1.size(), ptr1, ptr2, pr, cuda_device, repeat);
        return result;
      },
      py::arg("v1"), py::arg("v2"), py::arg("cuda_device") = 0,
      py::arg("repeat") = 1,
      R"pbdoc(Computes the additions of two vectors
of the same size with CUDA and CUDA streams.

:param v1: array
:param v2: array
:param cuda_device: device id (if mulitple one)
:param repeat: number of times to repeat the addition
:return: addition of the two arrays
)pbdoc");

  m.def(
      "vector_sum0",
      [](const py_array_float &vect, int max_threads,
         int cuda_device) -> float {
        if (vect.size() == 0)
          return 0;
        auto ha = vect.request();
        const float *ptr = reinterpret_cast<float *>(ha.ptr);
        return vector_sum0(static_cast<unsigned int>(vect.size()), ptr,
                           max_threads, cuda_device);
      },
      py::arg("vect"), py::arg("max_threads") = 256, py::arg("cuda_device") = 0,
      R"pbdoc(Computes the sum of all coefficients with CUDA. Naive method.

:param vect: array
:param max_threads: number of threads to use (it must be a power of 2)
:param cuda_device: device id (if mulitple one)
:return: sum
)pbdoc");

  m.def(
      "vector_sum_atomic",
      [](const py_array_float &vect, int max_threads,
         int cuda_device) -> float {
        if (vect.size() == 0)
          return 0;
        auto ha = vect.request();
        const float *ptr = reinterpret_cast<float *>(ha.ptr);
        return vector_sum_atomic(static_cast<unsigned int>(vect.size()), ptr,
                                 max_threads, cuda_device);
      },
      py::arg("vect"), py::arg("max_threads") = 256, py::arg("cuda_device") = 0,
      R"pbdoc(Computes the sum of all coefficients with CUDA. Uses atomicAdd

:param vect: array
:param max_threads: number of threads to use (it must be a power of 2)
:param cuda_device: device id (if mulitple one)
:return: sum
)pbdoc");

  m.def(
      "vector_sum6",
      [](const py_array_float &vect, int max_threads,
         int cuda_device) -> float {
        if (vect.size() == 0)
          return 0;
        auto ha = vect.request();
        const float *ptr = reinterpret_cast<float *>(ha.ptr);
        return vector_sum6(static_cast<unsigned int>(vect.size()), ptr,
                           max_threads, cuda_device);
      },
      py::arg("vect"), py::arg("max_threads") = 256, py::arg("cuda_device") = 0,
      R"pbdoc(Computes the sum of all coefficients with CUDA. More efficient method.

:param vect: array
:param max_threads: number of threads to use (it must be a power of 2)
:param cuda_device: device id (if mulitple one)
:return: sum
)pbdoc");
}
