#
# module: teachcompute.torch_extensions.piecewise_linear_c
# site-packages/torch/include/torch/extension.h
#

if(CUDA_AVAILABLE)

  message(STATUS "+ teachcompute.torch_extension.piecewise_linear_c (CUDA)")

  find_package(Torch REQUIRED)

  message(STATUS "TORCH_INCLUDE_DIRS=${TORCH_INCLUDE_DIRS}")
  message(STATUS "TORCH_LIBRARIES=${TORCH_LIBRARIES}")
  message(STATUS "TORCH_CXX_FLAGS=${TORCH_CXX_FLAGS}")

  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
  get_target_property(TORCH_INTERFACE_LIB torch INTERFACE_LINK_LIBRARIES)
  string(REPLACE "/usr/local/cuda" ${CUDA_TOOLKIT_ROOT_DIR} TORCH_INTERFACE_LIB "${TORCH_INTERFACE_LIB}")
  set_target_properties(torch PROPERTIES INTERFACE_LINK_LIBRARIES ${TORCH_INTERFACE_LIB})

  cuda_pybind11_add_module(
    piecewise_linear_c
    ../teachcompute/torch_extensions/piecewise_linear_c.cpp)

  target_include_directories(
    piecewise_linear_c
    PRIVATE
    ${TORCH_INCLUDE_DIRS})
  target_link_libraries(piecewise_linear_c PRIVATE ${TORCH_LIBRARIES})

else()

  message(STATUS "+ teachcompute.torch_extension.piecewise_linear_c (CPU)")
  # set(NO_CUDA 1)

  # find_package(Torch REQUIRED)

  message(STATUS "TORCH_INCLUDE=${TORCH_INCLUDE}")
  message(STATUS "TORCH_LIBRARIES=${TORCH_LIBRARIES}")
  # message(STATUS "TORCH_CXX_FLAGS=${TORCH_CXX_FLAGS}")
  # message(STATUS "TORCH_INCLUDE_DIRS=${TORCH_INCLUDE_DIRS}")

  # Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
  # [1/1] c++ -MMD -MF /home/xadupre/github/td3a_cpp_deep/build/temp.linux-x86_64-cpython-310/td3a_cpp_deep/fcts/piecewise_linear_c.o.d
  # -Wno-unused-result -Wsign-compare -DNDEBUG -g 
  # -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g
  # -fwrapv -O2 -fPIC
  # -I/home/xadupre/.local/lib/python3.10/site-packages/torch/include 
  # -I/home/xadupre/.local/lib/python3.10/site-packages/torch/include/torch/csrc/api/include 
  # -I/home/xadupre/.local/lib/python3.10/site-packages/torch/include/TH 
  # -I/home/xadupre/.local/lib/python3.10/site-packages/torch/include/THC 
  # -I/usr/include/python3.10 -c -c /home/xadupre/github/td3a_cpp_deep/td3a_cpp_deep/fcts/piecewise_linear_c.cpp 
  # -o /home/xadupre/github/td3a_cpp_deep/build/temp.linux-x86_64-cpython-310/td3a_cpp_deep/fcts/piecewise_linear_c.o 
  # -DTORCH_API_INCLUDE_EXTENSION_H 
  # '-DPYBIND11_COMPILER_TYPE="_gcc"' 
  # '-DPYBIND11_STDLIB="_libstdcpp"' 
  # '-DPYBIND11_BUILD_ABI="_cxxabi1011"' 
  # -DTORCH_EXTENSION_NAME=piecewise_linear_c 
  # -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++17
  # creating build/lib.linux-x86_64-cpython-310
  #creating build/lib.linux-x86_64-cpython-310/td3a_cpp_deep
  # creating build/lib.linux-x86_64-cpython-310/td3a_cpp_deep/fcts
  # x86_64-linux-gnu-g++ -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -g -fwrapv
  # -O2 /home/xadupre/github/td3a_cpp_deep/build/temp.linux-x86_64-cpython-310/td3a_cpp_deep/fcts/piecewise_linear_c.o
  # -L/home/xadupre/.local/lib/python3.10/site-packages/torch/lib 
  # -L/usr/lib/x86_64-linux-gnu -lc10 -ltorch -ltorch_cpu -ltorch_python 
  # -o build/lib.linux-x86_64-cpython-310/td3a_cpp_deep/fcts/piecewise_linear_c.cpython-310-x86_64-linux-gnu.so
  # copying build/lib.linux-x86_64-cpython-310/td3a_cpp_deep/fcts/piecewise_linear_c.cpython-310-x86_64-linux-gnu.so -> /home/xadupre/github/td3a_cpp_deep/td3a_cpp_deep/fcts

  # /usr/bin/c++ -Dpiecewise_linear_c_EXPORTS 
  # -I/home/xadupre/.local/lib/python3.10/site-packages/numpy/core/include 
  # -I/home/xadupre/.local/lib/python3.10/site-packages/torch/include 
  # -I/home/xadupre/.local/lib/python3.10/site-packages/torch/include/torch/csrc/api/include 
  # -I/home/xadupre/.local/lib/python3.10/site-packages/torch/include/TH 
  # -I/home/xadupre/.local/lib/python3.10/site-packages/torch/include/THC 
  # -isystem /usr/include/python3.10 -isystem /home/xadupre/github/teachcompute/build/temp.linux-x86_64-cpython-310/_deps/pybind11-src/include 
  # -Wall -Wno-unknown-pragmas -Wextra -mavx2 -mf16c -fPIC -O3 -DNDEBUG -O3 -std=gnu++20 -flto=auto 
  # -fno-fat-lto-objects -fPIC -fvisibility=hidden -fvisibility-inlines-hidden 
  # -fPIC -fopenmp -MD -MT 
  # CMakeFiles/piecewise_linear_c.dir/home/xadupre/github/teachcompute/teachcompute/torch_extensions/piecewise_linear_c.cpp.o 
  # -MF CMakeFiles/piecewise_linear_c.dir/home/xadupre/github/teachcompute/teachcompute/torch_extensions/piecewise_linear_c.cpp.o.d -o 
  # CMakeFiles/piecewise_linear_c.dir/home/xadupre/github/teachcompute/teachcompute/torch_extensions/piecewise_linear_c.cpp.o 
  # -c /home/xadupre/github/teachcompute/teachcompute/torch_extensions/piecewise_linear_c.cpp

  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

  local_pybind11_add_module(
    piecewise_linear_c
    OpenMP::OpenMP_CXX
    ../teachcompute/torch_extensions/piecewise_linear_c.cpp)

  target_include_directories(
    piecewise_linear_c
    PRIVATE
    ${TORCH_INCLUDE})
  target_compile_options(
    piecewise_linear_c
    PRIVATE
    -fPIC -lc10 -ltorch -ltorch_cpu -ltorch_python
    "-L${TORCH_LIBRARIES}")
  # target_link_libraries(piecewise_linear_c PRIVATE ${TORCH_LIBRARIES})

endif()
