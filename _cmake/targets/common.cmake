#
# module: common C++ libraries
#
message(STATUS "+ KERNEL teachcompute.common")
add_library(common STATIC ../teachcompute/cpp/teachcompute_helpers.cpp)
target_compile_definitions(common PRIVATE PYTHON_MANYLINUX=${PYTHON_MANYLINUX})
target_include_directories(common PRIVATE "${ROOT_INCLUDE_PATH}")
