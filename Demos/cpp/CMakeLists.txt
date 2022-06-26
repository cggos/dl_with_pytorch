cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(demo00)

# set(Torch_DIR "/home/cg/projects/3rdparty/pytorch/torch/lib/tmp_install/share/cmake/Torch/")
set(CMAKE_PREFIX_PATH "/home/cg/projects/3rdparty/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

message("++ TORCH_LIBRARIES: " ${TORCH_LIBRARIES})

add_executable(demo00 demo00.cpp)
target_link_libraries(demo00 "${TORCH_LIBRARIES}")
set_property(TARGET demo00 PROPERTY CXX_STANDARD 14)

add_executable(demo01 demo01.cpp)
target_link_libraries(demo01 "${TORCH_LIBRARIES}")
set_property(TARGET demo01 PROPERTY CXX_STANDARD 14)

add_executable(lenet5_test lenet5_test.cpp)
target_link_libraries(lenet5_test "${TORCH_LIBRARIES}")
set_property(TARGET lenet5_test PROPERTY CXX_STANDARD 14)