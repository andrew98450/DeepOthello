cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(DeepOthello)

find_package(Torch REQUIRED)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

add_executable(DeepOthello main.cpp)
target_link_libraries(DeepOthello ${TORCH_LIBRARIES})
set_property(TARGET DeepOthello PROPERTY CXX_STANDARD 11)