cmake_minimum_required(VERSION 3.18)
project(AircraftAntiIcing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(Eigen3)

if(MSVC)
    add_compile_options(/W3 /O2 /openmp:llvm)
    add_definitions(-D_USE_MATH_DEFINES -DNOMINMAX)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native)
    find_package(OpenMP REQUIRED)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/**/*.cpp
)

add_library(anti_icing_core STATIC ${SOURCES})

target_link_libraries(anti_icing_core PUBLIC Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(anti_icing_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(anti_icing_solver ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(anti_icing_solver PRIVATE anti_icing_core)

enable_testing()
add_subdirectory(tests)
