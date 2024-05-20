cmake_minimum_required(VERSION 3.10)

project(imo_pcd_reader)

set(CMAKE_BUILD_TYPE Release)

# Find PCL
find_package(PCL 1.10 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

# Find PyBind11
set(pybind11_DIR ${CONDA_PREFIX}/share/cmake/pybind11)
find_package(pybind11 REQUIRED) 

execute_process(
    COMMAND "python" -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Add your C++ source file
add_library(${PROJECT_NAME} MODULE pcd_reader.cpp)

# Link against necessary libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
                      ${PCL_LIBRARIES}
                      pybind11::module)

pybind11_extension(${PROJECT_NAME})

install(
    TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
)