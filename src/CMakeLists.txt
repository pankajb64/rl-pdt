cmake_minimum_required(VERSION 3.6)
set(PROJ_NAME pdt)
project(${PROJ_NAME})

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        src/Env.cpp
        src/CMakeLists.txt
        src/main.cpp
        src/QLearningTab.cpp
        src/QLearningTLF.cpp
        src/NLGA.cpp
        )

include_directories(
        header
        include
)

option(USE_MULTITHREADING "Use multithreading options" ON)
option(DISPLAY_GRAPHICS "Displaying graphics" ON)
option(USE_X11 "use x11 for cimg display" ON)

# Find the location of the openmp package (needed for parallel operations)
# required for building
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
# add O3 Optimization commands to the compiler
add_definitions(-O3)
if (USE_MULTITHREADING)
    # add the multithreading flag to the header
    add_definitions(-D_USE_MULTITHREADING)
endif(USE_MULTITHREADING)

if (DISPLAY_GRAPHICS)
    # define the _DISPLAY_GRAPHICS option for the header files to use
    add_definitions(-D_DISPLAY_GRAPHICS)
    if (USE_X11)
        # have to find the X11 package for displaying graphics
        # remove references to X11 and _DISPLAY_GRAPHICS if you don't want to draw the environments
        find_package(X11 REQUIRED)
    else()
        # set the display to not a window (this isn't necessary)
        target_compile_definitions(${PROJ_NAME} PRIVATE cimg_display=0)
    endif()
endif(DISPLAY_GRAPHICS)

add_definitions(-D_USE_MULTITHREADING)
add_definitions(-O3)

# create the executable named as the project name
add_executable(${PROJ_NAME} ${SOURCE_FILES})

# link the X11 library to the project
if (USE_X11 AND DISPLAY_GRAPHICS)
    target_link_libraries(${PROJ_NAME} ${X11_LIBRARIES})
endif()