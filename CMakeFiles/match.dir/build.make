# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/d09/Downloads/clion-2017.3.4/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/d09/Downloads/clion-2017.3.4/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/d09/CLionProjects/shanxi_project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/d09/CLionProjects/shanxi_project

# Include any dependencies generated for this target.
include CMakeFiles/match.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/match.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/match.dir/flags.make

CMakeFiles/match.dir/test.cpp.o: CMakeFiles/match.dir/flags.make
CMakeFiles/match.dir/test.cpp.o: test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/d09/CLionProjects/shanxi_project/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/match.dir/test.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/match.dir/test.cpp.o -c /home/d09/CLionProjects/shanxi_project/test.cpp

CMakeFiles/match.dir/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/match.dir/test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/d09/CLionProjects/shanxi_project/test.cpp > CMakeFiles/match.dir/test.cpp.i

CMakeFiles/match.dir/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/match.dir/test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/d09/CLionProjects/shanxi_project/test.cpp -o CMakeFiles/match.dir/test.cpp.s

CMakeFiles/match.dir/test.cpp.o.requires:

.PHONY : CMakeFiles/match.dir/test.cpp.o.requires

CMakeFiles/match.dir/test.cpp.o.provides: CMakeFiles/match.dir/test.cpp.o.requires
	$(MAKE) -f CMakeFiles/match.dir/build.make CMakeFiles/match.dir/test.cpp.o.provides.build
.PHONY : CMakeFiles/match.dir/test.cpp.o.provides

CMakeFiles/match.dir/test.cpp.o.provides.build: CMakeFiles/match.dir/test.cpp.o


CMakeFiles/match.dir/sim.cpp.o: CMakeFiles/match.dir/flags.make
CMakeFiles/match.dir/sim.cpp.o: sim.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/d09/CLionProjects/shanxi_project/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/match.dir/sim.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/match.dir/sim.cpp.o -c /home/d09/CLionProjects/shanxi_project/sim.cpp

CMakeFiles/match.dir/sim.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/match.dir/sim.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/d09/CLionProjects/shanxi_project/sim.cpp > CMakeFiles/match.dir/sim.cpp.i

CMakeFiles/match.dir/sim.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/match.dir/sim.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/d09/CLionProjects/shanxi_project/sim.cpp -o CMakeFiles/match.dir/sim.cpp.s

CMakeFiles/match.dir/sim.cpp.o.requires:

.PHONY : CMakeFiles/match.dir/sim.cpp.o.requires

CMakeFiles/match.dir/sim.cpp.o.provides: CMakeFiles/match.dir/sim.cpp.o.requires
	$(MAKE) -f CMakeFiles/match.dir/build.make CMakeFiles/match.dir/sim.cpp.o.provides.build
.PHONY : CMakeFiles/match.dir/sim.cpp.o.provides

CMakeFiles/match.dir/sim.cpp.o.provides.build: CMakeFiles/match.dir/sim.cpp.o


# Object files for target match
match_OBJECTS = \
"CMakeFiles/match.dir/test.cpp.o" \
"CMakeFiles/match.dir/sim.cpp.o"

# External object files for target match
match_EXTERNAL_OBJECTS =

match: CMakeFiles/match.dir/test.cpp.o
match: CMakeFiles/match.dir/sim.cpp.o
match: CMakeFiles/match.dir/build.make
match: /usr/local/lib/libopencv_cudabgsegm.so.3.3.0
match: /usr/local/lib/libopencv_cudaobjdetect.so.3.3.0
match: /usr/local/lib/libopencv_cudastereo.so.3.3.0
match: /usr/local/lib/libopencv_dnn.so.3.3.0
match: /usr/local/lib/libopencv_ml.so.3.3.0
match: /usr/local/lib/libopencv_shape.so.3.3.0
match: /usr/local/lib/libopencv_stitching.so.3.3.0
match: /usr/local/lib/libopencv_superres.so.3.3.0
match: /usr/local/lib/libopencv_videostab.so.3.3.0
match: /usr/local/lib/libopencv_cudafeatures2d.so.3.3.0
match: /usr/local/lib/libopencv_cudacodec.so.3.3.0
match: /usr/local/lib/libopencv_cudaoptflow.so.3.3.0
match: /usr/local/lib/libopencv_cudalegacy.so.3.3.0
match: /usr/local/lib/libopencv_calib3d.so.3.3.0
match: /usr/local/lib/libopencv_cudawarping.so.3.3.0
match: /usr/local/lib/libopencv_features2d.so.3.3.0
match: /usr/local/lib/libopencv_flann.so.3.3.0
match: /usr/local/lib/libopencv_highgui.so.3.3.0
match: /usr/local/lib/libopencv_objdetect.so.3.3.0
match: /usr/local/lib/libopencv_photo.so.3.3.0
match: /usr/local/lib/libopencv_cudaimgproc.so.3.3.0
match: /usr/local/lib/libopencv_cudafilters.so.3.3.0
match: /usr/local/lib/libopencv_cudaarithm.so.3.3.0
match: /usr/local/lib/libopencv_video.so.3.3.0
match: /usr/local/lib/libopencv_videoio.so.3.3.0
match: /usr/local/lib/libopencv_imgcodecs.so.3.3.0
match: /usr/local/lib/libopencv_imgproc.so.3.3.0
match: /usr/local/lib/libopencv_core.so.3.3.0
match: /usr/local/lib/libopencv_cudev.so.3.3.0
match: CMakeFiles/match.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/d09/CLionProjects/shanxi_project/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable match"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/match.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/match.dir/build: match

.PHONY : CMakeFiles/match.dir/build

CMakeFiles/match.dir/requires: CMakeFiles/match.dir/test.cpp.o.requires
CMakeFiles/match.dir/requires: CMakeFiles/match.dir/sim.cpp.o.requires

.PHONY : CMakeFiles/match.dir/requires

CMakeFiles/match.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/match.dir/cmake_clean.cmake
.PHONY : CMakeFiles/match.dir/clean

CMakeFiles/match.dir/depend:
	cd /home/d09/CLionProjects/shanxi_project && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/d09/CLionProjects/shanxi_project /home/d09/CLionProjects/shanxi_project /home/d09/CLionProjects/shanxi_project /home/d09/CLionProjects/shanxi_project /home/d09/CLionProjects/shanxi_project/CMakeFiles/match.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/match.dir/depend
