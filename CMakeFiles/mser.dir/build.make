# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jack/textdetect-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jack/textdetect-master

# Include any dependencies generated for this target.
include CMakeFiles/mser.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mser.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mser.dir/flags.make

CMakeFiles/mser.dir/extend.cpp.o: CMakeFiles/mser.dir/flags.make
CMakeFiles/mser.dir/extend.cpp.o: extend.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jack/textdetect-master/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/mser.dir/extend.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/mser.dir/extend.cpp.o -c /home/jack/textdetect-master/extend.cpp

CMakeFiles/mser.dir/extend.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mser.dir/extend.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jack/textdetect-master/extend.cpp > CMakeFiles/mser.dir/extend.cpp.i

CMakeFiles/mser.dir/extend.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mser.dir/extend.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jack/textdetect-master/extend.cpp -o CMakeFiles/mser.dir/extend.cpp.s

CMakeFiles/mser.dir/extend.cpp.o.requires:
.PHONY : CMakeFiles/mser.dir/extend.cpp.o.requires

CMakeFiles/mser.dir/extend.cpp.o.provides: CMakeFiles/mser.dir/extend.cpp.o.requires
	$(MAKE) -f CMakeFiles/mser.dir/build.make CMakeFiles/mser.dir/extend.cpp.o.provides.build
.PHONY : CMakeFiles/mser.dir/extend.cpp.o.provides

CMakeFiles/mser.dir/extend.cpp.o.provides.build: CMakeFiles/mser.dir/extend.cpp.o

CMakeFiles/mser.dir/fast_clustering.cpp.o: CMakeFiles/mser.dir/flags.make
CMakeFiles/mser.dir/fast_clustering.cpp.o: fast_clustering.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jack/textdetect-master/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/mser.dir/fast_clustering.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/mser.dir/fast_clustering.cpp.o -c /home/jack/textdetect-master/fast_clustering.cpp

CMakeFiles/mser.dir/fast_clustering.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mser.dir/fast_clustering.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jack/textdetect-master/fast_clustering.cpp > CMakeFiles/mser.dir/fast_clustering.cpp.i

CMakeFiles/mser.dir/fast_clustering.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mser.dir/fast_clustering.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jack/textdetect-master/fast_clustering.cpp -o CMakeFiles/mser.dir/fast_clustering.cpp.s

CMakeFiles/mser.dir/fast_clustering.cpp.o.requires:
.PHONY : CMakeFiles/mser.dir/fast_clustering.cpp.o.requires

CMakeFiles/mser.dir/fast_clustering.cpp.o.provides: CMakeFiles/mser.dir/fast_clustering.cpp.o.requires
	$(MAKE) -f CMakeFiles/mser.dir/build.make CMakeFiles/mser.dir/fast_clustering.cpp.o.provides.build
.PHONY : CMakeFiles/mser.dir/fast_clustering.cpp.o.provides

CMakeFiles/mser.dir/fast_clustering.cpp.o.provides.build: CMakeFiles/mser.dir/fast_clustering.cpp.o

CMakeFiles/mser.dir/group_classifier.cpp.o: CMakeFiles/mser.dir/flags.make
CMakeFiles/mser.dir/group_classifier.cpp.o: group_classifier.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jack/textdetect-master/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/mser.dir/group_classifier.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/mser.dir/group_classifier.cpp.o -c /home/jack/textdetect-master/group_classifier.cpp

CMakeFiles/mser.dir/group_classifier.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mser.dir/group_classifier.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jack/textdetect-master/group_classifier.cpp > CMakeFiles/mser.dir/group_classifier.cpp.i

CMakeFiles/mser.dir/group_classifier.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mser.dir/group_classifier.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jack/textdetect-master/group_classifier.cpp -o CMakeFiles/mser.dir/group_classifier.cpp.s

CMakeFiles/mser.dir/group_classifier.cpp.o.requires:
.PHONY : CMakeFiles/mser.dir/group_classifier.cpp.o.requires

CMakeFiles/mser.dir/group_classifier.cpp.o.provides: CMakeFiles/mser.dir/group_classifier.cpp.o.requires
	$(MAKE) -f CMakeFiles/mser.dir/build.make CMakeFiles/mser.dir/group_classifier.cpp.o.provides.build
.PHONY : CMakeFiles/mser.dir/group_classifier.cpp.o.provides

CMakeFiles/mser.dir/group_classifier.cpp.o.provides.build: CMakeFiles/mser.dir/group_classifier.cpp.o

CMakeFiles/mser.dir/test_mser.cpp.o: CMakeFiles/mser.dir/flags.make
CMakeFiles/mser.dir/test_mser.cpp.o: test_mser.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jack/textdetect-master/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/mser.dir/test_mser.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/mser.dir/test_mser.cpp.o -c /home/jack/textdetect-master/test_mser.cpp

CMakeFiles/mser.dir/test_mser.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mser.dir/test_mser.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jack/textdetect-master/test_mser.cpp > CMakeFiles/mser.dir/test_mser.cpp.i

CMakeFiles/mser.dir/test_mser.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mser.dir/test_mser.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jack/textdetect-master/test_mser.cpp -o CMakeFiles/mser.dir/test_mser.cpp.s

CMakeFiles/mser.dir/test_mser.cpp.o.requires:
.PHONY : CMakeFiles/mser.dir/test_mser.cpp.o.requires

CMakeFiles/mser.dir/test_mser.cpp.o.provides: CMakeFiles/mser.dir/test_mser.cpp.o.requires
	$(MAKE) -f CMakeFiles/mser.dir/build.make CMakeFiles/mser.dir/test_mser.cpp.o.provides.build
.PHONY : CMakeFiles/mser.dir/test_mser.cpp.o.provides

CMakeFiles/mser.dir/test_mser.cpp.o.provides.build: CMakeFiles/mser.dir/test_mser.cpp.o

CMakeFiles/mser.dir/max_meaningful_clustering.cpp.o: CMakeFiles/mser.dir/flags.make
CMakeFiles/mser.dir/max_meaningful_clustering.cpp.o: max_meaningful_clustering.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jack/textdetect-master/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/mser.dir/max_meaningful_clustering.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/mser.dir/max_meaningful_clustering.cpp.o -c /home/jack/textdetect-master/max_meaningful_clustering.cpp

CMakeFiles/mser.dir/max_meaningful_clustering.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mser.dir/max_meaningful_clustering.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jack/textdetect-master/max_meaningful_clustering.cpp > CMakeFiles/mser.dir/max_meaningful_clustering.cpp.i

CMakeFiles/mser.dir/max_meaningful_clustering.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mser.dir/max_meaningful_clustering.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jack/textdetect-master/max_meaningful_clustering.cpp -o CMakeFiles/mser.dir/max_meaningful_clustering.cpp.s

CMakeFiles/mser.dir/max_meaningful_clustering.cpp.o.requires:
.PHONY : CMakeFiles/mser.dir/max_meaningful_clustering.cpp.o.requires

CMakeFiles/mser.dir/max_meaningful_clustering.cpp.o.provides: CMakeFiles/mser.dir/max_meaningful_clustering.cpp.o.requires
	$(MAKE) -f CMakeFiles/mser.dir/build.make CMakeFiles/mser.dir/max_meaningful_clustering.cpp.o.provides.build
.PHONY : CMakeFiles/mser.dir/max_meaningful_clustering.cpp.o.provides

CMakeFiles/mser.dir/max_meaningful_clustering.cpp.o.provides.build: CMakeFiles/mser.dir/max_meaningful_clustering.cpp.o

CMakeFiles/mser.dir/min_bounding_box.cpp.o: CMakeFiles/mser.dir/flags.make
CMakeFiles/mser.dir/min_bounding_box.cpp.o: min_bounding_box.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jack/textdetect-master/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/mser.dir/min_bounding_box.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/mser.dir/min_bounding_box.cpp.o -c /home/jack/textdetect-master/min_bounding_box.cpp

CMakeFiles/mser.dir/min_bounding_box.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mser.dir/min_bounding_box.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jack/textdetect-master/min_bounding_box.cpp > CMakeFiles/mser.dir/min_bounding_box.cpp.i

CMakeFiles/mser.dir/min_bounding_box.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mser.dir/min_bounding_box.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jack/textdetect-master/min_bounding_box.cpp -o CMakeFiles/mser.dir/min_bounding_box.cpp.s

CMakeFiles/mser.dir/min_bounding_box.cpp.o.requires:
.PHONY : CMakeFiles/mser.dir/min_bounding_box.cpp.o.requires

CMakeFiles/mser.dir/min_bounding_box.cpp.o.provides: CMakeFiles/mser.dir/min_bounding_box.cpp.o.requires
	$(MAKE) -f CMakeFiles/mser.dir/build.make CMakeFiles/mser.dir/min_bounding_box.cpp.o.provides.build
.PHONY : CMakeFiles/mser.dir/min_bounding_box.cpp.o.provides

CMakeFiles/mser.dir/min_bounding_box.cpp.o.provides.build: CMakeFiles/mser.dir/min_bounding_box.cpp.o

CMakeFiles/mser.dir/mser.cpp.o: CMakeFiles/mser.dir/flags.make
CMakeFiles/mser.dir/mser.cpp.o: mser.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jack/textdetect-master/CMakeFiles $(CMAKE_PROGRESS_7)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/mser.dir/mser.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/mser.dir/mser.cpp.o -c /home/jack/textdetect-master/mser.cpp

CMakeFiles/mser.dir/mser.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mser.dir/mser.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jack/textdetect-master/mser.cpp > CMakeFiles/mser.dir/mser.cpp.i

CMakeFiles/mser.dir/mser.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mser.dir/mser.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jack/textdetect-master/mser.cpp -o CMakeFiles/mser.dir/mser.cpp.s

CMakeFiles/mser.dir/mser.cpp.o.requires:
.PHONY : CMakeFiles/mser.dir/mser.cpp.o.requires

CMakeFiles/mser.dir/mser.cpp.o.provides: CMakeFiles/mser.dir/mser.cpp.o.requires
	$(MAKE) -f CMakeFiles/mser.dir/build.make CMakeFiles/mser.dir/mser.cpp.o.provides.build
.PHONY : CMakeFiles/mser.dir/mser.cpp.o.provides

CMakeFiles/mser.dir/mser.cpp.o.provides.build: CMakeFiles/mser.dir/mser.cpp.o

CMakeFiles/mser.dir/nfa.cpp.o: CMakeFiles/mser.dir/flags.make
CMakeFiles/mser.dir/nfa.cpp.o: nfa.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jack/textdetect-master/CMakeFiles $(CMAKE_PROGRESS_8)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/mser.dir/nfa.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/mser.dir/nfa.cpp.o -c /home/jack/textdetect-master/nfa.cpp

CMakeFiles/mser.dir/nfa.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mser.dir/nfa.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jack/textdetect-master/nfa.cpp > CMakeFiles/mser.dir/nfa.cpp.i

CMakeFiles/mser.dir/nfa.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mser.dir/nfa.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jack/textdetect-master/nfa.cpp -o CMakeFiles/mser.dir/nfa.cpp.s

CMakeFiles/mser.dir/nfa.cpp.o.requires:
.PHONY : CMakeFiles/mser.dir/nfa.cpp.o.requires

CMakeFiles/mser.dir/nfa.cpp.o.provides: CMakeFiles/mser.dir/nfa.cpp.o.requires
	$(MAKE) -f CMakeFiles/mser.dir/build.make CMakeFiles/mser.dir/nfa.cpp.o.provides.build
.PHONY : CMakeFiles/mser.dir/nfa.cpp.o.provides

CMakeFiles/mser.dir/nfa.cpp.o.provides.build: CMakeFiles/mser.dir/nfa.cpp.o

CMakeFiles/mser.dir/region_classifier.cpp.o: CMakeFiles/mser.dir/flags.make
CMakeFiles/mser.dir/region_classifier.cpp.o: region_classifier.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jack/textdetect-master/CMakeFiles $(CMAKE_PROGRESS_9)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/mser.dir/region_classifier.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/mser.dir/region_classifier.cpp.o -c /home/jack/textdetect-master/region_classifier.cpp

CMakeFiles/mser.dir/region_classifier.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mser.dir/region_classifier.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jack/textdetect-master/region_classifier.cpp > CMakeFiles/mser.dir/region_classifier.cpp.i

CMakeFiles/mser.dir/region_classifier.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mser.dir/region_classifier.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jack/textdetect-master/region_classifier.cpp -o CMakeFiles/mser.dir/region_classifier.cpp.s

CMakeFiles/mser.dir/region_classifier.cpp.o.requires:
.PHONY : CMakeFiles/mser.dir/region_classifier.cpp.o.requires

CMakeFiles/mser.dir/region_classifier.cpp.o.provides: CMakeFiles/mser.dir/region_classifier.cpp.o.requires
	$(MAKE) -f CMakeFiles/mser.dir/build.make CMakeFiles/mser.dir/region_classifier.cpp.o.provides.build
.PHONY : CMakeFiles/mser.dir/region_classifier.cpp.o.provides

CMakeFiles/mser.dir/region_classifier.cpp.o.provides.build: CMakeFiles/mser.dir/region_classifier.cpp.o

CMakeFiles/mser.dir/region.cpp.o: CMakeFiles/mser.dir/flags.make
CMakeFiles/mser.dir/region.cpp.o: region.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jack/textdetect-master/CMakeFiles $(CMAKE_PROGRESS_10)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/mser.dir/region.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/mser.dir/region.cpp.o -c /home/jack/textdetect-master/region.cpp

CMakeFiles/mser.dir/region.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mser.dir/region.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jack/textdetect-master/region.cpp > CMakeFiles/mser.dir/region.cpp.i

CMakeFiles/mser.dir/region.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mser.dir/region.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jack/textdetect-master/region.cpp -o CMakeFiles/mser.dir/region.cpp.s

CMakeFiles/mser.dir/region.cpp.o.requires:
.PHONY : CMakeFiles/mser.dir/region.cpp.o.requires

CMakeFiles/mser.dir/region.cpp.o.provides: CMakeFiles/mser.dir/region.cpp.o.requires
	$(MAKE) -f CMakeFiles/mser.dir/build.make CMakeFiles/mser.dir/region.cpp.o.provides.build
.PHONY : CMakeFiles/mser.dir/region.cpp.o.provides

CMakeFiles/mser.dir/region.cpp.o.provides.build: CMakeFiles/mser.dir/region.cpp.o

# Object files for target mser
mser_OBJECTS = \
"CMakeFiles/mser.dir/extend.cpp.o" \
"CMakeFiles/mser.dir/fast_clustering.cpp.o" \
"CMakeFiles/mser.dir/group_classifier.cpp.o" \
"CMakeFiles/mser.dir/test_mser.cpp.o" \
"CMakeFiles/mser.dir/max_meaningful_clustering.cpp.o" \
"CMakeFiles/mser.dir/min_bounding_box.cpp.o" \
"CMakeFiles/mser.dir/mser.cpp.o" \
"CMakeFiles/mser.dir/nfa.cpp.o" \
"CMakeFiles/mser.dir/region_classifier.cpp.o" \
"CMakeFiles/mser.dir/region.cpp.o"

# External object files for target mser
mser_EXTERNAL_OBJECTS =

mser: CMakeFiles/mser.dir/extend.cpp.o
mser: CMakeFiles/mser.dir/fast_clustering.cpp.o
mser: CMakeFiles/mser.dir/group_classifier.cpp.o
mser: CMakeFiles/mser.dir/test_mser.cpp.o
mser: CMakeFiles/mser.dir/max_meaningful_clustering.cpp.o
mser: CMakeFiles/mser.dir/min_bounding_box.cpp.o
mser: CMakeFiles/mser.dir/mser.cpp.o
mser: CMakeFiles/mser.dir/nfa.cpp.o
mser: CMakeFiles/mser.dir/region_classifier.cpp.o
mser: CMakeFiles/mser.dir/region.cpp.o
mser: CMakeFiles/mser.dir/build.make
mser: /usr/lib/i386-linux-gnu/libopencv_videostab.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_video.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_ts.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_superres.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_stitching.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_photo.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_ocl.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_objdetect.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_ml.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_legacy.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_imgproc.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_highgui.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_gpu.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_flann.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_features2d.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_core.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_contrib.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_calib3d.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_videostab.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_video.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_ts.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_superres.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_stitching.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_photo.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_ocl.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_objdetect.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_ml.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_legacy.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_imgproc.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_highgui.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_gpu.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_flann.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_features2d.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_core.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_contrib.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_calib3d.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_photo.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_legacy.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_video.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_objdetect.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_ml.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_calib3d.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_features2d.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_highgui.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_imgproc.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_flann.so.2.4.8
mser: /usr/lib/i386-linux-gnu/libopencv_core.so.2.4.8
mser: CMakeFiles/mser.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable mser"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mser.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mser.dir/build: mser
.PHONY : CMakeFiles/mser.dir/build

CMakeFiles/mser.dir/requires: CMakeFiles/mser.dir/extend.cpp.o.requires
CMakeFiles/mser.dir/requires: CMakeFiles/mser.dir/fast_clustering.cpp.o.requires
CMakeFiles/mser.dir/requires: CMakeFiles/mser.dir/group_classifier.cpp.o.requires
CMakeFiles/mser.dir/requires: CMakeFiles/mser.dir/test_mser.cpp.o.requires
CMakeFiles/mser.dir/requires: CMakeFiles/mser.dir/max_meaningful_clustering.cpp.o.requires
CMakeFiles/mser.dir/requires: CMakeFiles/mser.dir/min_bounding_box.cpp.o.requires
CMakeFiles/mser.dir/requires: CMakeFiles/mser.dir/mser.cpp.o.requires
CMakeFiles/mser.dir/requires: CMakeFiles/mser.dir/nfa.cpp.o.requires
CMakeFiles/mser.dir/requires: CMakeFiles/mser.dir/region_classifier.cpp.o.requires
CMakeFiles/mser.dir/requires: CMakeFiles/mser.dir/region.cpp.o.requires
.PHONY : CMakeFiles/mser.dir/requires

CMakeFiles/mser.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mser.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mser.dir/clean

CMakeFiles/mser.dir/depend:
	cd /home/jack/textdetect-master && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jack/textdetect-master /home/jack/textdetect-master /home/jack/textdetect-master /home/jack/textdetect-master /home/jack/textdetect-master/CMakeFiles/mser.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mser.dir/depend
