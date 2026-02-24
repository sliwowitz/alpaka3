#
# Copyright 2025 René Widera
# SPDX-License-Identifier: MPL-2.0
#


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was alpakaConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

#-------------------------------------------------------------------------------
# Common.

# alpaka root directory
set(_alpaka_ROOT_DIR "/usr/local")
# Compiler feature tests.
set(_alpaka_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Tests, this is required because the used compiler can differ compared to the compiler used for installation.
set(_alpaka_FEATURE_TESTS_DIR "${_alpaka_CMAKE_DIR}/tests")
set(_alpaka_TESTING_DIR "/usr/local/share/tests")

set(_alpaka_INCLUDE_DIRECTORY "/usr/local/include")

include("${_alpaka_CMAKE_DIR}/alpakaCommon.cmake")

check_required_components(alpaka)

# CMake Options to activate testing with the currently selected environment
option(alpaka_TESTING "Enable/Disable testing" OFF)

if(alpaka_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory("${_alpaka_TESTING_DIR}" "${CMAKE_BINARY_DIR}/alpaka/tests")
endif()
