# ##############################################################################
#
# triqs_ana_cont_interface - A TRIQS front-end for the ana_cont analytic
# continuation library
#
# Copyright (C) 2026, H. LaBollita
#
# triqs_ana_cont_interface is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# triqs_ana_cont_interface is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# triqs_ana_cont_interface (in the file COPYING.txt in this directory). If not,
# see <http://www.gnu.org/licenses/>.
#
# ##############################################################################

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
cmake_policy(VERSION ${CMAKE_VERSION})

# ############
# Define Project
#
# This application is pure Python: there is no c++ subdirectory, no compiled
# extension module and no exported cmake target. LANGUAGES CXX is kept only
# because the imported TRIQS targets carry c++ properties.
#
# The repository and the documentation are hosted as ana_cont_interface, without
# the triqs_ prefix, as the other TRIQS applications do (TRIQS/cthyb hosts the
# triqs_cthyb module). Urls therefore spell out ana_cont_interface rather than
# interpolating PROJECT_NAME.
project(triqs_ana_cont_interface VERSION 4.0.0 LANGUAGES CXX)
get_directory_property(IS_SUBPROJECT PARENT_DIRECTORY)

# version displayed in the documentation
set(BRANCH_FOR_DOC unstable)

# ############
# Load TRIQS
find_package(TRIQS 4.0 REQUIRED)

# Get the git hash & print status
triqs_get_git_hash_of_source_dir(PROJECT_GIT_HASH)
message(STATUS "${PROJECT_NAME} version : ${PROJECT_VERSION}")
message(STATUS "${PROJECT_NAME} Git hash: ${PROJECT_GIT_HASH}")

# Enforce Consistent Versioning
if(NOT ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR} VERSION_EQUAL ${TRIQS_VERSION_MAJOR}.${TRIQS_VERSION_MINOR})
  message(FATAL_ERROR "The ${PROJECT_NAME} version ${PROJECT_VERSION} is not compatible with TRIQS version ${TRIQS_VERSION}.")
endif()

# There is nothing to install without python support
if(NOT TRIQS_WITH_PYTHON_SUPPORT)
  message(FATAL_ERROR "${PROJECT_NAME} is a python application, but TRIQS was installed without python support.")
endif()

# ana_cont is a runtime dependency. It is not pip-installable, so it cannot be
# declared as a package requirement and is only reported on here.
execute_process(COMMAND ${TRIQS_PYTHON_EXECUTABLE} -c "import ana_cont"
  RESULT_VARIABLE ana_cont_NOT_FOUND OUTPUT_QUIET ERROR_QUIET)
if(ana_cont_NOT_FOUND)
  message(WARNING "The python module 'ana_cont' was not found by ${TRIQS_PYTHON_EXECUTABLE}. "
                  "${PROJECT_NAME} installs without it, but every continuation will fail at "
                  "import time. See https://github.com/josefkaufmann/ana_cont")
endif()

# Default Install directory to TRIQS_ROOT if not given or when provided as relative path.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR (NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX}))
  message(STATUS "No install prefix given (or invalid). Defaulting to TRIQS_ROOT")
  set(CMAKE_INSTALL_PREFIX ${TRIQS_ROOT} CACHE PATH "default install path" FORCE)
  set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT FALSE)
endif()
if(NOT IS_SUBPROJECT)
  message(STATUS "-------- CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX} --------")
endif()
set(${PROJECT_NAME}_BINARY_DIR ${PROJECT_BINARY_DIR} CACHE STRING "Binary directory of the ${PROJECT_NAME} Project")

# Make additional Find Modules available
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/share/cmake/Modules)

# ############
# CMake Options

# Documentation
option(Build_Documentation "Build documentation" OFF)

# Testing
option(Build_Tests "Build tests" ON)
if(Build_Tests)
  enable_testing()
endif()

# Provide GNU Installation directories
include(GNUInstallDirs)

# #############
# Build Project

# Install the python package
add_subdirectory(python/${PROJECT_NAME})

# Tests
if(Build_Tests)
  add_subdirectory(test)
endif()

# Docs
if(NOT IS_SUBPROJECT AND Build_Documentation)
  add_subdirectory(doc)
endif()

# Additional configuration files
add_subdirectory(share)
