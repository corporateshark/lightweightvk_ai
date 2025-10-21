cmake_minimum_required(VERSION 3.19)

project(SamplesAI CXX C)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(SAMPLESAI_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SAMPLESAI_DEPS_CACHE_FILE "${SAMPLESAI_ROOT_DIR}/third-party/.bootstrap.json")

if($ENV{CLION_IDE})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${SAMPLESAI_ROOT_DIR})
endif()

find_package(Python3 COMPONENTS Interpreter)

execute_process(COMMAND ${Python3_EXECUTABLE} deploy_deps.py WORKING_DIRECTORY ${SAMPLESAI_ROOT_DIR} COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND ${Python3_EXECUTABLE} deploy_deps.py WORKING_DIRECTORY ${SAMPLESAI_ROOT_DIR}/third-party/deps/src/lightweightvk COMMAND_ERROR_IS_FATAL ANY)

add_custom_command(COMMAND ${Python3_EXECUTABLE} deploy_deps.py OUTPUT ${SAMPLESAI_DEPS_CACHE_FILE}
                   DEPENDS "${SAMPLESAI_ROOT_DIR}/third-party/bootstrap.json" WORKING_DIRECTORY ${SAMPLESAI_ROOT_DIR})
add_custom_target(Dependencies DEPENDS ${SAMPLESAI_DEPS_CACHE_FILE})
set_property(TARGET Dependencies PROPERTY FOLDER "third-party")

if(NOT EXISTS ${SAMPLESAI_DEPS_CACHE_FILE})
  message(FATAL_ERROR "Cannot bootstrap dependencies.")
endif()

set(BUILD_SHARED_LIBS OFF)

# cmake-format: off
add_subdirectory(third-party/deps/src/lightweightvk/third-party/deps/src/ldrutils)
set(KTX_FEATURE_DOC            OFF CACHE BOOL "")
set(KTX_FEATURE_GL_UPLOAD      OFF CACHE BOOL "")
set(KTX_FEATURE_JNI            OFF CACHE BOOL "")
set(KTX_FEATURE_KTX1            ON CACHE BOOL "")
set(KTX_FEATURE_KTX2            ON CACHE BOOL "")
set(KTX_FEATURE_LOADTEST_APPS  OFF CACHE BOOL "")
set(KTX_FEATURE_STATIC_LIBRARY  ON CACHE BOOL "")
set(KTX_FEATURE_TESTS          OFF CACHE BOOL "")
set(KTX_FEATURE_TOOLS          OFF CACHE BOOL "")
set(KTX_FEATURE_VK_UPLOAD      OFF CACHE BOOL "")
add_subdirectory(third-party/deps/src/lightweightvk/third-party/deps/src/ktx-software)
set(LVK_WITH_SAMPLES   OFF CACHE BOOL "")
set(LVK_WITH_TRACY     OFF  CACHE BOOL "")
set(LVK_WITH_TRACY_GPU OFF  CACHE BOOL "")
add_subdirectory(third-party/deps/src/lightweightvk)
lvk_set_folder(ktx "third-party/ktx-software")
lvk_set_folder(ktx_read "third-party/ktx-software")
lvk_set_folder(ktx_version "third-party/ktx-software")
lvk_set_folder(obj_basisu_cbind "third-party/ktx-software")
lvk_set_folder(objUtil "third-party/ktx-software")
if(TARGET astcenc-avx2-static)
  lvk_set_folder(astcenc-avx2-static "third-party/ktx-software")
endif()
# cmake-format: on

if(WIN32)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

include_directories(.)
include_directories(third-party/deps/src)
include_directories(third-party/deps/src/lightweightvk/third-party/deps/src/glm)
include_directories(third-party/deps/src/lightweightvk/third-party/deps/src/imgui)
include_directories(third-party/deps/src/lightweightvk/third-party/deps/src/3D-Graphics-Rendering-Cookbook)

set_property(TARGET LUtils          PROPERTY FOLDER "third-party")

add_dependencies(LVKLibrary Dependencies)

add_subdirectory(samples)
