# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

set(PROJECT_NAME "LVK AI Samples")

if(MSVC)
  add_definitions(-D_CONSOLE)
else()
  add_compile_options(-Wno-deprecated-volatile)
endif()

if(WIN32)
  add_definitions("-DVK_USE_PLATFORM_WIN32_KHR=1")
  add_definitions("-DNOMINMAX")
endif()

add_library(LVKVulkanApp STATIC
  ${SAMPLESAI_ROOT_DIR}/third-party/deps/src/lightweightvk/samples/VulkanApp.cpp
  ${SAMPLESAI_ROOT_DIR}/third-party/deps/src/lightweightvk/samples/VulkanApp.h
)
lvk_set_cxxstd(LVKVulkanApp 20)
lvk_set_folder(LVKVulkanApp "LVK")
target_link_libraries(LVKVulkanApp PUBLIC minilog glfw)
target_link_libraries(LVKVulkanApp PUBLIC LVKstb LVKLibrary)
target_include_directories(LVKVulkanApp PUBLIC "${SAMPLESAI_ROOT_DIR}/third-party/deps/src/lightweightvk/samples")

macro(ADD_DEMO app)
  add_executable(${app} "${app}.cpp")
  lvk_set_cxxstd(${app} 20)
  lvk_set_folder(${app} ${PROJECT_NAME})
  target_link_libraries(${app} PRIVATE LVKVulkanApp)
endmacro()

ADD_DEMO("GAME_TerrainRunner")
ADD_DEMO("GAME_DungeonCrawler")
