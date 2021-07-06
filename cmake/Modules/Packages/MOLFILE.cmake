set(MOLFILE_INCLUDE_DIR "${LAMMPS_LIB_SOURCE_DIR}/molfile" CACHE STRING "Path to VMD molfile plugin headers")
add_library(molfile INTERFACE)
target_include_directories(molfile INTERFACE ${MOLFILE_INCLUDE_DIR})
if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Windows"))
  target_link_libraries(molfile INTERFACE ${CMAKE_DL_LIBS})
endif()
target_link_libraries(lammps PRIVATE molfile)