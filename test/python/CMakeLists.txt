# Copy h5 files to binary dir
file(GLOB_RECURSE all_h5_ref_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.ref.h5)
foreach(file ${all_h5_ref_files})
  configure_file(${file} ${file} COPYONLY)
endforeach()

# List of all tests. toy.py is a helper module, not a test: the tests pick it up
# from the source directory, which python puts on sys.path as the script's own.
set(all_tests
  test_gf
  test_h5
  test_imtime
  test_offdiag
  test_options
  test_sigma
  test_validate
  test_warnings
)

foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  add_test(NAME Py_${test_name} COMMAND ${TRIQS_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}/${test_name}.py WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  set_property(TEST Py_${test_name} APPEND PROPERTY ENVIRONMENT_MODIFICATION
    "PYTHONPATH=path_list_prepend:${PROJECT_BINARY_DIR}/python")
endforeach()
