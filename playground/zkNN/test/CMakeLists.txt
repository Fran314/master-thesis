#Testing macro
macro (add_test_executable_with_lib _name)
	add_executable(test_${_name} "${_name}.cpp")
	set_target_properties(test_${_name} PROPERTIES
		COMPILE_FLAGS "-g"
	)
	target_link_libraries(test_${_name} ${EMP-OT_LIBRARIES} emp-zk)
	target_link_libraries(
		test_${_name}

		ff
  		${GMP_LIBRARIES}
  		${GMPXX_LIBRARIES}
  		${PROCPS_LIBRARIES}
	)
endmacro()

macro (add_test_case _name)
	add_test_executable_with_lib(${_name})
  	add_test(NAME ${_name} COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${_name}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
endmacro()

macro (add_test_case_with_run _name)
	add_test_executable_with_lib(${_name})
	add_test(NAME ${_name} COMMAND "./run" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${_name}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")

endmacro()

add_executable(test_resnet18 "cnns.cpp")
target_link_libraries(test_resnet18 ${EMP-OT_LIBRARIES} emp-zk)
target_link_libraries(
	test_resnet18

	ff
	${GMP_LIBRARIES}
	${GMPXX_LIBRARIES}
	${PROCPS_LIBRARIES}
)
TARGET_COMPILE_DEFINITIONS(test_resnet18 PRIVATE -DRESNET18)

add_executable(test_resnet50 "cnns.cpp")
target_link_libraries(test_resnet50 ${EMP-OT_LIBRARIES} emp-zk)
target_link_libraries(
	test_resnet50

	ff
	${GMP_LIBRARIES}
	${GMPXX_LIBRARIES}
	${PROCPS_LIBRARIES}
)
TARGET_COMPILE_DEFINITIONS(test_resnet50 PRIVATE -DRESNET50)

add_executable(test_resnet101 "cnns.cpp")
target_link_libraries(test_resnet101 ${EMP-OT_LIBRARIES} emp-zk)
target_link_libraries(
	test_resnet101

	ff
	${GMP_LIBRARIES}
	${GMPXX_LIBRARIES}
	${PROCPS_LIBRARIES}
)
TARGET_COMPILE_DEFINITIONS(test_resnet101 PRIVATE -DRESNET101)

add_executable(test_vgg11 "cnns.cpp")
target_link_libraries(test_vgg11 ${EMP-OT_LIBRARIES} emp-zk)
target_link_libraries(
	test_vgg11

	ff
	${GMP_LIBRARIES}
	${GMPXX_LIBRARIES}
	${PROCPS_LIBRARIES}
)
TARGET_COMPILE_DEFINITIONS(test_vgg11 PRIVATE -DVGG_11)

# Test cases
add_test_case_with_run(transformer)
add_test_case_with_run(range)
add_test_case_with_run(lookup)
add_test_case(three_square)
add_test_case(nonlinear)
