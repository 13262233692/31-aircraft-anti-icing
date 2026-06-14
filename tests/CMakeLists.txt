find_package(GTest QUIET)
if(GTest_FOUND)
    include(GoogleTest)
    
    add_executable(test_fvm test_fvm.cpp)
    target_link_libraries(test_fvm PRIVATE anti_icing_core GTest::gtest GTest::gtest_main)
    gtest_discover_tests(test_fvm)
    
    add_executable(test_fem test_fem.cpp)
    target_link_libraries(test_fem PRIVATE anti_icing_core GTest::gtest GTest::gtest_main)
    gtest_discover_tests(test_fem)
    
    add_executable(test_coupling test_coupling.cpp)
    target_link_libraries(test_coupling PRIVATE anti_icing_core GTest::gtest GTest::gtest_main)
    gtest_discover_tests(test_coupling)
endif()
