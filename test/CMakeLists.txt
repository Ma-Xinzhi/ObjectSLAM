add_executable(TestInitializeQuadric TestInitializeQuadric.cpp)
target_link_libraries(TestInitializeQuadric ${PROJECT_NAME} ${THIRD_PARTY_LIBS} )

add_executable(TestBuildQuadric TestBuildQuadric.cpp)
target_link_libraries(TestBuildQuadric ${PROJECT_NAME} ${THIRD_PARTY_LIBS})

add_executable(TestProjectQuadric TestProjectQuadric.cpp)
target_link_libraries(TestProjectQuadric ${PROJECT_NAME} ${THIRD_PARTY_LIBS} )

add_executable(TestDetector TestDetector.cpp)
target_link_libraries(TestDetector ${PROJECT_NAME} ${THIRD_PARTY_LIBS} )