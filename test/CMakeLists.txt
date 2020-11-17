add_executable(TestInitializeQuadric TestInitializeQuadric.cpp)
target_link_libraries(TestInitializeQuadric ${PROJECT_NAME} ${THIRD_PARTY_LIBS} )

add_executable(TestProjectQuadric TestProjectQuadric.cpp)
target_link_libraries(TestProjectQuadric ${PROJECT_NAME} ${THIRD_PARTY_LIBS} )