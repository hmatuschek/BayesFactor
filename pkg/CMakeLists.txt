#
# NOTE:
#  This CMake build script does nothing but allows me to use my preferred IDE (QtCreator). So
#  simply ignore it.

cmake_minimum_required(VERSION 2.6)
PROJECT(BayesFactor)

INCLUDE_DIRECTORIES(
  "/Users/hannes/Library/R/3.0/library/Rcpp/include" 
  "/Users/hannes/Library/R/3.0/library/RcppEigen/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/src")

SET(BF_PACKAGE BayesFactor/DESCRIPTION BayesFactor/NAMESPACE
    BayesFactor/R/oneWayAOV-utility.R BayesFactor/R/nWayAOV-utility.R)
add_custom_target(bf-pkg SOURCES ${BF_PACKAGE})

# Template library headers (indp. of R runtime)
SET(BF_HEADERS BayesFactor/src/gibbsonewayanova.h BayesFactor/src/rrng.h
    BayesFactor/src/gibbsnwayanova.h
    BayesFactor/src/sparseonewayaov.h )
SET(BF_SOURCES
    BayesFactor/src/sparseonewayaov.cpp
    BayesFactor/src/oneWayAOV.c BayesFactor/src/nWayAOV.c)

add_custom_target(bf-src SOURCES ${BF_HEADERS} ${BF_SOURCES})




