# Copyright 2016 Anna Sinelnikova and Kristofer Björnson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

## @file Makefile
## @brief ...
##
## Run $make list_of_objects to see
## the rule for compilation only onle file.
##
## @author Anna Sinelnikova
## @author Kristofer Björnson
##
## @cond

# Main compiler
CC:= g++

# CUDA compiler
NVCC:= nvcc

# Compiler flags
CFLAGS:= -std=c++11 -Wall -fopenmp -O3

# CUDA compiler flags
NVCCFLAGS:= -D_FORCE_INLINES -std=c++11 --compiler-options "-fopenmp"

# Libraries (can also include path to these libraries as -L/SOME_PATH)
LIB:= -lm

#
CUDA_LIB:=

# Optimization (-O3 or -O2)
OPT:= -O3

# Optimization (-O3 or -O2)
NVCCOPT:= -O3

# Directory for executables
BIN_DIR:= .

# Name for exectutable
STATIC_LIB = build/libTBTK.a

# Root directories for .h files
INC_DIR = include/Builders/
INC_DIR += include/Core/
INC_DIR += include/Lattices/
INC_DIR += include/Lattices/D2/
INC_DIR += include/Lattices/D3/
INC_DIR += include/ManyBody/
INC_DIR += include/ManyBody/FockStateMap/
INC_DIR += include/ManyBody/FockStateRule/
INC_DIR += include/Properties/
INC_DIR += include/PropertyExtractors/
INC_DIR += include/Solvers/
INC_DIR += include/StatesAndOperators/
INC_DIR += include/Utilities/
INC_DIR += include/Uncategorized/
INC_DIR += $(TBTK_dir)/hdf5/hdf5-build/hdf5/include/
INC_DIR += $(TBTK_dir)/hdf5/hdf5-1.8.16/include/

# Main source directory
SRC_DIR = src/Builders
SRC_DIR += src/Core
SRC_DIR += src/Lattices
SRC_DIR += src/Lattices/D2
SRC_DIR += src/Lattices/D3
SRC_DIR += src/ManyBody
SRC_DIR += src/ManyBody/FockStateRule
SRC_DIR += src/Properties
SRC_DIR += src/PropertyExtractors
SRC_DIR += src/Solvers
SRC_DIR += src/StatesAndOperators
SRC_DIR += src/Utilities
SRC_DIR += src/Uncategorized

# CUDA source directory
CUDA_SRC_DIR:= src/cuda

#No CUDA source directory
NOCUDA_SRC_DIR:= src/nocuda

#ArnoldiSolver source directory
ARNOLDI_SRC_DIR = src/Solvers/ArnoldiSolver
ARNOLDI_SRC_DIR += src/PropertyExtractors/APropertyExtractor

#FourierTransform source directory
FOURIER_TRANSFORM_SRC_DIR = src/Utilities/FourierTransform

#RayTracer source directory
RAY_TRACER_SRC_DIR = src/Utilities/RayTracer

#Plotter source directory
PLOTTER_SRC_DIR = src/Utilities/Plotter

#Resource source directory
RESOURCE_SRC_DIR = src/Utilities/Resource

#Canvas source directory
CANVAS_SRC_DIR = src/Utilities/Canvas

# Directory for Objects
OBJ_DIR:= build


#---------------------No need to change the following!-------------------

# Include files are needed fot Includes
INC_FILES :=  $(wildcard $(INC_DIR)/*.h))

# All include directories are needed for Includes
INC_DIRS:= $(dir $(INC_FILES))

# Includes: -I ...
LDLIBS:=$(addprefix -I, $(INC_DIRS))

# Source files
#SRC :=  $(wildcard $(SRC_DIR)/*.cpp)
SRC :=  $(foreach srcDir, $(SRC_DIR), $(wildcard $(srcDir)/*.cpp))

# CUDA source files
CUDA_SRC := $(wildcard $(CUDA_SRC_DIR)/*.cu)

# No CUDA source files
NOCUDA_SRC := $(wildcard $(NOCUDA_SRC_DIR)/*.cpp)

# ArnoldiSolver source files
ARNOLDI_SRC := $(foreach srcDir, $(ARNOLDI_SRC_DIR), $(wildcard $(srcDir)/*.cpp))

# FourierTransform source files
FOURIER_TRANSFORM_SRC := $(foreach srcDir, $(FOURIER_TRANSFORM_SRC_DIR), $(wildcard $(srcDir)/*.cpp))

# RayTracer source files
RAY_TRACER_SRC := $(foreach srcDir, $(RAY_TRACER_SRC_DIR), $(wildcard $(srcDir)/*.cpp))

# Plotter source files
PLOTTER_SRC := $(foreach srcDir, $(PLOTTER_SRC_DIR), $(wildcard $(srcDir)/*.cpp))

# Resource source files
RESOURCE_SRC := $(foreach srcDir, $(RESOURCE_SRC_DIR), $(wildcard $(srcDir)/*.cpp))

# Canvas source files
CANVAS_SRC := $(foreach srcDir, $(CANVAS_SRC_DIR), $(wildcard $(srcDir)/*.cpp))

# Object files
OBJ := $(addprefix $(OBJ_DIR)/, $(notdir $(SRC:.cpp=.o)))

# CUDA object files
CUDA_OBJ := $(addprefix $(OBJ_DIR)/, $(addprefix cuda, $(notdir $(CUDA_SRC:.cu=.o))))

# CUDA object files
NOCUDA_OBJ := $(addprefix $(OBJ_DIR)/, $(addprefix nocuda, $(notdir $(NOCUDA_SRC:.cpp=.o))))

# ArnoldiSolver object files
ARNOLDI_OBJ := $(addprefix $(OBJ_DIR)/, $(notdir $(ARNOLDI_SRC:.cpp=.o)))

# FourierTransform object files
FOURIER_TRANSFORM_OBJ := $(addprefix $(OBJ_DIR)/, $(notdir $(FOURIER_TRANSFORM_SRC:.cpp=.o)))

# RayTracer object files
RAY_TRACER_OBJ := $(addprefix $(OBJ_DIR)/, $(notdir $(RAY_TRACER_SRC:.cpp=.o)))

# Plotter object files
PLOTTER_OBJ := $(addprefix $(OBJ_DIR)/, $(notdir $(PLOTTER_SRC:.cpp=.o)))

# Resource object files
RESOURCE_OBJ := $(addprefix $(OBJ_DIR)/, $(notdir $(RESOURCE_SRC:.cpp=.o)))

# Canvas object files
CANVAS_OBJ := $(addprefix $(OBJ_DIR)/, $(notdir $(CANVAS_SRC:.cpp=.o)))

#All object files
ALL_OBJ := $(OBJ) $(CUDA_OBJ)

# File names of objects without directory name
OBJ_PURE := $(notdir $(OBJ))

# CUDA file names of objects without directory name
CUDA_OBJ_PURE := $(notdir $(CUDA_OBJ))

# CUDA file names of objects without directory name
NOCUDA_OBJ_PURE := $(notdir $(NOCUDA_OBJ))

# ArnoldiSolver file name of object without directory name
ARNOLDI_OBJ_PURE := $(notdir $(ARNOLDI_OBJ))

# FourierTransform file name of object without directory name
FOURIER_TRANSFORM_OBJ_PURE := $(notdir $(FOURIER_TRANSFORM_OBJ))

# RayTracer file name of objects without directory name
RAY_TRACER_OBJ_PURE := $(notdir $(RAY_TRACER_OBJ))

# Plotter file name of objects without directory name
PLOTTER_OBJ_PURE := $(notdir $(PLOTTER_OBJ))

# Resource file name of objects without directory name
RESOURCE_OBJ_PURE := $(notdir $(RESOURCE_OBJ))

# Canvas file name of objects without directory name
CANVAS_OBJ_PURE := $(notdir $(CANVAS_OBJ))

all: $(STATIC_LIB)

# Linking
$(STATIC_LIB): $(OBJ) $(CUDA_OBJ)
	@echo "Linking: " $(notdir $(STATIC_LIB))
	@ar rcs $(STATIC_LIB) $(OBJ) $(CUDA_OBJ)

cuda: $(OBJ) $(CUDA_OBJ)
	@echo "Linking: " $(notdir $(STATIC_LIB))
	@ar rcs $(STATIC_LIB) $(OBJ) $(CUDA_OBJ)

nocuda: $(OBJ) $(NOCUDA_OBJ)
	@echo "Linking: " $(notdir $(STATIC_LIB))
	@ar rcs $(STATIC_LIB) $(OBJ) $(NOCUDA_OBJ)

arnoldi: $(ARNOLDI_OBJ)
	@echo "Linking: " $(notdir $(STATIC_LIB))
	@echo $(ARNOLDI_OBJ)
	@ar rcs $(STATIC_LIB) $(ARNOLDI_OBJ)

fourier: $(FOURIER_TRANSFORM_OBJ)
	@echo "Linking: " $(notdir $(STATIC_LIB))
	@echo $(FOURIER_TRANSFORM_OBJ)
	@ar rcs $(STATIC_LIB) $(FOURIER_TRANSFORM_OBJ)

raytracer: $(RAY_TRACER_OBJ)
	@echo "Linking: " $(notdir $(STATIC_LIB))
	@echo $(RAY_TRACER_OBJ)
	@ar rcs $(STATIC_LIB) $(RAY_TRACER_OBJ)

plotter: $(PLOTTER_OBJ)
	@echo "Linking: " $(notdir $(STATIC_LIB))
	@echo $(PLOTTER_OBJ)
	@ar rcs $(STATIC_LIB) $(PLOTTER_OBJ)

resource: $(RESOURCE_OBJ)
	@echo "Linking: " $(notdir $(STATIC_LIB))
	@echo $(RESOURCE_OBJ)
	@ar rcs $(STATIC_LIB) $(RESOURCE_OBJ)

canvas: $(CANVAS_OBJ)
	@echo "Linking: " $(notdir $(STATIC_LIB))
	@echo $(CANVAS_OBJ)
	@ar rcs $(STATIC_LIB) $(CANVAS_OBJ)

# Compilation rule
define app_compile_template
 $(1)_OBJ = $$(addprefix $$(OBJ_DIR)/, $$(notdir $$(patsubst %.cpp, %.o, $(1))))

$$($(1)_OBJ): $(1)
	@echo "Compiling: $(1)"
	@$$(CC) $$(CFLAGS) $$(OPT) -c $(1) -o $$($(1)_OBJ) $$(LDLIBS)
endef

# CUDA compilation rule
define cuda_app_compile_template
 $(1)_OBJ = $$(addprefix $$(OBJ_DIR)/, $$(addprefix cuda, $$(notdir $$(patsubst %.cu, %.o, $(1)))))

$$($(1)_OBJ): $(1)
	@echo "Compiling: $(1)"
	@$$(NVCC) $$(NVCCFLAGS) $$(NVCCOPT) -c $(1) -o $$($(1)_OBJ) $$(LDLIBS)
endef

# No CUDA compilation rule
define nocuda_app_compile_template
 $(1)_OBJ = $$(addprefix $$(OBJ_DIR)/, $$(addprefix nocuda, $$(notdir $$(patsubst %.cpp, %.o, $(1)))))

$$($(1)_OBJ): $(1)
	@echo "Compiling: $(1)"
	@$$(CC) $$(CFLAGS) $$(OPT) -c $(1) -o $$($(1)_OBJ) $$(LDLIBS)
endef

# ArnoldiSolver compilation rule
define arnoldi_app_compile_template
 $(1)_OBJ = $$(addprefix $$(OBJ_DIR)/, $$(notdir $$(patsubst %.cpp, %.o, $(1))))

$$($(1)_OBJ): $(1)
	@echo "Compiling: $(1)"
	@$$(CC) $$(CFLAGS) $$(OPT) -c $(1) -o $$($(1)_OBJ) $$(LDLIBS)
endef

# FourierTransform compilation rule
define fourier_transform_app_compile_template
 $(1)_OBJ = $$(addprefix $$(OBJ_DIR)/, $$(notdir $$(patsubst %.cpp, %.o, $(1))))

$$($(1)_OBJ): $(1)
	@echo "Compiling: $(1)"
	@$$(CC) $$(CFLAGS) $$(OPT) -c $(1) -o $$($(1)_OBJ) $$(LDLIBS)
endef

# RayTracer compilation rule
define ray_tracer_app_compile_template
 $(1)_OBJ = $$(addprefix $$(OBJ_DIR)/, $$(notdir $$(patsubst %.cpp, %.o, $(1))))

$$($(1)_OBJ): $(1)
	@echo "Compiling: $(1)"
	@$$(CC) $$(CFLAGS) $$(OPT) -c $(1) -o $$($(1)_OBJ) $$(LDLIBS)
endef

# Plotter compilation rule
define plotter_app_compile_template
 $(1)_OBJ = $$(addprefix $$(OBJ_DIR)/, $$(notdir $$(patsubst %.cpp, %.o, $(1))))

$$($(1)_OBJ): $(1)
	@echo "Compiling: $(1)"
	@$$(CC) $$(CFLAGS) $$(OPT) -c $(1) -o $$($(1)_OBJ) $$(LDLIBS)
endef

# Resource compilation rule
define resource_app_compile_template
 $(1)_OBJ = $$(addprefix $$(OBJ_DIR)/, $$(notdir $$(patsubst %.cpp, %.o, $(1))))

$$($(1)_OBJ): $(1)
	@echo "Compiling: $(1)"
	@$$(CC) $$(CFLAGS) $$(OPT) -c $(1) -o $$($(1)_OBJ) $$(LDLIBS)
endef

# Canvas compilation rule
define canvas_app_compile_template
 $(1)_OBJ = $$(addprefix $$(OBJ_DIR)/, $$(notdir $$(patsubst %.cpp, %.o, $(1))))

$$($(1)_OBJ): $(1)
	@echo "Compiling: $(1)"
	@$$(CC) $$(CFLAGS) $$(OPT) -c $(1) -o $$($(1)_OBJ) $$(LDLIBS)
endef

# Compile
$(foreach app, $(SRC), $(eval $(call app_compile_template,$(app))))

# CUDA compile
$(foreach app, $(CUDA_SRC), $(eval $(call cuda_app_compile_template,$(app))))

# No CUDA compile
$(foreach app, $(NOCUDA_SRC), $(eval $(call nocuda_app_compile_template,$(app))))

# ArnoldiSolver compile
$(foreach app, $(ARNOLDI_SRC), $(eval $(call arnoldi_app_compile_template,$(app))))

# FourierTransform compile
$(foreach app, $(FOURIER_TRANSFORM_SRC), $(eval $(call fourier_transform_app_compile_template,$(app))))

# RayTracer compile
$(foreach app, $(RAY_TRACER_SRC), $(eval $(call ray_tracer_app_compile_template,$(app))))

# Plotter compile
$(foreach app, $(PLOTTER_SRC), $(eval $(call plotter_app_compile_template,$(app))))

# Resource compile
$(foreach app, $(RESOURCE_SRC), $(eval $(call resource_app_compile_template,$(app))))

# Canvas compile
$(foreach app, $(CANVAS_SRC), $(eval $(call canvas_app_compile_template,$(app))))

# Cleaning
clean:
	@echo "Cleaning: build/"
	@rm -r build/*

list_of_objects:
	@echo $(OBJ)

# For debugging makefile
print:
	@echo $(SRC_DIR)
	@echo $(SRC)
	@echo $(CUDA_SRC)
	@echo $(LDLIBS)
	@echo INC_DIR
	@echo $(ARNOLDI_SRC_DIR)
	@echo $(ARNOLDI_SRC)
	@echo $(ARNOLDI_OBJ)
	@echo $(FOURIER_TRANSFORM_SRC_DIR)
	@echo $(FOURIER_TRANSFORM_SRC)
	@echo $(FOURIER_TRANSFORM_OBJ)
	@echo $(RAY_TRACER_SRC)
	@echo $(RAY_TRACER_OBJ)
	@echo $(PLOTTER_SRC_DIR)
	@echo $(PLOTTER_SRC)
	@echo $(PLOTTER_OBJ)
	@echo $(RESOURCE_SRC_DIR)
	@echo $(RESOURCE_SRC)
	@echo $(RESOURCE_OBJ)
	@echo $(CANVAS_SRC_DIR)
	@echo $(CANVAS_SRC)
	@echo $(CANVAS_OBJ)
#	@echo $(OBJ)
#	@echo $(addprefix ./build/, $(notdir $(patsubst %.cpp, %.o, $(SRC))))

## @endconf

