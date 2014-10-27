################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/DataFrame.cpp \
../src/IOHelper.cpp \
../src/Node.cpp \
../src/RandomForest.cpp \
../src/Tree.cpp \
../src/smuRF.cpp 

OBJS += \
./src/DataFrame.o \
./src/IOHelper.o \
./src/Node.o \
./src/RandomForest.o \
./src/Tree.o \
./src/smuRF.o 

CPP_DEPS += \
./src/DataFrame.d \
./src/IOHelper.d \
./src/Node.d \
./src/RandomForest.d \
./src/Tree.d \
./src/smuRF.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/home/loschen/programs/Eigen3_2 -O3 -floop-optimize  -funroll-loops -march=native -fomit-frame-pointer -Wall -c -fmessage-length=0 -std=c++0x -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


