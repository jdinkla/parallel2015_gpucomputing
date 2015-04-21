# (c) 2015 Joern Dinkla, www.dinkla.com
#

CC = clang++
INCLUDES =  -I../opencl -I../cuda -I../gpu_basics
CCFLAGS = -std=c++11 -stdlib=libc++
LDFLAGS = -std=c++11 -stdlib=libc++ -framework OpenCL -lgpu_basics -L../gpu_basics

CPP = OpenCLMac.cpp ../opencl/OpenCLUtilities.cpp ../opencl/SimpleTest.cpp ../opencl/opencl_beispiel.cpp ../opencl/HeatDemoOpenCL.cpp ../opencl/SimpleBenchmark.cpp ../opencl/DemoOpenCL.cpp
CPP_OS = $(patsubst %.cpp, %.o, $(CPP))

CPP_OS2 = $(notdir $(CPP_OS))

%.o : %.cpp 
	$(EXEC) $(CC) $(INCLUDES) $(CCFLAGS) -o $(notdir $@) -c $<

./%.o : ../cuda/%.cpp 
	$(EXEC) $(CC) $(INCLUDES) $(CCFLAGS) -o $(notdir $@) -c $<

./%.o : ../opencl/%.cpp 
	$(EXEC) $(CC) $(INCLUDES) $(CCFLAGS) -o $(notdir $@) -c $<

all:	opencl_mac

opencl_mac: $(CPP_OS2)
	$(EXEC) $(CC) $(INCLUDES) $(LDFLAGS) $(CPP_OS2) -o $@

clean:
	rm -f opencl_mac *.o
