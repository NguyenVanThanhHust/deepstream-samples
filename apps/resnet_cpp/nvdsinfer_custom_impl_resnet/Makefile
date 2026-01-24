CC:= g++

CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes/ -I/usr/local/cuda/include -I/usr/local/include/opencv4/ -I/usr/lib/gcc/x86_64-linux-gnu/11/include/ 
CXXFLAGS:= -fopenmp

LIBS:= -lnvinfer -lnvparsers
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= resnet_output_parser.cpp

TARGET_LIB:= libnvds_infercustomparser.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)