# simple cpp Makefile
CXX      := g++
CXXFLAGS := -std=c++11 -Wall -Wextra 
LDFLAGS  := 
TARGET   := resnet_cpp
PKGS:= gstreamer-1.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

SRCS:= $(wildcard *.cpp)
INCS:= $(wildcard *.h)

CFLAGS+= -I /opt/nvidia/deepstream/deepstream/sources/includes/ \
         -I /usr/local/cuda-12.2/include \
	 	 -I /usr/local/include/opencv4/ \
		 -DWITH_OPENCV

CFLAGS+= $(shell pkg-config --cflags $(PKGS))
LIBS:= $(shell pkg-config --libs $(PKGS))

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvds_inferutils -lnvds_utils -lm \
       -L/usr/local/cuda-12.2/lib64/ -lcudart \
       -lcuda -Wl,-rpath,$(LIB_INSTALL_DIR) \
	   -lopencv_objdetect -lopencv_imgproc -lopencv_core \
	   -L$(LIB_INSTALL_DIR) -lnvds_infercustomparser

OBJS:= $(SRCS:.cpp=.o)

all: $(TARGET)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(TARGET): $(OBJS) Makefile
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

clean:
	rm -rf $(OBJS) $(TARGET)