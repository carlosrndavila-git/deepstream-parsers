################################################################################
# LPR custom DeepStream bbox parser
# Handles single-class YOLO output: [1, 5, N] (x1, y1, x2, y2, score)
#
# Usage (on Jetson, find CUDA version with: nvcc --version):
#   make CUDA_VER=11.4
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
	$(error "CUDA_VER is not set. Usage: make CUDA_VER=11.4")
endif

CC := g++

CFLAGS := -Wall -std=c++14 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS += -I/opt/nvidia/deepstream/deepstream/sources/includes
CFLAGS += -I/usr/local/cuda-$(CUDA_VER)/include

LIBS    := -lnvinfer
LFLAGS  := -shared -Wl,--start-group $(LIBS) -Wl,--end-group

TARGET_LIB := libnvdsinfer_custom_impl_Yolo_LPR.so

all: $(TARGET_LIB)

$(TARGET_LIB): nvdsparsebbox_LPR.cpp
	$(CC) -o $@ $(CFLAGS) $< $(LFLAGS)

clean:
	rm -f $(TARGET_LIB) *.o
