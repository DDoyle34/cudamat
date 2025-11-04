# Build libcudamat.so from cudamat.cu (not .c) using nvcc

NVCC := nvcc
NVCCFLAGS := -Xcompiler -fPIC -O3
INCLUDES := -Iinclude

TARGET := libcudamat.so
SRC := src/cudamat.cu
OBJ := $(SRC:.cu=.o)

$(TARGET): $(OBJ)
	$(NVCC) -shared -o $@ $^ $(NVCCFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

