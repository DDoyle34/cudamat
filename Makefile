# Compiler and flags
NVCC := nvcc
NVCCFLAGS := -Xcompiler -fPIC -O3 -g
INCLUDES := -Iinclude
LIBS := -lcublas

# Targets
TARGET := libcudamat.so
OBJ := cudamat.o

# Default target
all: $(TARGET)

# Build shared library and link with cuBLAS
$(TARGET): $(OBJ)
	$(NVCC) -shared -o $@ $^ $(NVCCFLAGS) $(LIBS)

# Compile CUDA source
cudamat.o: src/cudamat.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Example: build add.cu linking against libcudamat
add: examples/add.cu $(TARGET)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -L. -lcudamat -lcublas -Xlinker -rpath \
    -Xlinker . -lcurand $< -o $@

# Clean up
clean:
	rm -f $(OBJ) $(TARGET) add

