#-------------------------------------------------------------------------------
# makefile for neural network framework
#
# sections marked "TEMPLATE" describe how to implement the nueral network
# framework in your own code
#-------------------------------------------------------------------------------
# Matt Welch
#-------------------------------------------------------------------------------

FC = gfortran
FFLAGS = -O3

# uncomment FFLAGS below for debugging (overrides FFLAGS above)
# FFLAGS = -Wall -Wextra -fsanitize=undefined -fsanitize=address -O3

# uncomment FFLAGS below for profiling (overrides FFLAGS above); after running:
# gprof EXECUTABLE_NAME gmon.out > profile.txt
# FFLAGS = -Wall -Wextra -pg -O3

# uncomment to ignore warning from comparing reals, which occurs in BLAS;
# be sure to first run your code with this commented, to ensure your non-BLAS
# code does not have this warning
# FFLAGS += -Wno-compare-reals

# directory structure; put all source files in src
SDIR = src
ODIR = obj
MDIR = mod
BDIR = BLAS

# framework source files; must specify dependency order in Fortran
BASE_DEPS = net_helper_procedures.f08 \
			pool_layer_definitions.f08 \
			dense_layer_definitions.f08 conv_layer_definitions.f08 \
			dense_neural_net.f08 conv_neural_net.f08 \
			sequential_neural_net.f08

# build blas objects
BLAS_DEPS = lsame.f xerbla.f dgemm.f
BLAS_SRC = $(addprefix $(BDIR)/, $(BLAS_DEPS))
BLAS_OBJ = $(patsubst $(BDIR)/%.f08, $(ODIR)/%.o, $(BLAS_SRC))
BLAS_MOD = $(patsubst $(BDIR)/%.f08, $(ODIR)/%.mod, $(BLAS_SRC))

#-------------------------------------------------------------------------------
# TEMPLATE: for implementing the framework in your own code;
# simply substitute the assigned "YOUR_" variables to the names your want,
# with your primary source that uses the framework in place of YOUR_FILE.f08.
# you should include your source file in the src folder, along with all the
# other source files for the framework:
#
# YOUR_DEPS = $(BASE_DEPS) YOUR_FILE.f08
# YOUR_SRC = $(addprefix src/, $(YOUR_DEPS))
# YOUR_OBJ = $(patsubst $(SDIR)/%.f08, $(ODIR)/%.o, $(YOUR_SRC))
# YOUR_MOD = $(patsubst $(SDIR)/%.f08, $(ODIR)/%.mod, $(YOUR_SRC))
# YOUR_OBJ += $(BLAS_OBJ)
#
# see below for next step in implementation
#-------------------------------------------------------------------------------

# xor test
XOR_DEPS = $(BASE_DEPS) test_xor.f08
XOR_SRC = $(addprefix $(SDIR)/, $(XOR_DEPS))
XOR_OBJ = $(patsubst $(SDIR)/%.f08, $(ODIR)/%.o, $(XOR_SRC))
XOR_MOD = $(patsubst $(SDIR)/%.f08, $(ODIR)/%.mod, $(XOR_SRC))
XOR_OBJ += $(BLAS_OBJ)

# mnist test
MNIST_DEPS = $(BASE_DEPS) test_mnist.f08
MNIST_SRC = $(addprefix $(SDIR)/, $(MNIST_DEPS))
MNIST_OBJ = $(patsubst $(SDIR)/%.f08, $(ODIR)/%.o, $(MNIST_SRC))
MNIST_MOD = $(patsubst $(SDIR)/%.f08, $(ODIR)/%.mod, $(MNIST_SRC))
MNIST_OBJ += $(BLAS_OBJ)

# autoencoder test
AUTOENC_DEPS = $(BASE_DEPS) test_autoenc.f08
AUTOENC_SRC = $(addprefix $(SDIR)/, $(AUTOENC_DEPS))
AUTOENC_OBJ = $(patsubst $(SDIR)/%.f08, $(ODIR)/%.o, $(AUTOENC_SRC))
AUTOENC_MOD = $(patsubst $(SDIR)/%.f08, $(ODIR)/%.mod, $(AUTOENC_SRC))
AUTOENC_OBJ += $(BLAS_OBJ)

#-------------------------------------------------------------------------------
# TEMPLATE: add usage statement here
#
# see below for next step in implementation
#-------------------------------------------------------------------------------

default:
	@echo "----------------------"
	@echo "for xor test:   compile with 'make xor',   run with './xor'."
	@echo "for mnist test: compile with 'make mnist', run with './mnist'."
	@echo "for autoenc test: compile with 'make autoenc', run with './autoenc'."
	@echo
	@echo "remove all compiled files and executables with 'make allclean'."
	@echo "remove all compiled object and module files with 'make clean'."
	@echo "----------------------"

#-------------------------------------------------------------------------------
# TEMPLATE: define how you want to compile your executable with this structure:
#
# YOUR_EXECUTABLE: $(YOUR_OBJ)
# 	$(FC) $(FFLAGS) -o $@ $^
#
# then, compile and run as:
#
# compile: make YOUR_EXECUTABLE
# run:	   ./YOUR_EXECUTABLE
#
# see below for final step in the implementation
#-------------------------------------------------------------------------------

xor: $(XOR_OBJ)
	$(FC) $(FFLAGS) -o $@ $^

mnist: $(MNIST_OBJ)
	$(FC) $(FFLAGS) -o $@ $^

autoenc: $(AUTOENC_OBJ)
	$(FC) $(FFLAGS) -o $@ $^

# create object files; -J specifies directory for mod files
$(ODIR)/%.o: $(SDIR)/%.f08
	$(FC) $(FFLAGS) -c -o $@ $< -J $(MDIR)

clean:
	-rm -f $(MDIR)/*.mod
	-rm -f $(ODIR)/*.o

#-------------------------------------------------------------------------------
# TEMPLATE: define how to delete your file in allclean, as
#
# -rm -f YOUR_EXECUTABLE
#-------------------------------------------------------------------------------

allclean:
	-rm -f $(MDIR)/*.mod
	-rm -f $(ODIR)/*.o
	-rm -f xor
	-rm -f mnist
	-rm -f autoenc

.PHONY: default xor mnist autoenc clean allclean
