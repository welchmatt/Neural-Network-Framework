#-------------------------------------------------------------------------------
# makefile for xor and mnist neural network tests;
#
# to implement the nueral network framework with your own code, see the template
# sections marked and commented out below
#-------------------------------------------------------------------------------
# Matt Welch
#-------------------------------------------------------------------------------

FC = gfortran
FFLAGS = -O3

# uncomment FFLAGS below for debugging (overrides FFLAGS above).
#
# only warning is 'Equality comparison for REAL(4)' for comparing
# this%output%d == 0 in conv_neural_net.f08:
# this%output%d is explicitly set to 0 for this check, so there is no unexpected
# behavior here (since the program does not check equality with arbitrary reals)
#
# outside of optimization flags O1, O2, O3, Wall and Wextra may complain that
# an array "may be used uninitialized in this function"; the usage is actually
# correct based on the Fortran standard, which allows allocatable
# arrays to be set without being explicitly allocated first (in which case,
# they are implicitly allocated). the compiler complains in such cases, but the
# output is unaffected (because it abides by the Fortran standard):

# FFLAGS = -Wall -Wextra -fsanitize=undefined -fsanitize=address -O3

SDIR = src
ODIR = obj
MDIR = mod

# must specify dependency order for Fortran
BASE_DEPS = net_helper_procedures.f08 \
			pool_layer_definitions.f08 \
			dense_layer_definitions.f08 conv_layer_definitions.f08 \
			dense_neural_net.f08 conv_neural_net.f08 \
			sequential_neural_net.f08

# xor test
XOR_DEPS = $(BASE_DEPS) test_xor.f08
XOR_SRC = $(addprefix src/, $(XOR_DEPS))
XOR_OBJ = $(patsubst $(SDIR)/%.f08, $(ODIR)/%.o, $(XOR_SRC))
XOR_MOD = $(patsubst $(SDIR)/%.f08, $(ODIR)/%.mod, $(XOR_SRC))

# mnist test
MNIST_DEPS = $(BASE_DEPS) test_mnist.f08
MNIST_SRC = $(addprefix src/, $(MNIST_DEPS))
MNIST_OBJ = $(patsubst $(SDIR)/%.f08, $(ODIR)/%.o, $(MNIST_SRC))
MNIST_MOD = $(patsubst $(SDIR)/%.f08, $(ODIR)/%.mod, $(MNIST_SRC))

# autoencoder test
AUTOENC_DEPS = $(BASE_DEPS) test_autoenc.f08
AUTOENC_SRC = $(addprefix src/, $(AUTOENC_DEPS))
AUTOENC_OBJ = $(patsubst $(SDIR)/%.f08, $(ODIR)/%.o, $(AUTOENC_SRC))
AUTOENC_MOD = $(patsubst $(SDIR)/%.f08, $(ODIR)/%.mod, $(AUTOENC_SRC))

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
#
# see below for next step in the implementation
#-------------------------------------------------------------------------------

default:
	@echo "----------------------"
	@echo "for xor test:   compile with 'make xor',   run with './xor'."
	@echo "for mnist test: compile with 'make mnist', run with './mnist'."
	@echo
	@echo "remove all compiled files and executables with 'make allclean'."
	@echo "remove all compiled object and module files with 'make clean'."
	@echo "----------------------"

xor: $(XOR_OBJ)
	$(FC) $(FFLAGS) -o $@ $^

mnist: $(MNIST_OBJ)
	$(FC) $(FFLAGS) -o $@ $^

autoenc: $(AUTOENC_OBJ)
	$(FC) $(FFLAGS) -o $@ $^

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

.PHONY: default xor mnist clean allclean