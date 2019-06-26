ifndef XILINX_SDX
$(error XILINX_SDX is not set)
endif

ifndef AWS_PLATFORM
$(error AWS_PLATFORM is not set)
endif

# Host compiler global settings
CC = g++ -O3 -Wno-deprecated-declarations

CLCC = xocc

BITSTREAM_NAME = Gradients
HOST_EXE = ${BITSTREAM_NAME}

PLATFORM = ${AWS_PLATFORM}

HOST_DIR = host_srcs
KERNEL_DIR = kernel_srcs
KERNEL_TYPE = cpp

# Host and Kernel sources
HOST_SRCS = $(wildcard $(HOST_DIR)/*/*.cpp) $(wildcard $(HOST_DIR)/*.cpp)
KERNEL_SRCS_CPP = $(wildcard $(KERNEL_DIR)/*.cpp)

HOST_OBJECTS := $(HOST_SRCS:.cpp=.o)
KERNEL_OBJECTS := $(KERNEL_SRCS_CPP:.cpp=.xo)
ESTIMATE_OBJCTS := $(KERNEL_SRCS_CPP:.cpp=.estimate)

# Include Libraries
HOST_CFLAGS = -O3 -Wall -I${XILINX_SDX}/runtime/include/1_2 -Ihost_srcs
HOST_LFLAGS = -L${XILINX_XRT}/lib -lxilinxopencl

# Connecting kernels to specific memory banks
BANKS = --sp Gradients_0_1.m_axi_gmem0:bank0 --sp Gradients_0_1.m_axi_gmem1:bank0 --sp Gradients_0_1.m_axi_gmem2:bank0 --sp Gradients_0_1.m_axi_gmem3:bank0 --sp Gradients_1_1.m_axi_gmem0:bank1 --sp Gradients_1_1.m_axi_gmem1:bank1 --sp Gradients_1_1.m_axi_gmem2:bank1 --sp Gradients_1_1.m_axi_gmem3:bank1 --sp Gradients_2_1.m_axi_gmem0:bank2 --sp Gradients_2_1.m_axi_gmem1:bank2 --sp Gradients_2_1.m_axi_gmem2:bank2 --sp Gradients_2_1.m_axi_gmem3:bank2 --sp Gradients_3_1.m_axi_gmem0:bank3 --sp Gradients_3_1.m_axi_gmem1:bank3 --sp Gradients_3_1.m_axi_gmem2:bank3 --sp Gradients_3_1.m_axi_gmem3:bank3

# Additional Vivado options
VIVADO_OPTS = --xp misc:enableGlobalHoldIter="True" --xp vivado_prop:run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=NoTimingRelaxation

SDA_FLOW = sw_emu
ifeq (${SDA_FLOW},sw_emu)
	TARGET = -t sw_emu
else ifeq (${SDA_FLOW},hw_emu)
	TARGET = -t hw_emu
else ifeq (${SDA_FLOW},hw)
	TARGET = -t hw
endif

all:
	make _TEST_="-D _TEST_" host

host: ${HOST_EXE}

xbin_sw_em:
	@+make SDA_FLOW=sw_emu xbin

xbin_hw_em:
	@+make SDA_FLOW=hw_emu xbin

xbin_hw :
	@+make SDA_FLOW=hw xbin

run_sw_em:
	@+make SDA_FLOW=sw_emu run_sem

run_hw_em:
	@+make SDA_FLOW=hw_emu run_hem

run_sem: xconfig host xbin
	XCL_EMULATION_MODE=sw_emu ./${HOST_EXE} 1

run_hem: xconfig host xbin
	XCL_EMULATION_MODE=hw_emu ./${HOST_EXE} 1

xconfig:
	emconfigutil --platform ${PLATFORM} --od . --nd 1

# Building host
${HOST_EXE}: ${HOST_OBJECTS}
	${CC} ${HOST_OBJECTS} ${HOST_LFLAGS} -o $@
	${RM} -rf ${HOST_OBJECTS}

xbin: ${KERNEL_OBJECTS}
	${CLCC} ${TARGET} --link -s --platform ${PLATFORM} ${VIVADO_OPTS} ${BANKS} ${KERNEL_OBJECTS} -o ${BITSTREAM_NAME}.xclbin
	${RM} -rf ${KERNEL_OBJECTS}

estimate: ${ESTIMATE_OBJCTS}
	${RM} -rf $(patsubst %.estimate,%.xo,$(ESTIMATE_OBJCTS))

%.o: %.cpp
	${CC} ${_TEST_} ${HOST_CFLAGS} -c $< -o $@

# Building kernel
%.xo: %.cpp
	${CLCC} ${TARGET} --save-temps --platform ${PLATFORM} --kernel $(notdir $(basename $<)) -c $< -o $@

%.estimate: %.${KERNEL_TYPE}
	${CLCC} --target hw_emu --report_level estimate --save-temps --platform ${PLATFORM} --kernel $(notdir $(basename $<)) -c $< -o $(basename $<).xo

clean:
	${RM} -rf ${HOST_EXE} $(patsubst %.estimate,%.xo,$(ESTIMATE_OBJCTS)) ${KERNEL_OBJECTS} ${HOST_OBJECTS} emconfig.json *.log *.dir *.xml *.dcp *.dat _sds iprepo *.tcl xilinx_aws-vu9p-f1_dynamic_5_0.hpfm .Xil sdaccel_* system_estimate.xtxt _x top_sp.ltx

cleanall: clean
	${RM} -rf ${BITSTREAM_NAME}*

help:
	@echo "Compile and run CPU emulation"
	@echo "make run_sw_em"
	@echo ""
	@echo "Compile and run hardware emulation"
	@echo "make run_hw_em"
	@echo ""
	@echo "Compile host executable only"
	@echo "make host"
	@echo ""
	@echo "Compile host executable only for SW version"
	@echo "make"
	@echo ""
	@echo "Compile .xclbin file for system run only"
	@echo "make xbin_hw"
	@echo ""
	@echo "Compile .xclbin file for sw emulation"
	@echo "make xbin_sw_em"
	@echo ""
	@echo "Compile .xclbin file for hw emulation"
	@echo "make xbin_hw_em"
	@echo ""
	@echo "Clean working diretory"
	@echo "make clean"
	@echo "Clean working diretory and bitstream files"
	@echo "make cleanall"
