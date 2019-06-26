<a href="https://www.inaccel.com/">
<p align="center">
<img src="https://www.inaccel.com/wp-content/uploads/logo-horizontal1200px.png" width=60% height=60% align="middle" alt="InAccel"/>
</p>
</a>

# Logistic Regression


This is an FPGA accelerated solution for Logistic Regression BGD algorithm. It can provide up to **70x** speedup compared to a single threaded execution and up to **12x** compared to an 8 threaded Intel Xeon CPU execution respectively.

## Specifications

|  Classes |  Features  |
| :------: | :--------: |
| up to 64 | up to 2047 |

## Supported Platforms

|            Board            |
| :-------------------------: |
|      Xilinx Alveo U200      |
|      Xilinx Alveo U250      |
|   AWS VU9P (F1 instances)   |
| Alibaba VU9P (F3 instances) |
| Any other Xilinx platform with at least the same amount of VU9P resources |

## Design Files

-   The application code is located in the hosts_srcs directory. Accelerator kernel files are located under the kernel_srcs directory while any accelerator binaries will be compiled to the current directory.
-   The Makefile will help you generate any host executable and accelerator _.xclbin_ files.

A listing of all the files in this repository is shown below:

    - Makefile
    - hosts_srcs/
    	- LogisticRegression.cpp
    	- common/
    		- INcl.cpp (OpenCL wrapper functions)
    		- INcl.h
    	- inaccel/
    		- runtime-api.cpp (InAccel runtime abstraction layer)
    		- runtime-api.h
    		- runtime.cpp (InAccel runtime abstraction layer)
    		- runtime.h
    - kernel_srcs/
    	- Gradients_0.cpp (Accelerated kernel)
    	- Gradients_1.cpp (Accelerated kernel)
    	- Gradients_2.cpp (Accelerated kernel)
    	- Gradients_3.cpp (Accelerated kernel)
    - data/

## Preparation

**!** Before invoking any of the Makefile targets make sure you have sourced Xilinx **XRT** setup script.  
**!** Make sure you have set **XILINX_SDX** environment variable pointing to the SDx installation directory.

As far as the **platform** (or board) is concerned, Makefile uses **AWS_PLATFORM** environment variable as the target platform for the kernels compilation. If you are running this on AWS make sure AWS_PLATFORM environment variable is present and points to the platform DSA files<sup>1</sup>. Otherwise you can set Makefile `PLATFORM` variable to point to your platform DSA files.

1.  To obtain the AWS platform DSA files make sure you have cloned the aws-fpga github repository

Download train letters train dataset to data directory. Navigate to data directory and execute the following commands:

``` bash
	wget https://s3.amazonaws.com/inaccel-demo/data/nist/letters_csv_train.dat
	wget https://s3.amazonaws.com/inaccel-demo/data/nist/letters_csv_test.dat
```

## Compiling the kernels

To compile the kernels for hardware target you just need to execute `make xbin_hw` while for software and hardware emulation you must execute  `make xbin_sw` and `make xbin_hw` respectively.  
A full list of all the available Makefile targets can be found using `make help` command.

## Single-thread - Single-application Execution

To test the generated xclbin file you can simply run `make host` command to create the host application. The host application takes only one input argument, the number of iterations.  
Example execution: `./Gradients 100`

## Scaling Up and Out with InAccel Coral

<p align="center">
<img src="https://www.inaccel.com/wp-content/uploads/coral_logo_big-1-e1561553344239.png" width=60% height=60% align="middle" alt="InAccel Coral"/>
</p>

The above example application spawns a single thread and can train a model using a single FPGA device which **is not viable for datacenter-scale needs**. Data scientists rely on frameworks like Scikit Learn and Apache Spark to create and test their machine learning pipelines.  
**InAccel Coral** FPGA resource manager is able to automatically **scale** and **schedule** any acceleration requests to a **cluster of FPGAs**, perform **load balancing** techniques, **reconfigure** the FPGA devices, perform **memory management** etc., yet providing a simple to use **high level API** in Java, CPP and Python.  
We have also ready-to-use **integrations** with broadly used open source frameworks like Apache Spark to seamlessly accelerate your pipelines.  
Finally, shaping cutting edge technology, Coral is fully compatible with **Kubernetes** and using InAccel's device plugin you can set up a Kubernetes cluster aware of hardware accelerated resources or take advantage of **Serverless architecture** and provide acclerated serverless solutions to your own customers.

* You can **create a free InAccel Coral license** [here](https://www.inaccel.com/license/).
* You can **download** InAccel Coral docker from [dockerhub](https://hub.docker.com/r/inaccel/coral).
* You can find **full documentation** as well as a **quick starting guide** in [InAccel Docs](https://docs.inaccel.com/latest/).
