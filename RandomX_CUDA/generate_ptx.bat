@echo off
call "%VS140COMNTOOLS%/../../VC/vcvarsall.bat" amd64
nvcc --gpu-architecture=compute_35 -ptx -prec-div=true -prec-sqrt=true -o kernel.ptx kernel.cu
