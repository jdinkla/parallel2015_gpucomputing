@echo off
set BENCH=..\x64\Release\cuda.exe
set BENCH2=..\x64\Release\opencl_intel.exe
set BENCH3=..\x64\Release\amp.exe
%BENCH% benchmark_single_gpu
REM %BENCH% benchmark_multi_gpu
REM %BENCH% benchmark_multi_seq
REM %BENCH% benchmark_multi_async
REM %BENCH% benchmark_multi_threads
REM %BENCH% benchmark_single_cpu
%BENCH% benchmark_multi_cpu

%BENCH2% benchmark_single
REM %BENCH2% benchmark_multi

%BENCH3% benchmark_single
REM %BENCH3% benchmark_multi
