================================================================================
(c) 2015 Joern Dinkla. www.dinkla.com

parallel2015_gpucomputing
================================================================================

Code für den Vortrag auf der Parallel 2015.

Benötigt

	CUDA
		Windows: 
			CUDA 7.0 und Visual Studio 2013
		Mac: 
			CUDA 7.0 und Xcode command line tools
		Linux: 
			CUDA 7.0 und die Abhängigkeiten

	OpenCL		

		Intel OpenCL SDK oder INDE
		AMD APP SDK
		oder NVIDIA OpenCL (wird mit CUDA installiert)

	C++ AMP
		Windows: 
			Visual Studio 2013 (evtl. auch 2012, aber nicht getestet)

--------------------------------------------------------------------------------

OpenCL

Für opencl_intel muss die Umgebungsvariable INTELOCLSDKROOT gesetzt sein. Diese 
wird z. B. bei der Installation der INDE Starter-Edition gesetzt.

Für opencl_amd muss die Umgebungsvariable AMDAPPSDKROOT gesetzt sein. Diese wird 
z. B. bei der Installation des AMD APP SDK Starter-Edition gesetzt.

Außerdem muss Bolt installiert werden und die Umgebungsvariable BOLTLIB_DIR
gesetzt werden, so dass BOLTLIB_DIR/include auf die Include-Dateinen verweist.

--------------------------------------------------------------------------------


