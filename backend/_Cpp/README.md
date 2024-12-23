# Customizing a PyTorch operation

Implement a custom operation in PyTorch that uses Metal kernels to improve performance.

## Overview

- Note: This sample code project is associated with WWDC23 session 10050: [Optimize machine learning for Metal apps](https://developer.apple.com/wwdc23/10050).

## Configure the sample code project

Before you run the sample code project:

1. Follow the instructions in [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/).

2. Install PyTorch nightly (Python 3.7 or later is required).

	```
	pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
	```

3. Install Ninja

	```
	pip3 install Ninja
	```

4. Run the sample.

	```
	python3 run_sample.py
	```