# Fast Fourier Transform

**Jack Haek**

---

## What is the Fast Fourier Transform Algorithm

The Fast Fourier Transform Algorithm (FFT) is possibly one of the most important and elegant algorithms published in the 20th century. The effects of this algorithm stretch to many industries including, but not limited to medical imaging, telecommunications, electromagnetic wave and quantum mechanics, as well as nearly anything to do with signal processing. For the demo today, we will see the algorithm identify what composition of sin waves is used to create a more complex wave.

**Discrete Fourier Transform (DFT)**

With regards to real world applications, the Fourier Transform is not always super helpful as functional forms of signals are not all that common. Thus, the Discrete Fourier Transform (DFT) was roughly derived from the FT.


**FT**
$$
f(k) = \int_{-\infty}^{\infty}f(x)e^{-2\pi ikx}dx
$$

**DFT**
$$
f_k=\sum_{0}^{N-1}x_ne^{\frac{-2\pi ikn}{N}}
$$

In DFT, $x_n$ refers to an element of our discrete signal. This is a vector. $F_k$ is an element of our transformed signal. $n$ and $k$ are indexes corresponding to the original and transformed signal vectors. To make this easier to see and for easier notation later, we are going to introduce a new term. 

$$
\omega_{kn} = e^{\frac{-2\pi ikn}{N}}
$$

Note that with the two coordinate values for $\omega$ we can now think of this as a two dimensional matrix. This is what our new equation looks like:

**DFT**
$$
f_k=\sum_{0}^{N-1}\omega_{kn}x_n
$$

To show the vectors and matrices

$$
f_k \rightarrow \begin{bmatrix}
f_0
\\
f_1
\\
...
\\ 
f_{k-1}
\end{bmatrix}
x_n \rightarrow \begin{bmatrix}
x_0
\\
x_1
\\
...
\\ 
x_{N-1}
\end{bmatrix}
\omega_{kn} \rightarrow\begin{bmatrix}
\omega_{0,0} & \omega_{0,1} & ... & \omega_{0,N-1} \\ 
\omega_{1,0} & \omega_{1,1} &  & ... \\
... &  &  & ... \\ 
\omega_{k-1,0} & ...  & ... & \omega_{k-1,N-1} 
\end{bmatrix}
$$

Filling in the equation:

$$
\begin{bmatrix}
f_0
\\
f_1
\\
...
\\ 
f_{k-1}
\end{bmatrix} = \begin{bmatrix}
\omega_{0,0} & \omega_{0,1} & ... & \omega_{0,N-1} \\ 
\omega_{1,0} & \omega_{1,1} &  & ... \\
... &  &  & ... \\ 
\omega_{k-1,0} & ...  & ... & \omega_{k-1,N-1} 
\end{bmatrix}\begin{bmatrix}
x_0
\\
x_1
\\
...
\\ 
x_{N-1}
\end{bmatrix}
$$

While it is great that we can represent the equation in matrix form as it makes it easier to map to GPU's for parallelization. Currently, 2 for loops are still needed. This gives $O(n^2)$ for the computational time complexity. To put this into context, that would mean that a 4 minute song would have the following number of computations:

$$
(4min * \frac{60sec}{1min} * \frac{44100}{1sec})^2 \approx 1.12 * 10^{14}
$$

**Fast Fourier Transform (FFT)**

Fast Fourier Transform describes an efficient algorithm for computing DFT. The main concept of FFT is instead of using the matrices described above, we will use 3 sparse matrices where one is a permutation matrix. This takes advantage of symmetrical patters that emerge and brings the computational time complexity down to $O(nlog(n))$!

**Sparse Matrix**
$$
\omega_N = \begin{bmatrix}
I_{N/2} & D_{N/2} \\ 
I_{N/2} & -D_{N/2}
\end{bmatrix}\begin{bmatrix}
\omega_{N/2} & 0 \\ 
0 & \omega_{N/2}
\end{bmatrix}(P)
$$

**Permutation Matrix**
$$
P_x = \begin{bmatrix}
x_{even} \\
x_{odd}
\end{bmatrix}
$$

```
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from scipy.fft import fft, fftfreq, fftshift
```
