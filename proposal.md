# Fast Fourier Transform

A) Fast Fourier Transform is an algorithm that computes the discrete Fourier transform of a sequence. This is often done to convert a signal from its original domain (usually time or space) to the frequency domain. The concept of decomposing a sequence of values into components of different frequencies is valuable in many fields, however performing discrete fourier transform is often too slow and therefore not practical $O(n^2)$. Fast Fourier transform provides a way to do these calculations in $O(nlog(n))$ time.

B) While there are no other algorithms that perform as well as FFT, there are other's that perform the same task. They are detailed here in this [paper](https://apps.dtic.mil/sti/tr/pdf/ADA058049.pdf) if you wish to read more in depth about them. I will simply be comparing and contrasting the memory complexity and the programming complexity. The two other algorithms I'll be comparing are the Prime Factor Algorithm and the Cooley-Tukey Algorithm. Details regarding the Cooley-Tukey Algorithm can be found [here](https://bookdown.org/rdpeng/timeseriesbook/the-fast-fourier-transform-fft.html) in section 3.6.2.

|  Algorithm  |     Memory     |          Time Complexity       |
| ----------- | -------------- | ------------------------------ |
|    FFT      |                |            $O(nlog(n))         |
|   Prime     |                |           $O(n\sqrt{n})$       |
| Cooly-Tukey |                | $O(\frac{nlog_2(n)r}{log_2(r)} |
