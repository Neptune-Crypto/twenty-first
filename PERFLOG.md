# Performance Log

At every release, a dedicated benchmark suite is run to ensure that the performance of the library
is not regressing. The results are recorded here.

## 0.40.0 – 2024-04-16

Various Evaluations in 2^10 Points/Parallel/10
                        time:   [240.98 µs 242.96 µs 244.63 µs]
                        thrpt:  [4.1859 Melem/s 4.2147 Melem/s 4.2493 Melem/s]

Various Evaluations in 2^10 Points/Fast/10
                        time:   [1.4162 ms 1.4207 ms 1.4255 ms]
                        thrpt:  [718.33 Kelem/s 720.76 Kelem/s 723.05 Kelem/s]

Various Evaluations in 2^10 Points/Faster of the two/10
                        time:   [239.10 µs 242.47 µs 244.92 µs]
                        thrpt:  [4.1810 Melem/s 4.2231 Melem/s 4.2827 Melem/s]

Various Evaluations in 2^14 Points/Parallel/14
                        time:   [9.9679 ms 10.030 ms 10.094 ms]
                        thrpt:  [1.6231 Melem/s 1.6335 Melem/s 1.6437 Melem/s]

Various Evaluations in 2^14 Points/Fast/14
                        time:   [24.795 ms 24.883 ms 24.972 ms]
                        thrpt:  [656.11 Kelem/s 658.45 Kelem/s 660.77 Kelem/s]

Various Evaluations in 2^14 Points/Faster of the two/14
                        time:   [10.152 ms 10.212 ms 10.275 ms]
                        thrpt:  [1.5945 Melem/s 1.6044 Melem/s 1.6139 Melem/s]

Various Evaluations in 2^16 Points/Parallel/16
                        time:   [144.72 ms 145.04 ms 145.37 ms]
                        thrpt:  [450.81 Kelem/s 451.84 Kelem/s 452.84 Kelem/s]

Various Evaluations in 2^16 Points/Fast/16
                        time:   [264.67 ms 265.13 ms 265.62 ms]
                        thrpt:  [246.73 Kelem/s 247.18 Kelem/s 247.61 Kelem/s]

Various Evaluations in 2^16 Points/Faster of the two/16
                        time:   [149.98 ms 150.32 ms 150.67 ms]
                        thrpt:  [434.96 Kelem/s 435.97 Kelem/s 436.95 Kelem/s]

Various Interpolations in 2^8 Points/Lagrange/8
                        time:   [453.25 µs 454.12 µs 455.00 µs]
                        thrpt:  [562.64 Kelem/s 563.73 Kelem/s 564.81 Kelem/s]

Various Interpolations in 2^8 Points/Fast/8
                        time:   [555.07 µs 557.50 µs 560.43 µs]
                        thrpt:  [456.79 Kelem/s 459.19 Kelem/s 461.21 Kelem/s]

Various Interpolations in 2^8 Points/Faster of the two/8
                        time:   [455.93 µs 456.03 µs 456.13 µs]
                        thrpt:  [561.25 Kelem/s 561.37 Kelem/s 561.48 Kelem/s]

Various Interpolations in 2^9 Points/Lagrange/9
                        time:   [1.6728 ms 1.6795 ms 1.6886 ms]
                        thrpt:  [303.21 Kelem/s 304.85 Kelem/s 306.07 Kelem/s]

Various Interpolations in 2^9 Points/Fast/9
                        time:   [1.2351 ms 1.2368 ms 1.2388 ms]
                        thrpt:  [413.32 Kelem/s 413.96 Kelem/s 414.54 Kelem/s]

Various Interpolations in 2^9 Points/Faster of the two/9
                        time:   [1.2329 ms 1.2354 ms 1.2379 ms]
                        thrpt:  [413.59 Kelem/s 414.44 Kelem/s 415.27 Kelem/s]

Various Interpolations in 2^10 Points/Lagrange/10
                        time:   [5.8141 ms 5.8532 ms 5.8987 ms]
                        thrpt:  [173.60 Kelem/s 174.95 Kelem/s 176.12 Kelem/s]

Various Interpolations in 2^10 Points/Fast/10
                        time:   [1.8744 ms 1.8801 ms 1.8855 ms]
                        thrpt:  [543.09 Kelem/s 544.65 Kelem/s 546.31 Kelem/s]

Various Interpolations in 2^10 Points/Faster of the two/10
                        time:   [1.8671 ms 1.8724 ms 1.8779 ms]
                        thrpt:  [545.29 Kelem/s 546.88 Kelem/s 548.45 Kelem/s]

Multiplication of Polynomials of Degree 2^7 (Product Degree: 2^8)/Naïve/8
                        time:   [22.937 µs 22.946 µs 22.956 µs]

Multiplication of Polynomials of Degree 2^7 (Product Degree: 2^8)/Fast/8
                        time:   [25.125 µs 25.142 µs 25.164 µs]

Multiplication of Polynomials of Degree 2^7 (Product Degree: 2^8)/Faster of the two/8
                        time:   [25.141 µs 25.159 µs 25.181 µs]

Multiplication of Polynomials of Degree 2^8 (Product Degree: 2^9)/Naïve/9
                        time:   [89.027 µs 89.065 µs 89.109 µs]

Multiplication of Polynomials of Degree 2^8 (Product Degree: 2^9)/Fast/9
                        time:   [54.522 µs 54.586 µs 54.667 µs]

Multiplication of Polynomials of Degree 2^8 (Product Degree: 2^9)/Faster of the two/9
                        time:   [54.516 µs 54.555 µs 54.605 µs]

Multiplication of Polynomials of Degree 2^9 (Product Degree: 2^10)/Naïve/10
                        time:   [353.09 µs 353.30 µs 353.53 µs]

Multiplication of Polynomials of Degree 2^9 (Product Degree: 2^10)/Fast/10
                        time:   [118.70 µs 118.74 µs 118.79 µs]

Multiplication of Polynomials of Degree 2^9 (Product Degree: 2^10)/Faster of the two/10
                        time:   [118.73 µs 118.85 µs 119.03 µs]

polynomial coset of degree 2^10/coset-evaluate bfe-pol/1024
                        time:   [20.310 µs 20.338 µs 20.369 µs]
                        thrpt:  [50.273 Melem/s 50.348 Melem/s 50.417 Melem/s]

polynomial coset of degree 2^10/coset-evaluate xfe-pol/1024
                        time:   [38.897 µs 38.922 µs 38.969 µs]
                        thrpt:  [26.277 Melem/s 26.309 Melem/s 26.326 Melem/s]

polynomial coset of degree 2^10/coset-interpolate bfe-pol/1024
                        time:   [23.818 µs 23.833 µs 23.857 µs]
                        thrpt:  [42.922 Melem/s 42.965 Melem/s 42.992 Melem/s]

polynomial coset of degree 2^10/coset-interpolate xfe-pol/1024
                        time:   [47.766 µs 47.785 µs 47.806 µs]
                        thrpt:  [21.420 Melem/s 21.429 Melem/s 21.438 Melem/s]

polynomial coset of degree 2^17/coset-evaluate bfe-pol/131072
                        time:   [4.2689 ms 4.2713 ms 4.2738 ms]
                        thrpt:  [30.668 Melem/s 30.686 Melem/s 30.704 Melem/s]

polynomial coset of degree 2^17/coset-evaluate xfe-pol/131072
                        time:   [8.2951 ms 8.2993 ms 8.3057 ms]
                        thrpt:  [15.781 Melem/s 15.793 Melem/s 15.801 Melem/s]

polynomial coset of degree 2^17/coset-interpolate bfe-pol/131072
                        time:   [5.2468 ms 5.2537 ms 5.2602 ms]
                        thrpt:  [24.918 Melem/s 24.949 Melem/s 24.981 Melem/s]

polynomial coset of degree 2^17/coset-interpolate xfe-pol/131072
                        time:   [13.140 ms 13.145 ms 13.153 ms]
                        thrpt:  [9.9655 Melem/s 9.9713 Melem/s 9.9753 Melem/s]
