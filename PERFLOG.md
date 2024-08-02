# Performance Log

At every release, a dedicated benchmark suite is run to ensure that the performance of the library
is not regressing. The results are recorded here.

## 0.42.0-alpha.7 – 2024-08-02

Evaluation of degree-65535 polynomial in 64 Points/Iterative/16
                        time:   [12.029 ms 12.033 ms 12.037 ms]
                        change: [-2.9787% -1.6367% -0.6054%] (p = 0.01 < 0.05)

Evaluation of degree-65535 polynomial in 64 Points/Divide-and-Conquer/16
                        time:   [9.1122 ms 9.1148 ms 9.1202 ms]
                        change: [-1.4568% -0.1815% +0.7635%] (p = 0.81 > 0.05)

Evaluation of degree-65535 polynomial in 64 Points/Entrypoint/16
                        time:   [2.8813 ms 2.8858 ms 2.8898 ms]
                        change: [+1.9188% +2.0393% +2.1452%] (p = 0.00 < 0.05)

Evaluation of degree-65535 polynomial in 256 Points/Iterative/16
                        time:   [48.136 ms 48.180 ms 48.231 ms]
                        change: [-5.8425% -3.5099% -1.4431%] (p = 0.01 < 0.05)

Evaluation of degree-65535 polynomial in 256 Points/Divide-and-Conquer/16
                        time:   [36.856 ms 36.877 ms 36.894 ms]
                        change: [+1.7994% +2.0644% +2.2998%] (p = 0.00 < 0.05)

Evaluation of degree-65535 polynomial in 256 Points/Entrypoint/16
                        time:   [5.4282 ms 5.4305 ms 5.4336 ms]
                        change: [+0.3471% +0.4229% +0.5012%] (p = 0.00 < 0.05)

Evaluation of degree-524287 polynomial in 64 Points/Iterative/19
                        time:   [96.223 ms 96.244 ms 96.266 ms]
                        change: [-0.2668% -0.1859% -0.1136%] (p = 0.00 < 0.05)

Evaluation of degree-524287 polynomial in 64 Points/Divide-and-Conquer/19
                        time:   [72.846 ms 72.961 ms 73.042 ms]
                        change: [+2.0823% +2.1924% +2.3094%] (p = 0.00 < 0.05)

Evaluation of degree-524287 polynomial in 64 Points/Entrypoint/19
                        time:   [24.076 ms 25.246 ms 25.832 ms]
                        change: [+4.1914% +8.2024% +12.302%] (p = 0.00 < 0.05)

Evaluation of degree-524287 polynomial in 256 Points/Iterative/19
                        time:   [399.76 ms 412.94 ms 425.73 ms]
                        change: [+3.5595% +7.1601% +10.435%] (p = 0.00 < 0.05)

Evaluation of degree-524287 polynomial in 256 Points/Divide-and-Conquer/19
                        time:   [292.03 ms 293.01 ms 294.13 ms]
                        change: [+2.1771% +2.5349% +2.9162%] (p = 0.00 < 0.05)

Evaluation of degree-524287 polynomial in 256 Points/Entrypoint/19
                        time:   [37.190 ms 38.220 ms 38.919 ms]
                        change: [+1.0489% +2.4578% +4.3297%] (p = 0.00 < 0.05)

Various Interpolations in 2^10 Points/Lagrange/10
                        time:   [5.3441 ms 5.4135 ms 5.5191 ms]
                        thrpt:  [185.54 Kelem/s 189.16 Kelem/s 191.61 Kelem/s]

Various Interpolations in 2^10 Points/Fast sequential/10
                        time:   [7.7223 ms 7.7264 ms 7.7360 ms]
                        thrpt:  [132.37 Kelem/s 132.53 Kelem/s 132.60 Kelem/s]

Various Interpolations in 2^10 Points/Dispatcher sequential/10
                        time:   [5.2805 ms 5.2843 ms 5.2887 ms]
                        thrpt:  [193.62 Kelem/s 193.78 Kelem/s 193.92 Kelem/s]

Various Interpolations in 2^10 Points/Fast parallel/10
                        time:   [2.9051 ms 2.9193 ms 2.9349 ms]
                        thrpt:  [348.91 Kelem/s 350.77 Kelem/s 352.48 Kelem/s]

Various Interpolations in 2^10 Points/Dispatcher parallel/10
                        time:   [2.9103 ms 2.9176 ms 2.9265 ms]
                        thrpt:  [349.90 Kelem/s 350.97 Kelem/s 351.85 Kelem/s]

Various Interpolations in 2^15 Points/Fast parallel/15
                        time:   [117.94 ms 118.61 ms 119.61 ms]
                        thrpt:  [273.95 Kelem/s 276.27 Kelem/s 277.84 Kelem/s]

Various Interpolations in 2^15 Points/Dispatcher parallel/15
                        time:   [117.05 ms 118.43 ms 119.81 ms]
                        thrpt:  [273.50 Kelem/s 276.68 Kelem/s 279.94 Kelem/s]

Multiplication of Polynomials of Degree 2^7 (Product Degree: 2^8)/Naïve/8
                        time:   [22.725 µs 22.735 µs 22.746 µs]
                        change: [-2.4991% -2.4203% -2.3440%] (p = 0.00 < 0.05)

Multiplication of Polynomials of Degree 2^7 (Product Degree: 2^8)/Fast/8
                        time:   [25.026 µs 25.048 µs 25.074 µs]
                        change: [-1.6298% -1.2273% -0.8575%] (p = 0.00 < 0.05)

Multiplication of Polynomials of Degree 2^7 (Product Degree: 2^8)/Faster of the two/8
                        time:   [25.018 µs 25.038 µs 25.061 µs]
                        change: [-0.8143% -0.6218% -0.4528%] (p = 0.00 < 0.05)

Multiplication of Polynomials of Degree 2^8 (Product Degree: 2^9)/Naïve/9
                        time:   [98.177 µs 98.256 µs 98.349 µs]
                        change: [+9.0155% +9.2747% +9.5621%] (p = 0.00 < 0.05)

Multiplication of Polynomials of Degree 2^8 (Product Degree: 2^9)/Fast/9
                        time:   [54.329 µs 54.365 µs 54.414 µs]
                        change: [-0.4537% -0.1639% +0.0839%] (p = 0.25 > 0.05)

Multiplication of Polynomials of Degree 2^8 (Product Degree: 2^9)/Faster of the two/9
                        time:   [54.290 µs 54.320 µs 54.360 µs]
                        change: [-0.3229% -0.1698% +0.0024%] (p = 0.04 < 0.05)

Multiplication of Polynomials of Degree 2^9 (Product Degree: 2^10)/Naïve/10
                        time:   [370.77 µs 371.34 µs 372.03 µs]
                        change: [+0.8527% +1.1724% +1.5184%] (p = 0.00 < 0.05)

Multiplication of Polynomials of Degree 2^9 (Product Degree: 2^10)/Fast/10
                        time:   [122.64 µs 124.32 µs 126.14 µs]
                        change: [+1.7454% +2.8649% +4.0096%] (p = 0.00 < 0.05)

Multiplication of Polynomials of Degree 2^9 (Product Degree: 2^10)/Faster of the two/10
                        time:   [133.33 µs 134.45 µs 135.39 µs]
                        change: [+7.6508% +8.7307% +9.7862%] (p = 0.00 < 0.05)

polynomial coset of degree 2^10/coset-evaluate bfe-pol/1024
                        time:   [21.259 µs 21.689 µs 22.092 µs]
                        thrpt:  [46.351 Melem/s 47.213 Melem/s 48.168 Melem/s]

polynomial coset of degree 2^10/coset-evaluate xfe-pol/1024
                        time:   [40.084 µs 40.502 µs 40.906 µs]
                        thrpt:  [25.033 Melem/s 25.282 Melem/s 25.546 Melem/s]

polynomial coset of degree 2^10/coset-interpolate bfe-pol/1024
                        time:   [24.164 µs 24.369 µs 24.602 µs]
                        thrpt:  [41.623 Melem/s 42.020 Melem/s 42.377 Melem/s]

polynomial coset of degree 2^10/coset-interpolate xfe-pol/1024
                        time:   [48.749 µs 49.337 µs 50.177 µs]
                        thrpt:  [20.408 Melem/s 20.755 Melem/s 21.005 Melem/s]

polynomial coset of degree 2^17/coset-evaluate bfe-pol/131072
                        time:   [4.4026 ms 4.4655 ms 4.5735 ms]
                        thrpt:  [28.659 Melem/s 29.352 Melem/s 29.772 Melem/s]

polynomial coset of degree 2^17/coset-evaluate xfe-pol/131072
                        time:   [8.6050 ms 8.6826 ms 8.8606 ms]
                        thrpt:  [14.793 Melem/s 15.096 Melem/s 15.232 Melem/s]

polynomial coset of degree 2^17/coset-interpolate bfe-pol/131072
                        time:   [5.1599 ms 5.2848 ms 5.3734 ms]
                        thrpt:  [24.393 Melem/s 24.801 Melem/s 25.402 Melem/s]

polynomial coset of degree 2^17/coset-interpolate xfe-pol/131072
                        time:   [13.213 ms 13.487 ms 13.784 ms]
                        thrpt:  [9.5089 Melem/s 9.7181 Melem/s 9.9197 Melem/s]


## 0.42.0-alpha.2 – 2024-04-26

Various Evaluations in 2^10 Points/Parallel/10
                        time:   [239.30 µs 240.25 µs 241.10 µs]
                        thrpt:  [4.2472 Melem/s 4.2622 Melem/s 4.2791 Melem/s]

Various Evaluations in 2^10 Points/Fast/10
                        time:   [1.3964 ms 1.4012 ms 1.4064 ms]
                        thrpt:  [728.10 Kelem/s 730.78 Kelem/s 733.32 Kelem/s]

Various Evaluations in 2^10 Points/Faster of the two/10
                        time:   [241.82 µs 242.68 µs 243.47 µs]
                        thrpt:  [4.2059 Melem/s 4.2196 Melem/s 4.2345 Melem/s]

Various Evaluations in 2^14 Points/Parallel/14
                        time:   [9.2347 ms 9.2947 ms 9.3653 ms]
                        thrpt:  [1.7494 Melem/s 1.7627 Melem/s 1.7742 Melem/s]

Various Evaluations in 2^14 Points/Fast/14
                        time:   [23.746 ms 23.831 ms 23.916 ms]
                        thrpt:  [685.07 Kelem/s 687.50 Kelem/s 689.98 Kelem/s]

Various Evaluations in 2^14 Points/Faster of the two/14
                        time:   [8.9896 ms 9.0391 ms 9.0978 ms]
                        thrpt:  [1.8009 Melem/s 1.8126 Melem/s 1.8225 Melem/s]

Various Evaluations in 2^16 Points/Parallel/16
                        time:   [143.70 ms 143.89 ms 144.10 ms]
                        thrpt:  [454.80 Kelem/s 455.45 Kelem/s 456.06 Kelem/s]

Various Evaluations in 2^16 Points/Fast/16
                        time:   [249.88 ms 250.26 ms 250.66 ms]
                        thrpt:  [261.45 Kelem/s 261.87 Kelem/s 262.27 Kelem/s]

Various Evaluations in 2^16 Points/Faster of the two/16
                        time:   [140.42 ms 140.56 ms 140.72 ms]
                        thrpt:  [465.73 Kelem/s 466.24 Kelem/s 466.72 Kelem/s]

Various Interpolations in 2^8 Points/Lagrange/8
                        time:   [426.13 µs 431.10 µs 435.86 µs]
                        thrpt:  [587.35 Kelem/s 593.83 Kelem/s 600.75 Kelem/s]

Various Interpolations in 2^8 Points/Fast/8
                        time:   [553.62 µs 555.42 µs 557.12 µs]
                        thrpt:  [459.51 Kelem/s 460.91 Kelem/s 462.41 Kelem/s]

Various Interpolations in 2^8 Points/Faster of the two/8
                        time:   [420.91 µs 426.18 µs 432.49 µs]
                        thrpt:  [591.92 Kelem/s 600.68 Kelem/s 608.21 Kelem/s]

Various Interpolations in 2^9 Points/Lagrange/9
                        time:   [1.7243 ms 1.7261 ms 1.7274 ms]
                        thrpt:  [296.40 Kelem/s 296.63 Kelem/s 296.94 Kelem/s]

Various Interpolations in 2^9 Points/Fast/9
                        time:   [1.2241 ms 1.2257 ms 1.2274 ms]
                        thrpt:  [417.13 Kelem/s 417.72 Kelem/s 418.26 Kelem/s]

Various Interpolations in 2^9 Points/Faster of the two/9
                        time:   [1.2223 ms 1.2241 ms 1.2259 ms]
                        thrpt:  [417.66 Kelem/s 418.28 Kelem/s 418.87 Kelem/s]

Various Interpolations in 2^10 Points/Lagrange/10
                        time:   [5.7691 ms 5.7941 ms 5.8219 ms]
                        thrpt:  [175.89 Kelem/s 176.73 Kelem/s 177.50 Kelem/s]

Various Interpolations in 2^10 Points/Fast/10
                        time:   [1.8598 ms 1.8675 ms 1.8759 ms]
                        thrpt:  [545.88 Kelem/s 548.33 Kelem/s 550.60 Kelem/s]

Various Interpolations in 2^10 Points/Faster of the two/10
                        time:   [1.8384 ms 1.8421 ms 1.8460 ms]
                        thrpt:  [554.70 Kelem/s 555.88 Kelem/s 557.00 Kelem/s]

Multiplication of Polynomials of Degree 2^7 (Product Degree: 2^8)/Naïve/8
                        time:   [23.021 µs 23.033 µs 23.046 µs]

Multiplication of Polynomials of Degree 2^7 (Product Degree: 2^8)/Fast/8
                        time:   [25.126 µs 25.139 µs 25.155 µs]

Multiplication of Polynomials of Degree 2^7 (Product Degree: 2^8)/Faster of the two/8
                        time:   [25.044 µs 25.059 µs 25.074 µs]

Multiplication of Polynomials of Degree 2^8 (Product Degree: 2^9)/Naïve/9
                        time:   [88.986 µs 89.024 µs 89.068 µs]

Multiplication of Polynomials of Degree 2^8 (Product Degree: 2^9)/Fast/9
                        time:   [54.465 µs 54.487 µs 54.514 µs]

Multiplication of Polynomials of Degree 2^8 (Product Degree: 2^9)/Faster of the two/9
                        time:   [54.473 µs 54.524 µs 54.605 µs]

Multiplication of Polynomials of Degree 2^9 (Product Degree: 2^10)/Naïve/10
                        time:   [352.62 µs 352.76 µs 352.94 µs]

Multiplication of Polynomials of Degree 2^9 (Product Degree: 2^10)/Fast/10
                        time:   [118.75 µs 118.79 µs 118.84 µs]

Multiplication of Polynomials of Degree 2^9 (Product Degree: 2^10)/Faster of the two/10
                        time:   [118.74 µs 118.78 µs 118.83 µs]

polynomial coset of degree 2^10/coset-evaluate bfe-pol/1024
                        time:   [20.274 µs 20.289 µs 20.313 µs]
                        thrpt:  [50.411 Melem/s 50.471 Melem/s 50.507 Melem/s]

polynomial coset of degree 2^10/coset-evaluate xfe-pol/1024
                        time:   [39.516 µs 39.667 µs 39.869 µs]
                        thrpt:  [25.684 Melem/s 25.815 Melem/s 25.913 Melem/s]

polynomial coset of degree 2^10/coset-interpolate bfe-pol/1024
                        time:   [23.799 µs 23.809 µs 23.821 µs]
                        thrpt:  [42.988 Melem/s 43.009 Melem/s 43.027 Melem/s]

polynomial coset of degree 2^10/coset-interpolate xfe-pol/1024
                        time:   [47.679 µs 47.698 µs 47.725 µs]
                        thrpt:  [21.456 Melem/s 21.469 Melem/s 21.477 Melem/s]

polynomial coset of degree 2^17/coset-evaluate bfe-pol/131072
                        time:   [4.2981 ms 4.2998 ms 4.3018 ms]
                        thrpt:  [30.469 Melem/s 30.483 Melem/s 30.495 Melem/s]

polynomial coset of degree 2^17/coset-evaluate xfe-pol/131072
                        time:   [8.3497 ms 8.3672 ms 8.3889 ms]
                        thrpt:  [15.624 Melem/s 15.665 Melem/s 15.698 Melem/s]

polynomial coset of degree 2^17/coset-interpolate bfe-pol/131072
                        time:   [5.0171 ms 5.0179 ms 5.0195 ms]
                        thrpt:  [26.113 Melem/s 26.121 Melem/s 26.125 Melem/s]

polynomial coset of degree 2^17/coset-interpolate xfe-pol/131072
                        time:   [13.186 ms 13.205 ms 13.234 ms]
                        thrpt:  [9.9042 Melem/s 9.9258 Melem/s 9.9400 Melem/s]

## 0.41.0 – 2024-04-24

Various Evaluations in 2^10 Points/Parallel/10
                        time:   [239.93 µs 240.68 µs 241.38 µs]
                        thrpt:  [4.2423 Melem/s 4.2547 Melem/s 4.2679 Melem/s]

Various Evaluations in 2^10 Points/Fast/10
                        time:   [1.3966 ms 1.4002 ms 1.4039 ms]
                        thrpt:  [729.39 Kelem/s 731.34 Kelem/s 733.23 Kelem/s]

Various Evaluations in 2^10 Points/Faster of the two/10
                        time:   [241.08 µs 241.74 µs 242.39 µs]
                        thrpt:  [4.2246 Melem/s 4.2359 Melem/s 4.2476 Melem/s]

Various Evaluations in 2^14 Points/Parallel/14
                        time:   [9.2073 ms 9.2173 ms 9.2318 ms]
                        thrpt:  [1.7747 Melem/s 1.7775 Melem/s 1.7795 Melem/s]

Various Evaluations in 2^14 Points/Fast/14
                        time:   [23.926 ms 24.004 ms 24.086 ms]
                        thrpt:  [680.23 Kelem/s 682.55 Kelem/s 684.79 Kelem/s]

Various Evaluations in 2^14 Points/Faster of the two/14
                        time:   [9.2455 ms 9.2562 ms 9.2711 ms]
                        thrpt:  [1.7672 Melem/s 1.7701 Melem/s 1.7721 Melem/s]

Various Evaluations in 2^16 Points/Parallel/16
                        time:   [143.79 ms 143.95 ms 144.11 ms]
                        thrpt:  [454.77 Kelem/s 455.28 Kelem/s 455.79 Kelem/s]

Various Evaluations in 2^16 Points/Fast/16
                        time:   [256.71 ms 257.05 ms 257.41 ms]
                        thrpt:  [254.60 Kelem/s 254.95 Kelem/s 255.29 Kelem/s]

Various Evaluations in 2^16 Points/Faster of the two/16
                        time:   [144.80 ms 144.87 ms 144.93 ms]
                        thrpt:  [452.18 Kelem/s 452.38 Kelem/s 452.60 Kelem/s]

Various Interpolations in 2^8 Points/Lagrange/8
                        time:   [419.69 µs 420.95 µs 422.53 µs]
                        thrpt:  [605.88 Kelem/s 608.15 Kelem/s 609.97 Kelem/s]

Various Interpolations in 2^8 Points/Fast/8
                        time:   [553.19 µs 554.59 µs 555.84 µs]
                        thrpt:  [460.56 Kelem/s 461.61 Kelem/s 462.77 Kelem/s]

Various Interpolations in 2^8 Points/Faster of the two/8
                        time:   [421.68 µs 424.45 µs 427.52 µs]
                        thrpt:  [598.80 Kelem/s 603.13 Kelem/s 607.10 Kelem/s]

Various Interpolations in 2^9 Points/Lagrange/9
                        time:   [1.7140 ms 1.7152 ms 1.7163 ms]
                        thrpt:  [298.32 Kelem/s 298.51 Kelem/s 298.72 Kelem/s]

Various Interpolations in 2^9 Points/Fast/9
                        time:   [1.2160 ms 1.2182 ms 1.2201 ms]
                        thrpt:  [419.62 Kelem/s 420.30 Kelem/s 421.04 Kelem/s]

Various Interpolations in 2^9 Points/Faster of the two/9
                        time:   [1.2189 ms 1.2200 ms 1.2212 ms]
                        thrpt:  [419.27 Kelem/s 419.66 Kelem/s 420.06 Kelem/s]

Various Interpolations in 2^10 Points/Lagrange/10
                        time:   [5.7082 ms 5.7155 ms 5.7243 ms]
                        thrpt:  [178.89 Kelem/s 179.16 Kelem/s 179.39 Kelem/s]

Various Interpolations in 2^10 Points/Fast/10
                        time:   [1.8447 ms 1.8486 ms 1.8523 ms]
                        thrpt:  [552.82 Kelem/s 553.94 Kelem/s 555.12 Kelem/s]

Various Interpolations in 2^10 Points/Faster of the two/10
                        time:   [1.8373 ms 1.8422 ms 1.8474 ms]
                        thrpt:  [554.28 Kelem/s 555.85 Kelem/s 557.35 Kelem/s]

Multiplication of Polynomials of Degree 2^7 (Product Degree: 2^8)/Naïve/8
                        time:   [22.973 µs 22.983 µs 22.994 µs]

Multiplication of Polynomials of Degree 2^7 (Product Degree: 2^8)/Fast/8
                        time:   [25.106 µs 25.119 µs 25.133 µs]

Multiplication of Polynomials of Degree 2^7 (Product Degree: 2^8)/Faster of the two/8
                        time:   [25.127 µs 25.141 µs 25.155 µs]

Multiplication of Polynomials of Degree 2^8 (Product Degree: 2^9)/Naïve/9
                        time:   [89.025 µs 89.055 µs 89.089 µs]

Multiplication of Polynomials of Degree 2^8 (Product Degree: 2^9)/Fast/9
                        time:   [54.460 µs 54.476 µs 54.496 µs]

Multiplication of Polynomials of Degree 2^8 (Product Degree: 2^9)/Faster of the two/9
                        time:   [54.467 µs 54.482 µs 54.501 µs]
                        change: [-0.2581% -0.1616% -0.0781%] (p = 0.00 < 0.05)

Multiplication of Polynomials of Degree 2^9 (Product Degree: 2^10)/Naïve/10
                        time:   [352.32 µs 352.43 µs 352.56 µs]
                        change: [-0.2588% -0.1979% -0.1323%] (p = 0.00 < 0.05)

Multiplication of Polynomials of Degree 2^9 (Product Degree: 2^10)/Fast/10
                        time:   [118.72 µs 118.76 µs 118.81 µs]
                        change: [-0.1116% -0.0229% +0.0442%] (p = 0.61 > 0.05)

Multiplication of Polynomials of Degree 2^9 (Product Degree: 2^10)/Faster of the two/10
                        time:   [118.77 µs 118.93 µs 119.16 µs]
                        change: [-0.0836% +0.0666% +0.2156%] (p = 0.40 > 0.05)

polynomial coset of degree 2^10/coset-evaluate bfe-pol/1024
                        time:   [20.231 µs 20.237 µs 20.246 µs]
                        thrpt:  [50.578 Melem/s 50.599 Melem/s 50.615 Melem/s]

polynomial coset of degree 2^10/coset-evaluate xfe-pol/1024
                        time:   [38.810 µs 38.834 µs 38.860 µs]
                        thrpt:  [26.351 Melem/s 26.369 Melem/s 26.385 Melem/s]

polynomial coset of degree 2^10/coset-interpolate bfe-pol/1024
                        time:   [23.560 µs 23.571 µs 23.584 µs]
                        thrpt:  [43.419 Melem/s 43.443 Melem/s 43.463 Melem/s]

polynomial coset of degree 2^10/coset-interpolate xfe-pol/1024
                        time:   [47.874 µs 47.901 µs 47.925 µs]
                        thrpt:  [21.367 Melem/s 21.377 Melem/s 21.390 Melem/s]

polynomial coset of degree 2^17/coset-evaluate bfe-pol/131072
                        time:   [4.2960 ms 4.2972 ms 4.2985 ms]
                        thrpt:  [30.493 Melem/s 30.501 Melem/s 30.510 Melem/s]

polynomial coset of degree 2^17/coset-evaluate xfe-pol/131072
                        time:   [8.4138 ms 8.4565 ms 8.4999 ms]
                        thrpt:  [15.420 Melem/s 15.500 Melem/s 15.578 Melem/s]

polynomial coset of degree 2^17/coset-interpolate bfe-pol/131072
                        time:   [5.1712 ms 5.1726 ms 5.1741 ms]
                        thrpt:  [25.332 Melem/s 25.340 Melem/s 25.347 Melem/s]

polynomial coset of degree 2^17/coset-interpolate xfe-pol/131072
                        time:   [13.387 ms 13.413 ms 13.438 ms]
                        thrpt:  [9.7537 Melem/s 9.7724 Melem/s 9.7910 Melem/s]

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
