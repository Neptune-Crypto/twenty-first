# Performance Log

At every release, a dedicated benchmark suite is run to ensure that the performance of the library
is not regressing. The results are recorded here.

## 0.43.0 – 2024-10-29

Fast extrapolation of length-262144 codeword in 256 Points/Fast Codeword Extrapolation/18
                        time:   [29.942 ms 30.022 ms 30.175 ms]

Fast extrapolation of length-524288 codeword in 256 Points/Fast Codeword Extrapolation/19
                        time:   [60.089 ms 60.099 ms 60.112 ms]

Fast extrapolation of length-1048576 codeword in 256 Points/Fast Codeword Extrapolation/20
                        time:   [122.16 ms 122.20 ms 122.24 ms]

Fast extrapolation of length-2097152 codeword in 256 Points/Fast Codeword Extrapolation/21
                        time:   [254.43 ms 255.03 ms 255.69 ms]

Fast extrapolation of length-4194304 codeword in 256 Points/Fast Codeword Extrapolation/22
                        time:   [562.12 ms 562.89 ms 563.73 ms]

Fast extrapolation of length-8388608 codeword in 256 Points/Fast Codeword Extrapolation/23
                        time:   [1.2218 s 1.2241 s 1.2270 s]

Evaluation of degree-16383 polynomial in 16384 Points/Iterative/14
                        time:   [770.53 ms 770.69 ms 770.84 ms]

Evaluation of degree-16383 polynomial in 16384 Points/Entrypoint/14
                        time:   [653.24 ms 653.97 ms 654.66 ms]

Evaluation of degree-16383 polynomial in 16384 Points/Par batch-evaluate/14
                        time:   [6.4681 ms 6.6857 ms 7.0208 ms]

Evaluation of degree-65535 polynomial in 64 Points/Iterative/16
                        time:   [12.052 ms 12.054 ms 12.057 ms]

Evaluation of degree-65535 polynomial in 64 Points/Divide-and-Conquer/16
                        time:   [9.4094 ms 9.4167 ms 9.4197 ms]

Evaluation of degree-65535 polynomial in 64 Points/Entrypoint/16
                        time:   [2.9265 ms 2.9277 ms 2.9290 ms]

Evaluation of degree-65535 polynomial in 64 Points/Par batch-evaluate/16
                        time:   [4.9483 ms 4.9959 ms 5.0531 ms]

Evaluation of degree-65535 polynomial in 256 Points/Iterative/16
                        time:   [48.211 ms 48.222 ms 48.235 ms]

Evaluation of degree-65535 polynomial in 256 Points/Divide-and-Conquer/16
                        time:   [37.447 ms 37.454 ms 37.465 ms]

Evaluation of degree-65535 polynomial in 256 Points/Entrypoint/16
                        time:   [5.4677 ms 5.4687 ms 5.4700 ms]

Evaluation of degree-65535 polynomial in 256 Points/Par batch-evaluate/16
                        time:   [5.7579 ms 6.1380 ms 6.5214 ms]

Evaluation of degree-524287 polynomial in 64 Points/Iterative/19
                        time:   [96.431 ms 96.453 ms 96.480 ms]

Evaluation of degree-524287 polynomial in 64 Points/Divide-and-Conquer/19
                        time:   [74.126 ms 74.203 ms 74.240 ms]

Evaluation of degree-524287 polynomial in 64 Points/Entrypoint/19
                        time:   [23.229 ms 23.251 ms 23.266 ms]

Evaluation of degree-524287 polynomial in 64 Points/Par batch-evaluate/19
                        time:   [34.393 ms 34.598 ms 34.896 ms]

Evaluation of degree-524287 polynomial in 256 Points/Iterative/19
                        time:   [385.68 ms 385.76 ms 385.84 ms]

Evaluation of degree-524287 polynomial in 256 Points/Divide-and-Conquer/19
                        time:   [297.60 ms 297.70 ms 297.81 ms]

Evaluation of degree-524287 polynomial in 256 Points/Entrypoint/19
                        time:   [37.164 ms 37.170 ms 37.178 ms]

Evaluation of degree-524287 polynomial in 256 Points/Par batch-evaluate/19
                        time:   [41.463 ms 42.946 ms 44.529 ms]

Extrapolation of length-262144 codeword in 64 Points/INTT-then-Evaluate/18
                        time:   [22.409 ms 22.414 ms 22.419 ms]

Extrapolation of length-262144 codeword in 64 Points/Fast Codeword Extrapolation/18
                        time:   [22.462 ms 22.502 ms 22.522 ms]

Extrapolation of length-262144 codeword in 64 Points/Dispatcher (includes preprocessing)/18
                        time:   [22.676 ms 22.687 ms 22.700 ms]

Extrapolation of length-262144 codeword in 128 Points/INTT-then-Evaluate/18
                        time:   [28.180 ms 28.202 ms 28.220 ms]

Extrapolation of length-262144 codeword in 128 Points/Fast Codeword Extrapolation/18
                        time:   [28.997 ms 29.005 ms 29.010 ms]

Extrapolation of length-262144 codeword in 128 Points/Dispatcher (includes preprocessing)/18
                        time:   [28.182 ms 28.191 ms 28.203 ms]

Extrapolation of length-262144 codeword in 256 Points/INTT-then-Evaluate/18
                        time:   [29.918 ms 29.928 ms 29.939 ms]

Extrapolation of length-262144 codeword in 256 Points/Fast Codeword Extrapolation/18
                        time:   [32.043 ms 32.065 ms 32.097 ms]

Extrapolation of length-262144 codeword in 256 Points/Dispatcher (includes preprocessing)/18
                        time:   [29.828 ms 29.833 ms 29.838 ms]

Extrapolation of length-524288 codeword in 64 Points/INTT-then-Evaluate/19
                        time:   [45.853 ms 45.881 ms 45.899 ms]

Extrapolation of length-524288 codeword in 64 Points/Fast Codeword Extrapolation/19
                        time:   [45.792 ms 45.813 ms 45.826 ms]

Extrapolation of length-524288 codeword in 64 Points/Dispatcher (includes preprocessing)/19
                        time:   [46.197 ms 46.235 ms 46.274 ms]

Extrapolation of length-524288 codeword in 128 Points/INTT-then-Evaluate/19
                        time:   [57.022 ms 57.056 ms 57.093 ms]

Extrapolation of length-524288 codeword in 128 Points/Fast Codeword Extrapolation/19
                        time:   [59.828 ms 59.847 ms 59.867 ms]

Extrapolation of length-524288 codeword in 128 Points/Dispatcher (includes preprocessing)/19
                        time:   [56.948 ms 56.984 ms 57.018 ms]

Extrapolation of length-524288 codeword in 256 Points/INTT-then-Evaluate/19
                        time:   [59.867 ms 59.881 ms 59.904 ms]

Extrapolation of length-524288 codeword in 256 Points/Fast Codeword Extrapolation/19
                        time:   [66.475 ms 66.494 ms 66.507 ms]

Extrapolation of length-524288 codeword in 256 Points/Dispatcher (includes preprocessing)/19
                        time:   [59.845 ms 59.897 ms 59.977 ms]

Extrapolation of length-1048576 codeword in 64 Points/INTT-then-Evaluate/20
                        time:   [94.463 ms 94.538 ms 94.633 ms]

Extrapolation of length-1048576 codeword in 64 Points/Fast Codeword Extrapolation/20
                        time:   [93.903 ms 93.940 ms 93.963 ms]

Extrapolation of length-1048576 codeword in 64 Points/Dispatcher (includes preprocessing)/20
                        time:   [94.068 ms 94.089 ms 94.110 ms]

Extrapolation of length-1048576 codeword in 128 Points/INTT-then-Evaluate/20
                        time:   [117.07 ms 117.21 ms 117.31 ms]

Extrapolation of length-1048576 codeword in 128 Points/Fast Codeword Extrapolation/20
                        time:   [121.87 ms 121.91 ms 121.96 ms]

Extrapolation of length-1048576 codeword in 128 Points/Dispatcher (includes preprocessing)/20
                        time:   [117.14 ms 117.23 ms 117.32 ms]

Extrapolation of length-1048576 codeword in 256 Points/INTT-then-Evaluate/20
                        time:   [121.83 ms 121.86 ms 121.90 ms]

Extrapolation of length-1048576 codeword in 256 Points/Fast Codeword Extrapolation/20
                        time:   [137.01 ms 137.05 ms 137.08 ms]

Extrapolation of length-1048576 codeword in 256 Points/Dispatcher (includes preprocessing)/20
                        time:   [122.18 ms 122.20 ms 122.21 ms]

Formal power series ring inverse/fpsi/2^1
                        time:   [288.95 ns 289.05 ns 289.16 ns]

Formal power series ring inverse/fpsi/2^2
                        time:   [729.12 ns 729.50 ns 729.93 ns]

Formal power series ring inverse/fpsi/2^3
                        time:   [7.8449 µs 7.8508 µs 7.8564 µs]

Formal power series ring inverse/fpsi/2^4
                        time:   [72.651 µs 72.675 µs 72.705 µs]

Formal power series ring inverse/fpsi/2^5
                        time:   [259.89 µs 259.95 µs 260.02 µs]

Formal power series ring inverse/fpsi/2^6
                        time:   [1.2120 ms 1.2122 ms 1.2125 ms]

Formal power series ring inverse/fpsi/2^7
                        time:   [5.5276 ms 5.5289 ms 5.5305 ms]

Formal power series ring inverse/fpsi/2^8
                        time:   [27.100 ms 27.123 ms 27.158 ms]

Formal power series ring inverse/fpsi/2^9
                        time:   [122.12 ms 122.15 ms 122.19 ms]

Formal power series ring inverse/fpsi/2^10
                        time:   [611.94 ms 612.64 ms 613.28 ms]

Various Interpolations in 2^10 Points/Lagrange/10
                        time:   [5.4015 ms 5.4019 ms 5.4025 ms]
                        thrpt:  [189.54 Kelem/s 189.56 Kelem/s 189.58 Kelem/s]

Various Interpolations in 2^10 Points/Fast sequential/10
                        time:   [7.7894 ms 7.7908 ms 7.7924 ms]
                        thrpt:  [131.41 Kelem/s 131.44 Kelem/s 131.46 Kelem/s]

Various Interpolations in 2^10 Points/Dispatcher sequential/10
                        time:   [5.4132 ms 5.4147 ms 5.4157 ms]
                        thrpt:  [189.08 Kelem/s 189.11 Kelem/s 189.17 Kelem/s]

Various Interpolations in 2^10 Points/Fast parallel/10
                        time:   [3.1191 ms 3.1714 ms 3.2361 ms]
                        thrpt:  [316.43 Kelem/s 322.89 Kelem/s 328.30 Kelem/s]

Various Interpolations in 2^10 Points/Dispatcher parallel/10
                        time:   [3.1203 ms 3.1627 ms 3.2099 ms]
                        thrpt:  [319.02 Kelem/s 323.77 Kelem/s 328.17 Kelem/s]

Various Interpolations in 2^15 Points/Fast parallel/15
                        time:   [119.08 ms 120.45 ms 121.17 ms]
                        thrpt:  [270.44 Kelem/s 272.04 Kelem/s 275.17 Kelem/s]

Various Interpolations in 2^15 Points/Dispatcher parallel/15
                        time:   [118.91 ms 121.37 ms 124.53 ms]
                        thrpt:  [263.13 Kelem/s 269.98 Kelem/s 275.57 Kelem/s]

Inverses/Inverse/0      time:   [154.61 ms 154.63 ms 154.66 ms]

merkle_tree/merkle_tree/65536
                        time:   [5.0847 ms 5.1047 ms 5.1240 ms]

merkle_tree_auth_structure_size/auth_structure_size/4194304
                        time:   [3161.7 bfe 3162.2 bfe 3162.7 bfe]

gen_auth_structure      time:   [63.434 µs 63.591 µs 63.749 µs]

verify_auth_structure   time:   [469.49 µs 469.66 µs 469.83 µs]

chu_ntt_forward/bfield/3
                        time:   [72.869 ns 72.906 ns 72.962 ns]
                        thrpt:  [109.65 Melem/s 109.73 Melem/s 109.79 Melem/s]

chu_ntt_forward/bfield/7
                        time:   [1.5605 µs 1.5609 µs 1.5612 µs]
                        thrpt:  [81.989 Melem/s 82.006 Melem/s 82.027 Melem/s]

chu_ntt_forward/bfield/12
                        time:   [80.166 µs 80.188 µs 80.216 µs]
                        thrpt:  [51.062 Melem/s 51.080 Melem/s 51.094 Melem/s]

chu_ntt_forward/bfield/18
                        time:   [8.1458 ms 8.1473 ms 8.1484 ms]
                        thrpt:  [32.171 Melem/s 32.176 Melem/s 32.182 Melem/s]

chu_ntt_forward/bfield/23
                        time:   [495.41 ms 496.69 ms 497.95 ms]
                        thrpt:  [16.846 Melem/s 16.889 Melem/s 16.933 Melem/s]

chu_ntt_forward/xfield/3
                        time:   [122.28 ns 122.36 ns 122.45 ns]
                        thrpt:  [65.331 Melem/s 65.381 Melem/s 65.423 Melem/s]

chu_ntt_forward/xfield/7
                        time:   [3.0461 µs 3.0499 µs 3.0538 µs]
                        thrpt:  [41.916 Melem/s 41.968 Melem/s 42.021 Melem/s]

chu_ntt_forward/xfield/12
                        time:   [159.26 µs 159.48 µs 159.63 µs]
                        thrpt:  [25.660 Melem/s 25.683 Melem/s 25.718 Melem/s]

chu_ntt_forward/xfield/18
                        time:   [15.837 ms 15.861 ms 15.886 ms]
                        thrpt:  [16.501 Melem/s 16.528 Melem/s 16.553 Melem/s]

chu_ntt_forward/xfield/23
                        time:   [968.15 ms 969.42 ms 970.92 ms]
                        thrpt:  [8.6398 Melem/s 8.6532 Melem/s 8.6646 Melem/s]

Clean Division of Polynomials – Dividend Degree: 2^9, Divisor Degree: 2^8/Long/9\|8
                        time:   [89.025 µs 89.049 µs 89.076 µs]

Clean Division of Polynomials – Dividend Degree: 2^9, Divisor Degree: 2^8/Clean/9\|8
                        time:   [92.803 µs 92.826 µs 92.853 µs]

Clean Division of Polynomials – Dividend Degree: 2^9, Divisor Degree: 2^9/Long/9\|9
                        time:   [904.81 ns 905.06 ns 905.35 ns]

Clean Division of Polynomials – Dividend Degree: 2^9, Divisor Degree: 2^9/Clean/9\|9
                        time:   [208.55 µs 208.66 µs 208.77 µs]

Clean Division of Polynomials – Dividend Degree: 2^10, Divisor Degree: 2^8/Long/10\|8
                        time:   [262.18 µs 262.25 µs 262.34 µs]

Clean Division of Polynomials – Dividend Degree: 2^10, Divisor Degree: 2^8/Clean/10\|8
                        time:   [274.16 µs 274.24 µs 274.33 µs]

Clean Division of Polynomials – Dividend Degree: 2^10, Divisor Degree: 2^9/Long/10\|9
                        time:   [345.76 µs 345.84 µs 345.92 µs]

Clean Division of Polynomials – Dividend Degree: 2^10, Divisor Degree: 2^9/Clean/10\|9
                        time:   [427.35 µs 427.52 µs 427.70 µs]

Clean Division of Polynomials – Dividend Degree: 2^10, Divisor Degree: 2^10/Long/10\|10
                        time:   [1.6503 µs 1.6507 µs 1.6512 µs]

Clean Division of Polynomials – Dividend Degree: 2^10, Divisor Degree: 2^10/Clean/10\|10
                        time:   [433.76 µs 433.85 µs 433.95 µs]

Clean Division of Polynomials – Dividend Degree: 2^11, Divisor Degree: 2^8/Long/11\|8
                        time:   [612.07 µs 612.21 µs 612.38 µs]

Clean Division of Polynomials – Dividend Degree: 2^11, Divisor Degree: 2^8/Clean/11\|8
                        time:   [639.80 µs 639.96 µs 640.13 µs]

Clean Division of Polynomials – Dividend Degree: 2^11, Divisor Degree: 2^9/Long/11\|9
                        time:   [1.0362 ms 1.0366 ms 1.0372 ms]

Clean Division of Polynomials – Dividend Degree: 2^11, Divisor Degree: 2^9/Clean/11\|9
                        time:   [873.52 µs 873.72 µs 873.94 µs]

Clean Division of Polynomials – Dividend Degree: 2^11, Divisor Degree: 2^10/Long/11\|10
                        time:   [1.3723 ms 1.3727 ms 1.3731 ms]

Clean Division of Polynomials – Dividend Degree: 2^11, Divisor Degree: 2^10/Clean/11\|10
                        time:   [889.33 µs 889.57 µs 889.83 µs]

Clean Division of Polynomials – Dividend Degree: 2^11, Divisor Degree: 2^11/Long/11\|11
                        time:   [3.0543 µs 3.0554 µs 3.0567 µs]

Clean Division of Polynomials – Dividend Degree: 2^11, Divisor Degree: 2^11/Clean/11\|11
                        time:   [905.14 µs 905.34 µs 905.54 µs]

Clean Division of Polynomials – Dividend Degree: 2^12, Divisor Degree: 2^8/Long/12\|8
                        time:   [1.3097 ms 1.3100 ms 1.3103 ms]

Clean Division of Polynomials – Dividend Degree: 2^12, Divisor Degree: 2^8/Clean/12\|8
                        time:   [1.3682 ms 1.3686 ms 1.3690 ms]

Clean Division of Polynomials – Dividend Degree: 2^12, Divisor Degree: 2^9/Long/12\|9
                        time:   [2.4147 ms 2.4153 ms 2.4159 ms]

Clean Division of Polynomials – Dividend Degree: 2^12, Divisor Degree: 2^9/Clean/12\|9
                        time:   [1.8268 ms 1.8283 ms 1.8301 ms]

Clean Division of Polynomials – Dividend Degree: 2^12, Divisor Degree: 2^11/Long/12\|11
                        time:   [5.4713 ms 5.4724 ms 5.4738 ms]

Clean Division of Polynomials – Dividend Degree: 2^12, Divisor Degree: 2^11/Clean/12\|11
                        time:   [1.8533 ms 1.8538 ms 1.8544 ms]

Clean Division of Polynomials – Dividend Degree: 2^12, Divisor Degree: 2^12/Long/12\|12
                        time:   [6.1305 µs 6.1319 µs 6.1335 µs]

Clean Division of Polynomials – Dividend Degree: 2^12, Divisor Degree: 2^12/Clean/12\|12
                        time:   [1.8847 ms 1.8857 ms 1.8868 ms]

Clean Division of Polynomials – Dividend Degree: 2^14, Divisor Degree: 2^8/Long/14\|8
                        time:   [5.5329 ms 5.5343 ms 5.5360 ms]

Clean Division of Polynomials – Dividend Degree: 2^14, Divisor Degree: 2^8/Clean/14\|8
                        time:   [5.7459 ms 5.7472 ms 5.7487 ms]

Clean Division of Polynomials – Dividend Degree: 2^14, Divisor Degree: 2^9/Long/14\|9
                        time:   [10.700 ms 10.702 ms 10.705 ms]

Clean Division of Polynomials – Dividend Degree: 2^14, Divisor Degree: 2^9/Clean/14\|9
                        time:   [8.8220 ms 8.8345 ms 8.8519 ms]

Clean Division of Polynomials – Dividend Degree: 2^14, Divisor Degree: 2^13/Long/14\|13
                        time:   [87.523 ms 87.539 ms 87.557 ms]

Clean Division of Polynomials – Dividend Degree: 2^14, Divisor Degree: 2^13/Clean/14\|13
                        time:   [9.0440 ms 9.0542 ms 9.0680 ms]

Clean Division of Polynomials – Dividend Degree: 2^14, Divisor Degree: 2^14/Long/14\|14
                        time:   [23.579 µs 23.584 µs 23.591 µs]

Clean Division of Polynomials – Dividend Degree: 2^14, Divisor Degree: 2^14/Clean/14\|14
                        time:   [9.2469 ms 9.2560 ms 9.2696 ms]

Modular reduction of degree 1048575 by degree 15/long division/20
                        time:   [34.455 ms 34.469 ms 34.485 ms]

Modular reduction of degree 1048575 by degree 15/fast reduce/20
                        time:   [36.009 ms 36.025 ms 36.043 ms]

Modular reduction of degree 1048575 by degree 31/long division/20
                        time:   [56.624 ms 56.664 ms 56.722 ms]

Modular reduction of degree 1048575 by degree 31/fast reduce/20
                        time:   [38.500 ms 38.511 ms 38.523 ms]

Modular reduction of degree 1048575 by degree 63/long division/20
                        time:   [106.11 ms 106.24 ms 106.43 ms]

Modular reduction of degree 1048575 by degree 63/fast reduce/20
                        time:   [45.249 ms 45.269 ms 45.287 ms]

Modular reduction of degree 1048575 by degree 127/long division/20
                        time:   [194.67 ms 194.72 ms 194.77 ms]

Modular reduction of degree 1048575 by degree 127/fast reduce/20
                        time:   [66.765 ms 66.808 ms 66.852 ms]

Modular reduction of degree 1048575 by degree 255/long division/20
                        time:   [373.26 ms 373.44 ms 373.68 ms]

Modular reduction of degree 1048575 by degree 255/fast reduce/20
                        time:   [71.743 ms 71.760 ms 71.778 ms]

Multiplication of Polynomials of Degree 2^7 (Product Degree: 2^8)/Naïve/8
                        time:   [22.749 µs 22.754 µs 22.760 µs]

Multiplication of Polynomials of Degree 2^7 (Product Degree: 2^8)/Fast/8
                        time:   [25.177 µs 25.186 µs 25.197 µs]

Multiplication of Polynomials of Degree 2^7 (Product Degree: 2^8)/Faster of the two/8
                        time:   [25.196 µs 25.204 µs 25.212 µs]

Multiplication of Polynomials of Degree 2^8 (Product Degree: 2^9)/Naïve/9
                        time:   [104.92 µs 104.95 µs 104.99 µs]

Multiplication of Polynomials of Degree 2^8 (Product Degree: 2^9)/Fast/9
                        time:   [54.888 µs 54.901 µs 54.915 µs]

Multiplication of Polynomials of Degree 2^8 (Product Degree: 2^9)/Faster of the two/9
                        time:   [54.968 µs 54.983 µs 55.000 µs]

Multiplication of Polynomials of Degree 2^9 (Product Degree: 2^10)/Naïve/10
                        time:   [372.17 µs 372.26 µs 372.37 µs]

Multiplication of Polynomials of Degree 2^9 (Product Degree: 2^10)/Fast/10
                        time:   [119.97 µs 120.00 µs 120.03 µs]

Multiplication of Polynomials of Degree 2^9 (Product Degree: 2^10)/Faster of the two/10
                        time:   [119.96 µs 119.99 µs 120.02 µs]

Multiplication of Polynomial of Degree 2^1 with a Scalar/Mut/1
                        time:   [4.8983 ns 4.9052 ns 4.9131 ns]

Multiplication of Polynomial of Degree 2^1 with a Scalar/Immut/1
                        time:   [11.421 ns 11.457 ns 11.490 ns]

Multiplication of Polynomial of Degree 2^7 with a Scalar/Mut/7
                        time:   [124.60 ns 124.64 ns 124.68 ns]

Multiplication of Polynomial of Degree 2^7 with a Scalar/Immut/7
                        time:   [128.62 ns 128.69 ns 128.76 ns]

Multiplication of Polynomial of Degree 2^13 with a Scalar/Mut/13
                        time:   [7.5136 µs 7.5157 µs 7.5180 µs]

Multiplication of Polynomial of Degree 2^13 with a Scalar/Immut/13
                        time:   [7.4959 µs 7.4977 µs 7.4998 µs]

Scale Polynomials of Degree 2^5/bfe poly, bfe scalar/5
                        time:   [93.741 ns 93.779 ns 93.817 ns]

Scale Polynomials of Degree 2^5/bfe poly, xfe scalar/5
                        time:   [584.85 ns 585.84 ns 586.96 ns]

Scale Polynomials of Degree 2^5/xfe poly, bfe scalar/5
                        time:   [159.18 ns 159.35 ns 159.53 ns]

Scale Polynomials of Degree 2^5/xfe poly, xfe scalar/5
                        time:   [826.04 ns 826.80 ns 827.57 ns]

Scale Polynomials of Degree 2^10/bfe poly, bfe scalar/10
                        time:   [2.7450 µs 2.7457 µs 2.7465 µs]

Scale Polynomials of Degree 2^10/bfe poly, xfe scalar/10
                        time:   [17.960 µs 17.978 µs 18.000 µs]

Scale Polynomials of Degree 2^10/xfe poly, bfe scalar/10
                        time:   [4.5876 µs 4.5891 µs 4.5908 µs]

Scale Polynomials of Degree 2^10/xfe poly, xfe scalar/10
                        time:   [25.567 µs 25.578 µs 25.590 µs]

Scale Polynomials of Degree 2^15/bfe poly, bfe scalar/15
                        time:   [87.178 µs 87.196 µs 87.218 µs]

Scale Polynomials of Degree 2^15/bfe poly, xfe scalar/15
                        time:   [574.35 µs 575.03 µs 575.83 µs]

Scale Polynomials of Degree 2^15/xfe poly, bfe scalar/15
                        time:   [145.36 µs 145.40 µs 145.45 µs]

Scale Polynomials of Degree 2^15/xfe poly, xfe scalar/15
                        time:   [815.16 µs 815.37 µs 815.59 µs]

Scale Polynomials of Degree 2^20/bfe poly, bfe scalar/20
                        time:   [2.7843 ms 2.7849 ms 2.7857 ms]

Scale Polynomials of Degree 2^20/bfe poly, xfe scalar/20
                        time:   [18.403 ms 18.425 ms 18.447 ms]

Scale Polynomials of Degree 2^20/xfe poly, bfe scalar/20
                        time:   [4.6954 ms 4.6968 ms 4.6984 ms]

Scale Polynomials of Degree 2^20/xfe poly, xfe scalar/20
                        time:   [26.024 ms 26.040 ms 26.056 ms]

polynomial coset of degree 2^10/coset-evaluate bfe-pol/1024
                        time:   [20.313 µs 20.320 µs 20.328 µs]
                        thrpt:  [50.373 Melem/s 50.394 Melem/s 50.410 Melem/s]

polynomial coset of degree 2^10/coset-evaluate xfe-pol/1024
                        time:   [40.413 µs 40.423 µs 40.434 µs]
                        thrpt:  [25.325 Melem/s 25.332 Melem/s 25.338 Melem/s]

polynomial coset of degree 2^10/coset-interpolate bfe-pol/1024
                        time:   [22.047 µs 22.051 µs 22.056 µs]
                        thrpt:  [46.427 Melem/s 46.439 Melem/s 46.446 Melem/s]

polynomial coset of degree 2^10/coset-interpolate xfe-pol/1024
                        time:   [49.297 µs 49.310 µs 49.329 µs]
                        thrpt:  [20.759 Melem/s 20.767 Melem/s 20.772 Melem/s]

polynomial coset of degree 2^17/coset-evaluate bfe-pol/131072
                        time:   [4.2854 ms 4.2867 ms 4.2880 ms]
                        thrpt:  [30.567 Melem/s 30.576 Melem/s 30.586 Melem/s]

polynomial coset of degree 2^17/coset-evaluate xfe-pol/131072
                        time:   [8.5407 ms 8.5585 ms 8.5753 ms]
                        thrpt:  [15.285 Melem/s 15.315 Melem/s 15.347 Melem/s]

polynomial coset of degree 2^17/coset-interpolate bfe-pol/131072
                        time:   [4.4593 ms 4.4600 ms 4.4606 ms]
                        thrpt:  [29.384 Melem/s 29.388 Melem/s 29.393 Melem/s]

polynomial coset of degree 2^17/coset-interpolate xfe-pol/131072
                        time:   [11.132 ms 11.146 ms 11.155 ms]
                        thrpt:  [11.750 Melem/s 11.760 Melem/s 11.774 Melem/s]

tip5/hash_10/Tip5 / Hash 10/10
                        time:   [532.92 ns 533.09 ns 533.30 ns]

tip5/hash_pair/Tip5 / Hash Pair/pair
                        time:   [538.37 ns 538.79 ns 539.19 ns]

tip5/hash_varlen/Tip5 / Hash Variable Length/16384
                        time:   [1.0785 ms 1.0792 ms 1.0798 ms]

tip5/parallel/Tip5 / Parallel Hash/65536
                        time:   [1.1070 ms 1.1639 ms 1.2350 ms]

mul/nop/10              time:   [2.4289 ns 2.4295 ns 2.4302 ns]
                        thrpt:  [4.1149 Gelem/s 4.1161 Gelem/s 4.1171 Gelem/s]

mul/nop/100             time:   [26.514 ns 26.543 ns 26.597 ns]
                        thrpt:  [3.7598 Gelem/s 3.7675 Gelem/s 3.7716 Gelem/s]

mul/nop/1000            time:   [224.55 ns 224.61 ns 224.65 ns]
                        thrpt:  [4.4513 Gelem/s 4.4522 Gelem/s 4.4533 Gelem/s]

mul/nop/1000000         time:   [220.86 µs 220.98 µs 221.25 µs]
                        thrpt:  [4.5198 Gelem/s 4.5253 Gelem/s 4.5277 Gelem/s]

mul/(u32,u32)->u64/10   time:   [4.5478 ns 4.5496 ns 4.5521 ns]
                        thrpt:  [2.1968 Gelem/s 2.1980 Gelem/s 2.1989 Gelem/s]

mul/(u32,u32)->u64/100  time:   [47.921 ns 47.934 ns 47.946 ns]
                        thrpt:  [2.0857 Gelem/s 2.0862 Gelem/s 2.0868 Gelem/s]

mul/(u32,u32)->u64/1000 time:   [446.03 ns 446.12 ns 446.25 ns]
                        thrpt:  [2.2409 Gelem/s 2.2416 Gelem/s 2.2420 Gelem/s]

mul/(u32,u32)->u64/1000000
                        time:   [442.35 µs 442.42 µs 442.51 µs]
                        thrpt:  [2.2598 Gelem/s 2.2603 Gelem/s 2.2606 Gelem/s]

mul/(u64,u64)->u128/10  time:   [6.1824 ns 6.1844 ns 6.1860 ns]
                        thrpt:  [1.6165 Gelem/s 1.6170 Gelem/s 1.6175 Gelem/s]

mul/(u64,u64)->u128/100 time:   [70.664 ns 70.678 ns 70.697 ns]
                        thrpt:  [1.4145 Gelem/s 1.4149 Gelem/s 1.4151 Gelem/s]

mul/(u64,u64)->u128/1000
                        time:   [667.08 ns 667.27 ns 667.40 ns]
                        thrpt:  [1.4983 Gelem/s 1.4987 Gelem/s 1.4991 Gelem/s]

mul/(u64,u64)->u128/1000000
                        time:   [663.87 µs 664.07 µs 664.26 µs]
                        thrpt:  [1.5054 Gelem/s 1.5059 Gelem/s 1.5063 Gelem/s]

mul/(BFE,BFE)->BFE/10   time:   [9.9365 ns 9.9392 ns 9.9419 ns]
                        thrpt:  [1.0058 Gelem/s 1.0061 Gelem/s 1.0064 Gelem/s]

mul/(BFE,BFE)->BFE/100  time:   [113.93 ns 113.99 ns 114.13 ns]
                        thrpt:  [876.21 Melem/s 877.24 Melem/s 877.74 Melem/s]

mul/(BFE,BFE)->BFE/1000 time:   [1.1083 µs 1.1085 µs 1.1086 µs]
                        thrpt:  [902.01 Melem/s 902.12 Melem/s 902.26 Melem/s]

mul/(BFE,BFE)->BFE/1000000
                        time:   [1.1067 ms 1.1069 ms 1.1071 ms]
                        thrpt:  [903.24 Melem/s 903.40 Melem/s 903.56 Melem/s]

mul/(XFE,XFE)->XFE/10   time:   [85.458 ns 85.477 ns 85.499 ns]
                        thrpt:  [116.96 Melem/s 116.99 Melem/s 117.02 Melem/s]

mul/(XFE,XFE)->XFE/100  time:   [943.98 ns 944.26 ns 944.53 ns]
                        thrpt:  [105.87 Melem/s 105.90 Melem/s 105.93 Melem/s]

mul/(XFE,XFE)->XFE/1000 time:   [9.4964 µs 9.4978 µs 9.4992 µs]
                        thrpt:  [105.27 Melem/s 105.29 Melem/s 105.30 Melem/s]

mul/(XFE,XFE)->XFE/1000000
                        time:   [9.5169 ms 9.5245 ms 9.5360 ms]
                        thrpt:  [104.87 Melem/s 104.99 Melem/s 105.08 Melem/s]

mul/(XFE,BFE)->XFE/10   time:   [28.265 ns 28.275 ns 28.281 ns]
                        thrpt:  [353.59 Melem/s 353.67 Melem/s 353.79 Melem/s]

mul/(XFE,BFE)->XFE/100  time:   [306.51 ns 306.57 ns 306.61 ns]
                        thrpt:  [326.14 Melem/s 326.19 Melem/s 326.25 Melem/s]

mul/(XFE,BFE)->XFE/1000 time:   [3.0958 µs 3.0963 µs 3.0968 µs]
                        thrpt:  [322.92 Melem/s 322.96 Melem/s 323.01 Melem/s]

mul/(XFE,BFE)->XFE/1000000
                        time:   [3.1266 ms 3.1280 ms 3.1292 ms]
                        thrpt:  [319.57 Melem/s 319.69 Melem/s 319.84 Melem/s]

Various Zerofiers with 0 Roots/Naïve/0
                        time:   [9.0537 ns 9.0557 ns 9.0582 ns]

Various Zerofiers with 0 Roots/Smart/0
                        time:   [10.266 ns 10.280 ns 10.294 ns]

Various Zerofiers with 0 Roots/Fast/0
                        time:   [37.149 ns 37.184 ns 37.227 ns]

Various Zerofiers with 0 Roots/Dispatcher/0
                        time:   [10.233 ns 10.258 ns 10.288 ns]

Various Zerofiers with 10 Roots/Naïve/10
                        time:   [394.51 ns 394.79 ns 395.13 ns]

Various Zerofiers with 10 Roots/Smart/10
                        time:   [102.08 ns 102.13 ns 102.20 ns]

Various Zerofiers with 10 Roots/Fast/10
                        time:   [149.55 ns 149.72 ns 149.89 ns]

Various Zerofiers with 10 Roots/Dispatcher/10
                        time:   [102.00 ns 102.04 ns 102.09 ns]

Various Zerofiers with 100 Roots/Naïve/100
                        time:   [20.041 µs 20.047 µs 20.054 µs]

Various Zerofiers with 100 Roots/Smart/100
                        time:   [7.7972 µs 7.8013 µs 7.8054 µs]

Various Zerofiers with 100 Roots/Fast/100
                        time:   [7.8228 µs 7.8336 µs 7.8463 µs]

Various Zerofiers with 100 Roots/Dispatcher/100
                        time:   [7.8590 µs 7.8640 µs 7.8694 µs]

Various Zerofiers with 200 Roots/Naïve/200
                        time:   [76.620 µs 76.646 µs 76.675 µs]

Various Zerofiers with 200 Roots/Smart/200
                        time:   [100.79 µs 100.83 µs 100.87 µs]

Various Zerofiers with 200 Roots/Fast/200
                        time:   [29.393 µs 29.409 µs 29.426 µs]

Various Zerofiers with 200 Roots/Dispatcher/200
                        time:   [29.800 µs 29.809 µs 29.819 µs]

Various Zerofiers with 500 Roots/Naïve/500
                        time:   [461.57 µs 461.69 µs 461.82 µs]

Various Zerofiers with 500 Roots/Smart/500
                        time:   [738.14 µs 738.30 µs 738.47 µs]

Various Zerofiers with 500 Roots/Fast/500
                        time:   [164.67 µs 164.77 µs 164.92 µs]

Various Zerofiers with 500 Roots/Dispatcher/500
                        time:   [163.67 µs 163.73 µs 163.78 µs]

Various Zerofiers with 700 Roots/Naïve/700
                        time:   [888.76 µs 889.01 µs 889.29 µs]

Various Zerofiers with 700 Roots/Smart/700
                        time:   [1.4549 ms 1.4552 ms 1.4555 ms]

Various Zerofiers with 700 Roots/Fast/700
                        time:   [319.08 µs 319.15 µs 319.23 µs]

Various Zerofiers with 700 Roots/Dispatcher/700
                        time:   [318.89 µs 318.98 µs 319.07 µs]

Various Zerofiers with 1000 Roots/Naïve/1000
                        time:   [1.7864 ms 1.7868 ms 1.7872 ms]

Various Zerofiers with 1000 Roots/Smart/1000
                        time:   [2.9776 ms 2.9790 ms 2.9811 ms]

Various Zerofiers with 1000 Roots/Fast/1000
                        time:   [413.93 µs 414.04 µs 414.16 µs]

Various Zerofiers with 1000 Roots/Dispatcher/1000
                        time:   [419.75 µs 419.84 µs 419.95 µs]

Various Zerofiers with 10000 Roots/Naïve/10000
                        time:   [173.04 ms 173.06 ms 173.09 ms]

Various Zerofiers with 10000 Roots/Smart/10000
                        time:   [296.44 ms 296.47 ms 296.50 ms]

Various Zerofiers with 10000 Roots/Fast/10000
                        time:   [8.9665 ms 8.9687 ms 8.9713 ms]

Various Zerofiers with 10000 Roots/Dispatcher/10000
                        time:   [8.9582 ms 8.9601 ms 8.9623 ms]

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
