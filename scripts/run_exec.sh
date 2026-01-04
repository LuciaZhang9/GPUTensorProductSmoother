#!/bin/bash                                                                                                                           

cmake .
make release

# for p in {1..10}
# do
#     echo "/////////////////////"
#     echo "Starting 2D degree $p"
#     echo "/////////////////////"
#     python3 scripts/ct_parameter.py -DIM 2 -DEG $p -MAXSIZE 200000000 -REDUCE 1e-8
#     make poisson
#     ./apps/poisson
# done

for p in {2..8}
do
    echo "/////////////////////"
    echo "Starting 3D degree $p"
    echo "/////////////////////"
    python3 scripts/ct_parameter.py -DIM 3 -DEG $p -MAXSIZE 5000000000
    make poisson
    ./apps/poisson
done