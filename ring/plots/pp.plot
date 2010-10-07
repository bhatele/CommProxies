set xlabel "Message Size (Bytes)"
set ylabel "Latency (us)"
set title "Pingpong benchmark on 2 nodes (xt5)"
set logscale x 2
set logscale y 10
set key top left
set xtics ("4" 4, "16" 16, "64" 64, "256" 256, "1K" 1024, "4K" 4096, "16K" 16384, "64K" 65536, "256K" 262144, "1M" 1048576)

set terminal png enhanced
set output "xt5_pp_2.png"

set style line 1 lt rgb "black" lw 3 pt 2
set style line 2 lt rgb "magenta" lw 3 pt 4
set style line 3 lt rgb "grey" lw 3 pt 6
set style line 4 lt rgb "yellow" lw 3 pt 8
set style line 5 lt rgb "green" lw 3 pt 2
set style line 6 lt rgb "blue" lw 3 pt 4
set style line 7 lt rgb "red" lw 3 pt 6

plot "xt5.pp.ampi.2.out" using 1:($2*1e6) title "AMPI" with linespoints ls 2, \
"xt5.pp.mpi.2.out" using 1:($2*1e6) title "MPI" with linespoints ls 1
