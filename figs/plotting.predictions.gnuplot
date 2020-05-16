#!/usr/bin/gnuplot
set term qt persist size 1200,800
set auto
#plot '../data_fs/raw/archive.csv' u 3:4
set datafile separator "\s"
plot '../data_fs/raw/archive.csv.original' u 1:5 w points pt 7 ,'../data_fs/raw/archive.csv.predictions' u 1:5 w lines
