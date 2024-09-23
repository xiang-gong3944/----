# set terminal postscript eps enhanced color 28 lw 2
# set output "band.eps"
set terminal png enhanced lw 2 font ",20"
set output "band.png"
# set ylabel 'Energy (eV)'

# set ytics 5

unset key

x1 = 0.66666666
x2 = 1
xmax = 3.9873
ymin = -160
ymax = 60
ef = 156.1658

set xrange [0:xmax]
set yrange [ymin:ymax]
set xtics ("L" 0, "{/Symbol G}" x1, "X" x2, "{/Symbol G}" xmax)
set arrow 1 nohead from x1,ymin to x1,ymax lt 2 lc "black"
set arrow 2 nohead from x2,ymin to x2,ymax lt 2 lc "black"
set arrow 3 nohead from 0, 0 to xmax, 0 lt 2 lc "black"

plot 'Ag.band.gnu' using 1:($2-ef) w l