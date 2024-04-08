#!/bin/bash


for i in {0..50}
do
    rAB=$(echo "scale=4; 0+$i*-0.25" |bc)
    echo $rAB
    python3 bose_hubbard_ed.py three_site edge_site_left 2 2.0 $rAB 0 5000 3001
    python3 bose_hubbard_ed.py three_site edge_site_left 2 1.0 $rAB 0 5000 3001
    python3 bose_hubbard_ed.py three_site edge_site_left 1 0.0 $rAB 0 5000 3001
done
