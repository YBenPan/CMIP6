#!/bin/tcsh

python3 map.py PM25 &
# python3 map.py "Baseline Mortality" &
# python3 map.py Population &
# python3 map.py "Population Size" &
# python3 map.py Aging

python3 map.py "ssp126" &
python3 map.py "ssp245" &
python3 map.py "ssp370" &
python3 map.py "ssp585"