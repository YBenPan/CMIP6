#!/bin/tcsh -f

python3 decomposition.py SSP GBD_super absolute & 
python3 decomposition.py SSP GBD_super pct &
# python3 decomposition.py SSP SDI absolute &
# python3 decomposition.py SSP SDI pct &
# python3 decomposition.py SSP GBD absolute &
# python3 decomposition.py SSP GBD pct &
python3 decomposition.py Disease GBD_super absolute & 
python3 decomposition.py Disease GBD_super pct &
# python3 decomposition.py Disease SDI absolute &
# python3 decomposition.py Disease SDI pct &
# python3 decomposition.py Disease GBD absolute &
# python3 decomposition.py Disease GBD pct