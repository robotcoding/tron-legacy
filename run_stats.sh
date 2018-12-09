#!/bin/sh
WEIGHT=0.25
for i in {1..5}; do
    echo "doin one"
    OLD_WEIGHT=$WEIGHT
    #$WEIGHT=$(( $OLD_WEIGHT + 0.15 ))
    WEIGHT=`echo "$OLD_WEIGHT + 0.15" | bc -l`
    #$(cat bots.py | sed 's/ART_WT/ = $OLD_WEIGHT/ART_WT = $NEW_WEIGHT)') > bots.py
    cat bots.py | sed 's/ART_WT = $OLD_WEIGHT/ART_WT = $NEW_WEIGHT/' > bots.py
    python weights_tester.py > result_$WEIGHT.txt 
done
