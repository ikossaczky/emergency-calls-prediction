#!/bin/bash

INPUT_PATH="./data/Seattle_Real_Time_Fire_911_Calls.csv"
OUTPUT_PATH="./data/dataset.csv"

# string for extracting data with sed
SEDSTR='s_.*([0-9]{2}/[0-9]{2}/[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} (A|P)M)(.*)_\1_'

# format of out time features
FEATURE_FORMAT=',%Y,%m,%d,%H,%u,%V,%q'

# setup output file with header
echo "count,year,month,day,hour,week,weekday,quarter" > "$OUTPUT_PATH"

# load data except the header line, extract date with sed, create features with date,
# accumulate (groupby count equivalent) with uniq -c, remove leading zeros with sed and space chars with tr
tail -n +2 "$INPUT_PATH" | \
 sed -E "$SEDSTR" | \
 TZ=UTC date -f- +"$FEATURE_FORMAT" | \
 sort | uniq -c | \
 sed -E 's/,0([0-9])/,\1/g' | tr -d " \t"  >> "$OUTPUT_PATH"
