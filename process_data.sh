#!/bin/bash

INPUT_PATH="./Seattle_Real_Time_Fire_911_Calls.csv"
OUTPUT_PATH="./dataset.txt"

# string for extracting dat with sed
SEDSTR='s_.*([0-9]{2}/[0-9]{2}/[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} (A|P)M)(.*)_\1_'

# format of out time features
FEATURE_FORMAT=',%Y,%m,%d,%H,%u,%V,%q'

# print header
echo "count,year,month,day,hour,week,weekday,quarter" > "$OUTPUT_PATH"

# extract date with sed, create features with date, accumulate with uniq -c and remove leading zeros with sed
tail -n +2 "$INPUT_PATH" | sed -E "$SEDSTR" | TZ=UTC date -f- +"$FEATURE_FORMAT" | sort | uniq -c | sed -E 's/,0([0-9])/,\1/g' >> "$OUTPUT_PATH"
