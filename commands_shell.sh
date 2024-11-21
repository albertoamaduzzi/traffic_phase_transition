# Check usage memory by user
ps -eo user,%mem --sort=-%mem | awk 'NR>1 {a[$1]+=$2} END {for (i in a) print i, a[i]}' | sort -k2 -nr
# Check free memory available
awk '/MemFree/ { printf "%.3f \n", $2/1024/1024 }' /proc/meminfo

!/bin/bash

# Get total memory in kilobytes
total_mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)

# Convert total memory to gigabytes
total_mem_gb=$(echo "scale=3; $total_mem_kb / 1024 / 1024" | bc)

# Check memory used by users in gigabytes
ps -eo user,%mem --sort=-%mem | awk -v total_mem_gb="$total_mem_gb" '
NR>1 {a[$1]+=$2} 
END {
    for (i in a) 
        printf "%s %.3f\n", i, a[i] * total_mem_gb / 100
}' | sort -k2 -nr