#!/bin/bash

NOW_DATE_EPOCH=$(date +%s)

LATEST_FILE=$(ls -t /ws/lidar_data/combined_pcd/*)
UPDATE_FILE_DATE_EPOCH=$(date -r "$LATEST_FILE" +%s)

DIFF=$((NOW_DATE_EPOCH - UPDATE_FILE_DATE_EPOCH))

readonly REBOOT_INFO_LOG_FILE='/ws/reboot_info.log'

if [ $DIFF -gt 60 ]; then
    echo $(date '+%Y:%m:%d:%H:%M') >> $REBOOT_INFO_LOG_FILE
    pkill -f roslaunch
    sleep 5
    source /ws/ws_livox/devel/setup.bash
    screen -d -m roslaunch livox_ros_driver livox_lidar.launch
fi

exit 0