#!/bin/bash

cd /users/yhb18174/Recreating_DMTA/

git checkout stage
git add .
git commit -m "Automatic commit on $(date +"%Y-%m-%d %H:%M")"

# Attempt to push, and capture the exit status.
if git push -f origin stage; then
    subject="Automated git backup successful"
    body="Recreating DMTA backup successful"
else
    subject="Automated git backup FAILED"
    body="Recreating DMTA backup failed. Check logs for details."
fi

recipient="huw.williams.2018@uni.strath.ac.uk"
echo "$body" | mail -s "$subject" "$recipient"
