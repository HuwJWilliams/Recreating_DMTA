#!/bin/bash

# === Logging for Debugging ===
logfile="/tmp/git_backup_debug.log"
{
    echo "=== Backup run at $(date) ==="
    echo "User: $(whoami)"
    echo "Current dir: $(pwd)"
    echo "Script location: $(realpath "$0")"
    echo "Environment PATH: $PATH"
} > "$logfile"  # Overwrite log file on each run

# === Navigate to repo directory ===
cd /users/yhb18174/Recreating_DMTA/ || {
    echo "Failed to cd into project directory" >> "$logfile"
    subject="❌ Git Backup FAILED"
    body="Failed to cd into the project directory."
    sendlog=true
    goto email
}

# === Git operations ===
{
    git checkout stage
    git add .
    git commit -m "Automatic commit on $(date +"%Y-%m-%d %H:%M")"
} >> "$logfile" 2>&1

# === Attempt to push ===
if git push -f origin stage >> "$logfile" 2>&1; then
    subject="✅ Automated Git Backup Successful"
    body="Recreating DMTA backup was successful on $(date)."
else
    subject="❌ Automated Git Backup FAILED"
    body="Git push failed on $(date). See log below."
fi

# === Send Email (summary + log) ===
recipient="huw.williams.2018@uni.strath.ac.uk"
{
    echo "$body"
    echo ""
    echo "---- Debug Log ----"
    cat "$logfile"
} | mail -s "$subject" "$recipient"