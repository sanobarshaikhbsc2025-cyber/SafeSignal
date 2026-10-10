#!/usr/bin/env bash
# Usage: bash setup_github.sh <your-github-username>
# First create an EMPTY repo named "SafeSignal" on github.com (no README), then run this.
set -e
USER_NAME="${1:?Give your GitHub username}"
git init
git add .
git commit -m "Initial commit: SafeSignal NLP toolkit"
git branch -M main
git remote add origin "https://github.com/${USER_NAME}/SafeSignal.git"
git push -u origin main
echo "Done: https://github.com/${USER_NAME}/SafeSignal"
