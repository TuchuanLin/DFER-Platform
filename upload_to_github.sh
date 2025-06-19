#!/bin/bash

echo "============================================================"
echo "DFER-Platform GitHub Upload Script"
echo "============================================================"
echo

echo "Initializing Git repository..."
git init

echo "Adding all files..."
git add .

echo "Creating initial commit..."
git commit -m "Initial commit: MICACL Dynamic Facial Expression Recognition Platform"

echo "Adding remote repository..."
git remote add origin git@github.com:TuchuanLin/DFER-Platform.git

echo "Pushing to GitHub..."
git push -u origin main

echo
echo "============================================================"
echo "Upload completed! Your project is now available at:"
echo "https://github.com/TuchuanLin/DFER-Platform"
echo "============================================================" 