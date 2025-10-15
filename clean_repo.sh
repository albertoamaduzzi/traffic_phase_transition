# 1. Navigate to parent directory
cd /home/alberto/LPSim

# 2. Make sure your current repo has all changes committed
cd traffic_phase_transition
git add .
git commit -m "Save all current changes before cleaning"

# 3. Try to push (this will likely fail due to size, but we need to try)
git push origin main || git push origin master

# 4. Make a backup
#cd /home/alberto/LPSim
#cp -r traffic_phase_transition traffic_phase_transition_backup

# 5. Clean the current repo with --force
#cd traffic_phase_transition

# Clean up garbage objects first
git gc --prune=now --aggressive

# Remove large files from history with --force
git filter-repo --path data_2/ --invert-paths --force
git filter-repo --path MemoryUsage/ --invert-paths --force
git filter-repo --path cache/ --invert-paths --force
git filter-repo --path data/tiff_files/ --invert-paths --force
git filter-repo --path-glob '*.parquet' --invert-paths --force
git filter-repo --path-glob '*.tif' --invert-paths --force
git filter-repo --path-glob '*.gpkg' --invert-paths --force
git filter-repo --path-glob '*.gz' --invert-paths --force

# 6. Re-add the remote (filter-repo removes it)
git remote add origin https://github.com/albertoamaduzzi/traffic_phase_transition.git

# 7. Force push the cleaned history
git push origin --force --all

# 8. Verify the size reduction
echo "Repository size:"
du -sh .git/