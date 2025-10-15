cd /home/alberto/LPSim/traffic_phase_transition

# Find the 50 largest objects
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  sed -n 's/^blob //p' | \
  sort -k2 -n -r | \
  head -50 | \
  awk '{printf "%.2f MB\t%s\n", $2/1024/1024, $3}'

# Remove all Jupyter notebooks (they often have large outputs)
git filter-repo --path-glob '*.ipynb' --invert-paths --force

# Remove all images and plots
git filter-repo --path-glob '*.png' --invert-paths --force
git filter-repo --path-glob '*.jpg' --invert-paths --force
git filter-repo --path-glob '*.pdf' --invert-paths --force
git filter-repo --path-glob '*.svg' --invert-paths --force

# Remove citations folder
git filter-repo --path citations/ --invert-paths --force

# Remove data folder entirely
git filter-repo --path data/ --invert-paths --force

# Remove config JSONs
git filter-repo --path config/ --invert-paths --force


# Remove Jupyter notebooks with large outputs
git filter-repo --path-glob '*.ipynb' --invert-paths --force

# Remove PDF files (citations folder)
git filter-repo --path citations/ --invert-paths --force

# Remove specific large JSON files
git filter-repo --path-glob '*_geometric_info.json' --invert-paths --force

# Remove any CSV files from history
git filter-repo --path-glob '*.csv' --invert-paths --force
# Remove the large JSON file from history
git filter-repo --path tests/cache/cea1bc73daf7492add457a96834e980d587694a5.json --invert-paths --force

# Remove entire tests/cache directory to be safe
git filter-repo --path tests/cache/ --invert-paths --force

# Clean up ambiguous refs
git config advice.objectNameWarning false
git for-each-ref --format='%(refname)' refs/ | grep -E '[0-9a-f]{40}' | xargs -r git update-ref -d

# Final aggressive cleanup
git reflog expire --expire=now --all
git gc --prune=now --aggressive


# Check size
echo "New repository size:"
du -sh .git/


# Check the repository size
echo "Repository size after cleaning:"
du -sh .git/

# Re-add the remote (filter-repo removes it)
git remote add origin https://github.com/albertoamaduzzi/traffic_phase_transition.git 2>/dev/null || git remote set-url origin https://github.com/albertoamaduzzi/traffic_phase_transition.git

# Force push the cleaned history
echo "Pushing cleaned repository..."
git push origin --force --all
git push origin --force --tags

echo "Done! Repository cleaned and pushed."