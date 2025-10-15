# Find the 30 largest objects in your repository history
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  sed -n 's/^blob //p' | \
  sort -k2 -n -r | \
  head -30 | \
  awk '{print $2/1024/1024 " MB\t" $3}'
# Install git-filter-repo if not already installed
pip install git-filter-repo

# Remove large directories from history
git filter-repo --path data_2/ --invert-paths
git filter-repo --path MemoryUsage/ --invert-paths
git filter-repo --path cache/ --invert-paths
git filter-repo --path data/tiff_files/ --invert-paths

# Remove large file types from history
git filter-repo --path-glob '*.tif' --invert-paths
git filter-repo --path-glob '*.gpkg' --invert-paths
git filter-repo --path-glob '*.gz' --invert-paths
# Remove from current tracking
git rm -r --cached data_2/ 2>/dev/null || true
git rm -r --cached MemoryUsage/ 2>/dev/null || true
git rm -r --cached cache/ 2>/dev/null || true
git rm -r --cached .vscode/ 2>/dev/null || true
git rm -r --cached __pycache__/ 2>/dev/null || true
git rm -r --cached scripts/__pycache__/ 2>/dev/null || true
git rm -r --cached scripts/__pycache__old/ 2>/dev/null || true

# Remove specific large files
git rm --cached data/tiff_files/*.tif 2>/dev/null || true
git rm --cached data/tiff_files/*.gpkg 2>/dev/null || true
git rm --cached data/tiff_files/*.gz 2>/dev/null || true
git rm --cached scripts/od_node_map.html 2>/dev/null || true

# Add the updated .gitignore
git add .gitignore

# Commit the cleanup
git commit -m "Clean repository: remove large files and update .gitignore"

