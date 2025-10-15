# Install Git LFS
sudo apt-get install git-lfs  # or: brew install git-lfs on macOS
git lfs install

# Track large file types
git lfs track "*.fma"
git lfs track "*.dbf"
git lfs track "*.shp"
git lfs track "*.tif"
git lfs track "*.gpkg"
git lfs track "*.parquet"
git lfs track "data/**"

# Add .gitattributes (created by git lfs track)
git add .gitattributes
git commit -m "Configure Git LFS for large data files"

# Now add your data files
git add data/
git commit -m "Add data files via Git LFS"
git push