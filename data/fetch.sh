username=data-user

# January 2018
mkdir ./tmp
mkdir ./tmp/raw
mkdir ./tmp/unzipped

echo "Downloading from server"
rsync -azP $username@craydata.ps.uci.edu:/data/daq.crayfis.io/raw/2018/01/* ./tmp/raw
echo "Unzipping files"
find ./tmp/raw -name "*.tar.gz"  -exec tar -zxf {} -C tmp/unzipped \; -exec rm {} \;
echo "Consolidating files"
python3 consolidate.py --msg_dir ./tmp/unzipped/ --max_xb 1000
echo "Removing temp directory"
rm -r tmp
