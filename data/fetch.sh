username=crayfis

# data from samsung Rad226 run
mkdir -p sam-rad226
rsync -avzP $username@crayfis.ps.uci.edu:/data/raw/bin/2014.10/14/9582b9cc2cb95ea0*.bin.gz sam-rad226

# data from htc Rad226 run
mkdir -p htc-rad226
rsync -avzP $username@crayfis.ps.uci.edu:/data/raw/bin/2014.08/17/414e9759cf63a2dc*.bin.gz htc-rad226

# data from samsung comsic run
mkdir -p sam-cosmic
rsync -avzP $username@crayfis.ps.uci.edu:/data/raw/bin/2014.08/05/9582b9cc2cb95ea0*.bin.gz sam-cosmic
rsync -avzP $username@crayfis.ps.uci.edu:/data/raw/bin/2014.08/06/9582b9cc2cb95ea0*.bin.gz sam-cosmic

# data from htc cosmic run
mkdir -p htc-cosmic
rsync -avzP $username@crayfis.ps.uci.edu:/data/raw/bin/2014.08/05/414e9759cf63a2dc*.bin.gz htc-cosmic
rsync -avzP $username@crayfis.ps.uci.edu:/data/raw/bin/2014.08/06/414e9759cf63a2dc*.bin.gz htc-cosmic
