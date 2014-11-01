crayfis-emulator
================
The emulator plays back real data from a few different phone runs, with configurable rates.
It should probably be rewritten with python's `multiprocessing` package; at the moment you have to spin up many instances which is expensive on RAM.

installation
------------
To get the playback data:
```
cd data/
./fetch.sh
```
You will need permissions on the crayfis.ps.uci.edu server (pubkey); contact an admin.

use
---
see `./device.py --help`.
There is also a helper script `spawn.sh` to run a bunch of device instances in the background
