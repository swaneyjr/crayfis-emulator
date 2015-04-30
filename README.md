crayfis-emulator
================
The emulator plays back real data from a few different phone runs, with configurable rates.

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

errors
---
Sometimes you will want to debug errors that occur on the server side during data submission.
In the event of a non-2XX response code, the HTTP response will be dumped to `err.html`.
This works best if you set `DEBUG=True` in the django `settings.py` file.
