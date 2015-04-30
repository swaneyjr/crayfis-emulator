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

The interesting options are mostly `--interval`, which controls how often submit requests will be made (per device), `--rate` which essentially controls the size of the data uploaded, and `-N` which can be used to emulate up to several hundred devices (in my experience).
It uses threading (which has performance problems in python) so if you've got multiple cores you can probably do better by running multiple instances of the device.py program with -N set to a few hundred on each.

When simulating a single device, you can also use the `--nowait` command to commence sending data immediately.
By default each simulated device waits a random amount of time before sending, to prevent all devices from
trying to send data at the same time.

errors
---
Sometimes you will want to debug errors that occur on the server side during data submission.
In the event of a non-2XX response code, the HTTP response will be dumped to `err.html`.
This works best if you set `DEBUG=True` in the django `settings.py` file.
