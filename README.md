crayfis-emulator
================
The emulator plays back either real data from a live runs or simulated data given with randomized hotcells, lens-shade maps, etc.

Installation
------------
The emulator is intended to run inside the crayfis-docker environment.  After crayfis-docker has been installed, the emulator can be built with:
```
cd crayfis-docker
git clone https://github.com/crayfis/crayfis-emulator.git
./rebuild.sh -e
```

Use
---
The docker container basically runs `sleep $SLEEP_TIME && ./device.py [OPTIONS]`.  When run in the docker-compose environment, the options are taken from `docker-compose.yml` in the crayfis-docker repository and should be edited there.  Of particular interest is the `$SOURCE` option, which determines whether the emulator will stream live data or simulate data from a noise/hotcell distribution (default).  To change this behavior, `$SOURCE` can be set to a directory in the docker container with source files to stream; by default, `./data` will have such a selection.

Errors
---
Sometimes you will want to debug errors that occur on the server side during data submission.
In the event of a non-2XX response code, the HTTP response will be dumped to `err.html`.
This works best if you set `DEBUG=True` in the django `settings.py` file.
