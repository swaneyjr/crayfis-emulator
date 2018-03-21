ARG base

FROM $base

RUN apt-get update && apt-get install -y rsync

ADD ./*.py /crayfis-emulator/
ADD ./data /crayfis-emulator/data

# also make sure we trust the server key
RUN ssh-keyscan craydata.ps.uci.edu >> /root/.ssh/known_hosts

# fetch data if not already present
WORKDIR /crayfis-emulator/data
RUN if [ ! -e ./*.bin ]; then \
        echo "Installing files from craydata"; \
        ./fetch.sh; \
    fi

WORKDIR /crayfis-emulator

# move to the working directory and install requirements
ADD requirements.txt /crayfis-emulator/requirements.txt
RUN pip3 install -r /crayfis-emulator/requirements.txt

ENV SLEEP_TIME=30
ENV SERVER=crayfis-site
ENV NUM_DEVICES=10
ENV INTERVAL=30
ENV APPCODE=""

WORKDIR /crayfis-emulator

CMD sleep $SLEEP_TIME && ./device.py --server $SERVER -N $NUM_DEVICES --interval $INTERVAL --appcode "$APPCODE"
