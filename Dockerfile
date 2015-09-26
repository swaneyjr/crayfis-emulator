FROM dev.crayfis.io/crayfis-base

ADD ./*.py /crayfis-emulator/
ADD ./data/fetch.sh /crayfis-emulator/data/fetch.sh

# add a key to access the data from crayfis.ps.uci.edu
# (needed by the fetch.sh script)
ADD id_rsa /root/.ssh/id_rsa
ADD id_rsa.pub /root/.ssh/id_rsa.pub

# also make sure we trust the server key
RUN ssh-keyscan crayfis.ps.uci.edu >> /root/.ssh/known_hosts

WORKDIR /crayfis-emulator/data
RUN ./fetch.sh

WORKDIR /crayfis-emulator

ENV SLEEP_TIME=30
ENV SERVER=crayfis-site
ENV NUM_DEVICES=10
ENV INTERVAL=30

CMD sleep $SLEEP_TIME && ./device.py --server $SERVER -N $NUM_DEVICES --interval $INTERVAL
