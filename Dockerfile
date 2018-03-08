FROM dev.crayfis.io/crayfis-base

RUN apt-get update && apt-get install -y rsync

ADD ./*.py /crayfis-emulator/
ADD ./data/fetch.sh /crayfis-emulator/data/fetch.sh

# also make sure we trust the server key
RUN ssh-keyscan crayfis.ps.uci.edu >> /root/.ssh/known_hosts

WORKDIR /crayfis-emulator/data
RUN ./fetch.sh

WORKDIR /crayfis-emulator

# move to the working directory and install requirements
ADD requirements.txt /crayfis-emulator/requirements.txt
RUN pip install -r /crayfis-emulator/requirements.txt

ENV SLEEP_TIME=30
ENV SERVER=crayfis-site
ENV NUM_DEVICES=10
ENV INTERVAL=30
ENV APPCODE=""

CMD sleep $SLEEP_TIME && ./device.py --server $SERVER -N $NUM_DEVICES --interval $INTERVAL --appcode "$APPCODE"
