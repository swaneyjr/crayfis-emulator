#!/bin/bash

usage() { echo "Usage: $0 -s host [-n njobs] [-d source]"; exit 1; }

on_die()
{
	echo "killing workers"
	kill $(jobs -p)
	exit 0
}
trap 'on_die' INT

njob=100
dsource=htc-cosmic
while getopts "hn:s:" o; do
        case "${o}" in
                h)
                usage
                ;;
                n)
                njob="${OPTARG}"
                ;;
                d)
                dsource="${OPTARG}"
                ;;
		s)
		host="${OPTARG}"
		;;
                *)
                usage
                ;;
        esac
done

[ -z "$host" ] && usage

i=0
echo "Spawning workers..."
while [ "$i" -lt "$njob" ]; do
	i=$((i+1))

	./device.py --source $dsource --server $host &
done
echo "Done. Press Ctrl+C to kill workers."

wait
