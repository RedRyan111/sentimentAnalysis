#!/bin/bash

#initialize variables
<<<<<<< HEAD
after="2018-04-01"
before="2018-04-02"

for i in {1..120}
=======
after="2018-01-01"
before="2018-01-02"

for i in {1..3}
>>>>>>> e54aebb9a4623e172311b6b2b53be93eb550e190
do
	echo "Current Date: $after $before"

	python3 twitterApi/GetOldTweets-python-master/TwitterApi.py $after $before
	python3 redditApi/redditAPI.py $after $before

	#Running python scripts
	after="$(python3 Date/get_date.py $after)"
	before="$(python3 Date/get_date.py $before)"

	sleep 3

done








