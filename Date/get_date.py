import datetime
import time
import sys

date = datetime.datetime.strptime(sys.argv[1],'%Y-%m-%d')
date = time.mktime(date.timetuple())
date = date+86400

date = datetime.datetime.fromtimestamp(date)
date = str(date).split()
date = date[0]

print(date)
