import datetime

logmsg = ''
timemark = dict()
saveDefault = False
def log(msg, save=None, oneline=False):
	global logmsg
	global saveDefault
	time = datetime.datetime.now()
	tem = '%s: %s' % (time, msg)
	if save != None:
		if save:
			logmsg += tem + '\n'
	elif saveDefault:
		logmsg += tem + '\n'
	if oneline:
		print(tem, end='\r')
	else:
		print(tem)

def marktime(marker):
	global timemark
	timemark[marker] = datetime.datetime.now()

def get_human_like_epoch_run_time(elapsed_time) -> tuple:
	hours = int(elapsed_time // 3600)
	minutes = int((elapsed_time % 3600) // 60)
	seconds = int(elapsed_time % 60)
	milliseconds = int((elapsed_time * 1000) % 1000)
	return hours, minutes, seconds ,milliseconds

if __name__ == '__main__':
	log('')