import sys
import datetime as dt
import numpy as np
import scipy.stats as sp
import scipy.optimize as so
import matplotlib as mlab
import matplotlib.pyplot as plt

plt.rc('font', family='serif')
plt.rc('font', serif='CMU Serif')

def pprint(*args):
	print '\033[1;30m',
	#sys.stdout.flush()
	for i in args:
		print i,
		#sys.stdout.flush()
	print '\033[m'
	#sys.stdout.flush()
def LitersFromDrinks(d):
	return d * 0.6 * 0.0295735
def GalFromDrinks(d):
	return d * 0.6 * 0.0078125

# get data
# each entry of data is [date, beer, liquor, wine, beerBeforeLiquor]
f = open("data")
ldata = []
for line in f:
	s = line.split()
	liqs = [0.0, 0.0, 0.0]
	beerBeforeLiquor = False
	foundBeer = False
	foundLiquor = False
	for i,e in enumerate(s):
		if e == "B":
			foundBeer = True
			x = s[i+1]
			x = x.lstrip('(')
			x = x.rstrip(',')
			x = x.rstrip(')')
			liqs[0] = liqs[0] + float(x)
		elif e == "L":
			if foundBeer and not foundLiquor: beerBeforeLiquor = True
			foundLiquor = True
			x = s[i+1]
			x = x.lstrip('(')
			x = x.rstrip(',')
			x = x.rstrip(')')
			liqs[1] = liqs[1] + float(x)
		elif e == "W":
			x = s[i+1]
			x = x.lstrip('(')
			x = x.rstrip(',')
			x = x.rstrip(')')
			liqs[2] = liqs[2] + float(x)
	d = dt.date(int(s[0][0:4]), int(s[0][5:7]), int(s[0][8:10]))
	ldata.append([d,liqs[0],liqs[1],liqs[2],sum(liqs),beerBeforeLiquor])
data = np.array(ldata)

# Global Statistics
first = dt.date(2009, 11, 7)
today = dt.date(2016, 6, 30)
pprint("There have been %i days from November 7, 2009 to %s." % ((today - first).days,today.strftime('%B %d, %Y')))
pprint("I drank on %i of those days, or %.0f%%." % (len(data), float(len(data))/(today - first).days*100.0))
pprint("I have drunk approximately %.0fL (%.0f gal) of ethanol in my lifetime." % (LitersFromDrinks(sum(data[:,4])), GalFromDrinks(sum(data[:,4]))))
pprint("Of that, %.0fL (%.0f%%) was from beer, %.0fL (%.0f%%) was from liquor, and %.0fL (%.0f%%) was from wine." % (
		LitersFromDrinks(sum(data[:,1])),
		sum(data[:,1])/sum(data[:,4])*100.0,
		LitersFromDrinks(sum(data[:,2])),
		sum(data[:,2])/sum(data[:,4])*100.0,
		LitersFromDrinks(sum(data[:,3])),
		sum(data[:,3])/sum(data[:,4])*100.0)
		)
pprint("I drank beer before liquor %i times (%.0f%% of the times I drank both), but wasn't necessarily sicker." % (sum(data[:,5]),
		100.0*float(sum(data[:,5]))/sum((data[:,1:3] > 0).sum(axis=1)==2)))

# get main datasets
# xxData is the list of xx; suffix with Before or After for special plots
wdData    = [(d.weekday()+1)%7   for d in data[:,0]]
wdBefore  = [(d.weekday()+1)%7   for d in data[:,0] if d < dt.date(2015, 7, 1)]
moData    = [(d.month  )         for d in data[:,0]]
moBefore  = [(d.month  )         for d in data[:,0] if d < dt.date(2016, 1, 1)]
moAfter   = [(d.month  )         for d in data[:,0] if d >= dt.date(2016, 1, 1)]
wkData    = [d.isocalendar()[1]  for d in data[:,0]]
yrData    = [d.year              for d in data[:,0]]

# weights, and weights for befores
w       = LitersFromDrinks(data[:,1:4])
wBefore = LitersFromDrinks(data[data[:,0] < dt.date(2015, 7, 1)][:,1:4])
mBefore = LitersFromDrinks(data[data[:,0] < dt.date(2016, 1, 1)][:,1:4])
mBeforeS= LitersFromDrinks(data[data[:,0] < dt.date(2016, 1, 1)][:,4])
mAfter  = LitersFromDrinks(data[data[:,0] >= dt.date(2016, 1, 1)][:,4])

# for cumulative time plots -- fill in zeroes for days without records
# alldays is days and each entry, allsums is days and the sum of all entries above it
oneday = dt.timedelta(1)
curr = dt.date(2009,11,7)
alldays = []
allsums = [[0,0.0,0.0,0.0,0.0]]
i = 1
while curr < today:
	if curr in data[:,0]:
		thisEntry = [d for d in data if d[0] == curr][0]
		alldays.append([
			(curr-first).days,
			thisEntry[1],
			thisEntry[2],
			thisEntry[3],
			thisEntry[4]
			])
		allsums.append([
			(curr-first).days,
			allsums[i-1][1]+thisEntry[1],
			allsums[i-1][2]+thisEntry[2],
			allsums[i-1][3]+thisEntry[3],
			allsums[i-1][4]+thisEntry[4]
			])
	else:
		alldays.append([
			(curr-first).days,
			0.0,
			0.0,
			0.0,
			0.0
			])
		allsums.append([
			(curr-first).days,
			allsums[i-1][1],
			allsums[i-1][2],
			allsums[i-1][3],
			allsums[i-1][4]
			])
	curr = curr + oneday
	i += 1

alldays = np.array(alldays)
allsums = np.array(allsums)
avgdpd = np.array([[d[0], d[1]/float(i+1), d[2]/float(i+1), d[3]/float(i+1), d[4]/float(i+1)] for i,d in enumerate(allsums)])

# moving averages
def smoothed(a, n):
	if n < 0:
		n = 0
	if int(n) != n:
		n = 0
	if n%2 == 0:
		n += 1
	# n is an odd positive integer now
	# p ~ "plus minus"
	r = []
	for i, x in enumerate(a):
		p = n/2
		l = float(x)
		if i == 0 or i == len(a)-1:
			r.append(l)
			continue
		while i-p < 0 or i+p > len(a)-1:
			p -= 1
		for j in range(1,p+1):
			l += float(a[i+j]) + float(a[i-j])
		l = float(l) / float(2*p)
		r.append(l)
	return r

avgdays = np.transpose(np.array([
	alldays[:,0],
	smoothed(alldays[:,1]>0, 20),
	smoothed(alldays[:,2]>0, 20),
	smoothed(alldays[:,3]>0, 20),
	smoothed(alldays[:,4]>0, 20)
	]))

def rateTest(a, n):
	print len(a)
	r = []
	for i in range(len(a)/n):
		r.append(sum(a[0 + n*i : n + n*i])/float(n))
	if len(a)/n*n != len(a):
		r.append(sum(a[len(a)/n*n :]) / float(len(a[len(a)/n*n :])))
	return r

ratetest = rateTest(alldays[:,4]>0, 150)
#plt.plot(ratetest, 'o')
#plt.show()

# intertime. hoping for poisson
rally = False
ct = 0
interTime = []
for x in alldays:
	if not rally and x[4] == 0:
		rally = True
		ct = 1
	elif rally and x[4] == 0:
		ct += 1
	elif rally and x[4] > 0:
		rally = False
		interTime.append(ct)
	elif not rally and x[4] > 0:
		interTime.append(0)
interTime = np.array(interTime)

# plots start here
# colors
kOrange = '#ffcc00'
blwcol = ['#ff9933','#339933','#9933ff']

# pie chart
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.0f}%  ({v:d} L)'.format(p=pct,v=val)
    return my_autopct
plt.figure(figsize=plt.figaspect(1))
#pievals = [sum(data[:,1])/sum(data[:,4])*100.0, sum(data[:,2])/sum(data[:,4])*100.0, sum(data[:,3])/sum(data[:,4])*100.0]
pievals = [LitersFromDrinks(sum(data[:,1])), LitersFromDrinks(sum(data[:,2])), LitersFromDrinks(sum(data[:,3]))]
plt.pie(pievals,labels=['Beer','Liquor','Wine'],colors=blwcol,autopct=make_autopct(pievals))
plt.tight_layout()
plt.savefig('pie.pdf')
plt.close()

# wd and wd-alc
def makeWDPlots(which):
	if which == "wd-alc":
		xb = np.transpose(np.array([wdBefore, wdBefore, wdBefore]))
		xd = np.transpose(np.array([wdData, wdData, wdData]))
		fwB  = wBefore
		fwD  = w
		nor  = False
		stk  = True
		yl   = [0.0, 9.1]
		ytit = 'Volume of Ethanol [L]'
		pad  = 20
		fn   = 'wd-alc'
		dcol  = blwcol
		decol = 'none'
		dlbl  = ['Beer','Liquor','Wine']
		dtyp  = 'stepfilled'
		plt.hist(xd, weights=fwD, stacked=stk, label=dlbl, bins=7, range=(0,7), normed=nor, histtype=dtyp, color=dcol, edgecolor=decol)
	elif which == "wd":
		xb = wdBefore
		xd = wdData
		fwB  = np.ones(len(wdBefore))
		fwD  = np.ones(len(wdData))
		nor  = True
		stk  = False
		yl   = [0.0, 0.26]
		ytit = 'Normalized Frequency'
		pad  = None
		fn   = 'wd'
		bcol  = kOrange; dcol  = "k"
		becol = 'none'   ; decol = 'k'
		blbl  = "Before July 1, 2015"; dlbl  = "All"
		btyp  = 'stepfilled'         ; dtyp  = 'step'
		plt.hist(xb, weights=fwB, stacked=stk, label=blbl, bins=7, range=(0,7), normed=nor, histtype=btyp, color=bcol, edgecolor=becol)
		plt.hist(xd, weights=fwD, stacked=stk, label=dlbl, bins=7, range=(0,7), normed=nor, histtype=dtyp, color=dcol, edgecolor=decol)
	ax = plt.gca()
	ax.set_ylim(yl)
	ax.set_xticks([float(i)+0.5 for i in range(7)])
	ax.set_xticklabels(["Sun","Mon","Tue","Wed","Thu","Fri","Sat"])
	plt.tight_layout()
	plt.subplots_adjust(bottom=0.1, left=0.1)
	plt.xlabel('Day of Week')
	plt.ylabel(ytit,labelpad=pad)
	lg = plt.legend(loc='upper left')
	lg.draw_frame(False)
	plt.savefig(fn+'.pdf')
	plt.close()

makeWDPlots("wd")
makeWDPlots("wd-alc")

# mo, wk, yr and -alc
# mo is permanently changed to moBefore
def makeOtherPlots(which):
	if which[0:2] == 'mo':
		b     = 12
		rng   = [1, 13]
		xt    = [float(i)+0.5 for i in range(1,13)]
		xtlbl = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
		xlbl  = 'Month'
		fn    = which
		ytit  = 'Normalized Frequency'
		nor   = 1
		stk   = 0
		pad   = None
		if which == 'mo-alc':
			#dat = np.transpose(np.array([moData, moData, moData]))
			dat = np.transpose(np.array([moBefore, moBefore, moBefore]))
			#wgt = w
			wgt = mBefore
			nor = 0
			ytit = 'Volume of Ethanol [L]'
			col = blwcol
			stk = 1
			pad = 20
			lbl = ['Beer', 'Liquor', 'Wine']
		elif which == 'mo-pre':
			dat = moBefore
			wgt = mBeforeS
			#nor = 0
			ytit = 'Normalized Volume of Ethanol [L]'
			col = kOrange
			lbl = '2009-2015'
		else:
			#dat = moData
			dat = moBefore
			wgt = np.ones(len(moBefore))
			col = kOrange
			lbl = ""
	elif which[0:2] == 'wk':
		b     = 53
		rng   = [1, 54]
		xt    = None
		xtlbl = None
		xlbl  = 'Week'
		fn    = which
		ytit  = 'Normalized Frequency'
		nor   = 1
		stk   = 0
		pad   = None
		if len(which) > 2:
			dat = np.transpose(np.array([wkData, wkData, wkData]))
			wgt = w
			nor = 0
			ytit = 'Volume of Ethanol [L]'
			col = blwcol
			stk = 1
			pad = 20
			lbl = ['Beer', 'Liquor', 'Wine']
		else:
			dat = wkData
			wgt = np.ones(len(wkData))
			col = kOrange
			lbl = ""
	elif which[0:2] == 'yr':
		b     = 8
		rng   = [2009, 2017]
		xt    = [float(i)+0.5 for i in range(2009,2017)]
		xtlbl = ['2009','2010','2011','2012','2013','2014','2015','2016']
		xlbl  = 'Year'
		fn    = which
		ytit  = 'Normalized Frequency'
		nor   = 1
		stk   = 0
		pad   = None
		if len(which) > 2:
			dat = np.transpose(np.array([yrData, yrData, yrData]))
			wgt = w
			nor = 0
			ytit = 'Volume of Ethanol [L]'
			col = blwcol
			stk = 1
			pad = 20
			lbl = ['Beer', 'Liquor', 'Wine']
		else:
			dat = yrData
			wgt = np.ones(len(yrData))
			col = kOrange
			lbl = ""
	p = plt.hist(dat, bins=b, label=lbl, weights=wgt, stacked=stk, range=rng, normed=nor, color=col, histtype='stepfilled', edgecolor='none')
	if which == 'mo-pre':
		qh, qe = np.histogram(moAfter, bins=b, weights=mAfter/sum(mAfter)*sum(p[0][0:6])/sum(p[0]), range=rng)
		q = plt.plot(qe[:-1]+0.5, qh, 'ro', label='2016', color='k')
	ax = plt.gca()
	ax.set_xlim(rng)
	if xt is not None: ax.set_xticks(xt)
	if xtlbl is not None: ax.set_xticklabels(xtlbl)
	plt.tight_layout()
	plt.subplots_adjust(bottom=0.1, left=0.1)
	plt.xlabel(xlbl)
	plt.ylabel(ytit,labelpad=pad)
	if lbl != "":
		if which == 'mo-pre':
			lg = plt.legend(loc='upper left', handler_map={q[0]:mlab.legend_handler.HandlerLine2D(numpoints=1)})
		else:
			lg = plt.legend(loc='upper left')
		lg.draw_frame(False)
	plt.savefig(fn+'.pdf')
	if which == 'mo-alc':
		pprint("Alcohol through June =", sum(p[0][2][0:6]), "which is a fraction of", sum(p[0][2][0:6])/sum(p[0][2]))
	if which == 'yr-alc':
		pprint("I've drunk", p[0][2][-1], "liters of alcohol so far this year.")
		pprint("Compared to", p[0][2][-2], "liters of alcohol last year.")
	plt.close()

makeOtherPlots('mo')
makeOtherPlots('wk')
makeOtherPlots('yr')
makeOtherPlots('mo-alc')
makeOtherPlots('mo-pre')
makeOtherPlots('wk-alc')
makeOtherPlots('yr-alc')

def func(x, m, b):
	return np.exp(m*x + b)
def CC(x, y, name=''):
	pprint('The', name, 'correlation coefficient is', '%.4f' % np.corrcoef(x,y)[0,1])
def fitLine(p0, p1,name=''):
	y = p0[p0>1]
	xtemp = list(p0>1)
	xtemp.append(True)
	x = p1[np.array(xtemp)][:-1]
	# several ways
	m1, b1 = np.polyfit(x,np.log(y),1)
	m2, b2 = np.polyfit(x,np.log(y),1,w=1./y)
	m3, b3 = np.polyfit(x,np.log(y),1,w=1./np.log(y))
	#par4 = so.curve_fit(func, p1[:-1], p0)
	#par5 = so.curve_fit(func, p1[:-1], p0, sigma=1./p0)
	par6 = so.curve_fit(func, x, y)
	par7 = so.curve_fit(func, x, y, sigma=1./y)
	def printPars(m, b, extra, mod, xx, yy):
		pprint('%s: m = %.4f, b = %.4f, e^b = %.4f' % (name+extra, m, b, np.exp(b)))
		if mod == 'lin':
			chi2e = sum(((yy - np.exp(m*xx+b))**2.0)/(yy))
			chi2t = sum(((yy - np.exp(m*xx+b))**2.0)/(np.exp(m*xx+b)))
		elif mod == 'log':
			chi2e = sum(((yy - (m*xx+b))**2.0)/(np.exp(yy)/(np.exp(yy))**2.))
			chi2t = sum(((yy - (m*xx+b))**2.0)/(np.exp(m*xx+b)))
		elif mod == 'exp':
			chi2e = sum(((yy - func(xx,m,b))**2.0)/(yy))
			chi2t = sum(((yy - func(xx,m,b))**2.0)/(func(xx,m,b)))
		pprint('  Chi^2 (exp) =', chi2e, 'p =', 1 - sp.chi2.cdf(chi2e, len(yy)-1-2))
		#pprint('  Chi^2 (the) =', chi2t, 'p =', 1 - sp.chi2.cdf(chi2t, len(yy)-1-2))
	printPars(m1, b1,                 ' - lin, w=1       ','lin',x,y)
	printPars(m2, b2,                 ' - lin, w=1/y     ','lin',x,y)
	printPars(m3, b3,                 ' - lin, w=1/log(y)','lin',x,y)
	printPars(m1, b1,                 ' - log, w=1       ','log',x,np.log(y))
	printPars(m2, b2,                 ' - log, w=1/y     ','log',x,np.log(y))
	printPars(m3, b3,                 ' - log, w=1/log(y)','log',x,np.log(y))
	#printPars(par4[0][0], par4[0][1], ' - exp, w=1       ','exp',p1[:-1],p0)
	#printPars(par5[0][0], par5[0][1], ' - exp, w=1/y     ','exp',p1[:-1],p0)
	printPars(par6[0][0], par6[0][1], ' - exp, w=1  , cas','exp',x,y)
	printPars(par7[0][0], par7[0][1], ' - exp, w=1/y, cas','exp',x,y)

	CC(x,np.log(y),name)
	
	return m1, b1

# drinks per day
def makeDpdPlot(which):
	rng = [1,15]
	b = rng[1] - rng[0]
	if which == 'dpd':
		p = plt.hist(alldays[:,4], bins=b, range=rng, color=kOrange, normed=0, histtype='stepfilled', edgecolor='none')
		m, b = fitLine(p[0],p[1],which)
	elif which == 'dpd-alc':
		plt.gca().set_color_cycle(blwcol)
		p = plt.hist(alldays[:,1:4], bins=b, range=rng, stacked=1, normed=1, histtype='stepfilled', edgecolor='none', label=['Beer','Liquor','Wine'])
		m, b = fitLine(p[0][2],p[1],which)
	l, = plt.plot(np.linspace(rng[0],rng[1],50),np.exp(b)*np.exp(m * np.linspace(rng[0],rng[1],50)),color='k',linestyle='dashed', label='Fit')
	ax = plt.gca()
	ax.set_xlim(rng)
	plt.tight_layout()
	plt.yscale('log')
	plt.subplots_adjust(bottom=0.1, left=0.1)
	plt.xlabel('Drinks')
	plt.ylabel('Normalized Frequency',labelpad=10)
	if which == 'dpd-alc':
		lg = plt.legend(loc='upper right')
		lg.draw_frame(False)
	plt.savefig(which+'.pdf')
	plt.close()

makeDpdPlot('dpd')
#makeDpdPlot('dpd-alc')

# interTime
def makeInterPlot():
	rng = [0,20]
	b = rng[1]-rng[0]
	p = plt.hist(interTime, bins=b, range=rng, normed=0, histtype='stepfilled', edgecolor='none', color=kOrange)
	m, b = fitLine(p[0],p[1],'InterTime')
	l, = plt.plot(np.linspace(rng[0],rng[1],50),np.exp(b)*np.exp(m * np.linspace(rng[0],rng[1],50)),color='k',linestyle='dashed', label='Fit')
	ax = plt.gca()
	ax.set_xlim(rng)
	plt.tight_layout()
	plt.yscale('log')
	plt.subplots_adjust(bottom=0.1, left=0.1)
	plt.xlabel('Consecutive Dry Days')
	plt.ylabel('Normalized Frequency')
	plt.savefig('intertime.pdf')
	plt.close()
makeInterPlot()

# time plots
def makeTimePlots(data1, data2, ytit, fn):
	if '-' in fn:
		plt.gca().set_color_cycle(blwcol)
		p = plt.plot(data1, data2)
	else:
		p = plt.plot(data1,data2,color='r')
	ax = plt.gca()
	ax.set_xticks([(dt.date(i,1,1)-first).days for i in range(2010,today.year+1)])
	ax.set_xticklabels([str(i) for i in range(2010,today.year+1)])
	if fn == 'avg-alc':
		ax.set_ylim([0.0,0.3])
	plt.xlabel('Time')
	plt.ylabel(ytit,labelpad=10)
	if '-' in fn:
		lg = plt.legend((p[0], p[1], p[2]), ('Beer', 'Liquor', 'Wine'),loc='upper left')
		lg.draw_frame(False)
	plt.tight_layout()
	plt.savefig(fn+'.pdf')
	plt.close()

makeTimePlots(allsums[:,0], allsums[:,1:4]* 0.6 * 0.0295735               , 'Volume of Ethanol [L]'               , 'time-alc')
makeTimePlots(allsums[:,0],   allsums[:,4]* 0.6 * 0.0295735               , 'Volume of Ethanol [L]'               , 'time')
#makeTimePlots(avgdays[:,0],   avgdays[:,4]* 0.6 * 0.0295735               , 'Smoothed Ethanol Consumption [L]'    , 'rate')
makeTimePlots(avgdays[:,0],   avgdays[:,4]               , 'Smoothed Ethanol Consumption [L]'    , 'rate')
#makeTimePlots(avgdays[:,0], avgdays[:,1:4]* 0.6 * 0.0295735               , 'Smoothed Ethanol Consumption [L]'    , 'rate-alc')
#makeTimePlots( avgdpd[:,0], avgdpd[:,1:4]                               , 'Average Drinks per Day'              , 'avg-alc')

# correlations
#beer = np.array(data[:,1],dtype=float)
#liqr = np.array(data[:,2],dtype=float)
#wine = np.array(data[:,3],dtype=float)
atLeast2 =\
	np.logical_or(
		np.logical_or(
			np.logical_and(data[:,1]>0,data[:,2]>0),
			np.logical_and(data[:,1]>0,data[:,3]>0)
			),
		np.logical_and(data[:,2]>0,data[:,3]>0)
		)
beer = np.array(data[:,1][atLeast2],dtype=float)
liqr = np.array(data[:,2][atLeast2],dtype=float)
wine = np.array(data[:,3][atLeast2],dtype=float)

CC(beer,wine,'beer-wine')
CC(beer,liqr,'beer-liquor')
CC(liqr,wine,'liquor-wine')

#plt.plot(beer,liqr,'ro')
#m,b = np.polyfit(beer,liqr,1)
#plt.plot(np.arange(20), m*np.arange(20)+b)
#plt.show()
#
#plt.plot(beer,wine,'bo')
#m,b = np.polyfit(beer,wine,1)
#plt.plot(np.arange(20), m*np.arange(20)+b)
#plt.show()
#
#plt.plot(liqr,wine,'go')
#m,b = np.polyfit(liqr,wine,1)
#plt.plot(np.arange(20), m*np.arange(20)+b)
#plt.show()
