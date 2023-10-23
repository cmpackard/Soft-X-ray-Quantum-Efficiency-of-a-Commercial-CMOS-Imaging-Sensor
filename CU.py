"""
CMOS UTILS
"""
import sys
from scipy import optimize
from scipy import stats
import astropy.io.fits as pyfits
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import lmfit
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
'''
Defined constants, values, etc
'''
#values found from argonne data
aduM = 3.69546219 #slope
aduB = 5.85548827 #intercept

def aduToeV(adu):
	'''
	Converts ADU value to eV
	Parameters:
		ADU: given ADU value, either int or list
	'''
	return (aduM * adu) + aduB
def mean_2d(img):
	'''
	Returns the mean of an image
	Parameters:
		img - 2d array of image
	'''
	nx, ny = img.shape
	return np.mean(np.reshape(img, nx*ny))

def median_2d(img):
	'''
	Returns the median of an image
	Parameters:
		img - 2d array of image
	'''
	nx, ny = img.shape
	return np.median(np.reshape(img, nx*ny))

def std_2d(img):
	'''
	Returns the standard deviation of an image
	Parameters:
		img - 2d array of image
	'''
	nx, ny = img.shape
	return np.std(np.reshape(img, nx*ny))

def imStats(img, verbose=False):
	'''
	Returns the mean, median and standard deviation of an image in a tuple
	Will also print the statistics if verbose is True
	Parameters:
		img - 2d array of image
		verbose (default False) - print the image stats or not
	'''
	mean = mean_2d(img)
	med = median_2d(img)
	std = std_2d(img)
	if(verbose):
		print("Mean:", mean)
		print("Median:", med)
		print("Std:", std)
	return [mean, med, std]

def avgBias(biasPath, verbose=False):
	'''
	Computes the average of bias frames, returns it and writes out the averaged frame for future reuse
	Paramaters:
		biasPath - path to biasFrames, ex /Files/Images/Bias/BiasFrame
		verbose (default False) - print the process of compiling the average + statistics
	'''
	if (os.path.exists(biasPath + "avgBias.FTS")):
		if (verbose): print("Average bias already compiled")
		f = pyfits.open(biasPath + "avgBias.FTS")
		dat = f[0].data
		f.close()
		print(dat)

	files = glob.glob(biasPath + "?.FTS") + glob.glob(biasPath + "??.FTS") + glob.glob(biasPath + "???.FTS")
	shaping = pyfits.open(files[0])
	index = 0
	if(len(shaping) == 2):
		index = 1
	shape = shaping[index].data
	shaping.close()
	totBias = np.array(0 * shape).astype(np.int32)
	if (verbose): print("Found " + str(len(files)) + " images")
	for i in range(0, len(files)):
		dat = pyfits.open(files[i])
		imDat = dat[index].data
		dat.close()
		totBias += imDat
	medBias = totBias / len(files)
	medBias = medBias.astype(np.int32)
	if (verbose):
		print("Avg bias stats")
		imStats(medBias, verbose=True)
	pyfits.writeto(biasPath + "avgBias.FTS", medBias, None, overwrite=True)
	print(medBias)
	return medBias

def avgDark(darkPath, verbose):

        '''
        Computes the average of dark frames, returns it and writes out the averaged frame for future reuse
        Paramaters:
                biasPath - path to dark frames, ex /Files/Images/Darks/DarkFrame
                verbose (default False) - print the process of compiling the average + statistics
        '''
        if (os.path.exists(darkPath + "avgDark.FTS")):
                if (verbose): print("Average dark already compiled")
                f = pyfits.open(darkPath + "avgDark.FTS")
                dat = f[0].data
                f.close()
                return dat

        files = glob.glob(darkPath + "?.FTS") + glob.glob(darkPath + "??.FTS") + glob.glob(darkPath + "???.FTS")
        shaping = pyfits.open(files[0])
        index = 0
        if(len(shaping) == 2):
                index = 1
        shape = shaping[index].data
        shaping.close()
        totDark = np.array(0 * shape).astype(np.int32)
        if (verbose): print("Found " + str(len(files)) + " images")
        for i in range(0, len(files)):
                dat = pyfits.open(files[i])
                imDat = dat[index].data
                dat.close()
                totDark += imDat
        medDark = totDark / len(files)
        medDark = medDark.astype(np.int32)
        if (verbose):
                print("Avg dark stats")
                imStats(medDark, verbose=True)
        pyfits.writeto(darkPath + "avgDark.FTS", medDark, None, overwrite=True)
        return medDark

def collectFiles(filePath):
	'''
	A function more for internal convience, returns a set of images as an array containing the image data
	Parameters:
		filePath - Path to the files to be collected
	'''
	fs = glob.glob(filePath + "?.FTS") +  glob.glob(filePath + "??.FTS") + glob.glob(filePath + "???.FTS") + glob.glob(filePath + "????.FTS") 
	ps = []
	for i in range(0, len(fs)):
		dat = pyfits.open(fs[i])
		if (len(dat) == 1):
			ps.append(dat[0].data.astype(np.int32))
		else:
			ps.append(dat[1].data.astype(np.int32))
		dat.close()
	return ps


def getBadPix(darkPath, thr=10, rep=5, verbose=False):
	'''
	Finds hot and warm pixels from the dark frames
	Parameters:
		darkPath - path to the dark frames
		thr (default 10) - minimum threshold to be counted as bad pixel
		rep (default 5) - percentage of frames a pixel must occur in to be disqualified (on scale from 0-100)
		verbose (default False) - print information about process
	'''
	lpath = darkPath + "hotT" + str(thr) + "R" + str(rep) + ".csv"
	if (os.path.exists(lpath)):
		if (verbose): print("Pixel list already computed for these parameters")
		(x, y) = np.loadtxt(lpath, delimiter=',', unpack=True)
		pos = []
		for i in range(0, len(x)):
			pos.append([x[i], y[i]])
		return pos
	ps = collectFiles(darkPath)
	if (verbose): print("Parsing " + str(len(ps)) + " images for hot pixels.\nThr: " + str(thr) + "\nRep: " + str(rep))
	hots = []
	pos = []
	nx, ny = ps[0].shape
	pindex = 0
	for phot in ps:
		pindex += 1
		if (verbose): print(pindex)
		f1 = np.reshape(phot, nx*ny)
		q = np.argsort(f1)[::-1]
		bad = True
		i = 0
		while (bad):
			if(f1[q[i]] > thr):
				hots.append(q[i])
				i += 1
			else:
				bad = False
	h, c = np.unique(hots, return_counts=True)
	for i in range(0, len(h)):
		x = int(math.floor(h[i] / 1936))
		y = int((h[i] % 1936))
		if (c[i] > rep):
			pos.append([x, y])

	if (verbose):
		print(str(len(pos)) + " pixels found")
		print("Loss: %.2f" % ((len(pos)/(nx * ny)) * 100))
	f = open(lpath, 'w+')
	for i in range(0, len(pos)):
		f.write(str(pos[i][0]) + ", " + str(pos[i][1]) + "\n")
	f.close()
	return pos

def histogram(ep, saveImg=False, saveName='default.png'):
	'''
	Display the histogram of events
	Parameters:
		ep - list of event pulses
		saveImg (default False) - save histogram as image?
		saveName (default 'default.png') - where to save the image
	'''
	plt.ion()
	plt.figure('event histogram')
	plt.clf()
	unis = np.unique(ep)
	b = int(max(unis) - min(unis))
	plt.hist(ep, bins=b, range=[min(unis), max(unis)])
	if (saveImg):
		plt.savefig(saveName)
	plt.ioff()
	plt.show()

def spectrum(ep, saveImg=False, saveName='default.png', tickSpace=400):
	'''
	Produce a spectrum from event list
	Parameters:
		ep - list of event pulses
		saveImg (default False) - save the spectrum as an image?
		saveName (default 'default.png') - where to save the image
		tickSpace (default 400) - spacing for major ticks on spectrum
	'''
	unis, cnts = np.unique(ep, return_counts=True)
	unisList = dict(zip(unis, cnts))
	plt.ion()
	plt.figure('Spectrum', figsize=(16, 10))
	plt.clf()
	lists = sorted(unisList.items())
	x, y = zip(*lists)
	x = np.asarray(x)
	xdat = aduToeV(x)
	ydat = np.asarray(y)
	plt.xlabel("Energy (eV)")
	plt.ylabel("Intensity")
	plt.plot(xdat, ydat)
	if (saveImg):
		plt.savefig(saveName)
	plt.ioff()
	plt.show()

def eventsFromFile(fileName, forceNoShape=False):
	'''
	Returns the events and shape from a given filename
	Parameters:
		fileName - name of file with events (dont include .csv part of path)
		forceNoShape - return just a tuple of ep, ex, ey, don't return the shapes
	'''
	fP = fileName + ".csv"
	ep, ex, ey, sh = [], [], [], []
	hasShape = False
	with open(fP) as file:
		line = file.readline()
		while line:
			l1 = line
			bits = l1.split(":")
			vals = bits[0].split(",")
			if (str(line) != "ep, ex, ey\n"):
				ep.append(int(vals[0]))
				ex.append(int(vals[1]))
				ey.append(int(vals[2]))
				if(len(bits) == 2):
					hasShape = True
					sh.append(int(bits[1].split(",")[0]))
			line = file.readline()
	ep = np.array(ep)
	ex = np.array(ex)
	ey = np.array(ey)
	if (hasShape and (forceNoShape == False)):
		sh = np.array(sh)
		return [ep, ex, ey, sh]
	else:
		return [ep, ex, ey]



def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '='):
	'''
	Call in a loop to create terminal progress bar
	Parameters:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		length      - Optional  : character length of bar (Int)
		fill        - Optional  : bar fill character (Str)
	'''
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar =  fill * filledLength +'>' + '-' * (length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
	# Print New Line on Complete
	if iteration == total: 
		print()
		print()

def func_gaussc(x, norm, cent, width, cons):
	return norm*np.exp(-(x-cent)**2 / (2*width**2)) + cons

gaussc = lambda p, x: p[0]*np.exp(-(x-p[1])**2/(2*p[2]**2)) + p[3]

def fitgaus(ph, cent, fwhm, plot=0, binfactor=1.0, ADU=True):
	'''
	Fit a standard gaussian to event pulses
	Parameters:
		ph - event list
		cent, fwhm - initial guesses
		plot (default 0) - histogram the data? (0 - no 1 - yes)
		binfactor (default 1) - how much to bin the event list
		ADU (default True) - plot in ADU or eV
	'''
	bins = int((max(ph) - min(ph)) / binfactor)
	(counts, edges) = np.histogram(ph, bins=bins)
	n = len(counts)
	chan = 0.5 * (edges[0:n]+edges[1:n+1])
	tot = np.sum(abs(ph-cent) < fwhm)
	sig = fwhm / 2.35
	pinit = [tot/(sig*np.sqrt(2*np.pi)), cent, sig, 0.0]
	cerr = 1+np.sqrt(counts+0.75)
	p,t = optimize.curve_fit(func_gaussc, chan, counts, p0=pinit, sigma=cerr)
	if (plot > 0): 
		plt.figure(plot)
		plt.clf()
		plt.errorbar(chan, counts, cerr, fmt='.', capsize=0)
		plt.xlabel("Pulse Height (ADU)")
		plt.ylabel("Events / bin")
		if (ADU):
			plt.plot(chan, gaussc(p, chan))
		else:
			plt.plot(aduToeV(chan), gaussc(p, aduToeV(chan))) 
		plt.show()
	return p, t

def printEventShape(shape):
	'''
	Prints a visual representation of a shape code. X is the center pixel, 0 is a non-illuminated pixel and 1 is an illuminated pixel
	Parameters:
		shape - the shape code
	'''
	b = np.binary_repr(shape)
	padding = 25 - len(b)
	b = ("0" * padding) + b
	for i in range(0, 5):
		for j in range(0, 5):
			px = 5 * i + j
			if (px == 12):
				sys.stdout.write("X")
			else:
				sys.stdout.write(b[px])
	sys.stdout.flush()

def getEp(area, tp, diag=True):
	s = []
	p = 0
	for i in range(0, 5):
		temp = []
		for j in range(0, 5):
			temp.append(area[i][j] > tp)
		s.append(temp)
	valid = []
	for i in range(0, 5):
		temp = []
		for j in range(0, 5):
			temp.append(False)
		valid.append(temp)
	for i in range(1, 4):
		for j in range(1, 4):
			if(s[i][j]):
				if(i == 2 or j == 2):
					valid[i][j] =  True
				if(i == 1 and j == 1):
					if(diag):
						valid[i][j] =  True
					else:
						valid[i][j] = (s[2][1] or s[1][2])
				if(i == 3 and j == 1):
					if(diag):
						valid[i][j] =  True
					else:
						valid[i][j] = (s[2][1] or s[3][2])
				if(i == 1 and j == 3):
					if(diag):
						valid[i][j] =  True
					else:
						valid[i][j] = (s[1][2] or s[2][3])
				if(i == 3 and j == 3):
					if(diag):
						valid[i][j] =  True
					else:
						valid[i][j] = (s[2][3] or s[3][2])
			else:
				valid[i][j] = False
	#Group A - center edges
	if(s[0][2]):
		if(diag):
			valid[0][2] = (s[1][2] or s[1][1] or s[1][3])
		else:
			valid[0][2] = valid[1][2]
	if(s[2][0]):
		if(diag):
			valid[2][0] = (s[2][1] or s[1][1] or s[3][1])
		else:
			valid[2][0] = s[2][1]
	if(s[2][4]):
		if(diag):
			valid[2][4] = (s[2][3] or s[1][3] or s[3][3])
		else:
			valid[2][4] = s[2][3]
	if(s[4][2]):
		if(diag):
			valid[4][2] = (s[3][2] or s[3][1] or s[3][3])
		else:
			valid[4][2] = s[3][2]
	#Group B - offcenter edges
	if(s[0][1]):
		if(diag):
			valid[0][1] = (valid[0][2] or valid[1][1] or valid[1][2])
		else:
			valid[0][1] = (valid[0][2] or valid[1][1])
	if(s[0][3]):
		if(diag):
			valid[0][3] = (valid[0][2] or valid[1][3] or valid[1][2])
		else:
			valid[0][3] = (valid[0][2] or valid[1][3])
	if(s[1][0]):
		if(diag):
			valid[1][0] = (valid[1][1] or valid[2][0] or valid[2][1])
		else:
			valid[1][0] = (valid[1][1] or valid[2][0])
	if(s[1][4]):
		if(diag):
			valid[1][4] = (valid[1][3] or valid[2][4] or valid[2][3])
		else:
			valid[1][4] = (valid[1][3] or valid[2][4])
	if(s[3][0]):
		if(diag):
			valid[3][0] = (valid[3][1] or valid[2][0] or valid[2][1])
		else:
			valid[3][0] = (valid[3][1] or valid[2][0])
	if(s[3][4]):
		if(diag):
			valid[3][4] = (valid[2][4] or valid[3][3] or valid[2][3])
		else:
			valid[3][4] = (valid[2][4] or valid[3][3])
	if(s[4][1]):
		if(diag):
			valid[4][1] = (valid[3][1] or valid[4][2] or valid[3][2])
		else:
			valid[4][1] = (valid[3][1] or valid[4][2])
	if(s[4][3]):
		if(diag):
			valid[4][3] = (valid[3][3] or valid[4][2] or valid[3][2])
		else:
			valid[4][3] = (valid[3][3] or valid[4][2])
	#Group C - corners
	if(s[0][0]):
		if(diag):
			valid[0][0] = (valid[1][0] or valid[0][1] or valid[1][1])
		else:
			valid[0][0] = (valid[1][0] or valid[0][1])
	if(s[0][4]):
		if(diag):
			valid[0][4] = (valid[0][3] or valid[1][4] or valid[1][3])
		else:
			valid[0][4] = (valid[0][3] or valid[1][4])
	if(s[4][0]):
		if(diag):
			valid[4][0] = (valid[3][0] or valid[4][2] or valid[3][1])
		else:
			valid[4][0] = (valid[3][0] or valid[4][2])
	if(s[4][4]):
		if(diag):
			valid[4][4] = (valid[3][4] or valid[4][3] or valid[3][3])
		else:
			valid[4][4] = (valid[3][4] or valid[4][3])
	#print()
	#for i in range(0, 5):
	#	print(valid[i])
	for i in range(0, 5):
		for j in range(0, 5):
			if(valid[i][j]):
				p += area[i][j]
	shp = ""
	for i in range(4, -1, -1):
		for j in range(0, 5):
			if(valid[i][j]):
				shp += "1"
			else:
				shp += "0"
	s = int(shp, 2)
	return p, s, valid


def fileAnalysis(filePath, darkPath, thres, thresp, savePath='default', hthr=7, hrep=6, verbose=False, diagonals=True):
	'''
	Performs analysis of files
	Parameters:
		filePath - path to fits images (dont include number of .FTS in path)
		darkPath - path to dark frames
		thres - threshold of events
		thresp - threshold to be summed
		savePath (default 'output/default') - path to output file 
		hthr (default 10) - threshold for hotpixels
		hrep - percent of images hotpixels must occur in to be disqualified
		verbose (default False) - print out more than just progress bar
		diagonals (default True) - sum along diagonals in processing?
	'''

	ep, ex, ey, shp = [], [], [], []
	nx, ny = 1936, 1096
	dx, dy = 2, 2
	darkFrame = avgDark(darkPath, verbose=verbose)
	files = glob.glob(filePath + "?.FTS") + glob.glob(filePath + "??.FTS") + glob.glob(filePath + "???.FTS")+ glob.glob(filePath + "????.FTS") + glob.glob(filePath + "?????.FTS")
	if (verbose): print("Performing analysis of " + str(len(files)) + " images")
	hotPixels = getBadPix(darkPath, thr=hthr, rep=hrep, verbose=verbose)
	fname = []
	for c in range(0, len(files)):
		dat = pyfits.open(files[c])
		if (len(dat) == 1):
			imDat = dat[0].data
		else:
			imDat = dat[1].data
		dat.close()
		phot = imDat - darkFrame
		phot[phot < 0] = 0
		for ii in range(0, len(hotPixels)):
			phot[int(hotPixels[int(ii)][0])][int(hotPixels[int(ii)][1])] = 0
		f1 = np.reshape(phot, nx*ny)
		q = np.argsort(f1)[::-1]
		j = 0
		above_thres = True
		while above_thres:
			if (j >= 2121856):
				above_thres = False
				break
			i = q[j]
			j += 1
			if (f1[i] >= thres):
				x = (i % 1936) + 1 #x coordinate in image
				y = math.floor((i / 1936) + 1) #y coordinate in image
				xR = int(math.floor(i/1936)) #x coordinate in array
				yR = int(i % 1936) #y coordinate in array
				if (xR > dx) and (xR < ny-dx-1) and (yR > dy) and (yR < nx-dy-1):
					area = phot[(xR-dx):(xR+dx+1), (yR-dy):(yR+dy+1)]
					#print(area)
					#print(xR)
					#print(yR)
					p, s, v = getEp(area, thresp, diag=diagonals)
					for xi in range(xR - dx, xR + dx + 1):
						for yi in range(yR - dy, yR + dy + 1):
							if (v[xi-xR-dx][yi-yR-dy]):
								phot[xi, yi] = 0
					if (p > 0):
						ep.append(p)
						ex.append(x)
						ey.append(y)
						shp.append(s)
						fname.append(str(files[c]))
			else:
				above_thres = False
		prog = (c/len(files)) * 100
		printProgressBar(prog, 100, prefix = 'Progress: ', suffix='Files processed', length=50)
	ep = np.array(ep)
	ex = np.array(ex)
	ey = np.array(ey)
	shp = np.array(shp)
	fi = open(savePath + ".csv", 'w')
	for cntr in range(0, len(ep)):
		fi.write(str(ep[cntr]) + ', ' + str(ex[cntr]) + ', ' + str(ey[cntr]) + ':' + str(shp[cntr]) + ", " + str(fname[cntr]) + "\n")
	fi.close()
	if (verbose):
		print(str(len(ep)) + " total events written")
	return [ep, ex, ey, shp]

def skewgaus(data, saveName, title, saveImg=False, energy=True, plot=False):
	'''
	:param data: Input ep Array
	:param saveName: File Save Naame
	:param title: Plot Title
	:param saveImg: Default False, True saves Image according to SaveName
	:param energy: Default True, Returns in ADU space if False
	:return: Fit Parameters
	'''
	m = aduM #0.0037168339915133394   # FE-55 Calibration Data
	b = aduB # 0.002116795017471418
	bins = int((max(data) - min(data))/2)   # Binning
	(counts, edges) = np.histogram(data, bins=bins) # Counts/bin and bins
	n = len(counts)
	chan = 0.5 * (edges[0:n] + edges[1:n + 1])   # Center of bins
	if energy == True:
		chan = aduToeV(chan)   # Convert bin center to eV
		cerr = 1 + np.sqrt(counts + 0.75)	# Standard Error
		cerr = cerr[:-1]
		chan = chan[:-1]
		counts = counts[:-1]
		mod = lmfit.models.SkewedGaussianModel(prefix = 'skg')  # Define Skewed Model
		pars = mod.guess(counts,chan)	   # Let LMFIT guess intial values
		out = mod.fit(counts, pars, x=chan,weights = 1/cerr)  # Perform the fit
		skg_norm = out.best_values['skgamplitude']
		skg_cent = out.best_values['skgcenter']
		skg_sigma = out.best_values['skgsigma']
		skg_gamma = out.best_values['skggamma']
		skg_height = 0.3989423*skg_norm/max(2.220446049250313e-16, skg_sigma)  # The height (# counts/bin)
		r_chisq = out.redchi  # Reduced Chi Square
		skg_fwhm = 2.3548200*skg_sigma   # FWHM
		covar = out.covar  # Find Errors
		err_norm = np.sqrt(abs(covar[0][0]))
		err_cent = np.sqrt(abs(covar[1][1]))
		err_sigma = np.sqrt(abs(covar[2][2]))
		err_gamma = np.sqrt(abs(covar[3][3]))
		err_fwhm = np.sqrt((err_sigma*2.35848200)**2)
		err_height = np.sqrt(((0.3989423 * err_norm)/(skg_sigma))**2 + ((0.3989423 * err_sigma*skg_norm)/(skg_sigma**2))**2)
		print(out.fit_report(min_correl=0.5))   # Print out fitting results
		#fig, axes = plt.subplots(1, figsize=(10, 8))
		#plt.xlabel('Energy (eV)')
		#plt.ylabel('Counts/Bin')
		#plt.title(title)
		#textstr = '\n'.join((
		#	r'Height = %.2f +- %.2f' % (skg_height,err_height),
		#	r'Centroid = %.2f +- %.2f eV' % (skg_cent,err_cent),
		#	r'FWHM = %.2f +- %.2f eV' % (skg_fwhm,err_fwhm),
		#	r'Gamma = %.2f +- %.2f' % (skg_gamma, err_gamma),
		#	r'$\chi^2/dov $= %.2f' % (r_chisq)))
		#props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		#axes.text(.22, .89, textstr, fontsize=8, ha='center', va='center', transform=axes.transAxes, bbox=props)
		#axes.errorbar(chan, counts, cerr, fmt='.')
		#axes.plot(chan, out.best_fit, 'r-', label='best fit')
		#axes.legend(loc='best')
		if(plot):
			plt.ioff()
			plt.show()
		if (saveImg) == True:
			fig.savefig(saveName)
		return (skg_height,err_height,skg_cent,err_cent,skg_fwhm,err_fwhm,skg_gamma, err_gamma)
	else:
		chan = aduToeV(chan)
		cerr = 1 + np.sqrt(counts + 0.75)
		cerr = cerr[:-1]
		chan = chan[:-1]
		counts = counts[:-1]
		mod = lmfit.models.SkewedGaussianModel(prefix='skg')
		pars = mod.guess(counts, chan)
		out = mod.fit(counts, pars, x=chan, weights=1 / cerr)
		skg_norm = out.best_values['skgamplitude']
		skg_cent = out.best_values['skgcenter']
		skg_sigma = out.best_values['skgsigma']
		skg_gamma = out.best_values['skggamma']
		skg_height = 0.3989423 * skg_norm / max(2.220446049250313e-16, skg_sigma)
		r_chisq = out.redchi
		skg_fwhm = 2.3548200 * skg_sigma
		covar = out.covar
		err_norm = np.sqrt(abs(covar[0][0]))
		err_cent = np.sqrt(abs(covar[1][1]))
		err_sigma = np.sqrt(abs(covar[2][2]))
		err_gamma = np.sqrt(abs(covar[3][3]))
		err_fwhm = np.sqrt((err_sigma * 2.35848200) ** 2)
		err_height = np.sqrt(
			((0.3989423 * err_norm) / (skg_sigma)) ** 2 + ((0.3989423 * err_sigma * skg_norm) / (skg_sigma ** 2)) ** 2)
		print(out.fit_report(min_correl=0.5))
		#fig, axes = plt.subplots(1, figsize=(10, 8))
		#plt.xlabel('Energy (eV)')
		#plt.ylabel('Counts/Bin')
		#plt.title(title)
		#textstr = '\n'.join((
		#	r'Height = %.2f +- %.2f' % (skg_height, err_height),
		#	r'Centroid = %.2f +- %.2f eV' % (skg_cent, err_cent),
		#	r'FWHM = %.2f +- %.2f eV' % (skg_fwhm, err_fwhm),
		#	r'Gamma = %.2f +- %.2f' % (skg_gamma, err_gamma),
		#	r'$\chi^2/dov $= %.2f' % (r_chisq)))
		#props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		#axes.text(.22, .89, textstr, fontsize=8, ha='center', va='center', transform=axes.transAxes, bbox=props)
		#axes.errorbar(chan, counts, cerr, fmt='.')
		#axes.plot(chan, out.best_fit, 'r-', label='best fit')
		#axes.legend(loc='best')
		if(plot):
			plt.ioff()
			plt.show()
		if (saveImg) == True:
			fig.savefig(saveName)
		return (skg_height, err_height, skg_cent, err_cent, skg_fwhm, err_fwhm, skg_gamma, err_gamma)

