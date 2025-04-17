
# Robert Cudmore
# 20191119


import sys, time, math
import numpy as np

from skimage.measure import profile
from scipy.optimize import curve_fit
from skimage.draw import polygon
import scipy.signal

#from multiprocessing import Pool
import multiprocessing

class ShapeAnalysis:
	def __init__(self, data):
		"""
		data: 3d image data
		"""
		self.data = data

	def fitGaussian(self, x, y):
		"""
		x: np.ndarray of x, e.g. pixels or um
		y: np.ndarray of line intensity profile
        """
		n = len(x)                          #the number of data
		mean = sum(x*y)/n                   #
		sigma = sum(y*(x-mean)**2)/n        #
		#mean = sum(y)/n                   #
		#sigma = sum((y-mean)**2)/n        #
		'''
		print('fitGaussian mean:', mean)
		print('fitGaussian sigma:', sigma)
		'''

		def myGaussian(x, amplitude, mean, stddev):
		    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)

		# see: https://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
		def FWHM(X,Y):
			Y = scipy.signal.medfilt(Y, 3)
			half_max = max(Y) / 2.
			half_max = max(Y) * 0.7

			# for explanation of this wierd syntax
			# see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
			#whr = np.where(Y > half_max)
			#print('   half_max:', half_max)
			whr = np.asarray(Y > half_max).nonzero()
			if len(whr[0]) > 2:
				left_idx = whr[0][0]
				right_idx = whr[0][-1]
				fwhm = X[right_idx] - X[left_idx]
			else:
				left_idx = np.nan
				right_idx = np.nan
				fwhm = np.nan
			return fwhm, left_idx, right_idx #return the difference (full width)

		try:
			#print('x.shape:', x.shape)
			popt,pcov = curve_fit(myGaussian,x,y)
			yFit = myGaussian(x,*popt)
			myFWHM, left_idx, right_idx = FWHM(x,y)
			return yFit, myFWHM, left_idx, right_idx
		except RuntimeError as e:
			#print('... fitGaussian() error: ', e)
			return np.full((x.shape[0]), np.nan), np.nan, np.nan, np.nan
		except:
			print('\n... ... fitGaussian() error: exception in bAnalysis.fitGaussian() !!!')
			raise
			return np.full((x.shape[0]), np.nan), np.nan, np.nan, np.nan
			#return None, None, None, None

	@property
	def imageShape(self):
		""" return the shape of an individual image """
		if len(self.data.shape)==2:
			return self.data.shape
		elif len(self.data.shape)==3:
			return self.data[0,:,:].shape
		elif len(self.data.shape)==4:
			return self.data[0,0,:,:].shape

	@property
	def numImages(self):
		"""
		return number of images, either number of slice images in a stack or frames in a time-series
		"""
		if len(self.data.shape)==2:
			return 1
		elif len(self.data.shape)==3:
			# assuming (slice, row, col)
			return self.data.shape[0]
		elif len(self.data.shape)==4:
			# assuming (color, slice, row, col)
			return self.data.shape[1]

	def polygonAnalysis(self, slice, data):
		"""
		data: list of vertex points
		"""

		dataList = data.tolist()
		r = list(zip(*dataList))[0]
		c = list(zip(*dataList))[1]
		#channel = 0
		#myImageShape = self.stack.stack[channel,slice,:,:].shape
		#myImageShape = self.data[slice,:,:].shape
		(rr, cc) = polygon(r, c, shape=self.imageShape)
		if len(rr)==0 or len(cc)==0:
			return np.nan, np.nan, np.nan

		#roiImage = self.stack.stack[channel,slice,rr,cc] # extract the roi
		try:
			roiImage = self.data[slice,rr,cc] # extract the roi
			#print('roiImage:', roiImage, 'roiImage.shape', roiImage.shape, 'type(roiImage):', type(roiImage))
			theMin = np.nanmin(roiImage)
			theMax = np.nanmax(roiImage)
			theMean = np.nanmean(roiImage)
			return theMin, theMax, theMean
		except IndexError as e:
			print('*** IndexError exception in ShapeAnalysis.polygonAnalysis() e:', e)
			raise

	def polygonAnalysis2(self, slice):
		"""
		data: list of vertex points
		"""
		if slice % 300 == 0:
			print('   worker polygonAnalysis2() slice:', slice, 'of', self.numImages)
		try:
			roiImage = self.data[slice,self.rr, self.cc] # extract the roi
			#print('roiImage:', roiImage, 'roiImage.shape', roiImage.shape, 'type(roiImage):', type(roiImage))
			theMin = np.nanmin(roiImage)
			theMax = np.nanmax(roiImage)
			theMean = np.nanmean(roiImage)
			return theMin, theMax, theMean

		except IndexError as e:
			print('*** IndexError exception in ShapeAnalysis.polygonAnalysis() e:', e)
			raise

	def stackPolygonAnalysis(self, data):
		"""
		data: list of vertex points
		"""
		#numSlices = self.stack.numImages # will only work for [color,slice,x,y]
		minList = []
		maxList = [] # each element is intensity profile for one slice/image
		meanList = [] # each element is intensity profile for one slice/image
		#if numSlices < 500:
		doSingleThread= False
		if doSingleThread or self.numImages < 500:
			print('stackPolygonAnalysis performing loop through images')
			startTime = time.time()
			for idx, slice in enumerate(range(self.numImages)): # why do i need -1 ???
				if idx % 300 == 0:
					print('   idx:', idx, 'of', self.numImages)
				theMin, theMax, theMean = self.polygonAnalysis(slice, data)
				minList.append(theMin)
				maxList.append(theMax)
				meanList.append(theMean)
			stopTime = time.time()
			print(   '1) single-thread ', self.numImages, 'slices took', round(stopTime-startTime,3))
		else:
			numCPU = multiprocessing.cpu_count()
			chunksize = numCPU*200 #400 took 8.1 seconds, 200 takes 8 seconds, 100 takes 17 seconds, 50 takes 23 sec
			print('stackPolygonAnalysis using multiprocessing pool imape, num cpu:', numCPU, 'chunksize:', chunksize)
			print('   self.imageShape:', self.imageShape)
			#print('   ', self.data.shape, self.data.dtype)# create a list of parameters to function self.lineProfile as a tuple (slice, src, dst, linewidth)
			dataList = data.tolist()
			r = list(zip(*dataList))[0]
			c = list(zip(*dataList))[1]
			(rr, cc) = polygon(r, c, shape=self.imageShape)
			if len(rr)==0 or len(cc)==0:
				print('stackPolygonAnalysis() got empty analysis polygon: rr.shape:', rr.shape, 'cc.shape:', cc.shape)
				return None, None, None
			numImages = self.numImages
			#numImages = 100
			self.rr = rr # used by polygonAnalysis2 worker
			self.cc = cc
			startTime = time.time()
			myIterable = [a for a in range(numImages)]
			with multiprocessing.Pool(processes=numCPU-1) as p:
				# previously tried starmap but it always ran out of memory?
				minList, maxList, meanList = zip(*p.imap(self.polygonAnalysis2, myIterable, chunksize=chunksize))
			stopTime = time.time()
			print('2) multi-thread stackPolygonAnalysis for', self.numImages, 'slices took', round(stopTime-startTime,3))
		return np.asarray(minList), np.asarray(maxList), np.asarray(meanList)

	def lineProfile(self, slice, src, dst, linewidth=3, doFit=True):
		""" one slice

		Returns:

		x: ndarray, one point for each point in the profile (NOT images/slice in stack)
		"""
		#channel = 0
		#intensityProfile = profile.profile_line(self.stack.stack[channel,slice,:,:], src, dst, linewidth=linewidth)
		try:
			#print('self.data[slice,:,:].shape', self.data[slice,:,:].shape)
			intensityProfile = profile.profile_line(self.data[slice,:,:], src, dst, linewidth=linewidth)
			x = np.asarray([a for a in range(len(intensityProfile))]) # make alist of x points (todo: should be um, not points!!!)
			yFit, FWHM, left_idx, right_idx = self.fitGaussian(x,intensityProfile)
		except ValueError as e:
			print('!!!!!!!!!! *********** !!!!!!!!!!!!! my exception in lineProfile() ... too many values to unpack (expected 2)')
			return (None, None, None, None, None, None)
		
		return (x, intensityProfile, yFit, FWHM, left_idx, right_idx)

	def lineProfile2(self, slice):
		""" one slice

		Returns:

		x: ndarray, one point for each point in the profile (NOT images/slice in stack)
		"""
		if slice % 300 == 0:
			print('   worker lineProfile2() slice:', slice, 'of', self.numImages)
		try:
			intensityProfile = profile.profile_line(self.data[slice,:,:], self.src, self.dst, linewidth=self.linewidth)
			x = np.asarray([a for a in range(len(intensityProfile))]) # make alist of x points (todo: should be um, not points!!!)
			yFit, FWHM, left_idx, right_idx = self.fitGaussian(x,intensityProfile)
		except ValueError as e:
			print('!!!!!!!!!! *********** !!!!!!!!!!!!! my exception in lineProfile2() ... too many values to unpack (expected 2)')
			print('e:', e)
			return (None, None, None, None, None, None)
		return (x, intensityProfile, yFit, FWHM, left_idx, right_idx)

	def stackLineProfile(self, src, dst, linewidth=3):
		"""
		calculate line profile for each slice in a stack
		todo: fix the logic here, my self.lineProfile is returning too much
		"""
		print('stackLineProfile() src:', src, 'dst:', dst)
		print('   line length:', self.euclideanDistance(src, dst))

		xList = []
		intensityProfileList = [] # each element is intensity profile for one slice/image
		fwhmList = [] # each element is intensity profile for one slice/image
		# not sure what is going on here
		# a 3d stack of cd31 staining takes 3x longer when using multiprocessing?
		doSingleThread= False
		if doSingleThread or self.numImages < 500:
			print('   stackLineProfile performing loop through images')
			startTime = time.time()
			for idx, slice in enumerate(range(self.numImages)): # why do i need -1 ???
				if idx % 300 == 0:
					print('   idx:', idx, 'of', self.numImages)
				x, intensityProfile, yFit, fwhm, left_idx, right_idx = self.lineProfile(slice, src, dst, linewidth=linewidth, doFit=True)
				xList.append(x)
				intensityProfileList.append(intensityProfile)
				fwhmList.append(fwhm)
				#print(idx, fwhm)
			stopTime = time.time()
			print('1) single-thread line profile for', self.numImages, 'slices took', round(stopTime-startTime,3))
		else:
			# threaded
			print('   running line profile for all slices in PARALLEL, numImages:', self.numImages)
			numCPU = multiprocessing.cpu_count()
			chunksize = numCPU * 100 #50 takes 20 sec, 100 takes 15 sec, 200 takes 18.5 sec, 400 takes 41 sec
			print('   num cpu:', numCPU, 'chunksize:', chunksize)

			self.src = src
			self.dst = dst
			self.linewidth = linewidth
			myIterable = [a for a in range(self.numImages)] 
			startTime = time.time()
			with multiprocessing.Pool(processes=numCPU-1) as p:
				xList, intensityProfileList, yFit, fwhmList, left_idx, right_idx = zip(*p.imap(self.lineProfile2, myIterable, chunksize=chunksize))
			stopTime = time.time()
			print('2) multi-thread line-profile for', self.numImages, 'slices took', round(stopTime-startTime,3))

		return np.asarray(xList), np.asarray(intensityProfileList), np.asarray(fwhmList)

	def euclideanDistance(self, pnt1, pnt2):
		"""
		given 2d/3d points, return the straight line (euclidean) distance btween them

		remeber, this is not accurate for 3d points as our Z is huge in scanning microscopy!!!
		"""
		if len(pnt1) == 2:
			return math.sqrt((pnt1[0]-pnt2[0])**2 + (pnt1[1]-pnt2[1])**2)
		elif len(pnt1) == 3:
			return math.sqrt((pnt1[0]-pnt2[0])**2 + (pnt1[1]-pnt2[1])**2 + + (pnt1[2]-pnt2[2])**2)
