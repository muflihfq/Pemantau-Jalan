import numpy as np

class VideoAtribut():

	def __init__(self,IDVideo):
		# 0 = GAMBIR011
		# 1 = GAMBIR014
		# 2 = LENTENG
		# 3 = PULOGADUNG
		# 4 = KEBONSIRIH
		self.IDVideo = IDVideo
		self.roi_R = []
		self.roi_H = []


	#Mengambil 2 titik untuk garis hijau (x,y) 
	def getGreenPoint(self):
		#Array 3 Dimensi
		#[((PointX_1_Vid1, Point Y_1_Vid1), (Point X_2_Vid1, Point Y_2_Vid1))]
		greenPoint = [((1050,110),(70,550)),((1440,770),(0,415)),((1450,400),(600,300)),((1220,150),(550,500)),((900,0),(100,500))]

		if self.IDVideo <= len(greenPoint) - 1:
			return greenPoint[self.IDVideo]

		else :
			return 0

	#Mengambil 2 titik untuk garis merah (x,y)
	def getRedPoint(self):
		redPoint = [((1680,230),(565,1080)),((1450,400),(380,270)),((1440,770),(70,480)),((1690,100),(800,700)),((1500,120),(690,900))]
		
		if self.IDVideo <= len(redPoint) - 1:
			return redPoint[self.IDVideo]

		else :
			return 0

	#Mengambil region ROI hijau
	def getGreenROI(self):

		if self.IDVideo == 0:
			self.roi_H = np.array([(70,350),(580,140),(1100,190),(200,700)], np.int32)

		elif self.IDVideo == 1:
			self.roi_H = np.array([(0,415),(0,600),(1300,1080),(1380,650),(150,400)], np.int32)

		elif self.IDVideo == 2:
			self.roi_H = np.array([(1300,530),(490,380),(1100,200),(1370,350)], np.int32)

		elif self.IDVideo == 3:
			self.roi_H = np.array([(300,370),(700,200),(1330,150),(690,530)], np.int32)

		elif self.IDVideo == 4:
			self.roi_H = np.array([(0,300),(500,0),(1100,0),(370,580)], np.int32)

		else:
			return 0

		return self.roi_H

	#Mengambil region ROI merah
	def getRedROI(self):

		if self.IDVideo == 0:
			self.roi_R = np.array([(1350,200),(350,800),(870,1080),(1920,1080),(1920,280)], np.int32)

		elif self.IDVideo == 1:
			self.roi_R = np.array([(1380,550),(280,350),(980,130),(1450,150)], np.int32)

		elif self.IDVideo == 2:
			self.roi_R = np.array([(1270,600),(400,400),(0,480),(0,1080),(1100,1080)], np.int32)

		elif self.IDVideo == 3:
			self.roi_R = np.array([(1440,150),(800,600),(1920,1080),(1920,170)], np.int32)

		elif self.IDVideo == 4:
			self.roi_R = np.array([(1300,0),(530,680),(1200,1080),(1920,1080),(1920,0)], np.int32)

		else:
			return 0

		return self.roi_R

	#mengambil 1 titik di ROI
	def getROIValue(self,indicator):

		#mengambil titik keempat (hijau dan merah beda)
		if indicator == 'merah':
			return self.roi_R[1]

		#jika hijau
		else:
			return self.roi_H[1]


	def getTreshold(self,indicator):
		#(merah,hijau)
		treshold = [(10000,20000),(30000,40000),(30000,10000),(30000,10000),(30000,10000)]
		
		if indicator == 'merah':
			tresh = treshold[self.IDVideo][0]

		else:
			tresh = treshold[self.IDVideo][1]
		
		return tresh

	def getLength(self):
		if self.IDVideo == 0:
			length = 17.8

		elif self.IDVideo == 1:
			length = 7.8

		elif self.IDVideo == 2:
			length = 10.8
			
		elif self.IDVideo == 3:
			length = 10.4

		elif self.IDVideo == 4:
			length = 4.5

		else:
			return 0

		return length
		
	
	def getVersion(self):
	    
	    return 9

#GAMBIR 011
#HIJAU             #MERAH 				POLYGON
									   #//roi_H = np.array([(70,350),(580,140),(1660,230),(565,1080),(200,700)], np.int32)
#pt1_h = (1050,110)  pt1_r = (1680,230) roi_H = np.array([(70,350),(550,195),(1000,250),(200,700)], np.int32)
#pt2_h = (70,550)    pt2_r = (565,1080) roi_R = np.array([(1100,160),(200,700),(1000,1005),(1800,280)], np.int32)
#BATAS ======================
#20.000				#40.000

#GAMBIR 014
#HIJAU     			#MERAH
#pt1_h = (1440,770)  pt1_r = (1450,400)	 roi_H = np.array([(0,415),(0,600),(1300,1080),(1450,410),(400,290)], np.int32)
#pt2_h = (0,415)    pt2_r = (380,270)   roi_R = np.array([(1350,650),(70,380),(980,130),(1450,150)], np.int32)
#BATAS ======================
#40.000				#300.000

#LENTENG
#HIJAU              #MERAH
#pt1_h = (1450,400)  pt1_r = (1440,770) roi_H = np.array([(1250,730),(90,480),(1100,200),(1370,350)], np.int32)
#pt2_h = (600,300)   pt2_r = (70,480)	roi_R = np.array([(1300,500),(670,400),(100,670),(1200,900)], np.int32)
#BATAS ======================			//lima titik roi_R = np.array([(1300,500),(600,300),(70,480),(100,670),(1200,900)], np.int32)
#10.000				#30.000

#PULOGADUNG
#HIJAU				#MERAH
#pt1_h = (1220,150)  pt1_r = (1690,100) roi_H = np.array([(300,370),(700,200),(1630,150),(900,650)], np.int32)
#pt2_h = (550,500)   pt2_r = (800,700)  roi_R = np.array([(1270,150),(700,550),(1400,900),(1800,170)], np.int32)
#BATAS ======================
#10.000				#30.000

#KEBON SIRIH
#HIJAU 				#merah
#pt1_h = (900,0)     pt1_r = (1500,120) roi_H = np.array([(0,300),(500,0),(1600,0),(740,850)], np.int32)
#pt2_h = (100,500)   pt2_r = (690,900)  roi_R = np.array([(985,0),(300,550),(1200,1080),(1860,0)], np.int32)
#BATAS ============
#10.000             #30.000




	