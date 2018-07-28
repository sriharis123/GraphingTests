#!python
# 15 July 2018


import numpy as np

class SPEfile(object):

    def __init__(self, filename):

        self._f = open(filename, 'rb')

        self._footer = self._read_footer() # call _read_footer() method on file object
        self._version = self._footer['SPEversion']

        if self._version == 3:
            self._pixelFormat = self._footer['DataFormat']['Frame']['pixelFormat']
            self._numFrames = self._footer['DataFormat']['Frame']['count']
            self._FrameWidth = self._get_width() # aka xdim
            self._FrameHeight = int(self._get_height()) # aka ydim
            self._FrameSize = int(self._footer['DataFormat']['Frame']['size'])
            self._wavelengths = self._get_wvlngths()
        else:
            print("This file is not SPE version 3.")

    def read_at(self,pos,size,ntype):
        self._f.seek(pos)
        return np.fromfile(self._f,ntype,size)


# A single DataFormat element describes type and layout of image data.

# Data has the following layout:
# One frame of image data containing each region of interest (in the order defined)
# Followed by defined metadata for that frame (any combination of timestamps, frame tracking,
# gate tracking, gate tracking, and modulation tracking)
# Repeated for each frame.
# All frames have the same regions of interest in the same order.

# Frame size: total number of bytes required to store all pixels in all regions in one frame. Frame size is the sum of the (width x height x pixel size) of all ROIs in the frame. Pixel size depends on the pixel data type (pixelFormat) and will be either 2 or 4.
    # The following pixel data types are supported:
    # MonochromeUnsigned16 (2 bytes)
    # Monochrome Unsigned32 (4 bytes)
    # MonochromeFloating32 (4 bytes)
    # If pixelFormat="MonochromeUnsigned16",then for a region multiply width x height x 2 to get size.
    #  If pixelFormat were monochromeUnsigned32 or monochromeFloating32, then size would be width x height x 4.
# the ROI sizes add up to the Frame size
    # If for example, if there were two ROIs one of 175 x 1 and the other 125 x 1 and the pixel format was MonochromeUnsigned16 (2 bytes per pixel), the frame size would be (175 x 1 x 2) + (125 x 1 x 2 ) or 600 bytes.

# Frame stride: total number of bytes to skip to get to the beginning of the next frame from the start of a frame. A frame stride includes the frame pixel data and any frame metadata and/or padding.

# size of Frame = total number of bytes requires to store all pixels in all regions in 1 frame
# vs.
# height of region = number of rows of pixels in a region



# A frame contains the data read after the exposure time has ended. One frame equals one image. Number of frames determines how many images will be acquired and stored during the experiment. When Exposures per frame = 1, the number of exposures and number of frames are the same.

    def _read_footer(self):
        '''
        SPEformat = {'DataFormat':
                            {
                            'Frame':
                                {'count':"1",'pixelFormat':"MonochromeUnsigned16",'size':"11822",'stride'="11822",'calibrations'="1"},
                            'Region1':
                                {'calibrations':"2", 'count':"1", 'width':"5911", 'height':"1", 'size':"11822", 'stride':"11822"}
                            'Region2':
                                {'calibrations':"2", ...'}
                            }
                      'SPEversion':
                            3.0
                        }
        '''
        # Dict levels:
        # dict level1 is SPEformat
        # dict level2s are DataFormat, SPEversion
        # dict level3s are Frame, Region1, etc.


        # retrieve xml footer
        footer_pos = self.read_at(678,8,np.uint64)[0]
        self._f.seek(footer_pos)
        xmltext = self._f.read() #lines = f.read() returns all lines as one string in which lines’re separated by \n
        # (xmltext is all one line)

        SPEformat = {} # initialize SPEformat dict

        # separate DataFormat from the rest of the xml text
        splitAtDataFormat = str(xmltext).split("/DataFormat")
        xmltext = splitAtDataFormat[1] # remaining xml text after DataFormatBlock
        SPEformat['remaining xml text'] = xmltext

        # check SPE version
        preDataFormatBlock = splitAtDataFormat[0].split("DataFormat")[0].split()
        for i in preDataFormatBlock[1:]:
            pair = i.split("=")
            key = pair[0]
            if key == 'version':
                #key, value = pair[0], pair[1][1:-1]
                SPEformat['SPEversion'] = int( float(pair[1][1:-1]) )
            else:
                pass

        # make dictionary of DataFormat information
        DataFormatBlock = splitAtDataFormat[0].split("DataFormat")[1]
        DataFormatBlock = DataFormatBlock.split("><")[1:-2] # split DataFormatBlock into DataBlocks
        SPEformat['DataFormat'] = {} # initialize DataFormat dict
        for i in DataFormatBlock:
            if ("DataBlock" in i) and ("Frame" in i): # if DataBlock describes Frame
                DataBlock = i.split()
                #print("Frame DataBlock: ",DataBlock)
                for pair in DataBlock[1:]:
                    if "=" not in pair: #if it's not a pair
                        pass
                    else:
                        pairList = pair.split("=") # ['key','"value"']
                        key,value = pairList[0],pairList[1][1:-1]
                        if key == 'type':
                            keyname = value # these keys are level3 dicts
                            SPEformat['DataFormat'][keyname]=[]# =[(key,value),(key,value)]
                        else:
                            SPEformat['DataFormat'][keyname].append( (key,value) )
                SPEformat['DataFormat'][keyname] = dict( SPEformat['DataFormat'][keyname] )
            numRegions = 0
            if ("DataBlock" in i) and ("Region" in i): # if DataBlock describes a region
                numRegions += 1
                DataBlock = i.split()
                #print("Region DataBlock ",str(numRegions),": ",DataBlock)
                for pair in DataBlock[1:]:
                    if "=" not in pair:
                        pass
                    else:
                        pairList = pair.split("=")
                        key,value = pairList[0],pairList[1][1:-1]
                        if key == 'type':
                            keyname = value + str(numRegions)
                            SPEformat['DataFormat'][keyname] = []
                        else:
                            #keyname = "Region" + str(numRegions)
                            SPEformat['DataFormat'][keyname].append( (key,value) )
                SPEformat['DataFormat'][keyname] = dict(SPEformat['DataFormat'][keyname])
        SPEformat['DataFormat']['number of regions'] = numRegions

        return SPEformat

    def _get_width(self):
        '''returns width of Frame (sum of region widths) in pixels'''
        if self._numFrames == "1":
            DataFormat = self._footer['DataFormat']
            numRegions = DataFormat['number of regions']
            region_widths = []
            for i in range(1,numRegions+1):
                region_name = 'Region'+str(i)
                region_width = int(DataFormat[region_name]['width'])
                region_widths.append(region_width)
            frame_width = sum(region_widths)
            #print(region_widths)
            #print("frame width: "+str(frame_width))
        else:
            print("This program only works with data in 1 Frame.")
            frame_width = 0
        return frame_width

    def _get_height(self):
        '''returns height of Frame in pixels'''
        if self._numFrames == "1":
            DataFormat = self._footer['DataFormat']
            numRegions = DataFormat['number of regions']
            region_heights = []
            for i in range(1,numRegions+1):
                region_name = 'Region'+str(i)
                region_height = int(DataFormat[region_name]['height'])
                region_heights.append(region_height)
            ave_height = np.mean(region_heights)
            # print("region heights:")
            # print(region_heights)
            if ave_height == region_heights[0]:
                frame_height = ave_height
            else:
                print("ROIs have different heights.")
                frame_height = 0
        else:
            print("This program only works with data in 1 Frame.")
            frame_height = 0
        return frame_height


    def _get_wvlngths(self):
        wavelengths = []
        xmltext = self._footer['remaining xml text']
        WavelengthMapping = xmltext.split('</Wavelength>')[0]
        WavelengthMappingList = WavelengthMapping.split('><')
        for i in WavelengthMappingList:
            if 'Wavelength ' in i or 'Wavelength>' in i:
                wavelengthsStr = i.split('>')[1]
        wavelengths = np.array([float(n) for n in wavelengthsStr.split(',')]) # replace w apply fn)
        # print("wavelengths: ",wavelengths)
        # print("len(wavelengths): ", len(wavelengths))
        return wavelengths

    def load_img(self):
        '''returns the binary pixel data'''

        if self._pixelFormat == 'MonochromeUnsigned16':
            imageSize =  int(self._FrameSize / 2) # = self._FrameHeight * self._FrameWidth
            img = self.read_at(4100, imageSize, np.uint16)
            #img = self.read_at(4100, self._FrameWidth, np.uint16)
            #print(len(img))
        elif self._pixelFormat == 'MonochromeUnsigned32':
            imageSize =  int(self._FrameSize / 4)
            img = self.read_at(4100, imageSize, np.uint32)
            #img = self.read_at(4100, self._FrameWidth, np.uint32)
        else:
            print("pixel format is neither 16- not 32-bit unsigned integer")
            img = np.zeros(self._FrameHeight, self._FrameWidth)
        # print( '(FrameHeight, FrameWidth) = ' + str((self._FrameHeight, self._FrameWidth)) )

        #return np.reshape(img,(self._FrameHeight, self._FrameWidth))
        return img



def load(filename):
    '''returns wavelengths and image data'''
    f = SPEfile(filename) #f is initialized file object
    img = f.load_img()
    return (f._wavelengths, img)
    #return f._wavelengths
