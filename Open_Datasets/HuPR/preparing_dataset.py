# Description: This file is used to process the radar data from DCA1000 and generate the heatmap for the radar data.
# The code is based on the code from the following link: https://github.com/robert80203/HuPR-A-Benchmark-for-Human-Pose-Estimation-Using-Millimeter-Wave-Radar 
# ::HuPR-A-Benchmark-for-Human-Pose-Estimation-Using-Millimeter-Wave-Radar/preprocessing/process_iwr1843.py
# the radar configuration is from: HuPR-A-Benchmark-for-Human-Pose-Estimation-Using-Millimeter-Wave-Radar/config/mscsa_prgcn.yaml
# !!! we only process the radar data with the lables in the HuPR dataset



import json
import numpy as np
from PIL import Image
import os
import pickle
from tqdm import tqdm
from multiprocessing import Pool


class RadarObject():
    def __init__(self, start_file_idx=0, end_file_idx=1):
        '''
        start_file_idx: the start index of the processing list
        end_file_idx: the end index of the processing list

        the indexes of the radar and rgb samples for training, validation, and testing.
        the correpsonding 2D keypoints are stored in the json files in annotations folder.
        the number of the samples in trainset is:  193
        the number of the samples in testset is:  21
        the number of the samples in valset is:  21
        the items in the annotation file are:  dict_keys(['image', 'joints', 'bbox'])
        '''
        self.start_file_idx = start_file_idx
        self.end_file_idx = end_file_idx

        self.testName = [15, 16, 38, 40, 41, 42, 17, 39,  244, 245, 246, 249, 250, 251,
                         252, 253, 254, 247, 248, 255, 256
                         ]
        self.valName= [ 1, 14, 34, 57, 65,  98, 56, 99, 159, 178, 101, 120, 137, 156, 
                       161, 164, 181, 194, 197, 205, 257
                       ]
        self.trainName= [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 
                         22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 35, 36, 37, 43, 
                         44, 45, 46, 47, 48, 49, 50, 51, 52, 58, 59, 60, 61, 62, 
                         63, 64, 66, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 
                         84, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 31, 32, 
                         53, 54, 55, 67, 68, 69, 70, 72, 85, 86, 100, 157, 158, 
                         160, 167, 168, 169, 170, 177, 179, 180, 187, 188, 189, 190, 
                         228, 229, 230, 259, 260, 263, 264, 269, 270, 273, 274, 102,
                         103, 104, 105, 106, 107, 108, 109, 110, 119, 121, 122, 123,
                         124, 133, 134, 135, 138, 147, 148, 149, 150, 151, 152, 153,
                         154, 155, 162, 163, 165, 166, 171, 172, 173, 174, 175, 176, 
                         182, 183, 184, 185, 186, 191, 192, 193, 195, 196, 198, 199, 
                         200, 201, 202, 203, 204, 206, 207, 208, 209, 210, 211, 212,
                         213, 215, 216, 223, 224, 225, 226, 231, 232, 233, 234, 235, 
                         236, 258, 261, 262, 265, 266, 267, 268, 271, 272, 275, 276
                         ]
        self.idxToJoints= ["R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", "L_Ankle", 
                           "Neck", "Head", "L_Shoulder", "L_Elbow", "L_Wrist", "R_Shoulder", "R_Elbow", "R_Wrist"]
        
        self.sample_indexes = self.trainName + self.valName + self.testName
        # only processing the part we need:
        self.sample_indexes = self.sample_indexes[self.start_file_idx:self.end_file_idx]

        self.labels = None
        self.save_file_names = []
        self.radarDataFileNameGroup = []
        self.rgbFileNameGroup = []
        self.initialize()
        
        self.numADCSamples = 256
        self.adcRatio = 4
        self.numAngleBins = self.numADCSamples//self.adcRatio
        self.numEleBins = 8
        self.numRX = 4
        self.numLanes = 2
        self.framePerSecond = 10
        self.duration = 60
        self.numFrame = self.framePerSecond * self.duration
        self.numChirp = 64 * 3 
        self.idxProcChirp = 64
        self.numGroupChirp = 4

    def initialize(self):
        with open('annotations/hrnet_annot_train.json') as f:
            annot_train = json.load(f)
        with open('annotations/hrnet_annot_test.json') as f:
            annot_test = json.load(f)
        with open('annotations/hrnet_annot_val.json') as f:
            annot_val = json.load(f)

        self.labels = annot_train + annot_val + annot_test
        self.labels = self.labels[self.start_file_idx:self.end_file_idx]
        
        # if there is not dataset folder, create it to save the processed data
        if os.path.exists('radar_maps') == False:
            os.mkdir('radar_maps')
            
        for i in self.sample_indexes:
            radarDataFileName = ['radar_unzipped/single_' + str(i) + '/hori', 
                                'radar_unzipped/single_' + str(i) + '/vert']
            self.radarDataFileNameGroup.append(radarDataFileName)  # the raw radar data path
            
            rgb_image_folder = 'frames/single_' + str(i) + '/processed/images'
            self.rgbFileNameGroup.append(rgb_image_folder)  # the rgb image path
                
            save_file_name = 'radar_maps/single_' + str(i)
            # we save the radar heatmaps and the lables in the same file: a pickle file
            self.save_file_names.append(save_file_name)
    
    def postProcessFFT3D(self, dataFFT):
        dataFFT = np.fft.fftshift(dataFFT, axes=(0, 1,))
        dataFFT = np.transpose(dataFFT, (2, 0, 1))
        dataFFT = np.flip(dataFFT, axis=(1, 2))
        return dataFFT

    def getadcDataFromDCA1000(self, fileName):
        adcData = np.fromfile(fileName+'/adc_data.bin', dtype=np.int16)
        fileSize = adcData.shape[0]
        adcData = adcData.reshape(-1, self.numLanes*2).transpose()
        # # for complex data
        fileSize = int(fileSize/2)
        LVDS = np.zeros((2, fileSize))  # seperate each LVDS lane into rows

        temp = np.empty((adcData[0].size + adcData[1].size), dtype=adcData[0].dtype)
        temp[0::2] = adcData[0]
        temp[1::2] = adcData[1]
        LVDS[0] = temp
        temp = np.empty((adcData[2].size + adcData[3].size), dtype=adcData[2].dtype)
        temp[0::2] = adcData[2]
        temp[1::2] = adcData[3]
        LVDS[1] = temp

        adcData = np.zeros((self.numRX, int(fileSize/self.numRX)), dtype = 'complex_')
        iter = 0
        for i in range(0, fileSize, self.numADCSamples * 4):
            adcData[0][iter:iter+self.numADCSamples] = LVDS[0][i:i+self.numADCSamples] + np.sqrt(-1+0j)*LVDS[1][i:i+self.numADCSamples]
            adcData[1][iter:iter+self.numADCSamples] = LVDS[0][i+self.numADCSamples:i+self.numADCSamples*2] + np.sqrt(-1+0j)*LVDS[1][i+self.numADCSamples:i+self.numADCSamples*2]
            adcData[2][iter:iter+self.numADCSamples] = LVDS[0][i+self.numADCSamples*2:i+self.numADCSamples*3] + np.sqrt(-1+0j)*LVDS[1][i+self.numADCSamples*2:i+self.numADCSamples*3]
            adcData[3][iter:iter+self.numADCSamples] = LVDS[0][i+self.numADCSamples*3:i+self.numADCSamples*4] + np.sqrt(-1+0j)*LVDS[1][i+self.numADCSamples*3:i+self.numADCSamples*4]
            iter = iter + self.numADCSamples

        #correct reshape
        adcDataReshape = adcData.reshape(self.numRX, -1, self.numADCSamples)
        # print('Shape of radar data:', adcDataReshape.shape)
        return adcDataReshape

    def clutterRemoval(self, input_val, axis=0):
        """Perform basic static clutter removal by removing the mean from the input_val on the specified doppler axis.
        Args:
            input_val (ndarray): Array to perform static clutter removal on. Usually applied before performing doppler FFT.
                e.g. [num_chirps, num_vx_antennas, num_samples], it is applied along the first axis.
            axis (int): Axis to calculate mean of pre-doppler.
        Returns:
            ndarray: Array with static clutter removed.
        """
        # Reorder the axes
        reordering = np.arange(len(input_val.shape))
        reordering[0] = axis
        reordering[axis] = 0
        input_val = input_val.transpose(reordering)

        # Apply static clutter removal
        mean = input_val.transpose(reordering).mean(0)
        output_val = input_val - np.expand_dims(mean, axis=0)
        out = output_val.transpose(reordering)
        return out 

    def generateHeatmap(self, frame):
        # horizontal
        dataRadar = np.zeros((self.numRX*2, self.idxProcChirp, self.numADCSamples), dtype='complex_')
        # vertical
        dataRadar2 = np.zeros((self.numRX, self.idxProcChirp, self.numADCSamples), dtype='complex_')
        
        # Process radar data with TDM-MIMO
        for idxRX in range(self.numRX):
            for idxChirp in range(self.numChirp):
                if idxChirp % 3 == 0:
                    dataRadar[idxRX, idxChirp//3] = frame[idxRX, idxChirp]
                if idxChirp % 3 == 1:
                    dataRadar2[idxRX, idxChirp//3] = frame[idxRX, idxChirp]
                elif idxChirp % 3 == 2:
                    dataRadar[idxRX+4, idxChirp//3] = frame[idxRX, idxChirp]

        # step1: clutter removal
        dataRadar = np.transpose(dataRadar, (1, 0, 2))
        dataRadar = self.clutterRemoval(dataRadar, axis=0)
        dataRadar = np.transpose(dataRadar, (1, 0, 2))
        dataRadar2 = np.transpose(dataRadar2, (1, 0, 2))
        dataRadar2 = self.clutterRemoval(dataRadar2, axis=0)
        dataRadar2 = np.transpose(dataRadar2, (1, 0, 2))

        # step2: range-doppler FFT
        for idxRX in range(self.numRX * 2):
            dataRadar[idxRX, :, :] = np.fft.fft2(dataRadar[idxRX, :, :])
        for idxRX in range(self.numRX * 1):
            dataRadar2[idxRX, :, :] = np.fft.fft2(dataRadar2[idxRX, :, :])

        # step3: angle FFT
        padding = ((0, self.numAngleBins - dataRadar.shape[0]), (0,0), (0,0))
        dataRadar = np.pad(dataRadar, padding, mode='constant')
        padding2 = ((2, self.numAngleBins - 4 - 2), (0,0), (0,0))
        dataRadar2 = np.pad(dataRadar2, padding2, mode='constant')
        dataMerge = np.stack((dataRadar, dataRadar2))
        paddingEle = ((0, self.numEleBins - dataMerge.shape[0]), (0,0), (0,0), (0,0))
        dataMerge = np.pad(dataMerge, paddingEle, mode='constant')
        for idxChirp in range(self.idxProcChirp):
            for idxADC in range(self.numADCSamples):
                dataMerge[:, 2, idxChirp, idxADC] = np.fft.fft(dataMerge[:, 2, idxChirp, idxADC])
                dataMerge[:, 3, idxChirp, idxADC] = np.fft.fft(dataMerge[:, 3, idxChirp, idxADC])
                dataMerge[:, 4, idxChirp, idxADC] = np.fft.fft(dataMerge[:, 4, idxChirp, idxADC])
                dataMerge[:, 5, idxChirp, idxADC] = np.fft.fft(dataMerge[:, 5, idxChirp, idxADC])
                for idxEle in range(self.numEleBins):
                    dataMerge[idxEle, :, idxChirp, idxADC] = np.fft.fft(dataMerge[idxEle, :, idxChirp, idxADC])

        # select specific area of ADCSamples (containing signal responses)
        idxADCSpecific = [i for i in range(94, 30, -1)] # 84, 20
        rate = self.adcRatio

        # shift the velocity information
        dataTemp = np.zeros((self.idxProcChirp, self.numADCSamples//rate, self.numAngleBins, self.numEleBins), dtype='complex_')
        dataFFTGroup = np.zeros((self.idxProcChirp//self.numGroupChirp, self.numADCSamples//rate, self.numAngleBins, self.numEleBins), dtype='complex_')
        for idxEle in range(self.numEleBins):
            for idxRX in range(self.numAngleBins):
                for idxADC in range(self.numADCSamples//rate):
                    dataTemp[:, idxADC, idxRX, idxEle] = dataMerge[idxEle, idxRX, :, idxADCSpecific[idxADC]]
                    dataTemp[:, idxADC, idxRX, idxEle] = np.fft.fftshift(dataTemp[:, idxADC, idxRX, idxEle], axes=(0))
        
        # select specific velocity information
        chirpPad = self.idxProcChirp//self.numGroupChirp
        i = 0
        for idxChirp in range(self.idxProcChirp//2 - chirpPad//2, self.idxProcChirp//2 + chirpPad//2):
            dataFFTGroup[i, :, :, :] = self.postProcessFFT3D(np.transpose(dataTemp[idxChirp, :, :, :], (1, 2, 0)))
            i += 1
        # average along the velocity axis
        dataFFTGroup = np.mean(dataFFTGroup, axis=0)
        return dataFFTGroup  

    def processRadarDataHoriVert(self):
        # !!!!! To find the explaination of the processing steps, please refer to the README.ipynb
        for idxName in range(len(self.radarDataFileNameGroup)):
            print('Processing the radar data of the sample:', idxName, self.radarDataFileNameGroup[idxName])
            hori_map = []
            vert_map = []
    
            adcDataHori = self.getadcDataFromDCA1000(self.radarDataFileNameGroup[idxName][0])
            adcDataVert = self.getadcDataFromDCA1000(self.radarDataFileNameGroup[idxName][1])
            # for idxFrame in tqdm(range(0,self.numFrame)):
            for idxFrame in range(0,self.numFrame):
                frameHori = adcDataHori[:, self.numChirp*(idxFrame):self.numChirp*(idxFrame+1), 0:self.numADCSamples]
                frameVert = adcDataVert[:, self.numChirp*(idxFrame):self.numChirp*(idxFrame+1), 0:self.numADCSamples]
                outputHori = self.generateHeatmap(frameHori)
                outputVert = self.generateHeatmap(frameVert)
                
                hori_map.append(outputHori)
                vert_map.append(outputVert)
            
            result_dict = {
                'hori': hori_map,
                'vert': vert_map,
                'labels': self.labels[idxName],
                'rgb_image_folder': self.rgbFileNameGroup[idxName]
            }    
            # save the result as a pickle file
            with open(self.save_file_names[idxName] + '.pkl', 'wb') as f:
                pickle.dump(result_dict, f)
            print('The radar data of the sample:', idxName, 'is processed and saved successfully!')

def main_process(start_end_index):
    start_index = start_end_index[0]
    end_index = start_end_index[1]
    radarObject = RadarObject(start_index, end_index)
    radarObject.processRadarDataHoriVert()

if __name__ == "__main__":
    # multiple processes to process the radar data
    # with Pool(2) as p:
    #     p.map(main_process, [(1, 2), (3,4)])
        
    # processing from 4 to 235 with every process processing 10 samples
    a = [(i, i+10) for i in range(4, 236, 10)]
    b = a[:-2]
    b.append((224, 235))

    with Pool(23) as p:
        p.map(main_process, b)
    
