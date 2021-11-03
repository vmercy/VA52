import localizeDigits
import numpy as np
import math
import json
from json import JSONEncoder
import cv2

def groupDigitsByClass(digitsSamples, nbDigitsPerClass):
  """Reorders digit samples by class of digit

  Args:
      digitsSamples (list): raw list of digit samples positions
      nbDigitsPerClass (int): number of digit samples per class of digit
  
  Returns:
      list: reordered digit samples positions
  """
  digitSamples = []
  for i in range(len(digitsSamples)//nbDigitsPerClass):
    digitSamples.append([])
  for index, digit in enumerate(digitsSamples):
    digitSamples[index//nbDigitsPerClass].append(digit)
  return digitSamples

def getDigitZoning(imageSubMatrix, nbVertZones, nbHorZones):
  """Builds digit normalized zoning in one vector of dimension n*m where each element contains the number of black pixels

  Args:
      imageSubMatrix (np.array): The submatrix of the original set of digits containing only one digit to process
      nbVertZones (int): The number of vertical zoning areas (n)
      nbHorZones (int): The number of horizontal zoning areas (m)
  
  Returns:
      np.array: Vector of size n*m containing normalized densities for each zone
  """
  outputVector = []
  columnsWidth = math.ceil(np.size(imageSubMatrix,1)/nbHorZones)
  rowsHeight = math.ceil(np.size(imageSubMatrix, 0)/nbVertZones)
  surface = columnsWidth*rowsHeight
  for row in range(nbVertZones):
    rowStart = int(rowsHeight*row)
    for column in range(nbHorZones):
      columnStart = int(columnsWidth*column)
      subMatrix = imageSubMatrix[rowStart:rowStart+rowsHeight, columnStart:columnStart+columnsWidth]
      nbBlackPixels = np.size(subMatrix)-np.count_nonzero(subMatrix)
      nbBlackPixels/=surface
      outputVector.append(nbBlackPixels)
  return np.array(outputVector)

def learnDigits(learningBaseImage, nbDigitsByClass, nbVertZones, nbHorZones):
  digitPositions = localizeDigits.getDigitsCoordinates(learningBaseImage)
  im = cv2.imread(learningBaseImage, cv2.IMREAD_GRAYSCALE)
  digitPositions = groupDigitsByClass(digitPositions, nbDigitsByClass)
  digitVectors = []
  for digitClass in digitPositions:
    for digit in digitClass:
      digitSubMatrix = im[digit[0][1]:digit[1][1],digit[0][0]:digit[1][0]]
      profile = getDigitZoning(digitSubMatrix, nbVertZones, nbHorZones)
      digitVectors.append(profile)
  return digitVectors

def dist(vectorA, vectorB):
  """Computes the euclidean distance between two vectors A and B having the same dimension

  Args:
      vectorA (list): Vector A
      vectorB (list): Vector B

  Returns:
      int: The euclidean distance
  """
  return np.linalg.norm(np.array(vectorA)-np.array(vectorB))

def computeProbSum(profile, classesCenters):
  probSum = 0
  for classCenter in classesCenters:
    probSum+=math.exp(-dist(profile, classCenter))
  return probSum

def probability(digitClass, KNearestNeighbors):
  """Computes the probability for a profile to belong to the class of a digit

  Args:
  """
  return sum(neighbor[0]==digitClass for neighbor in KNearestNeighbors)/len(KNearestNeighbors)

def getKNearestNeighbors(K, digitDensityVector, digitsDensityVectors, nbDigitsPerClass):
  """Builds the list of k nearest neighbors of a digit in learning base

  Args:
      K (int): The number of expected nearest neighbors
      digitDensityVector (list): Density vector of digit to classify
      digitsDensityVectors (list): Flat list of density vectors of each digit in learning base
      nbDigitsPerClass (int): Number of digits per class of digits

  Returns:
      list: List of nearest neighbors as tuples in the form (class, distance)
  """
  distances = []
  for digitIndex, digitOfLearningBase in enumerate(digitsDensityVectors):
    distance = dist(digitDensityVector, digitOfLearningBase)
    distances.append((digitIndex//nbDigitsPerClass, distance))
  distances.sort(key=lambda i:i[1])
  return distances[0:K]

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def printClassifierStat(recognizedDigits, nbDigitsPerClass):
  """Displays stats about recognized digits

  Args:
      recognizedDigits (array): List of recognized digits
      nbDigitsPerClass (int): Number of digits in each class
  """
  nbSuccessByClass = [0]*10
  for index, recognizedDigit in enumerate(recognizedDigits):
    if(recognizedDigit==index//nbDigitsPerClass):
      nbSuccessByClass[index//nbDigitsPerClass]+=1
  nbSuccess = sum(nbSuccessByClass)
  print("%i digits successfully recognized = %4.2f %%"%(nbSuccess, nbSuccess))
  for (classIndex, successInClass) in enumerate(nbSuccessByClass):
    print('- class %i : %i success = %4.2f %%'%(classIndex, successInClass,100*successInClass//nbDigitsPerClass))

if __name__ == "__main__":
  n = 8
  m = 8
  nbDigitsPerClass = 20
  nbNearestNeighbors = 7
  digitsDensityVectors = learnDigits('app.tif', nbDigitsPerClass, n, m)
  with open('vectors.json','w') as vectorsOutfile:
    json.dump(digitsDensityVectors, vectorsOutfile, cls=NumpyArrayEncoder)
  testDigitsPositions = localizeDigits.getDigitsCoordinates('test.tif')
  im = cv2.imread('test.tif', cv2.IMREAD_GRAYSCALE)
  probVectors = []
  recognizedDigits = []
  for indexDigit, digit in enumerate(testDigitsPositions):
    probVectors.append([])
    digitSubMatrix = im[digit[0][1]:digit[1][1],digit[0][0]:digit[1][0]]
    profile = getDigitZoning(digitSubMatrix, n, m)
    nearestNeighbors = getKNearestNeighbors(nbNearestNeighbors, profile, digitsDensityVectors, nbDigitsPerClass)
    """ if(indexDigit==30):
      cv2.imwrite('testedDigit.png', digitSubMatrix)
      print(nearestNeighbors)
      break """
    recognizedDigit = -1
    maxProb = 0
    for digitCandidate in range(10):
      newProb = probability(digitCandidate, nearestNeighbors)
      probVectors[indexDigit].append(newProb)
      if newProb>maxProb:
        maxProb=newProb
        recognizedDigit=digitCandidate
    recognizedDigits.append(recognizedDigit)
    print(recognizedDigit)
  printClassifierStat(recognizedDigits, 10)
  with open('probabilities.json','w') as outfile:
    json.dump(probVectors,outfile)