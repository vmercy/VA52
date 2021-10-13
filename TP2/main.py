import localizeDigits
import numpy as np
import math
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

def getDigitProfile(imageSubMatrix, nbProfilingRows):
  """Builds digit left and right normalized profile in one vector where left profile is stored in even indexes and right profile in odd ones

  Args:
      imageSubMatrix (np.array): The submatrix of the original set of digits containing only one digit to process
      nbProfilingRows (int): The number of rows where digit must be profiled
  
  Returns:
      np.array: Vector of size 2*nbProfilingRows containing left and right normalized profiles of processed digit
  """
  outputVector = []
  subImageWidth = np.size(imageSubMatrix,1)
  rowsHeight = np.size(imageSubMatrix, 0)/nbProfilingRows
  for row in range(nbProfilingRows):
    rowPosition = int(rowsHeight*row)
    leftProfile = 0
    rightProfile = 0
    while(imageSubMatrix[rowPosition,leftProfile]):
      leftProfile+=1
    while(imageSubMatrix[rowPosition,subImageWidth-1-rightProfile]):
      rightProfile+=1
    outputVector.append(leftProfile/subImageWidth)
    outputVector.append(rightProfile/subImageWidth)
  return np.array(outputVector)

def getDigitClassCenterVector(vectors):
  """Computes the center vector for a digit class

  Args:
      vectors (list): List of vectors

  Returns:
      list: The center vector
  """
  centerVector = []
  nbElementsByVector = len(vectors[0])
  for i in range(nbElementsByVector):
    sum = 0
    for vector in vectors:
      sum+=vector[i]
    centerVector.append(sum/nbElementsByVector)
  return centerVector

def learnDigits(learningBaseImage, nbDigitsByClass, nbProfilingRows):
  digitPositions = localizeDigits.getDigitsCoordinates(learningBaseImage)
  im = cv2.imread(learningBaseImage, cv2.IMREAD_GRAYSCALE)
  digitPositions = groupDigitsByClass(digitPositions, nbDigitsByClass)
  digitCenterVectors = []
  for digitClass in digitPositions:
    digitProfiles=[]
    for digit in digitClass:
      digitSubMatrix = im[digit[0][1]:digit[1][1],digit[0][0]:digit[1][0]]
      profile = getDigitProfile(digitSubMatrix, nbProfilingRows)
      digitProfiles.append(profile)
    digitCenterVectors.append(getDigitClassCenterVector(digitProfiles))
  return digitCenterVectors

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

def probability(profile, digit, classesCenters):
  """Computes the probability for a profile to belong to the class of a digit

  Args:
      profile (np.array): The profile to be tested
      digit (int): The hypothetical digit
      classesCenters (list): Center vectors of all classes of digits
  """
  return math.exp(-dist(profile, classesCenters[digit]))/computeProbSum(profile, classesCenters)

if __name__ == "__main__":
  nbLines = 20
  digitCenterVectors = learnDigits('app.tif',20,nbLines)
  testDigitsPositions = localizeDigits.getDigitsCoordinates('test.tif')
  im = cv2.imread('test.tif', cv2.IMREAD_GRAYSCALE)
  for digit in testDigitsPositions:
    digitSubMatrix = im[digit[0][1]:digit[1][1],digit[0][0]:digit[1][0]]
    profile = getDigitProfile(digitSubMatrix, nbLines)
    recognizedDigit = -1
    maxProb = 0
    for digitCandidate in range(10):
      newProb = probability(profile, digitCandidate, digitCenterVectors)
      if newProb>maxProb:
        maxProb=newProb
        recognizedDigit=digitCandidate
    print(recognizedDigit)
  #print(probability(digitCenterVectors[3],6,digitCenterVectors))