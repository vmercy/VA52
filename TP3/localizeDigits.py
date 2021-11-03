import cv2
import numpy as np
from PIL import Image, ImageDraw

def getRanges(proj):
  """Turns horizontal or vertical pixels projection into ranges of row/columns containing non-zero elements

  Args:
      proj (np.array): Vector containing horizontal/vertical projection

  Returns:
      list: List of tuples containing start and end of non-zero elements
  """
  start = False
  end = False
  ranges = []
  for index, pixels in np.ndenumerate(proj):
    index = index[0]
    if pixels != 0 and not start:
      start = index
    if pixels == 0 and not end:
      end = index
    if start and end:
      if start<end:
        ranges.append([start, end])
      start = False
      end = False
  return ranges

def getDigitsCoordinates(imageFilename, outputFilename = None):
  """Localizes digits on learning base image

  Args:
      imageFilename (string): The learning base image filename
      outputFilename (string, optional): If non-empty, a copy of original image passed with imageFilename will be created under outputFilename with boxes around localized digits. Defaults to None.

  Returns:
      list: List of tuples of tuples containing coordinates of upper-left and bottom-right corners of each digit
  """
  im = cv2.imread(imageFilename, cv2.IMREAD_GRAYSCALE)
  im = 255 - im
  proj = np.sum(im,1)
  width = np.size(im, 1)
  horizontalRanges = getRanges(proj)
  i = Image.new('RGB', (width, proj.shape[0]), 'white')
  boxesCoordinates = []
  if outputFilename:
    originalImage = Image.open(imageFilename)
    draw = ImageDraw.Draw(originalImage)

  # Uncomment code below to save the horizontal projection histogram as 'result.png'
  """ m = np.max(proj)
  w = 500
  result = np.full((proj.shape[0],500),255)
  result = np.ascontiguousarray(result, dtype=np.uint8)
  for row in range(im.shape[0]):
    cv2.line(result, (0,row), (int(proj[row]*w/m),row), (0,0,0), 1)
  cv2.imwrite('result.png', result)"""

  for horizontalRange in horizontalRanges:
    lineOfDigits = im[horizontalRange[0]:horizontalRange[1], 0:width]
    verticalProj = np.sum(lineOfDigits,0)
    verticalRanges = getRanges(verticalProj)
    for verticalRange in verticalRanges:
      digitSubMatrix = im[horizontalRange[0]:horizontalRange[1], verticalRange[0]:verticalRange[1]]
      digitHorizontalProj = np.sum(digitSubMatrix, 1)
      digitTop = horizontalRange[0]
      digitBottom = horizontalRange[1]-1
      while not digitHorizontalProj[digitTop-horizontalRange[0]]:
        digitTop+=1
      while not digitHorizontalProj[digitBottom-horizontalRange[1]]:
        digitBottom-=1
      boxCoordinates = ((verticalRange[0],digitTop),(verticalRange[1],digitBottom))
      boxesCoordinates.append(boxCoordinates)
      if outputFilename:
        draw.rectangle(boxCoordinates,outline='red')
  if outputFilename:
    originalImage.save(outputFilename)
  return boxesCoordinates

if __name__ == "__main__":
  apprentissageFilename = "app.tif"
  imageWithBoxes = "chiffresEncadres.png"
  boxesCoordinates = getDigitsCoordinates(apprentissageFilename, imageWithBoxes)
  print(boxesCoordinates)