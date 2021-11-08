import json
import csv

def normalize(vector, max):
  newVector = []
  for element in vector:
    newVector.append(element/max)
  return newVector

def recognized(probVector, supposedDigit):
  maxProb = max(probVector)
  return (probVector.index(maxProb)==supposedDigit)

f = open('proba_profiling.json')
g = open('proba_zoning.json')

probas = []
for method in [f,g]:
  probas.append(json.load(method))

f.close()
g.close()

sum = []
product = []

for testedDigit in range(10):
  for digitClass in range(10):
    probSum = []
    probProd = []
    sumProbSum = 0
    sumProbProd = 0
    for proba in range(0,10):
      newProbSum = probas[0][10*testedDigit+digitClass][proba]+probas[1][10*testedDigit+digitClass][proba]
      newProbProd = probas[0][10*testedDigit+digitClass][proba]*probas[1][10*testedDigit+digitClass][proba]
      probSum.append(newProbSum)
      probProd.append(newProbProd)
      sumProbSum+=newProbSum
      sumProbProd+=newProbProd
    sum.append(normalize(probSum, sumProbSum))
    product.append(normalize(probProd, sumProbProd))

with open('sum.json','w') as outfile:
  json.dump(sum,outfile)

with open('product.json','w') as outfile:
  json.dump(product,outfile)

probas.append(sum)
probas.append(product)

#for profiling in probas:


#probas : profiling, zoning, sum, product

with open('histogramInput.csv','w',newline='') as csvfile:
  writer = csv.writer(csvfile)
  for digitClass in range(10):
    csvRow = []
    for testedMethod in probas:
      nbRecon = 0
      for digitSample in range(10):
        vector = testedMethod[10*digitClass+digitSample]
        nbRecon+=recognized(vector, digitClass)
      nbRecon*=10
      csvRow.append(nbRecon)
    writer.writerow(csvRow)