# boh vedi te
"""
media *
media ponderata *
stdev           *
chi quadro  *
error post  *
compatibilità
errore relativo
interpolazione prima
interpolazione seconda
interpolazione terza
# σ Δ τ α β γ θ λ μ χ π
# ·

#matplot default
blue:   #1f77b4
orange: #ff7f0e
green:  #2ca02c
red:    #d62728
purple: #9467bd
brown:  #8c564b
pink:   #e377c2
gray:   #7f7f7f
olive:  #bcbd22
cyan:   #17becf

usa gov
Nasa Red:     #FC3D21
Federal Blue: #001F3F
Postal Blue:  #1F4E79
Dark Green:   #006400
Burgundy:     #800020
Brown:        #654321
Gold:         #FFD700
White:        #FFFFFF
Black:        #000000

"""

pi = 3.141592653589793
e  = 2.718281828459045

# media aritmetica
def arythmAvg(array):
    sum = 0
    for x in array:
        sum += x

    return sum / len(array)


# media ponderata
def pondAvg(array, sArray):
    nomin = 0
    denom = 0
    i = 0
    for x in array:
        nomin += array[i] / sArray[i] ** 2
        denom += 1 / sArray[i] ** 2
        i += 1

    avgPond = nomin / denom
    sAvgPond = (1 / denom) ** 0.5
    return [avgPond, sAvgPond]


# dev std campionaria
def stDev(array):
    avg = arythmAvg(array)
    sum = 0
    for x in array:
        sum += (x - avg) ** 2

    return (sum / ((len(array) - 1))) ** 0.5

# dev std media
def stDevAvg(array):
    avg = arythmAvg(array)
    sum = 0
    for x in array:
        sum += (x - avg) ** 2

    return (sum / ((len(array) - 1)*len(array))) ** 0.5

#metodo casereccio per fare l'abs senza numpy
def toAbs(array):
    array2 = []
    for x in array:
        if x < 0:
            array2.append(-x)
        else:
            array2.append(x)
    return array2

# chi quadro
# ricorda valor vero di rif = y = a+bx quindi devi aggiungere x e
# quando si ha un valor vero di riferimento e nons erve calcolare l'equazione
# il valor vero viene salvato in a
def chi2(array, sArray, a, b=0, xArray=[1]):
    # se non viene dato un xarray si fa in modo che solamente il valore di a sia influente
    if len(xArray) == 1:
        xArray = [1] * len(array)
    sum = 0
    for i in range(0, len(array)):
        sum += ((array[i] - (a + b * xArray[i])) / sArray[i]) ** 2
        #print(((array[i] - (a + b * xArray[i])) / sArray[i]) ** 2)

    return sum

def chi2parab(x, y, sy, a, b, c):
    # se non viene dato un xarray si fa in modo che solamente il valore di a sia influente
    sum = 0
    for i in range(0, len(y)):
        sum += ((y[i] - (a*x[i]**2+b*x[i]+c)) / sy[i]) ** 2
        #print(((y[i] - (a*x[i]**2+b*x[i]+c)) / sy[i]) ** 2)

    return sum


def errorPost(array, k, a, b=0, xArray=1):
    # se non viene dato un xarray si fa in modo che solamente il valore di a sia influente
    if len(xArray) == 1:
        xArray = [1] * len(array)
    sum = 0
    for i in range(0, array):
        sum += (array[i] - (a + b * xArray[i])) ** 2

    return (sum / k) ** 0.5


def compatibility(x1, sx1, x2, sx2):
    return abs(x1 - x2) / (sx1 ** 2 + sx2 ** 2) ** 0.5


def interpol1(x, y, sy):
    sumx2 = 0
    for i in range(0, len(x)):
        sumx2 += x[i] ** 2

    sumx = 0
    for i in range(0, len(x)):
        sumx += x[i]

    sumy = 0
    for i in range(0, len(y)):
        sumy += y[i]

    sumxy = 0
    for i in range(0, len(x)):
        sumxy += x[i] * y[i]

    delta = len(x) * sumx2 - sumx ** 2

    a = (sumx2 * sumy - sumx * sumxy) / delta
    sa = sy * (sumx2 / delta) ** 0.5
    b = (len(x) * sumxy - sumx * sumy) / delta
    sb = sy * (len(x) / delta) ** 0.5
    return [a, sa, b, sb]


def interpol2(x, y, sy):
    sum1sy = 0
    for i in range(0, len(sy)):
        sum1sy += 1 / sy[i] ** 2

    sumysy = 0
    for i in range(0, len(sy)):
        sumysy += y[i] / sy[i] ** 2

    sumxysy = 0
    for i in range(0, len(sy)):
        sumxysy += (x[i] * y[i]) / sy[i] ** 2

    sumx2sy = 0
    for i in range(0, len(sy)):
        sumx2sy += x[i] ** 2 / sy[i] ** 2

    sumxsy = 0
    for i in range(0, len(sy)):
        sumxsy += x[i] / sy[i] ** 2

    delta = sum1sy * sumx2sy - sumxsy ** 2

    a = (sumx2sy * sumysy - sumxsy * sumxysy) / delta
    sa = (sumx2sy / delta) ** 0.5
    b = (sum1sy * sumxysy - sumxsy * sumysy) / delta
    sb = (sum1sy / delta) ** 0.5
    return [a, sa, b, sb]


def interpol3(x, sx, y, sy):
    si = []
    b = interpol2(x, y, sy)[2]
    for i in range(0, len(sy)):
        si.append((sy[i] ** 2 + b ** 2 * sx ** 2) ** .5)

    sum1si = 0
    for i in range(0, len(si)):
        sum1si += 1 / si[i] ** 2

    sumysi = 0
    for i in range(0, len(si)):
        sumysi += y[i] / si[i] ** 2

    sumxysi = 0
    for i in range(0, len(si)):
        sumxysi += (x[i] * y[i]) / si[i] ** 2

    sumx2si = 0
    for i in range(0, len(si)):
        sumx2si += x[i] ** 2 / si[i] ** 2

    sumxsi = 0
    for i in range(0, len(si)):
        sumxsi += x[i] / si[i] ** 2

    delta = sum1si * sumx2si - sumxsi ** 2

    a = (sumx2si * sumysi - sumxsi * sumxysi) / delta
    sa = (sumx2si / delta) ** 0.5
    b = (sum1si * sumxysi - sumxsi * sumysi) / delta
    sb = (sum1si / delta) ** 0.5
    #print(delta)
    return [a, sa, b, sb]

#when working with a multidimensional array
#just take one column in consideration
def columnizer(array,column=0):
    array2 = []
    for x in array:
        array2.append(x[column])
    return array2

def gauss(x,sigma,mu):
    return (1/((6.28**0.5)*sigma))*2.72**(-0.5*((x-mu)/sigma)**2)

def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def findMax(array):
    max = 0
    for x in array:
        if x > max:
            max = x
    return x
