import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

import mathFunctionis as mf

"""
Parametri Pendolo
massa pesetto= 0.1155 kg
diametro pesetto = 22.7 mm
L = 34 mm
lunghezza filo 0.75 m
angolo di rotazione pesetto radianti
tempo trascorso
frequenza forzante
"""
"""
L = 0.75
df = 0.0004
rf = df/2

dp = 0.0227
rp = dp/2
m = 0.115

I = (7.44*10**-6)
"""

# SETTINGS
# folder path
dir_path = r'C:source/'

#a      Grafico completo
#b      Fase smorzata
#c      Grafico completo in Absolute Value
#d      Fase smorzata in Absolute Value
#e      Grafico Fase di Stabilità
#f      Grafico Linearizzazione Massimi
#3s     #Istogramma Massimi e Minimi con   sistematiche in Absolute value
#g      #Istogramma Massimi e Minimi senza sistematiche in Absolute value
#gM     #Istogramma Massimi senza sistematiche in Absolute value
#gm     #Istogramma Minimi  senza sistematiche in Absolute value
#t      #Istogramma Periodi
#1      grafico contenente figura a b e
#2      grafico contenente figura c d f
#3      grafico contenente figrua gm g gM

#esempio di array contenente tutti i grafici
#charts = ["a","b","c","d","e","f","g","gM","gm","3s","o_s","g_s","t"]#scrivere il numero di figure da stampare, es["a","b","3s","gm"]
#charts = [1,2,3]
#Se si vuole solamente fare l'analisi dati allora è consigliato mettere l'array charts vuoto e slideshow e save come False
charts = [1]
#Se slideshow è False e Save è false i grafici verranno mostrati fino a chiusura manuale della figura
#slideshow mostra i grafici su schermo per 3 secondi prima di cambiare file
slideshow = False  #Impostare su True per vedere i grafici durante l'elaborazione ATTENZIONE: consigliato selezionare solamente se si hanno pochi grafici
save = True       #Impostare su True per salvare i grafici

overfitting = False#Funzione che forza il fitting dell'Ampiezza in modo che sia variabile"è solamente un effetto grafico nei calcoli viene comunque mantenuta l'Ampiezza originale"
compTrueMax = False #Calcola compatibilità tra massimi con fit parabolico e massimi registrati
                   #Disabilitare se si vuole fare un calcolo veloce dato che questo è molto molto molto dispendioso
start = 0          #Numero di file da saltare durante l'analisi

def findMax(array):
    startDeNoiser = 3
    deNoiser = 0  # con il buffer controllo che ci siano almeno 3 valori di fila dello stesso segno prima di iniziare a cercare il massimo
    max = 0.0
    maxIndex = 0
    maxArray = []
    i = 0
    for x in array:
        if x[2] >= 0:
            if deNoiser == 0:
                maxArray.append(maxIndex)  # salva la posizione del massimo
                max = x[2]
                deNoiser = startDeNoiser  # reset del buffer
            elif max < 0:  # se stavamo cercando il segno - e incontriamo il segno + diminuiamo il buffer
                deNoiser -= 1
            elif x[2] > max:
                max = x[2]
                maxIndex = i
        else:  # se negativo
            if deNoiser == 0:
                # print(max)#printa il massimo trovato con l'altro segno prima di iniziare a cercare con il meno
                # print("+")
                maxArray.append(maxIndex)  # salva la posizione del massimo
                max = x[2]
                deNoiser = startDeNoiser  # reset del buffer
            elif max >= 0:  # se stavamo cercando il segno + e incontriamo il segno - diminuiamo il buffer
                deNoiser -= 1
            elif x[2] < max:
                max = x[2]
                maxIndex = i
        i += 1
    return maxArray

#gli va dato l'array con i stabili
def findTrueMax(array,maxIndex):
    trueMax = []
    window = 10#quanti numeri a destra e sinistra prende dal massimo misurato

    for i in range(1,len(maxIndex)):
        arrayFitx = []#Array contenente i punti per fare il fit x
        arrayFity = []  # Array contenente i punti per fare il fit y
        for j in range(maxIndex[i]-window, maxIndex[i]+window):
            arrayFitx.append(array[j][0])
            arrayFity.append(abs(array[j][2]))

        parabx = np.arange(maxIndex[i]-window,maxIndex[i]+window,0.0001)
        popt, pcov = curve_fit(mf.parabola, arrayFitx, arrayFity,maxfev=1000000)
        paraby = mf.parabola(parabx,popt[0],popt[1],popt[2])

        #print(popt[0], popt[1])
        #print(popt[2]-(popt[1]**2)/(4*popt[0]))

        trueMax.append(popt[2]-(popt[1]**2)/(4*popt[0]))
        #print(mf.arythmAvg(paraby))
    #exit()

    avgMax = mf.arythmAvg(trueMax)
    s_avg = mf.stDevAvg(trueMax)
    return avgMax, s_avg


# cerca la fase di stabilità controlla do il primo punto di massimo essoluto e il primo punto di decadimento
def findStable(array, maxIndex, times = 0):
    # lavoriamo con massimi e minimi per semplificare il lavoro
    time = []
    angle = []
    for x in maxIndex:
        time.append(array[x][0])
        angle.append(abs(array[x][2]))

    # cerchiamo l'inizio della fase
    start = 0
    window = 20  # finestra per scorrere l'array
    # prendendo una finestra di x, scorre l'array dei massimi e calcola la pendenza
    # di una retta che passa per i x punti con la regressione lineare/interpolazione
    # se la pendenza è vicina a 0 allora abbiamo trovato la fase strabile
    for i in range(window, len(array) - (window + 1)):
        if not start:
            slope, intercept, r_value, p_value, std_err = linregress(time[i:i + window], angle[i:i + window])#np.polyfit(time[i:i + window], angle[i:i + window], 1)
            # Se la pendenza della retta è vicino a zero, sei nella fase di stabilità

            if abs(slope) < (5*10**-4):
                slope, intercept, r_value, p_value, std_err = linregress(time[i-window:i], angle[i-window:i])  # np.polyfit(time[i:i + window], angle[i:i + window], 1)
                # Se la pendenza della retta è vicino a zero, sei nella fase di stabilità

                if abs(slope) < (50 * 10 ** -4) and times == 0:
                    # siccome lavoravamo con l'array dei massimi dobbiamo trovare la posizione iniziale \
                    # nell'array originale
                    start = mf.columnizer(array, 0).index(array[maxArray[i]][0])
                    # Calcolo della deviazione standard dei residui
                    residuals = np.array(angle) - (slope * np.array(time) + intercept)
                    std_residuals = np.std(residuals)

                    # Calcolo dell'incertezza sulla slope
                    slope_error = std_residuals / np.sqrt(len(time) - 2)

                    #print(f"comp:{abs(slope)/slope_error}")
                    #print("Slope: {:.8f}".format(slope))
                    #print("Slope error: {:.10f}".format(slope_error))
                    break
                if times == 1:
                    # siccome lavoravamo con l'array dei massimi dobbiamo trovare la posizione iniziale \
                    # nell'array originale
                    start = mf.columnizer(array, 0).index(array[maxArray[i + window]][0])
                    # Calcolo della deviazione standard dei residui
                    residuals = np.array(angle) - (slope * np.array(time) + intercept)
                    std_residuals = np.std(residuals)

                    # print(f"comp:{abs(slope)/slope_error}")
                    # print("Slope: {:.8f}".format(slope))
                    # print("Slope error: {:.10f}".format(slope_error))
                    break


    # cerchiamo al fine della fase
    # cerca dove almeno 2 punti di fila hanno il torsore a 0 così cerca l'inizio del decadimento
    end = len(array) - 1
    for i in range(0, len(array)):
        if abs(array[i][1]) * 1024 == 0.0:
            if abs(array[i + 1][1]) * 1024 == 0.0:
                end = i
                break

    return array[start:end]


# calcola l'Ampiezza tra massimo e minimo
def findAlpha(maxArray):
    arrayAlpha = []
    i = 0
    # metodo 1 #calcoli distranza tra max e min
    while i < len(maxArray) - 1:
        #arrayAlpha.append((abs(maxArray[i]) + abs(maxArray[i + 1])))
        arrayAlpha.append(abs(maxArray[i]))
        i += 1

    # metodo 2 # prendi l'Ampiezza da ' e la raddoppi
    # for x in maxArray:
    #    arrayAlpha.append(abs(x)*2)

    return arrayAlpha


# calcola curva lorenziana approssimando dai dati
def breit_wigner(x, x0, gamma, A,d):
    # x0 è w0 oppure il picco della tua funzione
    # print(f"breit_wigner: {x0},{gamma},{A}")
    # gamma = 0.00882627448713556
    return A / np.sqrt((x0 ** 2 - x ** 2) ** 2 + 4*(gamma ** 2) * (x ** 2))+d


# Stima sperimentale energia oscillatore tramite larghezza a meta altezza: massimo e minimo
# della distribuzione necessari; fit Breit-Wigner / Cauchy / Lorentz facoltativo
def cauchy_lorentz(x, x0, gamma, A,d):
    # print(f"cauchy_lorentz: {x0}")
    return A * gamma ** 2 / ((x - x0) ** 2 + gamma ** 2)+d

#funzione usata per raddrizzare i dati
def pendulum_func_raddrizza(t, a, w, fase, d):
    return a * 2.71 ** (-0.0425 * t) * np.cos(w * t + fase)+d

#funzione usata per raddrizzare i dati
def pendulum_func_raddrizza_stable(t, a, w, fase, d):
    return a * np.cos(w * t + fase)+d

def pendulum_func(t, a, w, fase):
    return a * np.cos(w * t + fase)
    #return a*.8*np.cos(w * t + fase)#

def pendulum_func_smorz(t, a, w, fase):
    #0.045
    return a * 2.71 ** (-0.045 * t) * np.cos(w * t + fase)
    #return a*.8*2.71**(-0.045*t) * np.cos(w * t + fase)


def correzioneOffset(array,maxArray):

    # prima stima dei parametri
    stableArray = findStable(array, maxArray,0)
    if len(stableArray) == 0:
        stableArray = findStable(array, maxArray, 1)
    maxstableArray = findMax(stableArray)
    #stableAmp = mf.columnizer(stableArray, 2)
    stableAmp = []
    for x in maxstableArray:
        stableAmp.append(stableArray[x][2])
    #in alfa ci stanno i valori di max e min della fase di stabilità in absolute valuedsaxasdf
    alfa = findAlpha(stableAmp)  # calcola l'Ampiezza di rotazione prendendo massimi e minimi
    #METTERE QUESTA PARTE IN UNA FUNZIONE SOMEHOW


    ##controlla che almeno 2 numeri di fila abbiano il torsore a 0 per cercare il punto di decadimento
    i = len(array) - 1
    for j in range(0, len(array)):
        if abs(array[j][1]) * 1024 == 0.0:
            if abs(array[j + 1][1]) * 1024 == 0.0:
                i = j
                break

    # array contenenti i punti di smorzamento
    arraySmorz = array[i:]

    # Array conentende i massimi della decrescita
    maxArraySmorz = findMax(arraySmorz)

    SmorzMaxValsx = []
    SmorzMaxValsy = []
    #ILLEGALITà NON GUARDARE
    for x in maxArraySmorz[:-1]:
        SmorzMaxValsx.append(arraySmorz[x][0])
        SmorzMaxValsy.append(arraySmorz[x][2])


    stableColumn = mf.columnizer(stableArray, 0)
    maxStable = []
    # mette solo i massimi
    for i in maxstableArray:
        maxStable.append(stableColumn[i])

    for i in range(len(maxstableArray) - 1, 0, -1):  # calcoli singoli periodi oscillazione
        maxStable[i] -= maxStable[i - 1]

    timeArr = []
    for i in range(0, len(maxStable) - 2, 2):  # somma un periodo
        timeArr.append(maxStable[i + 1] + maxStable[i + 2])

    StableMaxValsx = []
    StableMaxValsy = []
    for x in maxstableArray:  # [:-1]
        StableMaxValsx.append(stableArray[x][0])
        StableMaxValsy.append(stableArray[x][2])

    temp_wf = title
    # print(alfa)
    temp_A = round(mf.arythmAvg(alfa), 5)
    #print(f"alfa_size:{len(alfa)}")
    temp_T = round(mf.arythmAvg(timeArr), 5)
    temp_sA = round(mf.stDevAvg(alfa), 5)
    temp_sT = round(mf.stDev(timeArr), 5)

    temp_puls = 2 * np.pi / (temp_T)  # Pulsazione prima stima per fare il fit
    popt, pcov = curve_fit(pendulum_func, StableMaxValsx, StableMaxValsy, p0=[temp_A, temp_puls, 0], maxfev=1000000)
    temp_puls = popt[1]  # Pulsazione dal fit, la più accurata
    popt, pcov = curve_fit(pendulum_func_raddrizza, StableMaxValsx, StableMaxValsy, p0=[temp_A, temp_puls, 0, 0], maxfev=1000000)
    perr = np.sqrt(np.diag(pcov))
    temp_w_s = popt[1]  # Pulsazione dal fit, la più accurata
    d = popt[3]  # sfasamento
    s_d = perr[3]

    d = mf.arythmAvg(mf.columnizer(array,2))

    temp_T4 = 2 * np.pi / (temp_puls)  # quarto metodo per calcolare il tempo, più accurato
    temp_T = temp_T4

    # correzione dei dati
    for i in range(0, len(array)):
        array[i][2] -= d
    #compTrueMaxVal = abs(mf.arythmAvg(StableMaxValsy)- d)/(s_d**2+mf.stDev(StableMaxValsy))**.5
    #compatibilità tra offset calcolato con media e offset calcolato con fit
    #print(abs(mf.arythmAvg(mf.columnizer(array,2))- d)/(s_d**2+mf.stDev(mf.columnizer(array,2)))**.5)
    #TROVARE UN MODO PER METTERE IN UNA FUNZIONE STA ROBA

    return array


def chi2sinus(x,y,sy,A,w,gamma):
    chi2 = 0
    #print(A)
    for i in range(0,int(len(x))):
        #f = np.arccos(A/y[i])
        #quando i valori sono maggiori dell'Ampiezza li resetti all'Ampiezza per evitare NaN
        #if (y[i]>A): y[i] = A*mf.e**(-gamma*(x[i]-x[0]))
        #elif (-y[i]>A): y[i] = -A*mf.e**(-gamma*(x[i]-x[0]))

        f = np.arccos((y[i]) / A)
        chi2 += ((y[i] - (A*mf.e**(-gamma*(x[i]-x[0])) * np.cos(w * (x[i]-x[0]) + f))) / sy) ** 2
        print(f" y {y[i]} : y* {(A *mf.e**(-gamma*(x[i]-x[0]))* np.cos(w * (x[i]-x[0]) + f))} : chi2 {((y[i] - (A *mf.e**(-gamma*(x[i]-x[0]))* np.cos(w * (x[i]-x[0]) + f))) / sy) ** 2}")

    #exit()

    return chi2

def overfit(x,y,puls):
    # facciamo un overfit
    pendulum_funcy = []
    if plot5y[0] < 0: temp_temp_fase = 3.14

    for i in range(0, len(x) - 1):
        A = (abs(y[i]) + abs(y[i + 1])) / 2  # calcola Ampiezza

        fase = 0
        if plot5y[i] < 0: fase = 3.14

        t2 = np.arange(0, x[i + 1] - x[i], 0.001)

        temp_pendulum_funcy = pendulum_func(t2, A, puls, fase)
        for j in temp_pendulum_funcy:
            pendulum_funcy.append(j)

    return pendulum_funcy

def clean3sigma(array):
    return array


def findAndMakeDir(nome_cartella):
    if not os.path.exists(nome_cartella):
        os.makedirs(nome_cartella)


# INIZIO DEL CODICE

# list to store files
res = []

#controlla e crea le cartelle per la distribuzione dei file
findAndMakeDir(dir_path + "dati")
findAndMakeDir(dir_path + "grafici")
findAndMakeDir(dir_path + "grafici/tot")
findAndMakeDir(dir_path + "grafici/totAbs")
findAndMakeDir(dir_path + "grafici/totGauss")
findAndMakeDir(dir_path + "grafici/gauss")
findAndMakeDir(dir_path + "grafici/gauss_max")
findAndMakeDir(dir_path + "grafici/gauss_min")
findAndMakeDir(dir_path + "grafici/gauss_3s")
findAndMakeDir(dir_path + "grafici/singole")
findAndMakeDir(dir_path + "grafici/singole/a")
findAndMakeDir(dir_path + "grafici/singole/b")
findAndMakeDir(dir_path + "grafici/singole/c")
findAndMakeDir(dir_path + "grafici/singole/d")
findAndMakeDir(dir_path + "grafici/singole/e")
findAndMakeDir(dir_path + "grafici/singole/f")

# Iterate directory
for path in os.listdir(dir_path + "dati/"):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path + "dati/", path)):
        res.append(path)
# print(res)
print("0|", end="")
for x in res[start:]:
    # print(f"{float(x[:-6])/10**(round(np.log10(float(x[:-6]))))}")#float(x[:-6])/
    print("─", end="")
print("|100%")

fileN = 0

print(" [", end="")
megaAvg = []


for fileN in range(start, len(res)):  # len(res))
    # print(f"File {fileN} : {res[fileN]}")
    max = "NAN"
    print("#", end="")
    title = float(res[fileN][:-6]) / 10 ** (round(np.log10(float(res[fileN][:-6]))))  # elabora il titolo
    title = title*2*mf.pi
    #print(title)
    f = False
    # Open the CSV file
    with open(dir_path + "dati/" + res[fileN], 'r') as file:
        # Create a file object to read the file
        contents = file.readlines()

        # Remove the first two rows
        contents = contents[2:]

        # Initialize the 3D array
        array = []

        # Iterate through each line in the file
        for line in contents:

            # Split the line into values
            values = line.strip().split('\t')

            # Extract the first three values
            value1 = value2 = value3 = value4 = value5 = 0
            cols = len(values)
            if cols > 4:
                value1, value2, value3, value4, value5 = values[:5]
            else:
                value1, value2, value3 = values[:4]

            value1 = float(value1.replace(",", "."))
            value2 = float(value2.replace(",", "."))
            value3 = float(value3.replace(",", "."))
            if cols > 4:
                value4 = float(value4.replace(",", "."))
                value5 = float(value5.replace(",", "."))

            #elimina valori anomali
            if max == "NAN" and len(array)>200:
                max = abs(value3)
            if max == "NAN" or abs(value3)<2*max:
                if not isinstance(max, str) and  abs(value3)>max:
                    max = abs(value3)

                # Append the values to the 3D array
                if not f and abs(value2) != 0.0:
                    f = True
                if f:
                    array.append([value1, value2, value3, value4, value5])
                    # print(len(array))
                    if len(array) > 1:
                        array[int(len(array) - 1)][0] = round(value1 - array[0][0], 2)
        #print(len(array))#numero di elementi non scartati

    array[0][0] = 0.0
    maxArray = findMax(array)

    #C'E' L'ABBIAMO FATTA LO ABBIAMO MESSO IN UNA FUNZIONE
    array = correzioneOffset(array,maxArray)

    #corregge le ampiezze e le mette in radianti
    for i in range(0,len(array)):
        array[i][2] = array[i][2]*mf.pi

    # seconda stima dei parametri
    stableArray = findStable(array, maxArray ,0)
    if len(stableArray) == 0:
        stableArray = findStable(array, maxArray, 1)
    #exit()
    maxstableArray = findMax(stableArray)
    maxstableArray = maxstableArray[1:]

    stableAbs = []
    for x in maxstableArray:
        stableAbs.append([abs(stableArray[x][2]),x])

    stableSigma = mf.stDev(mf.columnizer(stableAbs,0))
    stableAvg = mf.arythmAvg(mf.columnizer(stableAbs, 0))


    # rimuove i 3 sigma dall 'array di stabili
    #maxstableArray = []
    for i in range(0,len(stableAbs)):
        if stableAbs[i][0]>(stableAvg+3*stableSigma):
            maxstableArray.remove(stableAbs[i][1])

    #print(maxstableArray)
    stableAmp = mf.columnizer(stableArray, 2)
    stableAmp = []
    for x in maxstableArray:
        stableAmp.append(stableArray[x][2])
    alfa = findAlpha(stableAmp)  # calcola l'Ampiezza di rotazione prendendo massimi e minimi

    ##controlla che almeno 2 numeri di fila abbiano il torsore a 0 per cercare il punto di decadimento
    i = len(array) - 1
    for j in range(0, len(array)):
        if abs(array[j][1]) * 1024 == 0.0:
            if abs(array[j + 1][1]) * 1024 == 0.0:
                i = j
                break

    # array contenenti i punti di smorzamento
    arraySmorz = array[i:]


    # Array conentende i massimi della decrescita
    # Prendo solo i primi 10 massimi
    #salto il primo per imperfezioni nellmetodo del calcolo dei massimi
    maxArraySmorz = findMax(arraySmorz)[2:23]

    arraySmorz = arraySmorz[maxArraySmorz[0]:maxArraySmorz[-1]]
    maxArraySmorz = findMax(arraySmorz)

    SmorzMaxValsx = []
    SmorzMaxValsy = []
    # ILLEGALITà NON GUARDARE
    for x in maxArraySmorz[:-1]:
        SmorzMaxValsx.append(arraySmorz[x][0])
        SmorzMaxValsy.append(arraySmorz[x][2])

    stableColumn = mf.columnizer(stableArray, 0)
    maxStable = []
    # mette solo i massimi
    for i in maxstableArray:
        maxStable.append(stableColumn[i])

    for i in range(len(maxstableArray) - 1, 0, -1):  # calcoli singoli periodi oscillazione
        maxStable[i] -= maxStable[i - 1]



    timeArr = []
    for i in range(0, len(maxStable) - 2, 2):  # somma un periodo
        timeArr.append(maxStable[i + 1] + maxStable[i + 2])

    StableMaxValsx = []
    StableMaxValsy = []
    for x in maxstableArray:  # [:-1]
        StableMaxValsx.append(stableArray[x][0])
        StableMaxValsy.append(stableArray[x][2])

    temp_wf = title
    # print(alfa)
    temp_A = round(mf.arythmAvg(alfa) , 5)
    temp_T = round(mf.arythmAvg(timeArr), 5)
    temp_sA = round(mf.stDevAvg(alfa), 5)
    #print(f"{round(title/(2*mf.pi),5)}: sa:{temp_sA}")
    temp_sT = round(mf.stDevAvg(timeArr), 5)

    temp_puls = 2 * np.pi / (temp_T)  # Pulsazione z<prima stima per fare il fit
    popt, pcov = curve_fit(pendulum_func, StableMaxValsx, StableMaxValsy, p0=[temp_A, temp_puls, 0], maxfev=1000000)
    perr = np.sqrt(np.diag(pcov))
    temp_sw = perr[1]
    temp_puls = popt[1]  # Pulsazione dal fit, la più accurata
    #print(f"{round(title/(2*mf.pi),5)}: chi2 {chi2sinus(mf.columnizer(arraySmorz,0),mf.columnizer(arraySmorz,2), 0.001 / (12 ** .5), temp_A, popt[1],0.045)} : len {len(stableArray)}")
    StableMaxValsx2 = []
    for x in range(0,len(StableMaxValsx)):
        StableMaxValsx2.append(x-StableMaxValsx[0])

    #StableMaxValsx2 = StableMaxValsx


    popt, pcov = curve_fit(pendulum_func_smorz, StableMaxValsx, StableMaxValsy, p0=[temp_A, temp_puls, 0],maxfev=100000)
    perr = np.sqrt(np.diag(pcov))
    temp_sfase = perr[2]
    temp_w_s = popt[1]  # Pulsazione dal fit, la più accurata
    temp_sws = perr[1]
    #temp_w_smax =
    #temp_w_smin =
    #temp_gammasmorz = popt[3]#popt[3]*(0.0675)
    #print(f"gamma: {popt[3]}")
    #print(f"s_gamma: {perr[3]}")
    #temo_w_0 = temp_w_s

    temp_T4 = 2 * np.pi / (temp_puls)  # quarto metodo per calcolare il tempo, più accurato
    temp_T = temp_T4


    # massimi e minimi
    plot1x = []
    plot1y = []

    for x in maxArray[:-1]:
        plot1x.append(array[x][0])
        plot1y.append(array[x][2])

    # punti intermedi
    plot2x = []
    plot2y = []

    for x in array:
        plot2x.append(x[0])
        plot2y.append(x[2])

    # print(len(maxArray))
    # print(len(array))

    # massimi e minimi fase smorzata
    plot3x = SmorzMaxValsx
    plot3y = SmorzMaxValsy



    # punti intermedi
    plot4x = []
    plot4y = []

    for x in arraySmorz:
        plot4x.append(x[0])
        plot4y.append(x[2])

    # prepara la figura 2 con tutte abs
    plot1xabsmax = []  # max
    plot1xabsmin = []  # min
    plot1yabsmax = []  # max
    plot1yabsmin = []  # min
    plot2yabs = []

    plot3xabsmax = []  # max
    plot3xabsmin = []  # min
    plot3yabsmax = []  # max
    plot3yabsmin = []  # min
    plot4yabs = []

    for i in range(0, len(plot1y)):
        if plot1y[i] >= 0:
            plot1yabsmax.append(plot1y[i])
            plot1xabsmax.append(plot1x[i])
        else:
            plot1yabsmin.append(abs(plot1y[i]))
            plot1xabsmin.append(plot1x[i])

    for x in plot2y:
        plot2yabs.append(abs(x))

    for i in range(0, len(plot3y)):
        if plot3y[i] >= 0:
            plot3yabsmax.append(plot3y[i])
            plot3xabsmax.append(plot3x[i])
        else:
            plot3yabsmin.append(abs(plot3y[i]))
            plot3xabsmin.append(plot3x[i])

    for x in plot4y:
        plot4yabs.append(abs(x))

    # massimi e minimi fase stazionaria
    plot5x = StableMaxValsx
    plot5y = StableMaxValsy


    # punti intermedi fase stazionaria
    plot6x = []
    plot6y = []

    for x in stableArray:
        plot6x.append(x[0])
        plot6y.append(x[2])

    plot7x = []
    plot7y = []

    for i in range(0, len(plot3y)):
        if plot3y[i] >= 0:
            plot7y.append(np.log(plot3y[i]))
            plot7x.append(plot3x[i])
        else:
            plot7y.append(np.log(abs(plot3y[i])))
            plot7x.append(plot3x[i])

    # Salva i Dati Corretti Assoluti
    compTrueMaxVal = -1
    if compTrueMax:
        trueMaxAvg, s_trueMaxAvg = findTrueMax(stableArray,maxstableArray)
        absStableMaxValsy = []
        for x in StableMaxValsy:
            absStableMaxValsy.append(abs(x))
        maxAvg = mf.arythmAvg(absStableMaxValsy)
        s_maxAvg = mf.stDevAvg(absStableMaxValsy)
        #for x in absStableMaxValsy:
        #    print(x)
        #print(mf.arythmAvg(absStableMaxValsy))
        #exit()
        #print(f"{trueMaxAvg} : {maxAvg} : {s_trueMaxAvg} : {s_maxAvg}")
        #print(abs(trueMaxAvg-maxAvg)/(s_trueMaxAvg**2+s_maxAvg**2)**.5)
        compTrueMaxVal = abs(abs(trueMaxAvg-maxAvg)/(s_trueMaxAvg**2+s_maxAvg**2)**.5)#compatibilità tra massimi e massimi veri

        #compatibilità tra offset con fit e offse con media
        #compTrueMaxVal = abs(mf.arythmAvg(StableMaxValsy)- d)/(s_d**2+mf.stDev(StableMaxValsy))**.5

    smorzColumn = mf.columnizer(stableArray, 0)
    maxsmorz = []
    # mette solo i massimi
    for i in maxArraySmorz[:-1]:
        maxsmorz.append(smorzColumn[i])

    for i in range(len(maxArraySmorz[:-1]) - 1, 0, -1):  # calcoli singoli periodi oscillazione
        maxsmorz[i] -= maxsmorz[i - 1]
    timeArr = []
    for i in range(0, len(maxsmorz) - 2, 2):  # somma un periodo
        timeArr.append(maxsmorz[i + 1] + maxsmorz[i + 2])

    temp_Tws = round(mf.arythmAvg(timeArr), 5)
    #print(f"{temp_w_s} : ",end="")
    temp_w_s = 2 * np.pi / (temp_Tws)  # Pulsazione z<prima stima per fare il fit
    #print(f"{temp_w_s}")
    # plot3xabsmax plot3yabsmax
    #plot7x = plot3xabsmax
    #plot7y = plot3yabsmax
    # plot3xabsmin plot3yabsmin
    #plot7x = plot3xabsmin
    # plot7y = plot3yabsmin
    # plot7x plot7y
    splot7y = []
    for x in plot7y:
        splot7y.append((0.001 / (12 ** .5))/abs(x))
        #print(x)

    interpol = mf.interpol3(plot7x, 0.02 / (12 ** .5) ,plot7y, splot7y)

    chi2lin = 0
    for i in range(0,len(plot7x)):
        oldchi2 = chi2lin
        chi2lin += ((plot7y[i]-(interpol[0]+interpol[2]*plot7x[i]))/(splot7y[i]))**2
        #print(chi2lin-oldchi2)

    temp_sws = temp_sT/ temp_T * (2 * mf.pi) ** .5


    megaAvg.append([temp_wf, temp_A, temp_T, round(temp_puls, 5), round(temp_w_s, 5), temp_sA, temp_sT,interpol[0],interpol[2],interpol[1],interpol[3],temp_sw,temp_sfase,compTrueMaxVal,temp_sws,title/(2*mf.pi),chi2lin])
    #temp_A = round(mf.arythmAvg(mf.columnizer(array,3)), 5)/2#Ampiezza in caso di file con 4 colonne

    # print(f"File {fileN} : {res[fileN]}")

    # plotting
    if 1 in charts:
        fig1, (ax, ex, bx) = plt.subplots(1, 3, subplot_kw={'position': [0.1, 0.1, 0.4, 0.8]}, figsize=(16, 5))
    if 2 in charts:
        fig2, (cx, fx, dx) = plt.subplots(1, 3, subplot_kw={'position': [0.1, 0.1, 0.4, 0.8]}, figsize=(16, 5))
    if 3 in charts:
        fig3, (gmx, gx, gMx) = plt.subplots(1, 3, subplot_kw={'position': [0.1, 0.1, 0.4, 0.8]}, figsize=(16, 5))
    if "a" in charts:
        figa, ax = plt.subplots()

    if "b" in charts:
        figb, bx = plt.subplots()

    if "c" in charts:
        figc, cx = plt.subplots()

    if "d" in charts:
        figd, dx = plt.subplots()

    if "e" in charts:
        fige, ex = plt.subplots()

    if "f" in charts:
        figf, fx = plt.subplots()

    if "g" in charts:
        figG, gx = plt.subplots()

    if "gM" in charts:
        figGM, gMx = plt.subplots()

    if "gm" in charts:
        figGm, gmx = plt.subplots()

    if "3s" in charts:
        fig3s, sx = plt.subplots()

    if "3s" in charts:
        misure = []
        misure = mf.columnizer(stableAbs,0)



        nbins = int(len(misure) ** .5)

        sigma = mf.stDev(misure)
        mu = mf.arythmAvg(misure)
        x = np.arange(mu - 4 * sigma, mu + 4 * sigma, 0.001)  # Cambiamo il range così da dezoommare
        gauss = mf.gauss(x, sigma, mu)

        # print(misure)
        massimo = np.max(misure)
        minimo = np.min(misure)

        binwidth = (massimo - minimo) / nbins
        gauss = gauss * binwidth

        bidoni = [0] * nbins
        for i in range(0, nbins):
            for j in misure:
                if j > (minimo + binwidth * i) and j <= (minimo + binwidth * (i + 1)):
                    bidoni[i] += 1


        bidoni[0] += 1
        norm = []
        # print("bindwith")
        for i in range(0, nbins):
            norm.append(binwidth * bidoni[i] / (binwidth * len(misure)))
            # print(f"{round(minimo + binwidth * i, 3)}-{round(minimo + binwidth * (i + 1), 3)} : {norm[i]}")

        xnorm = []
        for i in range(0, nbins):
            xnorm.append(minimo + binwidth * i)
        # print(f"somma: {sum(norm)}")

        sx.bar(xnorm, norm, width=binwidth, alpha=0.8, edgecolor='blue', align='edge', linewidth=1,
               label=f"misurazioni")
        sx.plot(x, gauss, c="#d62728", label=f"")
        # print(x)
        # print(gauss)
        sx.axvline(mu, color='r', linestyle='--', label="Mu")
        sx.axvline(mu + sigma, color='orange', linestyle='-', label="sigma")
        sx.axvline(mu - sigma, color='orange', linestyle='-')
        sx.axvline(mu + 2 * sigma, color='cyan', linestyle=':', label="2 sigma")
        sx.axvline(mu - 2 * sigma, color='cyan', linestyle=':')
        sx.axvline(mu + 3 * sigma, color='green', linestyle='-.', label="3 sigma")
        sx.axvline(mu - 3 * sigma, color='green', linestyle='-.')
        sx.legend(loc='upper right')

        # omega_sx.set(xlabel='Ampiezza[rad]', ylabel='Frequenza',title=f"{round(title/(2*mf.pi),5)}"+"[Hz]")


    if 1 in charts or "a" in charts or "b" in charts:
        """ """
        if 1 in charts or "a" in charts:
            ax.errorbar(plot2x, plot2y, fmt='x', c="#ff7f0e", capsize=5, markersize=1, label="Misure")
            ax.errorbar(plot1x, plot1y, fmt='x', c="#1f77b4", capsize=5, markersize=2, label="Massimi")
            # print(f"{len(plot1x)} : {len(plot2x)}")


            ax.axvline(x=plot6x[0], color='#2ca02c', linestyle='--')
            ax.axvline(x=plot6x[-1], color='#2ca02c', linestyle='--')

            add_ylim = .15
            old_ylim = ax.get_ylim()
            old_ylim = (old_ylim[0]-add_ylim,old_ylim[1]+add_ylim)#aggiusta un po' il lim

            old_xlim = ax.get_xlim()
            # Plotta il rettangolo
            ax.fill_betweenx([-10,+10], -100, plot6x[0]          , color='#FC3D21', alpha=0.125,label="Transiente")#,label="Transiente"
            ax.fill_betweenx([-10, +10], plot6x[0],plot6x[-1]    , color='#2ca02c', alpha=0.125,label="Stabile")#,label="Stabile"
            ax.fill_betweenx([-10, +10], plot6x[-1] ,5*plot6x[-1], color='#1f77b4', alpha=0.125,label="Smorzato")#,label="Smorzato"

            # Aggiunta della scritta
            #plt.text(-plot6x[0]/2, old_ylim[0]+0.05, "$Transiente$", fontsize=12, color='#FC3D21')
            #plt.text(plot6x[0]+(plot6x[-1]-plot6x[0])/3, old_ylim[0] + 0.05, "$Stabile$", fontsize=12, color='#2ca02c')
            #plt.text(plot6x[-1]+12.5, old_ylim[0] + 0.05, "$Smorzato$", fontsize=12, color='#1f77b4')

            ax.set_ylim(old_ylim)
            ax.set_xlim(old_xlim)
            #ax.set_xlim((-plot6x[0]/1.5,old_xlim[1]))
            ax.legend(loc='upper right')

            ax.errorbar(plot2x, plot2y, fmt='x', c="#ff7f0e", capsize=5, markersize=1, label="Misure")
            ax.errorbar(plot1x, plot1y, fmt='x', c="#1f77b4", capsize=5, markersize=2, label="Massimi")
            # print(f"{len(plot1x)} : {len(plot2x)}")

        #fase smorzata
        """ """
        if 1 in charts or "b" in charts:
            bx.errorbar(plot4x, plot4y, fmt='x', c="#ff7f0e", capsize=5, markersize=1, label="Misure")
            bx.errorbar(plot3x, plot3y, fmt='x', c="#1f77b4", capsize=5, markersize=2, label="Massimi")
            bx.legend(loc='upper right')

            #############
            t = np.arange(plot4x[0], plot4x[-1], 0.001)  # plot5x[0]
       #temp_fase = -np.arccos(plot4y[0] / temp_A)
            if plot4y[0] > 0:
                temp_fase = 0
            else:
                temp_fase = mf.pi

            t2 = np.arange(0, plot4x[-1] - plot4x[0], 0.001)
            #t2 = t
            pendulum_func_smorzy = pendulum_func_smorz(t2, temp_A, temp_puls, temp_fase)
            #bx.plot(t, pendulum_func_smorzy, c="#2ca02c", linewidth="0.4")
        ###############
        #bx.axvline(x=plot3x[0])

    if 2 in charts or "c" in charts or "d" in charts:
        # figura 2
        """  """
        if 2 in charts or "c" in charts:
            cx.errorbar(plot2x, plot2yabs, fmt='x', c="#ff7f0e", capsize=5, markersize=1, label="Misure")
            cx.errorbar(plot1xabsmax, plot1yabsmax, fmt='x', c="#1f77b4", capsize=5, markersize=5, label="Massimi")
            cx.errorbar(plot1xabsmin, plot1yabsmin, fmt='x', c="#2ca02c", capsize=5, markersize=5, label="Minimi")
            cx.axvline(x=plot6x[0], color='#2ca02c', linestyle='--')
            cx.axvline(x=plot6x[-1], color='#2ca02c', linestyle='--')

            # se la fase è invers allora aggiungu in pi greco alla fase
            if (plot6y[0] / abs(plot6y[0])) < 0:
                temp_fase = mf.pi
            else:
                temp_fase = 0
            t2 = np.arange(0, plot6x[-1] - plot6x[0], 0.001)

            pendulum_funcy = pendulum_func(t2, temp_A, temp_puls, temp_fase)#Niente fase perchè inizia da un max
            dajetempx = plot1xabsmax
            dajetempy = plot1yabsmax
            for i in range(0,len(dajetempx)):
                if dajetempx[i]>=plot6x[0]:
                    dajetempx = dajetempx[i:]
                    dajetempy = dajetempy[i:]
                    break

            for i in range(0,len(dajetempx)):
                if dajetempx[i]>=plot6x[-1]:
                    dajetempx = dajetempx[:i]
                    dajetempy = dajetempy[:i]
                    break

            cx.axvline(x=dajetempx[0], color='red', linestyle='--')
            cx.axvline(x=dajetempx[-1], color='red', linestyle='--')

            popt, pcov = curve_fit(pendulum_func_raddrizza_stable, dajetempx, dajetempy, p0=[1, temp_puls, 0,dajetempy[0]],
                                   maxfev=1000000)
            #popt[0] = popt[0]/8
            #pendulum_funcy = pendulum_func_raddrizza_stable(t2, popt[0], popt[1], popt[2],popt[3])

            #for i in range(0, len(pendulum_funcy)):
            #    if pendulum_funcy[i] <= 0:
            #        pendulum_funcy[i] = 0#abs(pendulum_funcy[i])

            t3 = np.linspace(plot6x[0], plot6x[-1], len(pendulum_funcy))
            #cx.plot(t3, pendulum_funcy, c="#2ca02c", linewidth="0.4")

            cx.legend(loc='upper right')

        """ """
        if 2 in charts or "d" in charts:
            dx.errorbar(plot4x, plot4yabs, fmt='x', c="#ff7f0e", capsize=5, markersize=1, label="Misure")
            dx.errorbar(plot3xabsmax, plot3yabsmax, fmt='x', c="#1f77b4", capsize=5, markersize=5, label="Massimi")
            dx.errorbar(plot3xabsmin, plot3yabsmin, fmt='x', c="#2ca02c", capsize=5, markersize=5, label="Minimi")
            dx.legend(loc='upper right')

    if 3 in charts or "g" in charts:
        misure = []

        dajetempx = plot1xabsmax
        dajetempmaxy = plot1yabsmax
        dajetempminy = plot1yabsmin
        for i in range(0, len(dajetempx)):
            if dajetempx[i] >= plot6x[0]:
                dajetempx = dajetempx[i:]
                dajetempmaxy = dajetempmaxy[i:]
                dajetempminy = dajetempminy[i:]
                break

        for i in range(0, len(dajetempx)):
            if dajetempx[i] >= plot6x[-1]:
                dajetempx = dajetempx[:i]
                dajetempmaxy = dajetempmaxy[:i]
                dajetempminy = dajetempminy[:i]
                break

        for x in plot5y:
            misure.append(abs(x))
        #misure = dajetempmaxy
        #misure = dajetempminy

        #print(f"{(abs(mf.arythmAvg(dajetempmaxy)-mf.arythmAvg(dajetempminy))/(mf.stDev(dajetempmaxy)**2+mf.stDev(dajetempminy)**2)**.5)}")

        nbins = int(len(misure)**.5)

        sigma = mf.stDev(misure)
        mu = mf.arythmAvg(misure)
        x = np.arange(mu - 4 * sigma, mu + 4 * sigma, 0.000001)  # Cambiamo il range così da dezoommare
        gauss = mf.gauss(x,sigma,mu)

        #print(misure)
        massimo = np.max(misure)
        minimo = np.min(misure)

        binwidth = (massimo - minimo) / nbins
        gauss = gauss * binwidth

        bidoni = [0] * nbins
        for i in range(0, nbins):
            for j in misure:
                if j > (minimo + binwidth * i) and j <= (minimo + binwidth * (i + 1)):
                    bidoni[i] += 1

        bidoni[0] += 1
        norm = []
        #print("bindwith")
        for i in range(0, nbins):
            norm.append(binwidth * bidoni[i] / (binwidth * len(misure)))
            #print(f"{round(minimo + binwidth * i, 3)}-{round(minimo + binwidth * (i + 1), 3)} : {norm[i]}")

        xnorm = []
        for i in range(0, nbins):
            xnorm.append(minimo + binwidth * i)
        #print(f"somma: {sum(norm)}")

        gx.bar(xnorm, norm, width=binwidth, alpha=0.8, edgecolor='blue',align='edge' , linewidth=1, label=f"misurazioni")
        gx.plot(x, gauss, c="#d62728", label=f"")
        #print(x)
        #print(gauss)
        gx.axvline(mu, color='r', linestyle='--', label="Mu")
        gx.axvline(mu + sigma, color='orange', linestyle='-', label="sigma")
        gx.axvline(mu - sigma, color='orange', linestyle='-')
        gx.axvline(mu + 2 * sigma, color='cyan', linestyle=':', label="2 sigma")
        gx.axvline(mu - 2 * sigma, color='cyan', linestyle=':')
        gx.axvline(mu + 3 * sigma, color='green', linestyle='-.', label="3 sigma")
        gx.axvline(mu - 3 * sigma, color='green', linestyle='-.')
        gx.legend(loc='upper right')

    if 3 in charts or "gM" in charts:
        misure = []

        dajetempx = plot1xabsmax
        dajetempmaxy = plot1yabsmax
        dajetempminy = plot1yabsmin
        for i in range(0, len(dajetempx)):
            if dajetempx[i] >= plot6x[0]:
                dajetempx = dajetempx[i:]
                dajetempmaxy = dajetempmaxy[i:]
                dajetempminy = dajetempminy[i:]
                break

        for i in range(0, len(dajetempx)):
            if dajetempx[i] >= plot6x[-1]:
                dajetempx = dajetempx[:i]
                dajetempmaxy = dajetempmaxy[:i]
                dajetempminy = dajetempminy[:i]
                break

        for x in plot5y:
            misure.append(abs(x))
        misure = dajetempmaxy
        #misure = dajetempminy

        #print(f"{(abs(mf.arythmAvg(dajetempmaxy)-mf.arythmAvg(dajetempminy))/(mf.stDev(dajetempmaxy)**2+mf.stDev(dajetempminy)**2)**.5)}")

        nbins = int(len(misure)**.5)

        sigma = mf.stDev(misure)
        mu = mf.arythmAvg(misure)
        x = np.arange(mu - 4 * sigma, mu + 4 * sigma, 0.000001)  # Cambiamo il range così da dezoommare
        gauss = mf.gauss(x,sigma,mu)

        #print(misure)
        massimo = np.max(misure)
        minimo = np.min(misure)

        binwidth = (massimo - minimo) / nbins
        gauss = gauss * binwidth

        bidoni = [0] * nbins
        for i in range(0, nbins):
            for j in misure:
                if j > (minimo + binwidth * i) and j <= (minimo + binwidth * (i + 1)):
                    bidoni[i] += 1

        bidoni[0] += 1
        norm = []
        #print("bindwith")
        for i in range(0, nbins):
            norm.append(binwidth * bidoni[i] / (binwidth * len(misure)))
            #print(f"{round(minimo + binwidth * i, 3)}-{round(minimo + binwidth * (i + 1), 3)} : {norm[i]}")

        xnorm = []
        for i in range(0, nbins):
            xnorm.append(minimo + binwidth * i)
        #print(f"somma: {sum(norm)}")

        gMx.bar(xnorm, norm, width=binwidth, alpha=0.8, edgecolor='blue',align='edge' , linewidth=1, label=f"misurazioni")
        gMx.plot(x, gauss, c="#d62728", label=f"")
        #print(x)
        #print(gauss)
        gMx.axvline(mu, color='r', linestyle='--', label="Mu")
        gMx.axvline(mu + sigma, color='orange', linestyle='-', label="sigma")
        gMx.axvline(mu - sigma, color='orange', linestyle='-')
        gMx.axvline(mu + 2 * sigma, color='cyan', linestyle=':', label="2 sigma")
        gMx.axvline(mu - 2 * sigma, color='cyan', linestyle=':')
        gMx.axvline(mu + 3 * sigma, color='green', linestyle='-.', label="3 sigma")
        gMx.axvline(mu - 3 * sigma, color='green', linestyle='-.')
        if 3 not in charts:    gMx.legend(loc='upper right')

    if 3 in charts or "gm" in charts:
        misure = []

        dajetempx = plot1xabsmax
        dajetempmaxy = plot1yabsmax
        dajetempminy = plot1yabsmin
        for i in range(0, len(dajetempx)):
            if dajetempx[i] >= plot6x[0]:
                dajetempx = dajetempx[i:]
                dajetempmaxy = dajetempmaxy[i:]
                dajetempminy = dajetempminy[i:]
                break

        for i in range(0, len(dajetempx)):
            if dajetempx[i] >= plot6x[-1]:
                dajetempx = dajetempx[:i]
                dajetempmaxy = dajetempmaxy[:i]
                dajetempminy = dajetempminy[:i]
                break

        for x in plot5y:
            misure.append(abs(x))
        misure = dajetempmaxy
        #misure = dajetempminy

        #print(f"{(abs(mf.arythmAvg(dajetempmaxy)-mf.arythmAvg(dajetempminy))/(mf.stDev(dajetempmaxy)**2+mf.stDev(dajetempminy)**2)**.5)}")

        nbins = int(len(misure)**.5)

        sigma = mf.stDev(misure)
        mu = mf.arythmAvg(misure)
        x = np.arange(mu - 4 * sigma, mu + 4 * sigma, 0.000001)  # Cambiamo il range così da dezoommare
        gauss = mf.gauss(x,sigma,mu)

        #print(misure)
        massimo = np.max(misure)
        minimo = np.min(misure)

        binwidth = (massimo - minimo) / nbins
        gauss = gauss * binwidth

        bidoni = [0] * nbins
        for i in range(0, nbins):
            for j in misure:
                if j > (minimo + binwidth * i) and j <= (minimo + binwidth * (i + 1)):
                    bidoni[i] += 1

        bidoni[0] += 1
        norm = []
        #print("bindwith")
        for i in range(0, nbins):
            norm.append(binwidth * bidoni[i] / (binwidth * len(misure)))
            #print(f"{round(minimo + binwidth * i, 3)}-{round(minimo + binwidth * (i + 1), 3)} : {norm[i]}")

        xnorm = []
        for i in range(0, nbins):
            xnorm.append(minimo + binwidth * i)
        #print(f"somma: {sum(norm)}")

        gmx.bar(xnorm, norm, width=binwidth, alpha=0.8, edgecolor='blue',align='edge' , linewidth=1, label=f"misurazioni")
        gmx.plot(x, gauss, c="#d62728", label=f"")
        #print(x)
        #print(gauss)
        gmx.axvline(mu, color='r', linestyle='--', label="Mu")
        gmx.axvline(mu + sigma, color='orange', linestyle='-', label="sigma")
        gmx.axvline(mu - sigma, color='orange', linestyle='-')
        gmx.axvline(mu + 2 * sigma, color='cyan', linestyle=':', label="2 sigma")
        gmx.axvline(mu - 2 * sigma, color='cyan', linestyle=':')
        gmx.axvline(mu + 3 * sigma, color='green', linestyle='-.', label="3 sigma")
        gmx.axvline(mu - 3 * sigma, color='green', linestyle='-.')
        if 3 not in charts:
            gmx.legend(loc='upper right')



    if 1 in charts or "e" in charts:
        """ """
        t = np.arange(plot6x[0], plot6x[-1], 0.001)  # plot5x[0]

       #se la fase è invers allora aggiungu in pi greco alla fase
        if (plot6y[0] / abs(plot6y[0])) < 0:
            temp_fase = mf.pi
        else:
            temp_fase = 0

        #altri modi per clacolare T
        #temp_T2 = (plot5x[-1] - plot5x[0]) / (len(plot5x) / 2)#(TFinale-Tiniziale)/Nperiodi
        #temp_T3 = (plot5x[2] - plot5x[0])                     #Tempo primo periodo

        """print("stab")
        print(temp_T)
        print(temp_T2)
        print(temp_T3)
        """
        t2 = np.arange(0, plot6x[-1] - plot6x[0], 0.001)

        #pendulum_funcy = pendulum_func(t2, temp_A, temp_puls, temp_fase)#Niente fase perchè inizia da un max

        popt, pcov = curve_fit(pendulum_func, plot5x, plot5y, p0=[temp_A, temp_puls, temp_fase],maxfev=1000000)
        if plot5y[0]>0 : pendulum_funcy = pendulum_func(t2, temp_A,popt[1],0)#temp_A,temp_puls,temp_fase
        else           : pendulum_funcy = pendulum_func(t2, temp_A,popt[1],mf.pi)#temp_A,temp_puls,temp_fase


        #carlo mi ammazza se vede questo
        if overfitting:
            pendulum_funcy = overfit(plot5x,plot5y,temp_puls)

        t3 = np.linspace(plot5x[0], plot5x[-1], len(pendulum_funcy))
        #ex.plot(t3, pendulum_funcy, c="#2ca02c", linewidth="0.4")

        ex.errorbar(plot6x, plot6y, fmt='x', c="#ff7f0e", capsize=5, markersize=1, label="Misure")
        ex.errorbar(plot5x, plot5y, fmt='x', c="#1f77b4", capsize=5, markersize=2, label="Massimi")
        ex.legend(loc='upper right')

    if 2 in charts or "f" in charts:
        """ """
        t = np.arange(plot7x[0],plot7x[-1],0.001)
        #plot3xabsmax plot3yabsmax
        #plot3xabsmin plot3yabsmin
        #plot7x plot7y
        interpol = mf.interpol1(plot7x,plot7y,0.001/(12**.5))

        # plot3xabsmax plot3yabsmax
        # plot3xabsmin plot3yabsmin
        # plot7x plot7y
        plot7xc = plot7x
        plot7yc = plot7y

        plot7x = plot3xabsmax
        plot7y = plot3yabsmax
        splot7y = []
        for x in plot7y:
            splot7y.append((0.001 / (12 ** .5)) / abs(x))
            # print(x)

        interpol = mf.interpol3(plot7x, 0.02 / (12 ** .5), plot7y, splot7y)
        fx.plot(t, interpol[0] + interpol[2] * t, c="#2ca02c",linewidth = 0.9, label="y(max) = a + bt")
        fx.errorbar(plot7x, plot7y,xerr=[0.02 / (12 ** .5)]*len(plot7x), yerr=splot7y,fmt='o',  c="#1f77b4", capsize=5, markersize=1, label="Massimi")

        errorpost = 0
        chi2max = 0
        for i in range(0, len(plot7x)):
            errorpost += (plot7y[i] - (interpol[0] + interpol[2] * plot7x[i])) ** 2
            chi2max += ((plot7y[i] - (interpol[0] + interpol[2] * plot7x[i]))/splot7y[i]) ** 2
        #print(f"{title / (2 * mf.pi)} error-post-max: {(errorpost / (len(plot7y) - 2)) ** .5}")
        print(f"{title / (2 * mf.pi)} chi2max: {chi2max}")


        plot7x = plot3xabsmin
        plot7y = plot3yabsmin
        splot7y = []
        for x in plot7y:
            splot7y.append((0.001 / (12 ** .5)) / abs(x))
            # print(x)


        interpol = mf.interpol3(plot7x, 0.02 / (12 ** .5), plot7y, splot7y)
        fx.plot(t, interpol[0] + interpol[2] * t, c="#ff7f0e", linewidth=0.9, label="y(min) = a + bt")
        fx.errorbar(plot7x, plot7y, xerr=[0.02 / (12 ** .5)] * len(plot7x), yerr=splot7y, fmt='o', c="#d62728",
                    capsize=5, markersize=1, label="Minimi")

        errorpost = 0
        chi2min = 0
        for i in range(0, len(plot7x)):
            errorpost += (plot7y[i] - (interpol[0] + interpol[2] * plot7x[i])) ** 2
            chi2min += ((plot7y[i] - (interpol[0] + interpol[2] * plot7x[i]))/splot7y[i]) ** 2
        #print(f"{title / (2 * mf.pi)} error-post-min: {(errorpost / (len(plot7y) - 2)) ** .5}")
        print(f"{title / (2 * mf.pi)} chi2max: {chi2min}")

        plot7x = plot7xc
        plot7y = plot7yc

        errorpost = 0
        for i in range(0, len(plot7x)):
            errorpost += (plot7y[i] - (interpol[0] + interpol[2] * plot7x[i])) ** 2
        #print(f"{title/(2*mf.pi)} error-post-tot: {(errorpost/(len(plot7y)-2))**.5}")

        fx.legend(loc='upper right')

    ##settings del grafico
    if 1 in charts or "a" in charts or "e" in charts or "b" in charts:
        """ """
        if 1 in charts or "a" in charts:
            ax.set(xlabel='t[s]', ylabel='$Ampiezza$[rad]',
                   title=f"Completo")
            ax.yaxis.set_label_coords(-0.12, 0.5)
            ax.grid(linestyle='dotted', clip_on=True)

        if 1 in charts or "e" in charts:
            """ """
            ex.set(xlabel='t[s]', ylabel='$Ampiezza$[rad]',
                   title=f"Stazionario")
            ex.yaxis.set_label_coords(-0.12, 0.5)
            ex.grid(linestyle='dotted', clip_on=True)

        if 1 in charts or "b" in charts:
            bx.set(xlabel='t[s]', ylabel='$Ampiezza$[rad]',
                   title=f"Smorzato")
            bx.yaxis.set_label_coords(-0.12, 0.5)
            bx.grid(linestyle='dotted', clip_on=True)

    if 2 in charts or "c" in charts or "d" in charts or "f" in charts:
        """ """
        if 2 in charts or "c" in charts:
            cx.set(xlabel='t[s]', ylabel='$Ampiezza$[rad]',
                   title=f"")
            cx.yaxis.set_label_coords(-0.12, 0.5)
            cx.grid(linestyle='dotted', clip_on=True)

        """ """
        if 2 in charts or "f" in charts:
            fx.set(xlabel='t[s]', ylabel='$ln(Ampiezza)$[rad]',
                   title=f"Linearizzazione dei massimi")
            fx.yaxis.set_label_coords(-0.12, 0.5)
            fx.grid(linestyle='dotted', clip_on=True)

        if 2 in charts or "d" in charts:
            dx.set(xlabel='t[s]', ylabel='$Ampiezza$[rad]',
                   title=f"")
            dx.yaxis.set_label_coords(-0.12, 0.5)
            dx.grid(linestyle='dotted', clip_on=True)

    if 3 in charts or "gm" in charts or "g" in charts or "gM" in charts:
        if 3 in charts or "g" in charts:
            if 3 in charts: gx.set(title=f"Massimi e Minimi")
            else: gx.set(xlabel='Ampiezza[rad]', ylabel='Frequenza',
                   title=f"{round(title/(2*mf.pi),5)}" + "[Hz]")
            gx.yaxis.set_label_coords(-0.12, 0.5)
            gx.grid(linestyle='dotted', clip_on=True)

        if 3 in charts or "gM" in charts:
            """ """
            if 3 in charts:
                gMx.set(title=f"Massimi")
            else:
                gMx.set(xlabel='Ampiezza[rad]', ylabel='Frequenza',
                       title=f"{round(title/(2*mf.pi),5)}" + "[Hz]")
            gMx.yaxis.set_label_coords(-0.12, 0.5)
            gMx.grid(linestyle='dotted', clip_on=True)
            """ """
        if 3 in charts or "gm" in charts:
            """ """
            if 3 in charts:
                gmx.set(title=f"Minimi")
            else:
                gmx.set(xlabel='Ampiezza[rad]', ylabel='Frequenza',
                       title=f"{round(title/(2*mf.pi),5)}" + "[Hz]")
            gmx.yaxis.set_label_coords(-0.12, 0.5)
            gmx.grid(linestyle='dotted', clip_on=True)

    if "3s" in charts:
        sx.set(xlabel='Ampiezza[rad]', ylabel='Frequenza',
               title=f"{round(title/(2*mf.pi),5)}" + "[Hz]")
        sx.grid(linestyle='dotted', clip_on=True)

    # Aggiunta del titolo al subplot
    if 1 in charts:
        fig1.suptitle(f"{round(title/(2*mf.pi),5)}"+"[Hz]")
    if 2 in charts:
        fig2.suptitle(f"{round(title/(2*mf.pi),5)}"+"[Hz]")
    if 3 in charts:
        fig3.suptitle(f"{round(title/(2*mf.pi),5)}"+"[Hz]")

    if "a" in charts:
        ax.set(title=f"{round(title/(2*mf.pi),5)}"+"[Hz]")  # Completo
    if "b" in charts:
        bx.set(title=f"{round(title/(2*mf.pi),5)}"+"[Hz]")  # Smorzamento
    if "c" in charts:
        cx.set(title=f"{round(title/(2*mf.pi),5)}"+"[Hz]")  # Completo Abs
    if "d" in charts:
        dx.set(title=f"{round(title/(2*mf.pi),5)}"+"[Hz]")  # Smorzamento Abs
    if "e" in charts:
        ex.set(title=f"{round(title/(2*mf.pi),5)}"+"[Hz]")  # Stazionario
    if "f" in charts:
        fx.set(title=f"{round(title/(2*mf.pi),5)}"+"[Hz]")  # Linearizzazione



    # per calcolare l'Ampiezza devi trovare la fase
    # di risonanza e poi prendere in considerazione fino a che il torsionimetrometro è accesso

    # print(f"avg  : {mf.arythmAvg(columnized)}")
    # print(f"stdev: {mf.stDev(columnized)}")
    # print(f"$χ^2$: {mf.c(columnized)}")

    if save:
        if 1 in charts:
            fig1.savefig(f"{dir_path}grafici/tot/{round(title/(2*mf.pi),5)}.pdf")  # verisone con doppi grafici
        if 2 in charts:
            fig2.savefig(f"{dir_path}grafici/totAbs/{round(title/(2*mf.pi),5)}.pdf")  # versione con subplots
        if 3 in charts:
            fig3.savefig(f"{dir_path}grafici/totGauss/{round(title/(2*mf.pi),5)}.pdf")  # versione con subplots
        if "g" in charts:
            figG.savefig(f"{dir_path}grafici/Gauss/{round(title/(2*mf.pi),5)}.pdf")  # versione con subplots
        if "gM" in charts:
            figGM.savefig(f"{dir_path}grafici/Gauss_max/{round(title/(2*mf.pi),5)}.pdf")  # versione con subplots
        if "gm" in charts:
            figGm.savefig(f"{dir_path}grafici/Gauss_min/{round(title/(2*mf.pi),5)}.pdf")  # versione con subplots
        if "3s" in charts:
            fig3s.savefig(f"{dir_path}grafici/Gauss_3s/{round(title/(2*mf.pi),5)}.pdf")  # versione con subplots
        if "a" in charts:
            figa.savefig(f"{dir_path}grafici/singole/a/{round(title/(2*mf.pi),5)}.pdf")  # versione con subplots
        if "b" in charts:
            figb.savefig(f"{dir_path}grafici/singole/b/{round(title/(2*mf.pi),5)}.pdf")  # versione con subplots
        if "c" in charts:
            figc.savefig(f"{dir_path}grafici/singole/c/{round(title/(2*mf.pi),5)}.pdf")  # versione con subplots
        if "d" in charts:
            figd.savefig(f"{dir_path}grafici/singole/d/{round(title/(2*mf.pi),5)}.pdf")  # versione con subplots
        if "e" in charts:
            fige.savefig(f"{dir_path}grafici/singole/e/{round(title/(2*mf.pi),5)}.pdf")  # versione con subplots
        if "f" in charts:
            figf.savefig(f"{dir_path}grafici/singole/f/{round(title/(2*mf.pi),5)}.pdf")  # versione con subplots


    #plt.show()
    # Se il flag slideshow è attivo fa vedere i grafici
    if slideshow:
        """ """
        plt.show(block=False)
        # plt.close()
        plt.pause(3)
    elif not save:
        plt.show()

    for x in charts:
     plt.close()

print("]")

# istogramma omega_smorzata
if "o_s" in charts:
    figomega_s, omega_sx = plt.subplots()

# istogramma gamma_smorzata
if "g_s" in charts:
    figgamma_s, gamma_sx = plt.subplots()

if "t" in charts:
    figperiodi, tx = plt.subplots()

if "o_s" in charts:
    """ """
    misure = []

    misure = mf.columnizer(megaAvg,8)
    # print(f"{(abs(mf.arythmAvg(dajetempmaxy)-mf.arythmAvg(dajetempminy))/(mf.stDev(dajetempmaxy)**2+mf.stDev(dajetempminy)**2)**.5)}")

    nbins = int(len(misure) ** .5)

    sigma = mf.stDev(misure)
    mu = mf.arythmAvg(misure)
    x = np.arange(mu - 4 * sigma, mu + 4 * sigma, 0.000001)  # Cambiamo il range così da dezoommare
    gauss = mf.gauss(x, sigma, mu)

    # print(misure)
    massimo = np.max(misure)
    minimo = np.min(misure)

    binwidth = (massimo - minimo) / nbins
    gauss = gauss * binwidth * len(misure)
    # gauss = gauss * binwidth

    bidoni = [0] * nbins
    for i in range(0, nbins):
        for j in misure:
            if j > (minimo + binwidth * i) and j <= (minimo + binwidth * (i + 1)):
                bidoni[i] += 1

    bidoni[0] += 1
    norm = []
    # print("bindwith")
    for i in range(0, nbins):
        # norm.append(binwidth * bidoni[i] / (binwidth * len(misure)))
        norm.append(bidoni[i])
        # print(f"{round(minimo + binwidth * i, 3)}-{round(minimo + binwidth * (i + 1), 3)} : {norm[i]}")

    xnorm = []
    for i in range(0, nbins):
        xnorm.append(minimo + binwidth * i)
    # print(f"somma: {sum(norm)}")

    omega_sx.bar(xnorm, norm, width=binwidth, alpha=0.8, edgecolor='blue', align='edge', linewidth=1, label=f"misurazioni")
    omega_sx.plot(x, gauss, c="#d62728", label=f"")
    # print(x)
    # print(gauss)
    omega_sx.axvline(mu, color='r', linestyle='--', label="Mu")
    omega_sx.axvline(mu + sigma, color='orange', linestyle='-', label="sigma")
    omega_sx.axvline(mu - sigma, color='orange', linestyle='-')
    omega_sx.axvline(mu + 2 * sigma, color='cyan', linestyle=':', label="2 sigma")
    omega_sx.axvline(mu - 2 * sigma, color='cyan', linestyle=':')
    omega_sx.axvline(mu + 3 * sigma, color='green', linestyle='-.', label="3 sigma")
    omega_sx.axvline(mu - 3 * sigma, color='green', linestyle='-.')
    omega_sx.legend(loc='upper right')

    # omega_sx.set(xlabel='Ampiezza[rad]', ylabel='Frequenza',title=f"{round(title/(2*mf.pi),5)}"+"[Hz]")
    omega_sx.set(xlabel='$ω_s$[$rad*s^{-1}$]', ylabel='Frequenza', title=f"")
    omega_sx.grid(linestyle='dotted', clip_on=True)
    if save:
        figomega_s.savefig(f"{dir_path}grafici/omega_s.pdf")  # versione con subplots




if "g_s" in charts:
    """ """
    misure = []

    misure = plot3y

    misure = mf.columnizer(megaAvg, 4)
    # print(f"{(abs(mf.arythmAvg(dajetempmaxy)-mf.arythmAvg(dajetempminy))/(mf.stDev(dajetempmaxy)**2+mf.stDev(dajetempminy)**2)**.5)}")

    nbins = int(len(misure) ** .5)

    sigma = mf.stDev(misure)
    mu = mf.arythmAvg(misure)
    x = np.arange(mu - 4 * sigma, mu + 4 * sigma, 0.000001)  # Cambiamo il range così da dezoommare
    gauss = mf.gauss(x, sigma, mu)

    # print(misure)
    massimo = np.max(misure)
    minimo = np.min(misure)

    binwidth = (massimo - minimo) / nbins
    gauss = gauss * binwidth * len(misure)
    # gauss = gauss * binwidth

    bidoni = [0] * nbins
    for i in range(0, nbins):
        for j in misure:
            if j > (minimo + binwidth * i) and j <= (minimo + binwidth * (i + 1)):
                bidoni[i] += 1

    bidoni[0] += 1
    norm = []
    # print("bindwith")
    for i in range(0, nbins):
        # norm.append(binwidth * bidoni[i] / (binwidth * len(misure)))
        norm.append(bidoni[i])
        # print(f"{round(minimo + binwidth * i, 3)}-{round(minimo + binwidth * (i + 1), 3)} : {norm[i]}")

    xnorm = []
    for i in range(0, nbins):
        xnorm.append(minimo + binwidth * i)
    # print(f"somma: {sum(norm)}")

    gamma_sx.bar(xnorm, norm, width=binwidth, alpha=0.8, edgecolor='blue', align='edge', linewidth=1,
                 label=f"misurazioni")
    gamma_sx.plot(x, gauss, c="#d62728", label=f"")
    # print(x)
    # print(gauss)
    gamma_sx.axvline(mu, color='r', linestyle='--', label="Mu")
    gamma_sx.axvline(mu + sigma, color='orange', linestyle='-', label="sigma")
    gamma_sx.axvline(mu - sigma, color='orange', linestyle='-')
    gamma_sx.axvline(mu + 2 * sigma, color='cyan', linestyle=':', label="2 sigma")
    gamma_sx.axvline(mu - 2 * sigma, color='cyan', linestyle=':')
    gamma_sx.axvline(mu + 3 * sigma, color='green', linestyle='-.', label="3 sigma")
    gamma_sx.axvline(mu - 3 * sigma, color='green', linestyle='-.')
    gamma_sx.legend(loc='upper right')

    # omega_sx.set(xlabel='Ampiezza[rad]', ylabel='Frequenza',title=f"{round(title/(2*mf.pi),5)}"+"[Hz]")
    gamma_sx.set(xlabel='$γ$[$rad*s^{-1}$]', ylabel='Frequenza', title=f"")
    gamma_sx.grid(linestyle='dotted', clip_on=True)
    if save:
        figgamma_s.savefig(f"{dir_path}grafici/gamma_s.pdf")  # versione con subplots


if "t" in charts:
    """ """
    misure = []

    misure = mf.columnizer(megaAvg,2)
    # gamma
    # misure = [-0.0353,-0.0354,-0.0427,-0.0411,-0.0329,-0.045,-0.0456,-0.0443,-0.0371,-0.0335,-0.0412,-0.0434,-0.0429,-0.0446,-0.0344,-0.0456,-0.0454,-0.0419,-0.0381,-0.0431,-0.0411,-0.0463,-0.0418,-0.039,-0.0434,-0.04,-0.0417,-0.0375]
    # omega
    # misure = [5.69746,6.09051,5.94936,6.00506,3.61044,6.05258,6.60019,5.6624,6.05004,3.31468,4.80684,6.07242,6.07778,3.35719,6.08414,6.05262,6.08674,5.26014,6.06332,6.06614,6.1045,6.11711,6.05623,6.12808,6.1294,8.27336,6.22745,5.59242]
    # omega_s calcolato con
    # print(f"{(abs(mf.arythmAvg(dajetempmaxy)-mf.arythmAvg(dajetempminy))/(mf.stDev(dajetempmaxy)**2+mf.stDev(dajetempminy)**2)**.5)}")

    nbins = int(len(misure) ** .5)

    sigma = mf.stDev(misure)
    mu = mf.arythmAvg(misure)
    x = np.arange(mu - 4 * sigma, mu + 4 * sigma, 0.000001)  # Cambiamo il range così da dezoommare
    gauss = mf.gauss(x, sigma, mu)

    # print(misure)
    massimo = np.max(misure)
    minimo = np.min(misure)

    binwidth = (massimo - minimo) / nbins
    gauss = gauss * binwidth * len(misure)
    # gauss = gauss * binwidth

    bidoni = [0] * nbins
    for i in range(0, nbins):
        for j in misure:
            if j > (minimo + binwidth * i) and j <= (minimo + binwidth * (i + 1)):
                bidoni[i] += 1

    bidoni[0] += 1
    norm = []
    # print("bindwith")
    for i in range(0, nbins):
        # norm.append(binwidth * bidoni[i] / (binwidth * len(misure)))
        norm.append(bidoni[i])
        # print(f"{round(minimo + binwidth * i, 3)}-{round(minimo + binwidth * (i + 1), 3)} : {norm[i]}")

    xnorm = []
    for i in range(0, nbins):
        xnorm.append(minimo + binwidth * i)
    # print(f"somma: {sum(norm)}")

    tx.bar(xnorm, norm, width=binwidth, alpha=0.8, edgecolor='blue', align='edge', linewidth=1,
                 label=f"misurazioni")
    tx.plot(x, gauss, c="#d62728", label=f"")
    # print(x)
    # print(gauss)
    tx.axvline(mu, color='r', linestyle='--', label="Mu")
    tx.axvline(mu + sigma, color='orange', linestyle='-', label="sigma")
    tx.axvline(mu - sigma, color='orange', linestyle='-')
    tx.axvline(mu + 2 * sigma, color='cyan', linestyle=':', label="2 sigma")
    tx.axvline(mu - 2 * sigma, color='cyan', linestyle=':')
    tx.axvline(mu + 3 * sigma, color='green', linestyle='-.', label="3 sigma")
    tx.axvline(mu - 3 * sigma, color='green', linestyle='-.')
    tx.legend(loc='upper right')

    # omega_sx.set(xlabel='Ampiezza[rad]', ylabel='Frequenza',title=f"{round(title/(2*mf.pi),5)}"+"[Hz]")
    tx.set(xlabel='$periodi$[$s$]', ylabel='Frequenza', title=f"")
    tx.grid(linestyle='dotted', clip_on=True)
    if save:
        figperiodi.savefig(f"{dir_path}grafici/periodi.pdf")  # versione con subplots





megaAvg.sort()
"""
#calcolo omega megaAVG
print("wf, A , T, Pulsazione, Fase")
i = 0
for x in megaAvg:
    x.append(2*mf.pi/x[2])
    #x.append(np.arcsin(0.9673350192926328/x[1]))
    print(x)
    print(0.9673350192926328/x[1])
    megaAvg[i] = x

"""
x = mf.columnizer(megaAvg, 0)
y = mf.columnizer(megaAvg, 1)

x_functions = np.linspace(0.89*2*mf.pi, 1.01*2*mf.pi, 10000)
# popt, pcov = curve_fit(lorenzian, x, y)#, p0=[.96, 0.10, 0.01], maxfev=1000000
z = np.polyfit(x, y, 5)
f = np.poly1d(z)
normal_fit_y = f(x_functions)

# Fit the Lorentzian function to the data
popt, pcov = curve_fit(cauchy_lorentz, x, y, p0=[.96*2*mf.pi, 0.00955, 0.2,0],maxfev=10000)
cauchy_lorentz_y = cauchy_lorentz(x_functions, *popt)

chi2Brit = 0
for i in range(0,len(x)):
    oldchi2 = chi2Brit
    chi2Brit += ((y[i]-cauchy_lorentz(x[i], *popt))/(megaAvg[i][5]))**2
    #print(f"Sa[{i}]{megaAvg[i][5]}")
    #print(f"chi2Brit[{i}]{str(x[i]+0.000000001)[:7]}:brit:{str(y[i]+0.000000001)[:7]} y:{str(breit_wigner(x[i], *popt))[:7]} chi2:{chi2Brit-oldchi2}")
print(f"chi2Cauch: {chi2Brit}")


#popt, pcov = curve_fit(breit_wigner, x, y, p0=[.96*2*mf.pi, 0.10, 0.01,0],maxfev=10000)  # , p0=[.96, 0.10, 0.01], maxfev=1000000
popt, pcov = curve_fit(breit_wigner, x, y, p0=[.96*2*mf.pi, 0.00955, 0.2,0],maxfev=10000)  # , p0=[.96, 0.10, 0.01], maxfev=1000000
breit_wigner_y = breit_wigner(x_functions, *popt)

# plot fit
figauss, gaussx = plt.subplots()
figomega, omegax = plt.subplots()
figfase, fasex = plt.subplots()


# gaussx.plot(x_functions, normal_fit_y,c="#1f77b4", label='Aproximated Curve-Fit')
gaussx.plot(x_functions, breit_wigner_y, c="#ff7f0e", label='Breit-Wigner')
#gaussx.plot(x_functions, cauchy_lorentz_y, c="#2ca02c", label='Cauchy-Lorentz')

# different ranges gamma

#for gamma in np.arange(0.00, 0.1, 0.01):
#    gaussx.plot(x_functions, breit_wigner(x_functions, popt[0], gamma, popt[2]), linewidth="0.4")
#gaussx.plot(x_functions, breit_wigner(x_functions, popt[0], gamma, popt[2]), linewidth="0.4",
#            label=f'gamma range: 0.0-1.0')#+ fuori solo per mettere la label

# gaussx.plot(x_functions, cauchy_lorentz_y2,c="#ff7f0e", label='Cauchy-Lorentz2')
# gaussx.plot(x_functions, cauchy_lorentz_y3,c="#9467bd", label='Cauchy-Lorentz3')
gaussx.errorbar(x, y, c="#1f77b4",yerr=mf.columnizer(megaAvg,5), fmt='o', capsize=5, markersize=1, label="Dati Sperimentali")
gaussx.axhline(breit_wigner(popt[0], popt[0], popt[1], popt[2], popt[3])/2,c="#9467bd", linestyle="--",label=f'HM')

gaussx.set_xlim(0.89*2*mf.pi, 1.01*2*mf.pi)
#temp = round(breit_wigner(popt[0], popt[0], gamma, popt[2]) * 10 + .05, 2)+ .05
#print(temp)
#gaussx.set_ylim(0, (temp))

gaussx.legend(loc='upper left')
gaussx.set(xlabel='Frequenza Forzante[$rad/s$]', ylabel='Ampiezza[$rad$]',
           title=f"")
gaussx.grid(linestyle='dotted', clip_on=True)

#figura fasi
t = np.arange(0,2,0.0001)

fasex.plot(t/popt[0],np.arctan((popt[0]**2-t**2)/(2*popt[1]*popt[0])) )
for gamma in np.arange(0.2, 1.1, 0.2):
   fasex.plot(t/popt[0], np.arctan((popt[0]**2-t**2)/(2*gamma*popt[0])) , linewidth="0.8",label=f'γ={round(gamma,1)}')

t = np.arange(0,popt[0]-0.0001,0.0001)
fasex.plot(t/popt[0], -mf.pi*(t/popt[0]-1)/abs(t/popt[0]-1)/2, c="black")
t = np.arange(popt[0]+0.0001,2,0.0001)
fasex.plot(t/popt[0], -mf.pi*(t/popt[0]-1)/abs(t/popt[0]-1)/2, c="black")

fasex.set_ylim(-mf.pi,mf.pi)

fasex.legend()#loc='upper left'
fasex.set(ylabel=f'$β(ω)$', xlabel=f'$ω/ω_0$',
           title=f"")
fasex.grid(linestyle='dotted', clip_on=True)

omegax.errorbar(mf.columnizer(megaAvg,0), mf.columnizer(megaAvg,15), c="#1f77b4", fmt='+', capsize=5, markersize=6, label="$w_f*2pi$")
omegax.legend()#loc='upper left'
omegax.set(ylabel=f'$ω$', xlabel=f'$ω_f$',
           title=f"")
omegax.grid(linestyle='dotted', clip_on=True)

for i in range(0,len(megaAvg)):
    megaAvg[i][5]/=10

chi2Brit = 0
for i in range(0,len(x)):
    oldchi2 = chi2Brit
    chi2Brit += ((y[i]-breit_wigner(x[i], *popt))/(megaAvg[i][5]))**2
    #print(f"Sa[{i}]{megaAvg[i][5]}")
    #print(f"chi2Brit[{i}]{str(x[i]+0.000000001)[:7]}:brit:{str(y[i]+0.000000001)[:7]} y:{str(breit_wigner(x[i], *popt))[:7]} chi2:{chi2Brit-oldchi2}")
print(f"chi2Brit: {chi2Brit}")


plt.close()
plt.close()
plt.show()
#exit()
if save:
    figauss.savefig(f"{dir_path}grafici/HereWeGo.pdf")  # versione con subplots
    figfase.savefig(f"{dir_path}grafici/fase.pdf")  # versione con subplots

L = 0.75
df = 0.0004
rf = df / 2

dp = 0.0227
rp = dp / 2
m = 0.115

I = (7.44 * 10 ** -6)
avgwf = mf.arythmAvg(mf.columnizer(megaAvg, 0))
G = (4 * mf.pi ** 2 * I) / (L * avgwf ** 2)
K = 0.01  # (mf.pi/2)*(rf**4/L)*(G/m)
w0 = (K / I) ** .5

perr = np.sqrt(np.diag(pcov))

popt[1] = 1*popt[1]

#2*popt[1]/((popt[0]**2-(2*popt[1]**2))**.5)

print(f"ω_0={popt[0]}")
print(f"s_ω_0={perr[0]}")
print(f"gam={popt[1]}")
print(f"s_gam={perr[1]}")
print(f"s_A={perr[2]}")
print(f"s_D={perr[3]}")
print(f"ω_r={(popt[0]**2-(2*popt[1]**2))**.5}")
print(f"s_ω_r={((popt[0]/((popt[0]**2-(2*popt[1]**2))**.5))**2*perr[0]**2+(2*popt[1]/((popt[0]**2-(2*popt[1]**2))**.5))**2*perr[1]**2)**.5}")
#print(pcov[1][0])
print(f"s_y={0.001/(12**.5)}")
print(f"Q  ={popt[0]/(2*popt[1])}")
print(f" wf    :    A    :    s_A    :  Tempo  :    s_T    :   w_p   :  s_puls :   w_s   :   s_ws  :   w_0   :   sw0   :     a     :     b     :    s_a    :    s_b    :  s_fase :  compMx : omegaB : chi2lin")


for x in megaAvg:
    w_0 = (x[4]**2+0.045**2)**.5
    #(0.045/(x[4]**2+0.045**2))
    print(f"{str(x[0]+0.0000001)[:6]} : " # temp_wf, 
          f"{str(x[1]+0.0000001)[:7]} : "# temp_A, 
          f"{round(x[5],7)} : "# temp_sA,
          f"{str(x[2]+0.0000001)[:7]} : "# temp_T, 
          f"{round(x[6],7)} : "# temp_sT,
          f"{str(x[3]+0.0000001)[:7]} : "# temp_puls, 
          f"{str(x[11]+0.0000001)[:7]} : "# temp_sw,
          f"{str(x[4]+0.0000001)[:7]} : "# temp_w_s,
          f"{str(x[14]+0.0000001)[:7]} : "#s_ws
          f"{str(w_0+0.0000001)[:7]} : "#calcola w_o=(w_s**2+gamma**2)**.5
          f"{str((((0.045/(x[4]**2+0.045**2)))**2*perr[1]**2+(x[4]/w_0)**2*x[14]**2)**.5)[:7]} : "#calcola w_o=(w_s**2+gamma**2)**.5
          f"{round(x[7],7)} : "#a
          f"{round(x[8],7)} : "#b
          f"{round(x[9],7)} : "#sa
          f"{round(x[10],7)} : "#sb
          f"{str(x[12]+0.0000001)[:7]} : "# temp_sfase,
          f"{str(x[12]+0.0000001)[:7]} : "# compTrueMaxVal
          f"{str(x[15]+0.0000001)[:7]} : "# compTrueMaxVal
          f"{x[16]}")# omega,
print(f" wf    :    A    :    s_A    :  Tempo  :    s_T    :   w_p   :  s_puls :   w_s   :   s_ws  :   w_0   :   sw0   :     a     :     b     :    s_a    :    s_b    :  s_fase :  compMx : omegaB : chi2lin")

print("\nCalcoli con Valori scelti")
w_s = 6.42597
s_ws = 0.00409
gamma =   0.03127
s_gamma = 0.00003

w0 = (w_s**2+(gamma**2))**.5
sw0 = ((gamma/(w_s**2+gamma**2)**.5)**2*s_gamma**2+(w_s/(w_s**2+gamma**2)**.5)**2+s_ws**2)**.5

wr = (w0**2-(2*gamma**2))**.5
swr = ((w0/((w0**2-(2*gamma**2))**.5))**2*sw0**2+(2*gamma/((w0**2-(2*gamma**2))**.5))**2*s_gamma**2)**.5
perr = np.sqrt(np.diag(pcov))

a = popt[2]
b = popt[0]
c = popt[1]
d = popt[3]

pa = 1/(2*c*(b**2-(c**2))**.5)
pb = -(a*b)/(2*c*(b**2-(c**2))**(3/2))
pc = -(a*(b**2-2*(c**2)))/(2*(c**2)*(b**2-(c**2))**(3/2))

print(popt)

print(f"ω_0={w0}")
print(f"s_ω_0={sw0}")
print(f"ω_r={wr}")
print(f"s_ω_r={swr}")
print(f"ybrit_w_r: {breit_wigner(wr, *popt)}")
print(f"sBrit_w_0  {((pa)**2*perr[2]**2+(pb)**2*perr[0]**2+(pc)**2*perr[1]**2+(1)**2*perr[3]**2)**.5}")