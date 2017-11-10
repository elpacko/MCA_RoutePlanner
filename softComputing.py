# coding: utf-8

# Aplicacion de un algoritmo genetico para la solucion de el problema de Vehicle Routing Problem
### version 2.0

# ## Pre-procesamiento
# Primero se importan las librerias requeridas para el programa

# In[1]:


import math
import os
import numpy as np
import matplotlib.pyplot as plt
# from urllib.request import urlopen
from pylab import *
import random
import networkx as nx
import matplotlib.cbook as cbook
from scipy.misc import imread #sudo pip install Pillow==2.6.0


def calculateMagnitude(_coordenada):
    return math.sqrt(pow(_coordenada.y, 2) + pow(_coordenada.x, 2))


def calculatePositiveAngle(x, y):
    if x == 0:
        if y > 0:
            _angle = 90
        else:
            _angle = 270
    else:
        _angle = math.degrees(math.atan2(y, x))
        if(_angle<0):
            _angle = 360 + _angle
    return _angle


class FitnessUnfitness:
    def __init__(self):
        self.fitness = 0.0
        self.unfitness = 0.0

class CoordenadaPolar:
    angulo = 0
    magnitud = 0


class CoordenadaCartesiana:

    def __init__(self, _CoordenadaArray=None):
        if _CoordenadaArray is not None:
            self.x = int(_CoordenadaArray[0])
            self.y = int(_CoordenadaArray[1])
        else:
            self.x = 0
            self.y = 0


class RouteSettings:
    # GroupDeliverPoint = []
    def __init__(self, _routesettingsarray):
        self.CustomerQTY = int(_routesettingsarray[0])
        self.VehicleCapacity = int(_routesettingsarray[1])
        self.MaximumRouteTime = int(_routesettingsarray[2])
        self.DropTime = int(_routesettingsarray[3])


class ParadaCliente:

    def __init__(self, _ParadaDatos, _paradaID, _coordenadasDepot):
        self.peso = int(_ParadaDatos[2])
        self.paradaID = _paradaID
        self.coordenada = CoordenadaCartesiana(_ParadaDatos)
        self.coordenadaRespectoDepot = CoordenadaCartesiana(_ParadaDatos)
        self.coordenadaRespectoDepot = self.normailizaDepot(_coordenadasDepot)
        self.coordenadaPolarRespectoDepot = CoordenadaPolar()
        self.coordenadaPolarRespectoDepot.angulo = calculatePositiveAngle(self.coordenadaRespectoDepot.x, self.coordenadaRespectoDepot.y)
        self.coordenadaPolarRespectoDepot.magnitud = calculateMagnitude(self.coordenadaRespectoDepot)
        self.ordenAnguloPolar = 0

    def normailizaDepot(self, _coordenadasDepot):
        _coordenadaRespectoDepot = CoordenadaCartesiana()
        _coordenadaRespectoDepot.x = self.coordenada.x - _coordenadasDepot.x
        _coordenadaRespectoDepot.y = self.coordenada.y - _coordenadasDepot.y
        return _coordenadaRespectoDepot



# routeSettings
# number of customers, vehicle capacity, maximum route time, drop time
def generarPoblacionInicial(stopData, routeSettings):
    poblacionInicial = []
    pesosPoblacionInicial = []
    x = list(range(len(stopData)))
    for i in x:
        solucionActual = x[i:] + x[:i]
        rutaActual = 1
        cargaDeRuta = 0
        for j in solucionActual:
            cargaDeRuta += stopData[j].peso
            printForDebug('cargaDeRuta:' + str(cargaDeRuta))
            if cargaDeRuta > routeSettings.VehicleCapacity:
                rutaActual += 1
                cargaDeRuta = stopData[j].peso
            solucionActual[j] = rutaActual
        poblacionInicial.append(solucionActual)
    return poblacionInicial


# ## Mutacion
# The performance of our GAwas found to be improved by applying a simple mutation to new
# oFspring, in which two genes are selected at random and their values are exchanged. Thus, two
# randomly chosen customers are switched between vehicles, except in cases where the two customers
# happen to be on the same vehicle. Other types of mutation such as shifting one or more randomly
# chosen customers to a neighbouring vehicle were less eFective.

# In[11]:


def mutarSolucion(hijo1,hijo2):
    genAMutar = random.randint(0, len(hijo1)-1)
    retryCount = 0
    while hijo1[genAMutar] == hijo2[genAMutar]:
        retryCount += 1
        if (retryCount>100): #si la poblacion es muy chica se puede quedar iterando indefinidamente
            print ('mutar Solucion 100th retry')
            break
        genAMutar = random.randint(0, len(hijo1)-1)
    # print("Mutando de:" + str(hijo1[genAMutar]) + " a " + str(hijo2[genAMutar]))
    valorHijo1 = hijo1[genAMutar]
    hijo1[genAMutar] = hijo2[genAMutar]
    hijo2[genAMutar] = valorHijo1
    return hijo1, hijo2


# ## Reproduccion
# En esta funcion se toman 2 padres para generar 2 hijos combinando sus genes, se divide el cromosoma en 3 para poder intercalar los alelos

# In[12]:


def generarHijos(padre1, padre2, mutacion):
    # Calculo de los Crossover Points
    crossoverSize = int(len(padre1) / 3)
    # print (crossoverSize)
    hijo1 = padre1[:crossoverSize] + padre2[crossoverSize:(crossoverSize * 2)] + padre1[(crossoverSize * 2):]
    hijo2 = padre2[:crossoverSize] + padre1[crossoverSize:(crossoverSize * 2)] + padre2[(crossoverSize * 2):]
    if random.random() < mutacion:
        #print('Mutando...')
        hijo1, hijo2 = mutarSolucion(hijo1, hijo2)
    return hijo1, hijo2


# ## Validacion
# en esta funcion se reciben 2 soluciones (o individuos de la poblacion) para comparar su fitness y unfitnees y asi regresar la solucion optima

# In[13]:

def generateVectorDeCeros(columns):
    vectorARegresar = []
    for j in list(range(columns)):
        vectorARegresar.append(0)
    return vectorARegresar

def generateMatrizDeCeros(rows, columns):
    matrizARegresar = []
    for i in list(range(rows)):  # aqui hay que cambiar de soulucion a cant. de rutas
        solucionEnCeros = []
        matrizARegresar.append(generateVectorDeCeros(columns))
    return matrizARegresar


def matrixDeSumadeRutas(stopData,matrizDeSuma,rutas): #OPTIMIZAR
    for i, parada in enumerate(stopData):
        rutaActual = int(rutas[i]) - 1  # parada[4]-1
        matrizDeSuma[rutaActual][0] += parada.peso  # se incrementa en base al peso de esa parada
        matrizDeSuma[rutaActual][1] += routeSettings.DropTime  # se incrementa en base al drop time
    return matrizDeSuma

def sumaMatrizDeSuma(matrizDeSuma):
    for i, value in enumerate(matrizDeSuma):
        # el articulo dice que es la proporcion (carga/capacidad), pero es practicamente lo mismo
        matrizDeSauma[i][2] = matrizDeSuma[i][0] - routeSettings.VehicleCapacity  # se calcula el delta del peso
        matrizDeSuma[i][3] = matrizDeSuma[i][1] - routeSettings.MaximumRouteTime  # se calcula el delta del tiempo
    return matrizDeSuma


# routeSettings
# number of customers, vehicle capacity, maximum route time, drop time
def calculaUnfitness(rutas, stopData):
    # print(solucion)
    # las columnas de solucion son = polarData: magnitud,angulo,peso,id,ruta
    matrizDeSuma = generateMatrizDeCeros(np.amax(rutas), 4)
    # rutasNP = np.array(rutas)
    # np.argwhere(rutasNP == 4)
    # las columnas de matrizDeSuma son: peso, tiempo, delta Peso, delta Tiempo. y la posicion +1 es el id de ruta
    matrizDeSuma = matrixDeSumadeRutas(stopData,matrizDeSuma,rutas)
    matrizDeSuma = sumaMatrizDeSuma(matrizDeSuma)
    totales = np.sum(matrizDeSuma, axis=0)  # se suman las columnas
    return totales[2] + totales[3]  # se suman los totales de las deltas


def calculaFitness(solucion):

    distanciaDeRutas = calcularDistanciaRecorrida(solucion);

    #dibujaGrafo(grafoCompleto, solucion, None, 0, 0, True)
    return sum(distanciaDeRutas.values())
    # opcion B, la distancia recorrida, para esto hay que calcular la distancia entre nodos


    # opcion A, es la cantidad de las rutas distintas
    #return len(np.unique(solucion))  # cantidad de rutas distintas


def calcularDistanciaRecorrida(solucion):
    distanciaDeRutas={}
    solucionNP = np.asarray(solucion)
    for rutaNumero in np.unique(solucionNP):
        distanciaDeRutas[rutaNumero] = 0.0
        paradaAnterior = 0
        for paradaID in np.add(np.where(solucionNP == int(rutaNumero))[0],1): #TODO: aqui se asume que el orden es secuencial al polar, se necesita el 2opt para que nos de el orden
            distanciaDeRutas[rutaNumero]+=grafoCompleto[paradaAnterior][paradaID ]
            paradaAnterior = paradaID
        distanciaDeRutas[rutaNumero] += grafoCompleto[0][paradaAnterior]
    return distanciaDeRutas

def dosOPT():

    return
    #dibujaGrafo()
def solucionOrden(solucion):

    return


def calcularFitnessyUnfitnessDePoblacion(stopData, poblacionInicial):
    fitnessYUnfitnessDePoblacion.fitness = generateVectorDeCeros(len(poblacionInicial))
    fitnessYUnfitnessDePoblacion.unfitness = generateVectorDeCeros(len(poblacionInicial))
    for i, individuo in enumerate(poblacionInicial):  # aqui es por solucion..
        fitnessYUnfitnessDePoblacion.fitness[i] = calculaFitness(individuo)
        fitnessYUnfitnessDePoblacion.unfitness[i] = calculaUnfitness(individuo, stopData)
    return fitnessYUnfitnessDePoblacion


# regresa el set de soluciones, reemplazando un individuo por el nuevo si es que aplica
def compararHijo(fitnessYUnfitnessDePoblacion, solucionHijo, poblacion, stopData):
    hijoUnfitness = calculaUnfitness(solucionHijo, stopData)
    hijoFitness = calculaFitness(solucionHijo)
    #print ("hijo Unfitness:" + str(hijoUnfitness))
    poblacionFitnessMayorHijo = []
    poblacionUnitnessMayorHijo = []
    poblacionFitnessMenorHijo = []
    poblacionUnitnessMenorHijo = []
    fitnessNP = np.asarray(fitnessYUnfitnessDePoblacion.fitness)
    poblacionUnitnessMayorHijo = np.where(fitnessNP >= hijoFitness)[0].tolist()
    poblacionUnitnessMenorHijo = np.where(fitnessNP < hijoFitness)[0].tolist()
    unfitnessNP = np.asarray(fitnessYUnfitnessDePoblacion.unfitness)
    poblacionFitnessMayorHijo = np.where(unfitnessNP >= hijoUnfitness)[0].tolist()
    poblacionFitnessMenorHijo = np.where(unfitnessNP < hijoUnfitness)[0].tolist()
    individuoAReemplazarID = -1
    # s1 es donde el fitness y el unfitness de cualquier elemento de la solucion es mayor o igual al del hijo

    s1 = set(poblacionUnitnessMayorHijo).intersection(poblacionFitnessMayorHijo)
    if len(s1) > 0:
        individuoAReemplazarID = list(s1)[0]
    # s2 es donde el fitness del elemento es menor que el del hijo y unfitness es mayor o igual al hijo
    s2 = set(poblacionUnitnessMayorHijo).intersection(poblacionFitnessMenorHijo)
    if len(s2) > 0 and individuoAReemplazarID < 0:
        individuoAReemplazarID = list(s2)[0]
    # s3 es donde el fitness es mayor y el untfitness es menor
    s3 = set(poblacionUnitnessMenorHijo).intersection(poblacionFitnessMenorHijo)
    if len(s3) > 0 and individuoAReemplazarID < 0:
        individuoAReemplazarID = list(s3)[0]
    if individuoAReemplazarID >= 0:
        #print('se reemplaza:'+ str(individuoAReemplazarID))
        poblacion[individuoAReemplazarID] = solucionHijo
        fitnessYUnfitnessDePoblacion.fitness[individuoAReemplazarID] = hijoFitness
        fitnessYUnfitnessDePoblacion.unfitness[individuoAReemplazarID] = hijoUnfitness
    return poblacion, fitnessYUnfitnessDePoblacion


def dibujaGrafo( grafocompleto, individuo, ordenindividuo,generacionActual,individuoNumero,batchMode = True):

    G = nx.Graph()
    individuoNP = np.asarray(individuo)

    for numeroRutaActual in range(1, np.amax(individuoNP)+1):
        ultimaParadaEnRuta = '0'
        rutaActual = np.where(individuoNP == numeroRutaActual)[0].tolist()
        if(len(rutaActual)>=1):
            #G.add_edge('0', str(rutaActual[0]), color=numeroRutaActual)
            for i, paradaEnRuta in enumerate(rutaActual):
                fromParada = ultimaParadaEnRuta
                toParada = str(paradaEnRuta+1)
                ultimaParadaEnRuta = str(toParada)
                G.add_edge(fromParada, toParada, color=numeroRutaActual)
            G.add_edge(str(ultimaParadaEnRuta), '0', color=numeroRutaActual)
    nodeColors = [int(individuo[int(node)-1]) for node in G.nodes()]

    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]


    nx.draw_networkx_nodes(G, posicionAnguloPolar, node_color=nodeColors, cmap=plt.get_cmap('jet'), label='nodes', node_size=20)
    nx.draw_networkx_edges(G, posicionAnguloPolar, arrows=False, edge_color=colors,  )
    nx.draw_networkx_labels(G, posicionAnguloPolar, font_size=12)
    plt.grid(True)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



    #imgFile = open('juarez.jpg', 'r')
    #datafile = cbook.get_sample_data(os.path.abspath('')+'/juarez.png', asfileobj=True)
    #img = imread(datafile)
    #mapCoordinateExtension = 70.0
    #plt.imshow(img, zorder=0, extent=[-mapCoordinateExtension*1.42 , mapCoordinateExtension*1.42 , -mapCoordinateExtension  , mapCoordinateExtension  ])

    plt.title("Generacion "+str(generacionActual)+ " Individuo "+ str(individuoNumero))

    #batchMode = False
    if(batchMode or not debugmode):
        plt.savefig("OutputGraphs/Gen"+str(generacionActual)+"Ind" + str(individuoNumero) + ".png")
        plt.clf()
    else:
        show()


#Inicia Ejecucion de codigo


 



# Datos de prueba
lines = []
lines.append("12 20 20 2")
lines.append("35 35")

lines.append("7 43 27")
lines.append("62 48 15")
lines.append("45 45 10")
lines.append("25 45 10")
lines.append("25 25 10")
lines.append("45 25 10")

lines.append("41 49 10")
lines.append("35 17 7")
lines.append("55 45 13")
lines.append("55 20 19")
lines.append("15 30 26")
lines.append("25 30 3")
lines.append("20 50 5")
lines.append("10 43 9")
lines.append("33 44 20")
lines.append("9 56 13")
lines.append("66 14 22")
lines.append("44 13 28")
lines.append("26 13 12")
lines.append("11 28 6")
lines.append("17 64 14")

#fin de datos de prueba
verboseLevel = 0;
verboseLevel = True

def printForDebug(message):
    if verboseLevel > 0:
        print message
    


# vrpURL = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/vrpnc10.txt"


lines = tuple(open('vrpnc6.txt', 'r'))  # read from benchmark file

# la definicion de las lineas del archivo estan especificadas en: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/vrpinfo.html

routeSettingsParsed = False
depotCoordinatesParsed = False
stopData = []
posicionAnguloPolar = {}
for i, line in enumerate(lines):
    if not routeSettingsParsed:
        routeSettings = RouteSettings(line.split())
        routeSettingsParsed = True
        continue
    if not depotCoordinatesParsed:
        coordenadasDepot = CoordenadaCartesiana(line.split())
        depotCoordinatesParsed = True
        continue
    paradaCliente = ParadaCliente(line.split(), i - 1, coordenadasDepot)
    stopData.append(paradaCliente)

print('Coordenadas del Depot:' + str(coordenadasDepot.x) + ',' + str(coordenadasDepot.y))
print(len(stopData))

stopDataOrdenado = sorted(stopData, key=lambda stop: stop.coordenadaPolarRespectoDepot.angulo)

for i, stop in enumerate(stopDataOrdenado):
    printForDebug("angulo:" + str(stop.coordenadaPolarRespectoDepot.angulo) + " x:" +  str(stop.coordenadaRespectoDepot.x) + " y:" + str(stop.coordenadaRespectoDepot.y))
    stopData[stop.paradaID - 1].ordenAnguloPolar = i + 1
    posicionAnguloPolar[str(i + 1)] = (stop.coordenadaRespectoDepot.x, stop.coordenadaRespectoDepot.y)

# posicionAnguloPolar['0'] = (coordenadasDepot.x, coordenadasDepot.y)
posicionAnguloPolar['0'] = (0, 0)

grafoCompleto = {}
depotDistances = {}

for item in stopData:
    depotDistances[item.paradaID] = item.coordenadaPolarRespectoDepot.magnitud

# depotDistances.append((str(stopDataId + 1), magnitude))


grafoCompleto[0] = depotDistances
# el elemento 0 es el depot, el 1 es el primer elemento del stopData (indice 0)

for i, originPoint in enumerate(stopData):
    originPointDistances = {}
    for j, destinationPoint in enumerate(stopData[i + 1:]):
        sendCoordenate = CoordenadaCartesiana()
        sendCoordenate.x = destinationPoint.coordenadaRespectoDepot.x - originPoint.coordenadaRespectoDepot.x
        sendCoordenate.y = destinationPoint.coordenadaRespectoDepot.y - originPoint.coordenadaRespectoDepot.y
        originPointDistances[(j + i + 2)] = calculateMagnitude(sendCoordenate);
    grafoCompleto[(i + 1)] = originPointDistances

tempVar = 0

print grafoCompleto[(tempVar)]





# generacion de variables Iniciales
fitnessYUnfitnessDePoblacion = FitnessUnfitness()
poblacion = generarPoblacionInicial(stopDataOrdenado, routeSettings)
fitnessYUnfitnessDePoblacion = calcularFitnessyUnfitnessDePoblacion(stopDataOrdenado, poblacion)
#valores de afinacion
numeroDeHijosPorGeneracion = 200
numeroDeGeneraciones = 50
porcentajeMutacion = 0.01
numeroDeGeneracion = 0
mejoraEntreGeneraciones = 100.0
poblacionFitnessNP = np.asarray(fitnessYUnfitnessDePoblacion.fitness)


#fitness es la distancia Recorrida
#unfitness es la suma de los excesos de peso y distancia


#for i, individuoAImprimir in enumerate(poblacion):
#    dibujaGrafo(grafoCompleto, individuoAImprimir, None, 0,  i)


while numeroDeGeneracion <numeroDeGeneraciones: # pendiente agregar chequeo si se quedo estancado (local minima)
    fitnessYUnfitnessDeGeneracion = average(fitnessYUnfitnessDePoblacion.fitness)
    print ("Generacion," + str(numeroDeGeneracion) + ",Mejora," + str(mejoraEntreGeneraciones)+ ",Distancia Promedio," + str(fitnessYUnfitnessDeGeneracion))
    poblacionNueva = poblacion
    for i in range(numeroDeHijosPorGeneracion):
        if i % 100 == 0:
            printForDebug("Hijo " + str(i))
        idPadre1 = random.randint(1, len(poblacion))
        idPadre2 = random.randint(1, len(poblacion))
        while idPadre1 == idPadre2:
            idPadre2 = random.randint(1, len(poblacion))
        hijo1, hijo2 = generarHijos(poblacion[idPadre1-1], poblacion[idPadre2-1], porcentajeMutacion)
        poblacionNueva, fitnessYUnfitnessDePoblacion = compararHijo(fitnessYUnfitnessDePoblacion, hijo1, poblacionNueva, stopDataOrdenado)
        poblacionNueva, fitnessYUnfitnessDePoblacion = compararHijo(fitnessYUnfitnessDePoblacion, hijo2, poblacionNueva, stopDataOrdenado)
        printForDebug(poblacionNueva)
    numeroDeGeneracion += 1
    nuevamejoraEntreGeneraciones = 1 - (average(fitnessYUnfitnessDePoblacion.fitness) / fitnessYUnfitnessDeGeneracion)
    if nuevamejoraEntreGeneraciones  > 0:
        poblacion = poblacionNueva
        mejoraEntreGeneraciones = nuevamejoraEntreGeneraciones
        poblacionFitnessNP = np.asarray(fitnessYUnfitnessDePoblacion.fitness)
        minAvgDistanceID = np.argmin(poblacionFitnessNP)
        fitnessMejorIndividuo = str(fitnessYUnfitnessDePoblacion.fitness[minAvgDistanceID])
        dibujaGrafo(grafoCompleto,
                    poblacion[minAvgDistanceID],
                    None,
                    numeroDeGeneracion,
                    '_' + str(minAvgDistanceID) + '_avg_' + fitnessMejorIndividuo  )
        #dibujaGrafo(grafoCompleto, poblacion[np.argmax(poblacionFitnessNP)], None, numeroDeGeneracion,
         #           '_MayorFitness_' + str(np.argmax(poblacionFitnessNP)))
    else:
        printForDebug("brincando generacion, Mejora:" + str(nuevamejoraEntreGeneraciones))





