import random

# el formato de polarData es:
# longitud, angulo, carga, IDOriginal de StopData, RutaTemporal
MaxLimitCharge = 200



class DeliverPoint:
    # def _init_(self):
    # print('Helloh')

    def AppendDeliverPoint(self, _Magnitud, _Angle, _ChaargeValue, _ID, _Position):
        self.Magnitud = _Magnitud
        self.Angle = _Angle
        self.ChargeValue = _ChaargeValue
        self.ID = _ID
        self.Position = _Position


class Route:
    # GroupDeliverPoint = []
    def __init__(self, _RouteID):
        self.RouteID = _RouteID

        # print('RouteID: ' +str(RouteID) + ' Added')
        self.GroupDeliverPoint = []

    def AppendDelievrPoint(self, _DeliverPoint):
        self.GroupDeliverPoint.append(_DeliverPoint)
        # print('DeliverPoint Added ID: ' + str(_DeliverPoint.ID) + ' To RouteID: ' + str(self.RouteID))
        # print('Current Len: ' +str(self.RouteLen()))

    def RouteLen(self):
        return len(self.GroupDeliverPoint)


polarData = [[22.360679774997898, 26.56505117707799, 13, 6, 1],
 [14.142135623730951, 45.0, 10, 0, 1],
 [15.231546211727817, 66.80140948635182, 10, 4, 1],
 [26.248809496813376, 107.74467162505694, 9, 11, 1],
 [14.142135623730951, 135.0, 10, 1, 1],
 [21.213203435596427, 135.0, 5, 10, 1],
 [20.615528128088304, 194.03624346792648, 26, 8, 1],
 [11.180339887498949, 206.56505117707798, 3, 9, 1],
 [14.142135623730951, 225.0, 10, 2, 1],
 [18.0, 270, 7, 5, 1],
 [25.0, 306.86989764584405, 19, 7, 1],
 [14.142135623730951, 315.0, 10, 3, 1]]


# number of customers, vehicle capacity, maximum route time, drop time
routeSettings=['199', '200', '200', '10']

def generarPoblacionInicial(PolarData):
    poblacionInicial = []
    solucionActual = []
    paradasEnRutaTemporal = len(polarData) / 4
    # arbitrariamente se dividio en 4 rutas, aqui hay que calcular la carga de el vehiculo

    numeroElementosEnPolar = list(range(len(polarData)))
    for i in numeroElementosEnPolar:
        solucionActual = numeroElementosEnPolar[i:] + numeroElementosEnPolar[:i]
        rutaActual = 1
        currentIndex = 0
        DeliverPointList=[]
        for j in numeroElementosEnPolar[i:] + numeroElementosEnPolar[:i]:

            ###Empieza
            #for idxPolar in range(len(PolarData)):
            x = DeliverPoint()
            x.AppendDeliverPoint(PolarData[j][0], PolarData[j][1], PolarData[j][2],PolarData[j][3], j)
            DeliverPointList.append(x)
            # print('Fin Convertion')
            # print('Max Limit Charge Per Route: ' + str(MaxLimitCharge))
            RouteList = []
            RouteID = 1
            RouteList.append(Route(RouteID))
            CurrentCharge = 0
            for i in range(len(DeliverPointList)):
                CurrentCharge = CurrentCharge + DeliverPointList[i].ChargeValue
                if CurrentCharge <= MaxLimitCharge:
                    zeta = 0
                else:
                    RouteID = RouteID + 1
                    RouteList.append(Route(RouteID))
                    CurrentCharge = DeliverPointList[i].ChargeValue
                RouteList[RouteID - 1].AppendDelievrPoint(DeliverPointList[i])
            ###Termina
            currentIndex += 1
        poblacionInicial.append(RouteList)
    return poblacionInicial



poblacionInicial = generarPoblacionInicial(polarData)
#print(poblacionInicial)
#print (poblacionInicial[0].RouteID)


def mutar(lista1, lista2):
    lista3 = []
    lista4 = []
    y = random.randint(1, 10)
    x1 = lista1[y]
    x2 = lista2[y]
    for i in range(len(lista1)):
        if i == y:
            lista3.append(x2)
        else:
            lista3.append(lista1[i])

    for j in range(len(lista2)):
        if j == y:
            lista4.append(x1)
        else:
            lista4.append(lista2[j])
    return lista3, lista4


ind1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
ind2 = [9, 8, 7, 6, 1, 5, 4, 3, 2]

mutacion = mutar(ind1, ind2)
print (ind1, ind2)
print (mutacion[0], mutacion[1])

