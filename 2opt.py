import math, random, exceptions

#Get distance between 2 cities
#Cities stored as [ident, x, y]
def getDistance(city1, city2):
    return 	math.sqrt((int(city2[1]) - int(city1[1]))*2 + (int(city2[2])-int(city1[2]))*2)

# Distancia total del recorrido **
#A function to get the total weight of a path
#This function is messy because of an off-by-1 error introduced by the tour file starting at 1 instead of 0
def getWeight(perm):
#Set the initial distance to 0
    dist = 0
    #We now need to calculate and add the distance between each city in the path from 0 to n
    for index in range(len(perm)):
        try:
            #Pass the 2 cities to the distance formula and add the value to dist
            dist += getDistance(cities[perm[index]-1], cities[perm[index+1]-1])
        except:
            #We don't need to check bounds because the final pass will throw an out-of-bounds
            #exception so we just catch it and skip that calculation
            pass
    #All TSP solutions are cycles so we now have to add the final city back to the initial city to the total dist
    #Python has a nifty convention where list[-1] will return the last element, list[-2] will return second to last, etc.
    dist += getDistance(cities[perm[-1]-1], cities[perm[0]-1])
    #We now have the total distance so return it
    return dist

#Function to work with both float and int inputs
def num(s):
    try:
        #If there is an error it is due to precision loss
        return int(s)
    except exceptions.ValueError:
        return float(s)

#Calculate the 'best' tour for a set of cities using 2-opt
def two_opt(cities, numrounds, numiters):
    #Initialize the total weight to be averaged to 0
    results = 0
    #Create the initial 1...N permutation and find its weight
    curbest = [[], None]
    for city in cities:
        curbest[0].append(city[0])
    curbest[1] = getWeight(curbest[0])
    #Create the 'next' tour
    #[:] makes a copy of the array in curbest[0]
    next = [curbest[0][:], curbest[1]]

    for x in range(0, numrounds):
        for x in range(0, numiters):
            #Pick 2 random edges
            num1 = random.randint(0, len(cities)-1)
            num2 = (num1 + random.randint(0, len(cities)-2))%len(cities)
            #Swap the edges and get the new weight
            next[0][num1], next[0][num2] = next[0][num2], next[0][num1]
            next[1] = getWeight(next[0])
            #If the new tour is better than the old tour, set new tour as current best
            if next[1] < curbest[1]:
                curbest = next
          #Add the current best weight to results after each round
        results += curbest[1]
        print (curbest)
    #Return an arbitrary path and the average of all of the results of the rounds
    return [curbest[0], results/numrounds]

#-------------------The actual script begins here-----------------------

cities = [[1,288,149],[2,288,129],[3,270,133],[4,256,141],[5,256,157],[6,246,157],[7,236,169],[8,228,169],[9,228,161],[10,220,169]]
city = [[1,288,149],[2,288,129],[3,270,133],[4,256,141],[5,256,157],[6,246,157],[7,236,169],[8,228,169],[9,228,161],[10,220,169]]

#Set the default values for rounds and iters
rounds = 4
iters = 20

opt_tour = two_opt(cities, rounds, iters)
print ('The optimum tour is: %s (%f)' % (opt_tour[0], opt_tour[1]))
print ('There are %d cities in this tour.' % (len(opt_tour[0])))