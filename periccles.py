import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv

'''
Heatmap to visualize extremal points in 
b,c landscape of periodic parameter space
'''


def generateHeatmap(blb, bub, clb, cub, stepsize, x):
    brange = np.arange(blb, bub, stepsize)
    crange = np.arange(clb, cub, stepsize)
    prodlis = list(itertools.product(brange, crange))
    #print('Blb, Bub: ',blb,' ',bub)
    #print('B-range',brange)

    complis = []
    for e in prodlis:
        b, c = e
        val = np.sin(b * x - c)
        if val < 0.99 and val > 0.0:
            val = 0
        if val > -0.99 and val < 0.0:
            val = 0
        complis.append((b, c, val))

    #print('COMPLIS: ',complis)
    df = pd.DataFrame(complis)
    #print('~~~DEBUG~~~~~: ', df)
    #print('~~~~~~~~~~~~~')
    dfpivoted = df.pivot(index=0, columns=1, values=2)

    #ax = sns.heatmap(dfpivoted, cmap='coolwarm', annot=True)
    # print(dfpivoted)
    #plt.show()

    return dfpivoted


# hmrun1 = generateHeatmap(1.7,1.9,2,2.2,0.03,-5)


'''
Summary of PERICCLES intersection insights:

1. If at one corner the resulting D value is different then the others
--> intersection, even if the function does not intersect the edges

2. If all corner D values are above or below the expected D-bounds
we have to check if there are any min or max within the (A),B,C bounds

3. If there are none --> no intersection

4. If there are any min/max --> if the resulting D value at the min/max is equal
to the all-agreeing corner D-values --> no intersection

else: --> intersection

We are in step 4 in general only interested if there exists a min/max
not in -how many-. Use grid based approach with increasing resolution to determine existence
or abscence of extremal
'''

# boundaryX = [(1,3),(3,5),(2,4),(4,6)]
boundaryX = [(1, 3), (1.7, 1.9), (2, 2.2), (8.8, 9)]
# boundaryX = [(1,3),(1.7,11.9),(2,22.2),(8.8,9)]

from itertools import *


def boundaryCheck(bound, x, y):
    outofboundslis = []
    abound, bbound, cbound, dbound = bound

    AUTHORITYINTERSECTFLAG = False

    for e in product([0, 1], repeat=4):
        abit, bbit, cbit, dbit = e
        alock, block, clock, dlock = (abound[abit], bbound[bbit], cbound[cbit], dbound[dbit])
        # print(alock,block,clock,dlock)
        dcomp = y - alock * np.sin(block * x - clock)
        dexplb, dexpub = dbound

        # in-bound check
        if (dcomp >= dexplb and dcomp <= dexpub):
            # SET TRUE FLAG IF RETURNS TRUE
            #print('intersection detected')
            #print('dcomp: ', dcomp, ' dexplb: ', dexplb, ' dexpub: ', dexpub)
            #print('--------------')
            AUTHORITYINTERSECTFLAG = True
            return AUTHORITYINTERSECTFLAG
            break
        else:
            # SET TRUE FLAG IF RETURNS TRUE
            outofboundslis.append(dcomp)
            #print('dcomp: ', dcomp, ' dexplb: ', dexplb, ' dexpub: ', dexpub)
            #print('no intersection detected')

    # check if computed d value is below (-1), above (1), in-bound case is covered by conditional check above
    #print(outofboundslis)
    oobbitseq = np.asarray([1 if x > dexpub else -1 for x in outofboundslis])
    #print(oobbitseq)
    # if evaluates to true: function lies below or above the bounds of the hypercuboid we are looking at
    # else it means that somewhere the function intersects our hypercuboid
    # SET TRUE FLAG IF RETURNS TRUE
    allaboveorbelow = (np.all(oobbitseq == 1) or np.all(oobbitseq == -1))
    if (not allaboveorbelow):
        AUTHORITYINTERSECTFLAG = True
        return AUTHORITYINTERSECTFLAG
    #print('allaboveorbelow: ', allaboveorbelow)

    # if nothing is intercepting with the hypercuboid we now proceed to check for min/max within our param space bounds
    # start from low resolution and increase to high-res ...by how much/how far shall the resolution be increased?
    # best approach so far: get maximum of matrix ...if zero, proceed with higher resolution

    '''
    TODO: Implement adaptive resolution increase
    '''

    '''
    Go down to 80, 50 and 25% of boundary b and c resolution ...if at 25% all entries of the heatmap matrix are still
    at 0 or close to zero --> no intersection detected
    else w
    '''
    # compute resolution at 50% and 25%
    # 1) compute boundary ranges
    rangeBbound = np.abs(bbound[1] - bbound[0])
    rangeCbound = np.abs(cbound[1] - cbound[0])
    # now get the smallest range
    rangeMaxBounds = max(rangeBbound, rangeCbound)
    rangeIntersect = rangeMaxBounds
    fract = 1.0

    maxvalQ = 0.0
    minvalQ = 0.0
    minMaxInarea = False
    while (rangeIntersect > 0.1):
        rangeIntersect = rangeIntersect / fract
        pivotmatrixQuarter = generateHeatmap(bbound[0], bbound[1], cbound[0], cbound[1], rangeIntersect, x)
        maxvalQ = pivotmatrixQuarter.values.max()
        minvalQ = pivotmatrixQuarter.values.min()
        fract = fract + 1.0
        # SET TRUE FLAG IF RETURNS TRUE
        # If false: no min/max in area, else min/max in area
        minMaxInarea = (maxvalQ != 0) or (minvalQ != 0)
        if (minMaxInarea):
            break

    if (not minMaxInarea):
        AUTHORITYINTERSECTFLAG = False
        return AUTHORITYINTERSECTFLAG

    if (minMaxInarea):
        pivotmatrix = generateHeatmap(bbound[0], bbound[1], cbound[0], cbound[1], rangeIntersect,
                                      x)  # resolution curr 0.02
        #print('--------')
        # retrieve min and max value of heatmap
        maxval = pivotmatrix.values.max()
        #print('maxval: ', maxval)
        # retrieve parameter values (b,c) for maxval
        maxcoord = pivotmatrix.stack().idxmax()
        #print('maxcoord: ', maxcoord)

        # same for minval...
        minval = pivotmatrix.values.min()
        #print('minval: ', minval)
        mincoord = pivotmatrix.stack().idxmin()
        #print('mincoord: ', mincoord)

        #print('------------------minmax transition---------')
        # if minval or maxval is not equal to zero, check for intersection at that particular min/max point
        # at different amplitudes (alb, aub)
        # if the computed d value at that min/max point is NOT complying with the previous computed values at the vertices
        # --> intersection detected, ...else: no intersection

        if maxval > 0:
            bmax, cmax = maxcoord
            dmaxcomplb = y - abound[0] * np.sin(bmax * x - cmax)
            dmaxcompub = y - abound[1] * np.sin(bmax * x - cmax)
            # check if values are in-bound, if yes --> intersection detected, break!
            isinboundmax = (dmaxcomplb >= dbound[0] and dmaxcomplb <= dbound[1]) or (
                        dmaxcompub >= dbound[0] and dmaxcompub <= dbound[1])
            # SET TRUE FLAG IF RETURNS TRUE
            if (isinboundmax):
                AUTHORITYINTERSECTFLAG = True
                return AUTHORITYINTERSECTFLAG
            #print('max is inbound: ', isinboundmax)

            # if they are not in-bound, check if their encoding differs, if yes --> intersecting, else: no intersection
            dmaxcomplis = [dmaxcomplb, dmaxcompub]
            oobitseqmax = np.asarray([1 if x > dbound[1] else -1 for x in dmaxcomplis])
            #print(oobitseqmax)
            unionvertivesmax = np.concatenate((oobbitseq, oobitseqmax), axis=0)
            # SET TRUE FLAG IF RETURNS TRUE
            allmaxaboveorbelow = (np.all(unionvertivesmax == 1) or np.all(unionvertivesmax == -1))
            #print('>>> allmaxaboveorbelow: ', allmaxaboveorbelow)
            if (not allmaxaboveorbelow):
                AUTHORITYINTERSECTFLAG = True
                return AUTHORITYINTERSECTFLAG
            # if false --> intersection, else: no intersection
        else:
            print('No max in range at this resolution')

        #print('------------------minmax transition---------')

        if minval < 0:
            bmin, cmin = mincoord
            dmincomplb = y - abound[0] * np.sin(bmin * x - cmin)
            dmincompub = y - abound[1] * np.sin(bmin * x - cmin)
            # check if values are in-bound, if yes --> intersection detected, break!
            isinboundmin = (dmincomplb >= dbound[0] and dmincomplb <= dbound[1]) or (
                        dmincompub >= dbound[0] and dmincompub <= dbound[1])
            # SET TRUE FLAG IF RETURNS TRUE
            if (isinboundmin):
                AUTHORITYINTERSECTFLAG = True
                return AUTHORITYINTERSECTFLAG
            #print('min is inbound: ', isinboundmin)

            # if they are not in-bound, check if their encoding differs, if yes --> intersecting, else: no intersection
            dmincomplis = [dmincomplb, dmincompub]
            oobitseqmin = np.asarray([1 if x > dbound[1] else -1 for x in dmincomplis])
            #print(oobitseqmin)
            unionvertivesmin = np.concatenate((oobbitseq, oobitseqmin), axis=0)
            # SET TRUE FLAG IF RETURNS TRUE
            allminaboveorbelow = (np.all(unionvertivesmin == 1) or np.all(unionvertivesmin == -1))
            #print('>>> allminaboveorbelow: ', allminaboveorbelow)
            if (not allminaboveorbelow):
                AUTHORITYINTERSECTFLAG = True
                return AUTHORITYINTERSECTFLAG
            # if false --> intersection, else: no intersection
        else:
            print('No min in range at this resolution')

        return AUTHORITYINTERSECTFLAG


boundaryCheck(boundaryX, -5.0, 6.692440808350341)


'''
input: data dictionary, list of dpt ids, bounds 
returns a set of dpt ids which intersect within a given boundary
as well as a count of how many dpts intersect
'''
def numberDptsIntersecting(datadict, dptidlis, bounds):
    candidatedptids = []
    for dptid in dptidlis:
        dptX, dptY = datadict[dptid]
        if boundaryCheck(bounds, dptX, dptY):
            candidatedptids.append(dptid)
    return (len(candidatedptids), candidatedptids)





'''
function intersects space into two partitions, given a pre-defined axis in parameter space
input: number of candidates, list of dpt ids, bounds, axis split id, minpts parameter, resolution parameter
returns list of [number of candidates, list of dpt ids, new bounds, next axis split id, minpts parameter, curriter, maxiter parameter]
'''
def splitParamSpace(parentNumOfCand, parentDptidlis, bounds, axisSplitId, curriter, maxiter):
    if curriter < maxiter:
        leftsplit = []
        leftsummary = []

        rightsplit = []
        rightsummary = []

        i = 0
        for axis in bounds:
            if i != axisSplitId:
                rightsplit.append(axis)
                leftsplit.append(axis)
            else:
                axislb, axisub = axis
                axislb = float(axislb)
                axisub = float(axisub)
                axismid = float(axisub + axislb)/2.0
                leftsplit.append((axislb, axismid))
                rightsplit.append((axismid, axisub))
            i = i+1

        nextsplitaxis = (axisSplitId + 1) % 4
        curriter = curriter + 1

        leftsummary.append([parentNumOfCand, curriter , parentDptidlis, leftsplit, nextsplitaxis])
        rightsummary.append([parentNumOfCand, curriter, parentDptidlis, rightsplit, nextsplitaxis])

        return leftsummary[0], rightsummary[0]

    else:
        #if curriter == maxiter return true, use as flag for main function
        return True




'''
Final function splitting and determining number of functions intersecting
proceeds until priority queue is empty or maxiter is reached and thus at least one candidate is returned
'''

#list which captures the results from periccles; should be accessible by frequencytracker.py
RESGLOBAL = []


def periccles(datadict, initbounds, maxiter, minpts):
    #main priority queue --> first go by number of candidates, then as second attribute by curriter(=depth, =resolution)
    prioQ = []

    #generate initial dptidlis from datadict:
    dptidlis = datadict.keys()

    #'compute' initial number of candidates --> whole data set
    numOfCandidates = len(dptidlis)

    #initial axis split is assigned to the id 0
    axisSplitId = 0

    #current iteration
    curriter = 0

    #push initial bounds to the prioQ
    prioQ.append([numOfCandidates, curriter, dptidlis, initbounds, axisSplitId])
    #TODO: check this prioQ sorting!
    #originally: prioQ.sort(key=lambda element: (element[0], -element[1]), reverse=True), 09.10
    prioQ.sort(key=lambda element: (element[1], -element[0]), reverse=True)


    #----------TARGET LOOP BELOW--------

    while len(prioQ)>0:
        print('Next iter...')
        #initial check how many points and which points intersect with the full initial boundary
        topcandidate = prioQ.pop(0)
        print('topcandidate: ',topcandidate)
        initCandNum, initCurrIter, initDptlis, initSpaceBound, initAxisSplitId =topcandidate
        candNum, candIdLis = numberDptsIntersecting(datadict, initDptlis, initSpaceBound)
        print('INIT RES: ',candNum, candIdLis)


        #if minpts many candidate exist for the observed spatial section - proceed to split this
        if candNum >= minpts and initCurrIter < maxiter:
            print('Criteria matched - splitting parameter space.')

            splitres = splitParamSpace(candNum, candIdLis, initSpaceBound, initAxisSplitId, initCurrIter, maxiter)
            if splitres != True:
                prioQ.append(splitres[0])
                prioQ.append(splitres[1])
                prioQ.sort(key=lambda element: (element[0], -element[1]), reverse=True)



        elif candNum < minpts:
            print('insufficient dpts...no support for spatial partition')



        elif initCurrIter >= maxiter:
            print('-----------------')
            print('Maxiter reached - highest user defined resolution yields cluster: ')
            print('Number of dpts supporting: ', candNum,
                  'A-bound: ', initSpaceBound[0],
                  'B-bound: ', initSpaceBound[1],
                  'C-bound: ', initSpaceBound[2],
                  'D-bound: ', initSpaceBound[3],
                  'Iteration depth: ',initCurrIter)

            prioQ = []

        print('---------------Q-SIZE: ', len(prioQ), '---------------------')

    return (initSpaceBound[0], initSpaceBound[1], initSpaceBound[2], initSpaceBound[3], candNum, initCurrIter, candIdLis)






#areas which have below minpts shall not be put into priority queue anyway! --> put this into the next function, not in
#splitParamSpace!

#boundaryInArea = [(1.99,2.01),(3.99,4.01),(2.99,3.01),(4.99,5.01)]
#boundaryInArea = [(1,3),(3,5),(2,4),(4,6)]
#boundaryInArea = [(1,6),(2,6),(2,6),(3,6)]
boundaryInArea = [(0,8),(0,8),(0,8),(0,8)]

dptdict = {0: (-5.0, 6.692440808350341),
           1: (-3.0, 3.6994243196857663),
           2: (-1.0, 3.686026802562422),
           3: (1.0, 6.6829419696157935),
           4: (3.0, 5.824236970483513),
           5: (5.0, 3.077205016240886)}


'''
dpts [0,5] generated by (a,b,c,d) = 2,4,3,5
dpts [6,13] generated by (a,b,c,d) = 7,4,5,1
'''
dptdict2clus = {0: (-5.0, 6.692440808350341),
                1: (-3.0, 3.6994243196857663),
                2: (-1.0, 3.686026802562422),
                3: (1.0, 6.6829419696157935),
                4: (3.0, 5.824236970483513),
                5: (5.0, 3.077205016240886),
                6: (-5.0, 1.9264622506844113),
                7: (-3.571428571428571, -1.9572242292234563),
                8: (-2.142857142857143, -4.909351247797363),
                9: (-0.7142857142857144, -5.999965023367407),
                10: (0.7142857142857144, -4.8855097405667545),
                11: (2.1428571428571432, -1.9170515622745077),
                12: (3.571428571428571, 1.970311227708105),
                13: (5.0, 5.552014881099818)}

#res = numberDptsIntersecting(dptdict, [0,1,2,3,4], boundaryInArea)
#print(res)


#def splitParamSpace(parentNumOfCand, parentDptidlis, bounds, axisSplitId, curriter, maxiter):
#ressplit = splitParamSpace(5, [0,1,2,3,4], [(1, 3), (3, 5), (2, 4), (4, 6)], 0, 3, 4)
#for e in ressplit:
#    print(e)

'''
ori param: periccles(dptdict, boundaryInArea, 60, 5)
success:
Maxiter reached - highest user defined resolution yields cluster: 
Number of dpts supporting:  5 
A-bound:  (1.999481201171875, 1.999755859375) <- EXPECTED: 2
B-bound:  (5.42474365234375, 5.425018310546875) <- EXPECTED: 4, off by 1.42
C-bound:  (3.282684326171875, 3.282958984375)  <- EXPECTED: 3, off by 0.28
D-bound:  (4.99957275390625, 4.999847412109375) <- EXPECTED: 5
Iteration depth:  60

control computation yields for all data points:
dcomputed:  4.999722901683773  dexpected:  4.99957275390625
dcomputed:  4.9987450717909345  dexpected:  4.99957275390625
dcomputed:  5.000465870339995  dexpected:  4.99957275390625
dcomputed:  4.999488698802647  dexpected:  4.99957275390625
dcomputed:  4.999319871823036  dexpected:  4.99957275390625

aim for higher depths/resolution
'''
#single cluster test
#periccles(dptdict, boundaryInArea, 80, 6)

#double-cluster test
#periccles(dptdict2clus, boundaryInArea, 80, 6)

oktoberfest = [83, 39, 20, 12, 7, 4, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 4, 5, 5, 6, 7, 8, 10, 12, 15, 21, 26, 33, 41, 67, 86, 89, 43, 20, 14, 8, 4, 3, 3, 2, 2, 3, 2, 2, 2, 3, 3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4, 3, 3, 4, 4, 4, 5, 5, 4, 5, 6, 8, 9, 11, 14, 19, 25, 34, 47, 72, 100, 95, 52, 24, 15, 7, 4, 3, 3, 2, 2, 2, 2, 2, 2, 4, 3, 3, 2, 2, 3, 3, 2, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4, 3, 4, 4, 4, 4, 6, 7, 8, 9, 12, 16, 19, 24, 32, 66, 95, 84, 61, 25, 15, 9, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 3, 3, 9, 3, 3, 4, 3, 4, 3, 3, 4, 3, 4, 5, 5, 6, 7, 9, 9, 12, 15, 21, 24, 34, 65, 94, 84, 71, 28, 18, 10, 5, 3, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 3, 3, 3, 4, 3, 4, 3, 3, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 9, 11, 15, 20, 25, 33, 44, 75]

#generate pruned series where values from 0-5 are ommitted:
oktserieslist = []
i = 0
for e in oktoberfest:
    if e > 4:
        oktserieslist.append((i,e))
    i+=1

print(len(oktserieslist))

#generate dictionary from oktserieslist:
oktdict = {}
j = 0
for e in oktserieslist:
    oktdict[j] = e
    j+=1

#18, 10; use A beginning from 50 to max 100
#boundaryInArea = [(90,100),(0.1,1.0),(4,8),(0,0.25)]
#periccles(oktdict, boundaryInArea, 16, 15)