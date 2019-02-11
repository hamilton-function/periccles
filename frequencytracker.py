import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import periccles


dptdict = {0: (-5.0, 6.692440808350341),
           1: (-3.0, 3.6994243196857663),
           2: (-1.0, 3.686026802562422),
           3: (3.0, 5.824236970483513),
           4: (5.0, 3.077205016240886),
          5: (1.0, 6.6829419696157935)}


'''
dreal not to be confused with d: 
dreal = sin(B*x-C)
'''
def generateDptDLookup(datadict,b,c):
    valdict = {}
    for key in datadict:
        x, y = datadict[key]
        dreal = np.sin(b * x - c)
        truncatedD = float("%.4f" % dreal)
        #datapnt, dvalue
        valdict[key] = (datadict[key], truncatedD)
    return valdict




'''
create a lookup table of following structure:
dpt id: [f0, f1, ..., fn],
where each f is defined as (y,x,theta,n)
use as bcrange the original bc-bounds
'''
def generateDptLinFunctionLookup(dlookup, bcrange):
    # lookup table for functions
    funcdict = {k: [] for k in dlookup.keys()}
    bclb, bcub = bcrange

    for key in dlookup:
        x, y = (dlookup[key])[0]
        theta = (dlookup[key])[1]
        # y,x,theta,n
        #05.10 casted to integer, since on rare occasions it may have floats as lb/ub
        for n in range(int(bclb), int(bcub)):
            paramvect = (y, x, theta, n)
            (funcdict[key]).append(paramvect)

    return funcdict




'''
computes for a given bbound, cbound, x, theta and specific n
if the computed cbounds are located within the given cbounds

    return ([Bi*x+2*np.pi*n-np.arcsin(theta)       for Bi in Brange],
            [Bi*x+2*np.pi*n+np.arcsin(theta)+np.pi for Bi in Brange])
'''
def isInSinArea(bbounds, cbounds, x, theta, n):
    blb, bub = bbounds
    c_expected_lb, c_expected_ub = cbounds
    c_computed_blb1 = blb * x + 2 * np.pi * n - np.arcsin(theta)
    c_computed_blb2 = blb * x + 2 * np.pi * n + np.arcsin(theta) + np.pi
    c_computed_bub1 = bub * x + 2 * np.pi * n - np.arcsin(theta)
    c_computed_bub2 = bub * x + 2 * np.pi * n + np.arcsin(theta) + np.pi

    cond1 = (c_computed_blb1 >= c_expected_lb) and (c_computed_blb1 <= c_expected_ub)
    cond2 = (c_computed_blb2 >= c_expected_lb) and (c_computed_blb2 <= c_expected_ub)
    cond3 = (c_computed_bub1 >= c_expected_lb) and (c_computed_bub1 <= c_expected_ub)
    cond4 = (c_computed_bub2 >= c_expected_lb) and (c_computed_bub2 <= c_expected_ub)

    # conditions capturing the cases where on the lb the c value is below lb and ub the c value is above ub --> intersect
    cond5 = (c_computed_blb1 < c_expected_lb) and (c_computed_bub1 > c_expected_ub)
    cond6 = (c_computed_blb2 < c_expected_lb) and (c_computed_bub2 > c_expected_ub)

    # or when lb is above ub  and ub is below the lb
    cond7 = (c_computed_blb1 > c_expected_ub) and (c_computed_bub1 < c_expected_lb)
    cond8 = (c_computed_blb2 > c_expected_ub) and (c_computed_bub2 < c_expected_lb)

    return cond1 or cond2 or cond3 or cond4 or cond5 or cond6 or cond7 or cond8


'''
function computes for all n-factor tuples in lookup table if any of them intersects
within the expected/given cbounds
'''
def isInAreaForAllN(paramtuplelis, bbounds, cbounds):
    isInArea = False
    for e in paramtuplelis:
        y, x, theta, n = e
        # print(x,y,theta,n)
        intersects = isInSinArea(bbounds, cbounds, x, theta, n)
        # print(intersects)
        # print('------')
        if intersects:
            isInArea = True
            return isInArea
            break
    return isInArea


# 2.2831852436065674,6.424777865409851
# particularpoint = funcdict[0]
# for point in funcdict:
# print('--------BEGIN POINT---------')
# particularpoint = funcdict[point]
# res = isInAreaForAllN(particularpoint, (2.27, 2.29), (6.41, 6.43))
#print(res)
# print('--------END POINT---------')
# isInAreaForAllN(particularpoint, (3.99,4.01), (2.99,3.01))





'''
 0. create a grid of initial 2x2 resolution
 1. compute per cell the number of intersecting - unique - functions
 '''


# %matplotlib
def generateHeatmap(blb, bub, clb, cub, stepsize, functionlookup):

    brange = np.arange(blb, bub, stepsize)
    crange = np.arange(clb, cub, stepsize)
    prodlis = list(itertools.product(brange, crange))

    complis = []
    for e in prodlis:
        b, c = e
        bbounds = (b, b + stepsize)
        cbounds = (c, c + stepsize)

        intersectcounter = 0
        for key in functionlookup:
            pointlis = functionlookup[key]
            doesIntersect = isInAreaForAllN(pointlis, bbounds, cbounds)
            if doesIntersect:
                intersectcounter += 1

        val = intersectcounter
        complis.append((b, c, val))

    df = pd.DataFrame(complis)
    dfpivoted = df.pivot(index=0, columns=1, values=2)



    phase = list(dfpivoted.columns)
    freq = list(dfpivoted.index)
    gridcells = list(itertools.product(freq, phase))


    #retrieve all indives where we have a maxvalue in our heatmap
    #get max value from heatmap:
    hmapmax = dfpivoted.values.max()
    #iterate over all indices fetching the number of functions intersecting
    phasefreqlis = []
    for e in gridcells:
        row, col = e
        if dfpivoted.loc[row,col] == hmapmax:
            phasefreqlis.append((row,col))

    #sorted results list - by increasing frequency (second item of b,c-tuple)
    phasefreqsorted = sorted(phasefreqlis, key=lambda x: x[1])

    #plotting heatmap
    #ax = sns.heatmap(dfpivoted, cmap='coolwarm', annot=True)
    #plt.show()

    '''
    print('all phase-freq candidates:')
    for e in phasefreqsorted:
        print(e)
    print('end all phase-freq candidates')
    '''

    #return b,c-tuple with lowest frequency (second item of b,c,-tuple)
    return phasefreqsorted[0]





#generateHeatmap(1, 8, 1, 6, 0.9, funcdict)
#generateHeatmap(1, 8, 1, 8, 0.7, funcdict)
#generateHeatmap(1, 8, 1, 8, 0.5, funcdict)

#dptdict, b, c,
#valdict1 = generateDptDLookup(dptdict,4,3)
#max(bboundrange, cboundrange)
#funcdict = generateDptLinFunctionLookup(valdict1, (-15,15))
#blb bub, clb cub, resolution, funcdict
#generateHeatmap(1, 8, 1, 8, 0.1, funcdict)

#TODO: execute periccles.py with provided data from luketrendwalker.py here and determine the resulting parameters

sometrenddata = [83, 39, 20, 12, 7, 4, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 4, 5, 5, 6, 7, 8, 10, 12, 15, 21, 26, 33, 41, 67, 86, 89, 43, 20, 14, 8, 4, 3, 3, 2, 2, 3, 2, 2, 2, 3, 3, 3, 3, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4, 3, 3, 4, 4, 4, 5, 5, 4, 5, 6, 8, 9, 11, 14, 19, 25, 34, 47, 72, 100, 95, 52, 24, 15, 7, 4, 3, 3, 2, 2, 2, 2, 2, 2, 4, 3, 3, 2, 2, 3, 3, 2, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4, 3, 4, 4, 4, 4, 6, 7, 8, 9, 12, 16, 19, 24, 32, 66, 95, 84, 61, 25, 15, 9, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 3, 3, 9, 3, 3, 4, 3, 4, 3, 3, 4, 3, 4, 5, 5, 6, 7, 9, 9, 12, 15, 21, 24, 34, 65, 94, 84, 71, 28, 18, 10, 5, 3, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 3, 3, 3, 4, 3, 4, 3, 3, 3, 3, 3, 4, 4, 4, 5, 6, 6, 7, 9, 11, 15, 20, 25, 33, 44, 75]

somebounds = [(90,100),(0.1,1.0),(4,8),(0,0.25)]

import time
import copy
import csv
#15, 12
def bigbangpericcles(trendres, boundaryInArea, maxiter, minpts):
    #generate pruned series where values from 0-5 are ommitted:

    '''
    trendserieslist = []
    i = 0
    for e in trendres:
        if e > 4:
            trendserieslist.append((i,e))
        i+=1
    '''

    #generate dictionary from oktserieslist:
    trenddict = {}
    j = 0
    for e in trendres:
        trenddict[j] = e
        j+=1

    referencedict = copy.deepcopy(trenddict)

    ainitbound, binitbound, cinitbound, dinitbound = boundaryInArea
    #periccles.periccles(trenddict, boundaryInArea, 15, 12)
    #pericclesresults = periccles.RESGLOBAL
    pericclesresults = []
    detectedcounter = 0

    while len(trenddict.keys()) >= minpts:
        print('Number of remaining data point candidates: ',len(trenddict.keys()))
        time.sleep(7)
        res = periccles.periccles(trenddict, boundaryInArea, maxiter, minpts)
        pericclesresults.append(res)
        print('Detected periodic clusters: ',res)

        #writing clusters to csv...
        a, b, c, d, candnum, iter, candIdlis = res
        with open('detectedCluster' + str(detectedcounter) + '.csv', 'w', newline='') as csvfile:
            cluswriter = csv.writer(csvfile, delimiter=',')
            cluswriter.writerow(['Number of dpts supporting: ', candnum,
                                 'A-bound: ', a,
                                 'B-bound: ', b,
                                 'C-bound: ', c,
                                 'D-bound: ', d,
                                 'Iteration depth: ', iter])
            for e in candIdlis:
                cluswriter.writerow(trenddict[e])
            csvfile.close()
        detectedcounter += 1



        #get now the dpt candidate ids
        candidscurrclust = res[6]
        for e in candidscurrclust:
            del trenddict[e]



    #sample lowest frequency for each entry of RESGLOBAL
    finalresults = []
    for e in pericclesresults:
        a,b,c,d,candnum,iter,candIdlis = e
        alb, aub = a
        blb, bub = b
        clb, cub = c
        dlb, dub = d

        bavg = float((float(blb) + float(bub))/2.0)
        cavg = float((float(clb) + float(cub))/2.0)

        #dptdict, b, c,
        valdict1 = generateDptDLookup(trenddict,bavg,cavg)
        #max(bboundrange, cboundrange)
        funcdict = generateDptLinFunctionLookup(valdict1, cinitbound)
        #interval width
        ivsigma = 0.01

        print('Line which takes some computation time')
        #blb bub, clb cub, resolution, funcdict
        optfreq = generateHeatmap(binitbound[0], binitbound[1], cinitbound[0], cinitbound[1], ivsigma, funcdict) #param before funcdict, controls resolution.

        finalresults.append((a, (optfreq[0]-ivsigma, optfreq[0]+ivsigma), (optfreq[1]-ivsigma, optfreq[1]+ivsigma), d, candnum, iter))


    return finalresults




syndata = [(1.0, 1.1996668332936562),
 (2.9333333333333336, 2.257586048036937),
 (4.866666666666667, 2.904180683181032),
 (6.800000000000001, 2.9279659923048964),
 (8.733333333333334, 2.3211624025584014),
 (10.666666666666668, 1.2822400161197345),
 (12.600000000000001, 0.1510040610328348),
 (14.533333333333335, -0.7025468018711489),
 (16.46666666666667, -0.9992376800283709),
 (18.400000000000002, -0.642028493422494),
 (20.333333333333336, 0.25224667033952797),
 (22.26666666666667, 1.3910930302010884),
 (24.200000000000003, 2.402022754471197),
 (26.133333333333336, 2.954386306691646),
 (28.06666666666667, 2.8675195436353014),
 (3.0, 5.358678045449762),
 (3.8666666666666667, 5.484357550059133),
 (4.733333333333333, 5.4819914980762245),
 (5.6, 5.352205382885088),
 (6.466666666666667, 5.129309674830555),
 (7.333333333333334, 4.872229448986584),
 (8.2, 4.648926855634597),
 (9.066666666666666, 4.518434534713342),
 (9.933333333333334, 4.515249650273096),
 (10.8, 4.6402141635897465),
 (11.666666666666668, 4.860292250900537),
 (12.533333333333333, 5.11730370797404),
 (13.4, 5.343304564353819),
 (14.266666666666667, 5.478548862860209),
 (15.133333333333333, 5.487283198852218),
 (12.0, 3.5555675046944075),
 (13.0, 3.6315202836294054),
 (14.0, 3.0173427978173493),
 (15.0, 2.384212968019831),
 (16.0, 2.4240199835219043),
 (17.0, 3.093262428993962),
 (18.0, 3.660586968610873),
 (19.0, 3.5060169446583824),
 (20.0, 2.7984676783344544),
 (21.0, 2.3111544964428496),
 (22.0, 2.576617024315603),
 (23.0, 3.304755752261025),
 (24.0, 3.6998550300998687),
 (25.0, 3.330147302165935),
 (26.0, 2.5996520414233055),
 (27.0, 2.306660135371991),
 (28.0, 2.771355411726697),
 (29.0, 3.485915267776571),
 (30.0, 3.669463149883152),
 (31.0, 3.1214165086721244),
 (6.987049020820093, 7.3865143535500515),
 (27.35068722190563, 2.469511299187408),
 (1.530907705941189, 0.17305510267642532),
 (0.30022111465910606, 6.666788090118955),
 (9.397631978397712, 1.6274324589837263),
 (9.088311705220853, 6.035552428489278),
 (4.206059061260092, 1.1332967075333782),
 (10.847688176808457, 1.008648789266882),
 (7.228177783183232, 0.35027467306468996),
 (29.160863486785164, 6.435391652376428),
 (8.101261194707178, 0.6462115673070024),
 (6.285948349544524, 7.406165291897378),
 (1.074450636454627, 4.964124442974264),
 (25.00943706885315, 5.583669480900407),
 (18.593574711668687, 4.7960675716263825)]



synbounds = [(0.3,2.2),(0.1,1.3),(0,6.1),(1.5,5.1)]

#optres = bigbangpericcles(syndata, synbounds, 28, 12) #25,14
optres = bigbangpericcles(syndata, synbounds, 30, 13)

print('-----------OPTRES------------')
for e in optres:
    print(e)



'''
c0 = generate_periodicities(2,    0.3,  0.2,  1,(1,30),15)
c1 = generate_periodicities(0.5,  0.6,  1,    5,(3,16),15)
c2 = generate_periodicities(0.7,  1.1,  6,    3,(12,32),20)
'''


'''
import csv

trendserieslist = []
with open('realworld.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in reader:
        trendserieslist.append((i,int(row[1])))
        i += 1




real = [(0,100),(0,100),(0,100),(20,60)]
#previous setting: 28,40
optres = bigbangpericcles(trendserieslist, real, 31, 20)

print('-----------OPTRES------------')
for e in optres:
    print(e)
'''




'''
#real world freq-phase filter

import csv

trendserieslist = []
with open('realworld.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in reader:
        trendserieslist.append((i,int(row[1])))
        i += 1


trenddict = {}
j = 0
for e in trendserieslist:
    trenddict[j] = e
    j += 1


ainitbound, binitbound, cinitbound, dinitbound = [(72.265625, 72.65625),(0.0, 0.390625),(87.5, 87.890625),(27.1875, 27.5)]

alb, aub = (72.265625, 72.65625)
blb, bub = (0.0, 0.390625)
clb, cub = (87.5, 87.890625)
dlb, dub = (27.1875, 27.5)

bavg = float((float(blb) + float(bub))/2.0)
cavg = float((float(clb) + float(cub))/2.0)

#dptdict, b, c,
valdict1 = generateDptDLookup(trenddict,bavg,cavg)
#max(bboundrange, cboundrange)
funcdict = generateDptLinFunctionLookup(valdict1, cinitbound)
#interval width
ivsigma = 0.01


print('Line which takes some computation time')
#blb bub, clb cub, resolution, funcdict
optfreq = generateHeatmap(binitbound[0], binitbound[1], cinitbound[0], cinitbound[1], ivsigma, funcdict) #param before funcdict, controls resolution.

print(((alb, aub), (optfreq[0]-ivsigma, optfreq[0]+ivsigma), (optfreq[1]-ivsigma, optfreq[1]+ivsigma), ((27.1875, 27.5))))
'''