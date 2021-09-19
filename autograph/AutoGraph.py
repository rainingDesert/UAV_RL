from gremlin_python import statics
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.strategies import *
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.traversal import T
from gremlin_python.process.traversal import Order
from gremlin_python.process.traversal import Cardinality
from gremlin_python.process.traversal import Column
from gremlin_python.process.traversal import Direction
from gremlin_python.process.traversal import Operator
from gremlin_python.process.traversal import P
from gremlin_python.process.traversal import Pop
from gremlin_python.process.traversal import Scope
from gremlin_python.process.traversal import Barrier
from gremlin_python.process.traversal import Bindings
from gremlin_python.process.traversal import WithOptions

import time
import sys
import json 

def swapParams(swapList,s):
    swaps = len(swapList)/2
    i = 0
    newstr = s
    while i < swaps:
        try:
            newstr = newstr.replace(swapList[i*2].rstrip(),swapList[i*2+1].rstrip())
        except:
            print("len swaplist is odd: ",len(swapList))
        i = i + 1
    ll = ''.join([str(elem) for elem in newstr])
    #print('Swapped ['+s+'] to ['+ll+']')
    #print(type(ll))
    return ll
 
def execQuery(gremClient,q):
    print('SUBMIT:'+q)
    start = time.time()
    result_set = gremClient.submit(q)
    future_results = result_set.all() 
    results = future_results.result()
    end = time.time()
    print('ELAPSED TIME: ',end - start)
    print(results)
 
def saveToJson(lineNum,queryName,result):
    if(queryName[0:4]=='info'):
        fname = queryName+str(lineNum)+'.json'
        with open(fname, "w") as outfile: 
            json.dump(result, outfile)
      

opstart = time.time()

saveResults = True

#hostIP = '129.59.203.79'
hostIP = '192.168.1.124'
if len(sys.argv) > 1:
    hostIP = sys.argv[1]
fileName = 'graphOps.csv'
if len(sys.argv) > 2:
    fileName = sys.argv[2]
    
hostStr = 'ws://'+hostIP+':8182/gremlin'
#g = traversal().withRemote(DriverRemoteConnection('ws://localhost:8182/gremlin','g'))
print('Attempting to Connect to '+hostStr)
from gremlin_python.driver import client
client = client.Client(hostStr, 'g')
print('\nConnected to '+hostStr)
print('Executing file:',fileName)
print('---------------------------------------')
conntest=False
if(conntest==True):
    result_set = client.submit('g.V().hasLabel("[avm]Design").values("[]Name").toList()')
    future_results = result_set.all() 
    results = future_results.result()
    print(results)

opFile = open(fileName,'r')
opList = opFile.readlines()

opList.pop(0)
counter = 1
for op in opList:
    print('--------------------OP -------------------------------')
    print(op)

    l = op.split(',')
    qfname = 'scripts/'+l[0]+'.groovy'
    qname = l[0]
    l.pop(0)
    qswap = l
    qfile = open(qfname,'r')       
    qlines = qfile.readlines()
    for qline in qlines:
        query = swapParams(qswap,qline)
        #execQuery(client,query)
        if(len(query) > 2):
            print('SUBMIT: ',counter,' of ',len(opList),' to ',qfname)
            #print(['+query+']')
            start = time.time()
            result_set = client.submit(query)
            future_results = result_set.all() 
            results = future_results.result()
            print('RESULTS: ',results)
            end = time.time()
            saveToJson(counter,qname,results)
            print('ELAPSED TIME: ',end - start)
            #print(results)
    counter = counter + 1 
    
opend = time.time()
client.close()
print('FULL ELAPSED TIME: ',opend - opstart)        