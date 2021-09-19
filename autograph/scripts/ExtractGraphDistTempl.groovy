println("Extracting a design")
println args[0]

// tag::extraction[]
:remote connect tinkerpop.server conf/remote-objects.yaml

println("Design name is ......")
println "__DESIGNNAME__"

:> g.V().hasLabel("[avm]Design").has("[]Name","__DESIGNNAME__").as("root").inE('root').subgraph('dsg').select('root').repeat(__.inE('inside').subgraph('dsg').outV().as('sv').outE().subgraph('dsg').select('sv')).times(20).cap('dsg').next()
graph = result[0].getObject()

println("Query Complete")

graph.traversal().io('__DEST__/ADM/namedGraph.graphml').with(IO.writer, IO.graphml).write().iterate()


println("Write Complete to __DEST__/ADM/namedGraph.graphml")
//println args[0]

println("wait 3...")
sleep(3)



println ("Fix component names -- Disabled")
// copy []CName to []Name
 :> g.V().has('[]CName').property('[]Name',properties('[]CName').value())
 

println ("Writing out property-to-componentInstance links")
// build the mapping of design parameter to component parameter/name
:> g.V().hasLabel('[avm]Design').has('[]Name','__DESIGNNAME__').in('inside').in('inside').as('parts').hasLabel('[]Property').as('DESIGN_PARAM').select('DESIGN_PARAM').in('inside').in('value_source').out('inside').out('inside').as('COMPONENT_PARAM').out('inside').as('COMPONENT_NAME').select('DESIGN_PARAM','COMPONENT_PARAM','COMPONENT_NAME').by('[]Name')
paramMap = result.object
pf = new File("__DEST__/parameterMap.json")
jj = groovy.json.JsonOutput.toJson(paramMap)
pf.write(jj)
println ("Written to __DEST__/parameterMap.json")

println ("Writing out Connections: componentInstance-to-componentInstance") 
:> g.V().hasLabel('[avm]Design').has('[]Name','__DESIGNNAME__').in('inside').in('inside').hasLabel('[]ComponentInstance').as('FROM_COMP').select('FROM_COMP').in('inside').hasLabel('[]ConnectorInstance').as('FROM_CONN').out('connector_composition').as('TO_CONN').out('inside').as('TO_COMP').select('FROM_COMP','FROM_CONN','TO_COMP','TO_CONN').by('[]Name')
connectionMap = result.object
cf = new File("__DEST__/connectionMap.json")
jj = groovy.json.JsonOutput.toJson(connectionMap)
cf.write(jj)
println ("Written to __DEST__/connectionMap.json")


println("Generating componentInstance to Component mapping")
:> g.V().hasLabel('[avm]Design').has('[]Name','__DESIGNNAME__').in('inside').in('inside').hasLabel('[]ComponentInstance').as('FROM_COMP').select('FROM_COMP').out('component_id').out('component_instance').as('LIB_COMPONENT').select('FROM_COMP','LIB_COMPONENT').by('[]Name')
componentMap = result.object
cmapfile = new File("__DEST__/componentMap.json")
compMapJSON = groovy.json.JsonOutput.toJson(componentMap)
cmapfile.write(compMapJSON)
println ("Written to __DEST__/componentMap.json")


//:remote close
// end::extraction[]
