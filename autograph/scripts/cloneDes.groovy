
g.V().hasLabel("[avm]Design").has("[]Name","__SOURCENAME__").as('tmp').property('_duptag_','_SRC_').select('tmp').repeat(__.in('inside').as('tmp').property('_duptag_','_SRC_').select('tmp')).times(20)


g.V().has('_duptag_','_SRC_').as('x').select('x').addV(select('x').label()).as('y').property('_duptag_','_DUP_').addE('clone').from('x').to('y').iterate()

g.V().has('_duptag_','_SRC_').as('x').out('clone').where(__.has('_duptag_','_DUP_')).as('y').select('x').properties().as('xps').select('y').property(select('xps').key(),select('xps').value()).select('y').property('_duptag_','_DUP_').iterate()

g.V().has('_duptag_','_SRC_').as('orig').out('clone').where(__.has('_duptag_','_DUP_')).as('cloned').select('orig').inE().where(label().is(neq('clone'))).as('elabel').select('elabel').outV().out('clone').where(__.has('_duptag_','_DUP_')).as('inTarg').select('cloned').addE(select('elabel').label()).from('inTarg').to('cloned').iterate()


g.V().has('_duptag_','_SRC_').as('orig').out('clone').where(__.has('_duptag_','_DUP_')).as('cloned').select('orig').out('component_id').as('linkDest').addE('component_id').from('cloned').to('linkDest').iterate()

g.V().has('_duptag_','_SRC_').as('orig').out('clone').where(__.has('_duptag_','_DUP_')).as('cloned').select('orig').out('id_in_component_model').as('linkDest').addE('id_in_component_model').from('cloned').to('linkDest').iterate()


g.V().has('[]Name',"__SOURCENAME__").has('_duptag_','_DUP_').property('[]Name','__DESTNAME__')

g.V().as('a').out('root').as('b').where('a',neq('b')).inE('root').drop()
g.V().as('a').in('root').as('b').where('a',neq('b')).outE('root').drop()

g.V().has('_duptag_','_SRC_').property('_duptag_','_cpysrc_')
g.V().has('_duptag_','_DUP_').property('_duptag_','_cpydst_')

g.E().hasLabel("clone").drop()
g.tx().commit()
