g.V().hasLabel('[avm]Design').has('[]Name','__SOURCEDESIGN__').as('sd').in('inside').in('inside').hasLabel('[]ComponentInstance').has('[]Name','__SOURCECOMP__').in('inside').hasLabel('[]ConnectorInstance').has('[]CName','__SOURCECONN__').as('conn1').select('sd').in('inside').in('inside').hasLabel('[]ComponentInstance').has('[]Name','__DESTCOMP__').in('inside').hasLabel('[]ConnectorInstance').has('[]CName','__DESTCONN__').as('conn2').addE('connector_composition').from('conn1').to('conn2').addE('connector_composition').from('conn2').to('conn1')
