g.V().hasLabel('[avm]Design').has('[]Name','__SOURCEDESIGN__').as('root').sideEffect(select('root').in('inside').hasLabel('[]DomainFeature').drop()).select('root').in('inside').hasLabel('[]RootContainer').in('inside').has('[]Name','__ORIENTNAME__').as('orient').addV('[]DomainFeature').as('newv').property('[http://www.w3.org/2001/XMLSchema-instance]type','[cad]AssemblyRoot').property('_partition','baseDesign').select('newv').addE('inside').to('root').select('newv').addE('assembly_root_component_instance').to('orient')