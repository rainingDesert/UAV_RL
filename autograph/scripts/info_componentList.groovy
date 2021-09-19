g.V().hasLabel('[avm]Design').has('[]Name','__SOURCEDESIGN__').in('inside').in('inside').hasLabel('[]ComponentInstance').as('compInst').out('component_id').out('component_instance').as('comp').select('compInst','comp').by('[]Name')

