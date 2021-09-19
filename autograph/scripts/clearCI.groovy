
g.V().hasLabel('[avm]Design').has('[]Name',"__SOURCEDESIGN__").in('inside').in('inside').hasLabel('[]ComponentInstance').property('_tbd_','_DEL_').in('inside').property('_tbd_','_DEL_').in('inside').property('_tbd_','_DEL_').in('inside').property('_tbd_','_DEL_').in('inside').property('_tbd_','_DEL_').iterate()
g.V().has('_tbd_','_DEL_').drop()

g.tx().commit()

