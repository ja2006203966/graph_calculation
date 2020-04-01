import tensorflow as tf


##===============================================mapping coordinate function===================================================================
def index_0(bins = (28,28)):
    with tf.name_scope('index_0'):
        X = tf.range(0, bins[0], 1)
        Y = tf.range(0, bins[1], 1)
        X =tf.expand_dims(X, axis  = 1)
        Y = tf.expand_dims(Y, axis  = 0)
        X = tf.tile(X, [1,bins[1]])
        X =tf.expand_dims(X, axis  = -1)
        Y = tf.tile(Y, [bins[0],1])
        Y = tf.expand_dims(Y, axis  = -1)
        index = tf.concat([X,Y], axis=-1)
        return index


##-------------------------------------------creat coordination (bins,bins ) bins must be the same

def map_coor( xlm = tf.constant([0,0.4]), x0 = tf.constant([-14,0]), d=2, bins = [28,28], dtype = tf.float32):
    with tf.name_scope('map_coor'):
        b0 = bins[0]
        xlm = tf.cast(xlm, dtype )
        x0 = tf.cast(x0, dtype )
        bins = tf.constant(bins)
        bins = tf.cast(bins, dtype )
        delta = (xlm - x0)/bins
        coor0 = tf.ragged.range(x0,xlm,deltas = delta )
        # ----------------- custom 
        coor0x = tf.cast(coor0[0], dtype)
        coor0y = tf.cast(coor0[1], dtype)
        coor0 = tf.convert_to_tensor([coor0x,coor0y], dtype = dtype)
        return coor0
##=========================== mapping data x[i] (3, ) with coor(2,bins) ---> matrix index (tf.int8,tf.int8)
def within_ind(x,c, dtype = tf.float32):
    with tf.name_scope('within_ind'):
        x = tf.cast(x, dtype)
        xb = tf.math.greater_equal(x[0],c[0,:])
        xb = tf.cast(xb, tf.int8 )
        xb = tf.math.reduce_sum(xb) -1
        yb = tf.math.greater_equal(x[1],c[1,:])
        yb = tf.cast(yb, tf.int8 )
        yb = tf.math.reduce_sum(yb) -1
        return xb, yb
##=========================== mapping data (None, 2 + ?)  ---> matrix index (None,2)

def map_index( x ,coor0 = map_coor() , dtype = tf.int8):
    with tf.name_scope('map_index'):
        index = tf.convert_to_tensor([within_ind(x[0],coor0)], dtype = dtype)
        def cond(i, d, coor0, index, x):
            return  i< d
        def body(i, d , coor0, index, x):
            index = tf.constant([[]], dtype =dtype) if i == tf.constant(0) else index
            index = tf.cast(index, dtype = dtype)
            index = tf.concat([ index, tf.convert_to_tensor([within_ind(x[i],coor0)], dtype = dtype) ]  , axis = 0) 
            index = tf.cast(index, dtype = dtype)
            i = i+1
            return i, d, coor0, index, x
        i = tf.constant(1)
        td = tf.shape(x)[0]
        _, _, _, ind, _ = tf.while_loop(cond, body, [i, td , coor0, index, x], shape_invariants=[i.get_shape(), td.get_shape(), coor0.get_shape(),
                                                                                                 tf.TensorShape([None,2]), x.get_shape()]  )
        return ind
#==================================================
def map_matrix_2d(x, ind, dtype = tf.float32, bins = (28,28) ):
    with tf.name_scope('map_matrix_2d'):
        index0 = index_0(bins = bins)
        index0 = tf.cast(index0, dtype = tf.int32)
        matrix = tf.cast(index0[:, :,0]*0, dtype = dtype)
        def cond(i, d, ind, x, matrix):
            return  i< d
        def body(i, d, ind, x, matrix):
            a = index0
            col = ind[i]
            a = (a[:,:,0] <=col[0])&(a[:,:,0] >=col[0])&(a[:,:,1] <=col[1])&(a[:,:,1] >=col[1])
            a = tf.cast(a, tf.int32)
            matrix = matrix + tf.cast(a, dtype = dtype)*x[i][2]
            i = i+1
            return i, d, ind, x, matrix 
        i = tf.constant(0, dtype = tf.int32)
        d = tf.shape(x)[0]
        _, _, _, _, m = tf.while_loop(cond, body, [i, d , ind, x, matrix], shape_invariants=[i.get_shape(), d.get_shape(), ind.get_shape(),
                                                                                             x.get_shape(), matrix.get_shape()]  )
        return m


##=======================================================================
def get_coornum(x,axis = [0,1], bins = (28,28), dtype = tf.float32):
    a = tf.cast(x, dtype = dtype)
    coor0 = map_coor(bins = bins)
    ind = map_index(a, dtype = tf.int32, coor0 = coor0 )
    a = map_matrix_2d(a, ind, bins = bins)
    a =tf.math.logical_not(tf.math.equal(a,0))
    a = tf.cast(a, dtype = tf.int8)
    a = tf.transpose(a, axis )
    a = tf.reduce_sum(a, axis = -1)
    return a
    
