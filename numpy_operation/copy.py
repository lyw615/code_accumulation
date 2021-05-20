import numpy as np

def repeat():
    """
    在行列上产生重复的数
    Returns:

    """
    "产生5个3"
    rep=np.repeat(3,5)

    a2=[1,2,3]
    np.tile(a2,(5,1))#在行方向上重复一维数组  ===>[a2,a2,a2,a2,a2]
    np.tile(a2,(1,5))#在列方向上重复一维数组  ---》[ [ 1,2,3 ,1,2,3, ... ] ]

    np.tile(a2,(2,2))