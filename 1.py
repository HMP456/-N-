
import numpy as np
from scipy import linalg

def matmul_mulelms(*matrixs):
    '''
    连乘函数。将输入的矩阵按照输入顺序进行连乘。

    Parameters
    ----------
    *matrixs : 矩阵
        按计算顺序输入参数.

    Raises
    ------
    ValueError
        当参数个数小于2时，不满足乘法的要求.

    Returns
    -------
    res : 矩阵
        返回连乘的结果.

    '''
    if len(matrixs)<2:
        raise ValueError('Please input more than one parameters.')
    res = matrixs[0]
    for i in range(1,len(matrixs)):
        res = np.matmul(res, matrixs[i])
    return res

# 3.3.4 施密特正交化
def One_Col_Matrix(array):
    '''
    确保为列矩阵

    Parameters
    ----------
    array : 矩阵，向量或数组

    Raises
    ------
    ValueError
        获得的参数不是1xn或mx1时，报错.

    Returns
    -------
    TYPE
        返回列矩阵.

    '''
    mat = np.mat(array)
    if mat.shape[0] == 1:
        return mat.T
    elif mat.shape[1] == 1:
        return mat
    else:
        raise ValueError('Please input 1 row array or 1 column array')

def Transfor_Unit_Vector(matrix):
    '''
    将每列都转换为标准列向量，即模等于1

    Parameters
    ----------
    matrix : 矩阵

    Returns
    -------
    unit_mat : 矩阵

        每列模都为1的矩阵.

    '''
    col_num = matrix.shape[1]
    # 初始化为零矩阵
    unit_mat = np.zeros((matrix.shape))
    for col in range(col_num):
        vector = matrix[:,col]
        unit_vector = vector / np.linalg.norm(vector)
        unit_mat[:,col] = unit_vector.T
    return unit_mat

def Gram_Schmidt_Orthogonality(matrix):
    '''
    施密特正交化方法

    Parameters
    ----------
    matrix : 矩阵

    Returns
    -------
    标准正交化矩阵。

    '''
    col_num = matrix.shape[1]
    # 第一列无需变换
    gram_schmidt_mat = One_Col_Matrix(matrix[:,0])
    for col in range(1,col_num):
        raw_vector = One_Col_Matrix(matrix[:,col])
        orthogonal_vector = One_Col_Matrix(matrix[:,col])
        if len(gram_schmidt_mat.shape)==1:
            # 当矩阵为列向量是，shape的返回值为“(row,)”，没有col的值
            gram_schmidt_mat_col_num = 1
        else:
            gram_schmidt_mat_col_num = gram_schmidt_mat.shape[1]
        for base_vector_col in range(gram_schmidt_mat_col_num):
            base_vector = gram_schmidt_mat[:,base_vector_col]
            prejective_vector = matmul_mulelms(base_vector, linalg.inv(np.matmul(base_vector.T,base_vector)), base_vector.T, raw_vector)
            orthogonal_vector = orthogonal_vector - prejective_vector
        gram_schmidt_mat = np.hstack((gram_schmidt_mat,orthogonal_vector))
    #print(gram_schmidt_mat)
    return Transfor_Unit_Vector(gram_schmidt_mat)
# ### 测试用例
# import numpy as np

# d*A*c=f

#输入矩阵
# A = np.array([[1,0,1],[0,1,1],[0,0,1]])
A = np.array([[1,-1,-1],[-1,3,1],[-1,1,1]])
#A = np.array([[-2,1,1],[0,2,0],[-4,1,3]])
# A = np.array([[-2,1,1,1] ,[0,2,0,1],[-4,1,3,1],[-2.1,2,5]])
# A = np.array([[1,-2,-4],[-2,4,-2],[-4,-2,1]])
# print('打印A:{}'.format(A))
print('打印原矩阵B:')
print(A)
a, b = np.linalg.eig(A)
print('打印特征值a:{}'.format(a))
print('打印特征向量b:{}'.format(b))
#单位正交化特征向量
c=Gram_Schmidt_Orthogonality(b)
# A = np.array([[1,1,1],[-1,0,-1],[0,-1,1]])
# c=np.around(c, decimals=2, out=None)
print('打印施密特正交矩阵c:')
print(c)
print('打印施密特正交矩阵的转置d:')
# #d为c的转置
d=(c.T)
print(d)
# print(c.T)
# # np.dot(A,A)
print('打印对角矩阵f:')
# #f是对角矩阵Q-1*A*Q
# #np.linalg.inv(b)为Q的逆
f=np.dot(np.dot(d,A),c)
# f=np.dot(np.dot(np.linalg.inv(b),A),b)
#若保留两位用消除下面一行注释
f=np.around(f, decimals=2, out=None)
# print('gggg')
print(f)
# print('xxx')
# k=np.dot(np.dot(c,f),d)
# k=np.around(k, decimals=2, out=None)
# print(k)
e=f
print('打印对角矩阵开1/n次方（此处为1/3）:')
# #对角矩阵开1/n方根
for i in range(len(A)):
    for j in range(len(A[0])):
        if i==j :
            e[i][j]=pow(f[i][j], 1/3)
        else:
            e[i][j]=0
print(e)
print('打印X:')
g=np.dot(np.dot(c,e),d)
# g=np.around(g, decimals=2, out=None)
print(g)
print('打印X的n次方:')
h=g
for i in range (2):
    h=np.dot(h,g)
h=np.around(h, decimals=2, out=None)
print(h)
print('打印原矩阵B:')
print(A)
print('比较后结果正确')