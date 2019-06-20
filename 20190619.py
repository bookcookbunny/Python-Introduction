#!/usr/bin/env python
# coding: utf-8

# In[1]: 


import numpy as np


# In[2]:


# 1차원 배열 만들기
#넘피의 어레이라는 함수에 리스트를 넣으면 배열로 변환해준다.
#따라서 1차원 배열을 만드는 방법은 다음과 같다.


# In[3]:


np.array([0,1,2,3,4,5,6,7,8,9])


# In[4]:


ar=np.array([0,1,2,3,4,5,6,7,8,9])


# In[5]:


type(ar)


# In[6]:


#리스트와 비슷해 보이지만 type 명령으로 자료형을 살펴보면 ndarray임을 알 수 있다,
#리스트 클래스 객체는 각각의 원소가 다른 자료형이 될 수 있다. 그러나 배열 객체.
#객체는 C언어의 배열처럼 연속적인 매모리 배치를 가지기 때문에 모든 원소가 같은 자료형이어야 한다.
#이러한 제약사항이 있는 대신 원소에 대한 접근과 반복은 실행이 빨라진다.


# In[7]:


ar.dtype #ar의 데이터타입을 보여줌. 32는 32바이트라는 뜻.


# In[8]:


ar2=[0.1,5,4,12,0.5]


# In[9]:


np.array(ar2)


# In[10]:


#다차원 배열 만들기
np.array([[1,2],[3,4]])


# In[11]:


arr1=np.arange(12)


# In[12]:


arr1.shape


# In[13]:


arr1.reshape(2,6) # reshape는 배열의 차원을 바꿔줌. 1차원 배열을 2차원 배열로 바꾸기


# In[14]:


arr1.reshape(3,4)


# In[15]:


arr2=arr1.reshape(4,3)


# In[16]:


arr2.shape


# In[17]:


type(arr2.shape)


# In[18]:


arr1_1=arr2.reshape(12,)


# In[19]:


arr1_1


# In[20]:


#범위의 시작과 끝을 지정하ineal space 고 데이터의 개수를 지정해서 배열을 생성하는 방법: linear space = linspace()


# In[21]:


np.linspace(1,10,10)


# In[22]:


np.linspace(3,9,100)


# In[23]:


np.pi


# In[24]:


np.linspace(0,np.pi,20)


# In[25]:


np.zeros((3,4))


# In[26]:


np.ones(50)


# In[27]:


np.ones((3,5))


# In[28]:


no_one=np.ones((3,5))


# In[29]:


no_one.dtype


# In[30]:


#단위행렬(Identity Matrix)


# In[31]:


help(np.eye)


# In[32]:


np.eye(1)


# In[33]:


np.eye(4,k=1) # 주대각선을 기 주대각선 상으로부터 1만큼 옮긴 단위행렬


# In[34]:


np.array(['1.5','0.62','2','3.141592']) # 교재 p222


# In[35]:


tf=np.array([True,False])


# In[36]:


tf.dtype


# In[37]:


str_a1=np.array(['1.567','0.123','5.123','9','8'])


# In[38]:


str_a1.dtype


# In[41]:


num1=str_a1.astype(float)


# In[42]:


num1.dtype


# In[43]:


help(np.random.rand)


# In[44]:


np.random.rand(2)


# In[45]:


np.random.rand(2,3)


# In[46]:


type(np.random.rand())


# In[47]:


type(np.random.rand(2))


# In[48]:


np.random.rand(2,3,4) #2면의 3행 4열


# In[49]:


np_r=np.random.rand(2,3,4)


# In[51]:


np_r.shape


# In[56]:


np.random.randint(10,size=(3,4))# 3행 4열의 사이즈로 10 미만의 정수 난수 발생 


# In[57]:


np.arange(10,90,10)


# In[59]:


np.arange(10,90,10).reshape(2,4)


# In[60]:


ex1=np.arange(10,90,10).reshape(2,4)


# In[61]:


print(ex1)


# In[62]:


#벡터화 연산
data=[0,1,2,3,4,5,6,7,8,9]


# In[63]:


[2*i for i in data]


# In[66]:


answer = []
for i in data:
    answer.append(2*i)


# In[68]:


answer


# arr1=np.array([10,20,30,40])

# In[69]:


arr1=np.array([10,20,30,40])
arr3=np.array([1,2,3])


# In[70]:


arr1 - arr2


# In[72]:


arr1**2


# In[73]:


arr > 20


# In[74]:


arr3=np.arange(5)


# In[75]:


arr3.sum()


# In[76]:


arr3.mean()


# In[77]:


arr3.std()


# In[78]:


arr3.var()


# In[79]:


arr3.cumsum() #누적합 (Cumulative Sum)


# In[81]:


arr3.cumprod()


# In[82]:


np.array([0,1,2,3]).reshape(2,2)


# In[83]:


aaa=np.array([0,1,2,3]).reshape(2,2)


# In[84]:


aaa.shape


# In[85]:


A=np.array([0,1,2,3]).reshape(2,2)


# In[86]:


A


# In[87]:


B=np.array([3,2,0,1]).reshape(2,2)


# In[88]:


A.dot(B)


# In[92]:


np.transpose(A)


# In[91]:


a1=np.array([0,10,20,30,40,50])


# In[94]:


a1[0]


# In[95]:


a1[5]=70


# In[97]:


a1[[1,3,4]] # 리스트 안 리스트로 인덱싱을 해주어야 함(유의)


# In[104]:


np.arange(10,100,10).reshape(3,3)


# In[105]:


a2=np.arange(10,100,10).reshape(3,3)


# In[106]:


a2[1,1]


# In[107]:


a2[2,2]


# In[108]:


a2[1:3,1:3]


# In[109]:


x=np.arange(1,10001)


# In[110]:


x


# In[111]:


y=np.arange(10001,20001)


# In[113]:


x.shape


# In[115]:


z=np.zeros_like(x)


# In[116]:


get_ipython().run_cell_magic('time', '', 'print(z)')


# In[118]:


get_ipython().run_cell_magic('time', '', 'for i in range(10000):\n    z[i]=x[i]+y[i]')


# In[119]:


get_ipython().run_cell_magic('time', '', 'z = x + y')


# In[120]:


z


# In[121]:


a=np.array([1,2,3,4])
b=np.array([4,2,2,4])


# In[122]:


a==b


# In[123]:


#만약 배열의 각 원소들을 일일히 비교하는  것이 아니라 배열의 모든 원소가 다 같은지 알고 싶다면 all 명령을 사용하기


# In[132]:


a=np.array([1,2,3,4])
b=np.array([4,2,2,4])
c=np.array([1,2,3,4])


# In[133]:


np.all(a==b)


# In[134]:


np.all(a==c)


# In[138]:


#지수/로그 함수 등의 함수를 지원한다


# In[136]:


a=np.arange(5)


# In[137]:


np.exp(a)


# In[139]:


np.log(a+1)


# In[140]:


a=np.zeros((100,100))


# In[141]:


a.dtype


# In[142]:


help(np.zeros)
a=np.zeros((100,100), dtype=np.int)


# In[143]:


a.dtype


# In[144]:


np.any(a!=0)


# In[145]:


np.all(a==a)


# In[146]:


a=np.array([1,2,3,2])
b=np.array([2,2,3,2])
c=np.array([6,4,4,5])


# In[148]:


((a<=b))&((b<=c)).all()


# In[151]:


x=np.array([[1,1],[2,2]])


# In[154]:


x.shape


# In[155]:


x2=x.sum()


# In[156]:


x2


# In[157]:


x


# In[159]:


x.sum(axis=0) # 열의 합


# In[160]:


x.sum(axis=1) # 행의 합


# In[161]:


#실수로 이루어진 5*6 형태의 데이터 행렬을 만들고 이 데이터에 대해 다음과 같은 값을 구한다.

#1. 전체의 최대값
#2. 각 행의 합
#3. 각 열의 평균


# In[172]:


a=np.arange(1,31, dtype='float64')


# In[173]:


a=a.reshape(5,6)


# In[174]:


a.max()


# In[175]:


a


# In[176]:


a.sum(axis=1)


# In[177]:


a.mean(axis=0)


# In[183]:


[0 for col in range(10)]


# In[184]:


matrix=np.array([[0 for col in range(10)] for row in range(10)])


# In[185]:


matrix


# In[188]:


p_q=np.array([[1,0,0,0,0,0],[0,1,0,0,0,0,],[0,0,1,0,0,0],[1,0,0,-1,0,-2],[0,1,0,0,-1,-2]])


# In[189]:


matrix[:5,:6]=p_q


# In[190]:


matrix


# In[191]:


a1=np.ones((2,3))


# In[193]:


a2=np.zeros((2,2))


# In[194]:


a1


# In[195]:


a2


# In[196]:


np.hstack([a1,a2])


# In[197]:


b1 = np.ones((2,3))


# In[198]:


b2 = np.ones((3,3))


# In[199]:


b1


# In[200]:


b2


# In[201]:


np.stack([b2,b2])


# In[206]:


# 다음과 같은 행렬이 있다
m=np.array([[0,1,2,3,4],
            [5,6,7,8,9],
            [10,11,12,13,14]])
# 이 행렬에서 값 7을 인덱싱한다.
# 이 행렬에서 값 14를 인덱싱한다.
# 이 행렬에서 배열 [6,7]을 슬라이싱한다.
# 이 행렬에서 배열 [7,12]를 슬라이싱한다.
# 이 행렬에서 배열 [[3,4],[8,9]]을 슬라이싱한다.


# In[209]:


m[1,2]


# In[210]:


m[2,4]


# In[212]:


m[[1:1]:[1:2]]


# In[ ]:




