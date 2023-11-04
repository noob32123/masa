#我们用改进后的遗传算法求解函数1+sin(-1+x^2/2)+(x-1)^2/2在（-1~4上极大值）
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
(POP_SIZE,DNA_SIZE)=(48,16)   #DNA数量以及每个DNA长度
(min_range,max_range)=(-1,4)     #函数范围
(Pc1,Pc2,Pm1,Pm2)=(0.9,0.5,0.1,0.01)  ##表示遗传变异概率
get_val = np.array(2 ** np.arange(DNA_SIZE - 1, -1, -1))  ##由二进制求出对应值

def fun(x):
    return np.sin(-1+x**2/2)+1+(x-1)**2/2

def getfit(child):
    val=child.dot(get_val)
    return  fun(min_range + (max_range - min_range) * val / (2 ** DNA_SIZE))**2  ##我们的适应性函数为函数值的平方，便于扩大差异


def choose(pop,fit):
    index=np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace='True', p=fit / np.sum(fit))
    return pop[index]

def mutate(dna,mutate_chance=0.05):
    if np.random.rand()<mutate_chance:
        index=np.random.randint(DNA_SIZE)
        dna[index]=1^dna[index]

def cross_mutate(pop,cross_chance=0.5):
    new_pop=[]
    for child in pop:
        if np.random.rand()<cross_chance:
            father=pop[np.random.randint(POP_SIZE)]
            cross_point=np.random.randint(DNA_SIZE)
            child[cross_point]=father[cross_point]
        mutate(child)
        new_pop.append(child)
    return new_pop

def adapt_cross_mutate(pop,fit,heat):  ##自适应遗传交叉变异
    new_pop=[]
    (index,passon)=(0,0)  ##index用于标记child的fit,fit[index]就是child的适应度
    for child in pop:
        child_model=child  ##复制一个出来备用
        thisfit=fit[index]
        if thisfit<=np.average(fit):
            chance=Pc1
            flag=0      ##flag用于简化判断条件，在cross_mutate里判断过的就不需要在mutate里面继续判断
            passon=0    ##passon用于把带有fit的部分传给mutate函数，免去后续计算
        else:
            passon=(thisfit-np.average(fit)) / (max(fit) - np.average(fit))
            chance = Pc1 - (Pc1 - Pc2) * passon
            flag=1
        if np.random.rand()<chance:
            father=pop[np.random.randint(POP_SIZE)]
            cross_point=np.random.randint(DNA_SIZE)
            child[cross_point]=father[cross_point]
        if getfit(child)>getfit(child_model): pass  ##模拟退火判断是否接收本次变异
        else:
            if np.random.rand()>np.exp(-(getfit(child_model)-getfit(child))/heat):child=child_model  ##此时拒绝变异
        adapt_mutate(child,flag,passon)
        new_pop.append(child)
        index+=1
    return new_pop

def adapt_mutate(dna,flag,passon):
    if flag==0:
        chance=Pm1
    else:
        chance=Pm1-(Pm1-Pm2)*passon
    if np.random.rand()<chance:
        index=np.random.randint(DNA_SIZE)
        dna[index]=1^dna[index]

if __name__=='__main__':
    (heat,drop_rate)=(100,0.97)   ##模拟退火初始温度以及衰减率
    pop=np.random.randint(0,2,size=(POP_SIZE,DNA_SIZE))
    val=pop.dot(get_val)
    fit=fun(min_range+(max_range-min_range)*val/(2**DNA_SIZE))
    histiry_best=0
    temp_best=[]
    x=np.arange(1,201,1)

    pop1=pop
    val1=val
    fit1=fit
    histiry_best1=0
    temp_best1=[]

    for echo in range(200):
        pop=np.array(cross_mutate(choose(pop,fit*fit)))
        val = pop.dot(get_val)
        fit = fun(min_range + (max_range - min_range) * val / (2 ** DNA_SIZE))
        temp_max=np.max(fit)
        print('The ',echo+1,' times training get the best result ',temp_max)
        temp_best.append(temp_max)
        histiry_best=max(histiry_best,temp_max)

        pop1=np.array(adapt_cross_mutate(choose(pop1,fit1*fit1),fit1,heat))
        val1 = pop1.dot(get_val)
        fit1 = fun(min_range + (max_range - min_range) * val1 / (2 ** DNA_SIZE))
        temp_max1=np.max(fit1)
        print('The ',echo+1,' adapting times training get the best result ',temp_max1)
        temp_best1.append(temp_max1)
        histiry_best1=max(histiry_best1,temp_max1)
        heat=heat*drop_rate


    print('The normal histroy best= ',histiry_best)
    print('The adaption histroy best= ',histiry_best1)
    plt.xlabel('训练总次数')
    plt.ylabel('当前最佳值')
    plt.plot(x,temp_best,color='r',linewidth=1.2,label='正常遗传算法')
    plt.plot(x,temp_best1,color='b',linewidth=1.2,label='自适应模拟退火遗传算法')
    plt.legend()
    plt.show()

##问题：fit区分度太小了
##自适应遗传算法：
##模拟退火改进