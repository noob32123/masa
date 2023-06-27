#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import pandas as pd
import numpy as np
import math
import logging
import argparse
import sys

LOG = logging.getLogger(__name__)

__version__ = "1.0.0"
__author__ = ("Boya Xu",)   #输入作者信息
__email__ = "834786312@qq.com"
__all__ = []

def add_help_args(parser):   #帮助函数
    parser.add_argument('--file', type=str, default=False, help="输入文件")
    return parser

#初始参数
population_size = 50 #种群大小
Pc = 0.7 #交叉概率
Pm = 0.2 #变异概率
max_gen = 500 #最大进化代数
unimproved_gen = 50 #适应度连续无提升
initial_T = 100 #模拟退火初始温度
end_T = 10#凝结温度
drop_T = 3#每次下降温度


# 读取FASTA文件
def read_fasta_file(filename):
    sequences = {}
    with open(filename, 'r') as file:
        sequence = ''
        seq_name = ''
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if sequence != '':
                    sequences[seq_name] = sequence
                    sequence = ''
                seq_name = line[1:]
            else:
                sequence += line
        sequences[seq_name] = sequence
    return sequences


#产生初始化种群
def generate_initial_population(sequences, population_size):
    population = {}
    max_length = max(len(seq) for seq in sequences.values())
    # 遍历每个序列，并在末尾随机插入"-"，使其长度与最长序列相等
    for i in range(int(population_size)):
        population["unit_"+str(i)] = sequences.copy()
        for key, value in population["unit_" + str(i)].items():
            if len(value) < max_length:
                num_insertions = max_length - len(value)
                for _ in range(num_insertions):
                    insertion_index = random.randint(0, len(value))
                    value = value[:insertion_index] + '-' + value[insertion_index:]
            population["unit_" + str(i)][key] = value
    return population


def calculate_fitness(sequences): #计算适应度
    gap_open_penalty = -10  #对于gap采取仿射打分
    gap_extend_penalty = -0.5 #配对+1错配为0
    alignment_length = len(sequences[0])
    fitness = 0
    for i in range(alignment_length):
        column = [sequence[i] for sequence in sequences]

        pair_score = 0
        gap_penalty1 = 0
        gap_penalty2 = 0

        for j in range(len(column) - 1):
            for k in range(j + 1, len(column)):
                char1 = column[j]
                char2 = column[k]

                if char1 == '-' and char2 == '-':
                    continue
                elif char1 == '-':
                    if gap_penalty1 == 0:
                        gap_penalty1 = gap_open_penalty
                    else:
                        gap_penalty1 += gap_extend_penalty

                    pair_score += gap_penalty1
                elif char2 == '-':
                    if gap_penalty2 == 0:
                        gap_penalty2 = gap_open_penalty
                    else:
                        gap_penalty2 += gap_extend_penalty

                    pair_score += gap_penalty2
                elif char1 == char2:
                    pair_score += 1

                    gap_penalty1 = 0
                    gap_penalty2 = 0
                else:
                    gap_penalty1 = 0
                    gap_penalty2 = 0

        fitness += pair_score
    return fitness


def selection(population): #前15%最佳适应度选择，后85%轮盘赌选择
    # 计算适应度得分
    fitness_scores = [calculate_fitness(list(population[i].values())) for i in population]
    # 根据适应度得分进行排序
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
    sorted_fitness_scores = sorted(fitness_scores, reverse=True)

    # 计算前15%的个体数量
    top_percentage = int(population_size * 0.15)
    parent1 = sorted_population[0]

    # 计算剩余个体的适应度总和
    remaining_fitness_scores = sorted_fitness_scores[top_percentage:]
    total_fitness = sum(remaining_fitness_scores)

    # 计算剩余个体的选择概率
    probabilities = [score / total_fitness for score in remaining_fitness_scores]
    # 使用轮盘赌选择一个个体作为第二个父本
    parent2 = random.choices(sorted_population[top_percentage:], probabilities)[0]
    P1 = population[parent1]
    P2 = population[parent2]
    return P1, P2

# #多行横向交叉
# def breed(parent1,parent2):
#     matrix1 = pd.DataFrame({key: list(value) for key, value in parent1.items()})
#     matrix2 = pd.DataFrame({key: list(value) for key, value in parent2.items()})
#     matrix1 = matrix1.T
#     matrix2 = matrix2.T
#     ran_num = random.randint(1, len(parent1)-1)
#     matrix1[ran_num:], matrix2[ran_num:] = matrix2[ran_num:], matrix1[ran_num:]
#     return matrix1, matrix2
# #多行横向交叉
# def breed(parent1, parent2): #这里面采用矩阵操作，时间复杂度更低。前面使用字典操作，时间不太够了先不改了。
#     matrix1 = np.array([list(value) for value in parent1.values()]).T
#     matrix2 = np.array([list(value) for value in parent2.values()]).T
#
#     ran_num = random.randint(1, len(parent1)-1)
#     matrix1[ran_num:], matrix2[ran_num:] = matrix2[ran_num:], matrix1[ran_num:]
#     return matrix1, matrix2
def breed(parent1,parent2):#采用字典多行横向交叉
    child1 = {}
    child2 = {}

    keys = list(parent1.keys())

    for key in keys:
        seq1 = parent1[key]
        seq2 = parent2[key]

        crossover_point = random.randint(1, len(seq1) - 1)

        child1[key] = seq1[:crossover_point] + seq2[crossover_point:]
        child2[key] = seq2[:crossover_point] + seq1[crossover_point:]

    return child1, child2


def mutate_dictionary(dictionary): #进行变异，随机选一个含有“-”的序列，先随机删除一个“-”，再在该序列删除位置外的另一个位置随机插入一个"-".
    mutated_dict = {}
    # 随机选择一个含有连字符的序列进行变异
    keys_with_dash = [key for key, value in dictionary.items() if '-' in value]
    selected_key = random.choice(keys_with_dash)
    selected_value = dictionary[selected_key]

    # 随机删除一个连字符
    deletion_index = random.choice([i for i, char in enumerate(selected_value) if char == "-"])
    mutated_value = selected_value[:deletion_index] + selected_value[deletion_index + 1:]

    # 随机插入一个连字符
    insertion_index = random.choice([i for i in range(len(mutated_value)+1) if i != deletion_index])
    mutated_value = mutated_value[:insertion_index] + "-" + mutated_value[insertion_index:]

    # 复制字典并更新变异后的值
    mutated_dict = dict(dictionary)
    mutated_dict[selected_key] = mutated_value

    return mutated_dict


def run_GA(f):#遗传算法主函数
    un_fitness = 0 #连续适应度不增代数
    population_num = 0 #种群迭代次数
    population_dict = {} #初始100个种群
    child_size = 0 #下一代种群大小
    fitness_sum = 0 #适应度
    all_fitness = {}
    population_select = 100 #初始种群数大小
    file = read_fasta_file(f)#读取fasta格式文件
    for i in range(population_select): #从100个种群中筛选出最佳的种群，作为初始种群。
        initial_population = generate_initial_population(file, population_size) #生成初始种群
        for k in initial_population: #计算初始种群适应度
            sequence = list(initial_population[k].values())
            fitness = calculate_fitness(sequence)
            fitness_sum += fitness
        all_fitness[i] = fitness_sum
        population_dict[i] = initial_population
    max_fitness = max(all_fitness, key=all_fitness.get)
    initial_population = population_dict[max_fitness] #最佳初始种群
    while population_num<max_gen or un_fitness>unimproved_gen:#产生500代种群或者连续50代适应度无提升则结束
        n = 0
        P1, P2 = selection(initial_population)  # 从上一代种群选择一对作为父本
        new_population = []
        new_population_dict = {} #初始化新种群
        new_population.append(P1)
        new_population.append(P2)
        child_size = 0
        #形成下一代种群
        while child_size < population_size:  # 下一代种群达到初始种群时结束
            if random.random() < Pc:  # 判断是否进行交叉,并选择适应度最高加入种群
                child1, child2 = breed(P1, P2)
                dicts = [child1, child2, P1, P2]
                fitness_tmp = [calculate_fitness(list(child1.values())), calculate_fitness(list(child2.values())),
                               calculate_fitness(list(P1.values())), calculate_fitness(list(P2.values()))]
                arr = np.array(fitness_tmp)
                sorted_indices = np.argsort(arr)
                max_two_indices = sorted_indices[-2:]
                real_child1 = dicts[max_two_indices[0]]
                real_child2 = dicts[max_two_indices[1]]
            else:
                real_child1, real_child2 = P1, P2
            new_population.append(real_child1)
            new_population.append(real_child2)
            if random.random() < Pm:  # 判断是否进行变异
                mutate_child1 = mutate_dictionary(real_child1)
                mutate_child2 = mutate_dictionary(real_child2)
                new_population.append(mutate_child1)
                new_population.append(mutate_child2)
            # 删除种群中的重复个体
            unique_set = set(tuple(d.items()) for d in new_population)
            unique_list = [dict(t) for t in unique_set]
            new_population = unique_list
            child_size = len(new_population)

        for i in new_population:
            new_population_dict['unit_'+str(n)] = i
            n = n+1
        new_fitness_sum = 0
        fitness_sum = 0
        for i in new_population_dict:#计算新的种群适应度得分
            sequence = list(initial_population[k].values())
            fitness = calculate_fitness(sequence)
            new_fitness_sum += fitness
        for k in initial_population: #计算初始种群适应度
            sequence = list(initial_population[k].values())
            fitness = calculate_fitness(sequence)
            fitness_sum += fitness
        if fitness_sum > new_fitness_sum: #通过比较如果适应度连续50代都没有提升则结束循环,并选择最优的进入下一代
            un_fitness += 1
            new_population_dict = initial_population
        else:
            un_fitness = 0
            initial_population = new_population_dict#新种群作为初始种群进行下一轮迭代
        population_num += 1

    return initial_population


def run_SA(pop): #模拟遗传退火算法
    while initial_T <= end_T:
        for _ in range(100):
            keys = list(pop.keys())
            chosen_key = random.choice(keys)
            chosen_dict = pop[chosen_key]  # 随机选择一个个体进行变异
            mutate_dict = mutate_dictionary(chosen_dict)
            new_fit = calculate_fitness(list(mutate_dict.values()))
            fit = calculate_fitness(list(chosen_dict.values()))
            delta_e = new_fit - fit
            if delta_e > 0:  # 接受新解
                pop[chosen_key] = mutate_dict
            elif random.random() < math.exp(delta_e / initial_T):#根据概率接受新解
                pop[chosen_key] = mutate_dict
        initial_T = initial_T - drop_T

    return pop


def run_GA_SA(f):
    GA_pop = run_GA(f)
    SA_pop = run_SA(GA_pop)
    fitness_scores = [calculate_fitness(list(SA_pop[i].values())) for i in SA_pop]
    # 根据适应度得分进行排序
    sorted_population = [x for _, x in sorted(zip(fitness_scores, SA_pop), reverse=True)]
    sorted_fitness_scores = sorted(fitness_scores, reverse=True)
    best_msa = SA_pop[sorted_population[0]]
    best_scores = sorted_fitness_scores[0]
    print("MSA比对最佳的分:"+str(best_scores))
    out = open('MSA_result.fasta', w)
    for i in best_msa:
        out.write('>'+i+'\n'+best_msa[i]+'\n')
        

def main():   #主函数，执行函数
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=''' 
name:
    基于遗传算法和模拟退火算法的多序列比对
attention:
    MSA.py --file 
version: %s
contact: %s <%s>\ 

''' % (__version__, ' '.join(__author__), __email__))
    args = add_help_args(parser).parse_args()
    run_GA_SA(args.file)

if __name__ == "__main__":           #固定格式，使 import 到其他的 python 脚本中被调用（模块重用）执行
    main()

