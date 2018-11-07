#!/usr/bin/env python
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse


class Particle:
    def __init__(self, x_max, max_vel, dim):
        self.pos = np.random.uniform(low=-x_max, high=x_max, size=(dim, 1))
        self.vel = np.random.uniform(low=-max_vel, high=max_vel, size=(dim, 1))
        self.best_pos = self.pos
        self.fitness_value = 9999999

    def get_vel(self):
        return self.vel

    def set_vel(self, i_in, vel_in):
        self.vel[i_in] = vel_in

    def get_pos(self):
        return self.pos

    def set_pos(self, i_in, pos_in):
        self.pos[i_in] = pos_in

    def get_best_pos(self):
        return self.best_pos

    def set_best_pos(self, i_in, pos_in):
        self.best_pos[i_in] = pos_in

    def get_fitness_value(self):
        return self.fitness_value

    def set_fitness_value(self, fit_in):
        self.fitness_value = fit_in


class PSO:
    def __init__(self, size, iter_num, dim=5, x_max=5.12, max_vel=0.5,
                 best_fitness_value=float('Inf'), c1=2, c2=2, w=0.4):
        self.C1 = c1
        self.C2 = c2
        self.W = w                                      # 加权
        self.dim = dim                                  # 粒子的维度
        self.size = size                                # 粒子个数
        self.iter_num = iter_num                        # 迭代次数
        self.x_max = x_max                              # 距离区间
        self.max_vel = max_vel                          # 粒子最大速度
        self.best_fitness_value = best_fitness_value    # 最小值
        self.best_position = [0.0 for i in range(dim)]  # 种群最优位置
        self.fitness_val_list = []                      # 每次迭代最优适应值
        # 对种群进行初始化
        self.Particle_list = [Particle(self.x_max, self.max_vel, self.dim) for i in range(self.size)]

    def fit_fun(self, pos_in):
        sum_fit = 0
        for i in range(self.dim):
            cos_fit = np.cos(2 * math.pi * pos_in[i])
            sum_fit = sum_fit + pos_in[i] * pos_in[i] - cos_fit
        return sum_fit

    def get_best_fitness_value(self):
        return self.best_fitness_value

    def set_best_fitness_value(self, value_in):
        self.best_fitness_value = value_in

    def get_best_position(self):
        return self.best_position

    def set_best_position(self, i_in, pos_in):
        self.best_position[i_in] = pos_in

    #  更新速度
    def update_vel(self, part):
        for i in range(self.dim):
            vel_value = (self.W * part.get_vel()[i] + self.C1 * np.random.random() * (part.get_best_pos()[i] -
                         part.get_pos()[i]) + self.C2 * np.random.random() * (self.get_best_position()[i] -
                         part.get_pos()[i]))
            if vel_value > self.max_vel:
                vel_value = self.max_vel
            elif vel_value < -self.max_vel:
                vel_value = -self.max_vel
            part.set_vel(i, vel_value)

    # 更新位置
    def update_pos(self, part):
        for i in range(self.dim):
            pos_value = part.get_pos()[i] + part.get_vel()[i]
            if pos_value > self.x_max:
                pos_value = self.x_max
            elif pos_value < -self.x_max:
                pos_value = -self.x_max
            part.set_pos(i, pos_value)
        value = self.fit_fun(part.get_pos())
        if value < part.get_fitness_value():
            part.set_fitness_value(value)
            for j in range(self.dim):
                part.set_best_pos(j, part.get_pos()[j])
        if value < self.get_best_fitness_value():
            self.set_best_fitness_value(value)
            for j in range(self.dim):
                self.set_best_position(j, part.get_pos()[j])

    def update(self):
        for i in range(self.iter_num):
            for part in self.Particle_list:
                self.update_vel(part)                                         # 更新速度
                self.update_pos(part)                                         # 更新位置
            self.fitness_val_list.append(self.get_best_fitness_value())       # 每次迭代完把当前的最优适应度存到列表
        plt.plot(np.linspace(0, self.iter_num, self.iter_num), self.fitness_val_list, c="r")
        plt.xlabel("Iter Times")
        plt.ylabel("Best Adaption")
        plt.title("Search Minimum")
        plt.grid()
        plt.show()
        print("最佳适应值为:%.2f" % self.get_best_fitness_value())
        result_list = []
        for i in self.get_best_position():
            result_list.append(float(i))
        print("最佳x值为  :", result_list)


def pso(size_in, iter_num_in, max_vel_in, w_in, test_times):
    for i in range(test_times):
        pso_in = PSO(size=size_in, iter_num=iter_num_in, max_vel=max_vel_in, w=w_in)
        pso_in.update()


parser = argparse.ArgumentParser(prog="Search Minimum",
                                 usage="Use Particle Swarm Optimization(OPS) To Search Min")
parser.add_argument("-size", default=2000)
parser.add_argument("-iter", default=35)
parser.add_argument("-max_vel", default=0.1)
parser.add_argument("-w", default=0.2)
parser.add_argument("-test_times", default=1)
args = parser.parse_args()

pso(args.size, args.iter, args.max_vel, args.w, args.test_times)
