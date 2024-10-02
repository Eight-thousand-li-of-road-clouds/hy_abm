# A杭州某高校
import concurrent.futures
import os
import random
import pandas as pd
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from datetime import datetime, timedelta
import numpy as np
from joblib import Parallel, delayed
#时间分析
import cProfile


class StudentAgent(Agent):
    def __init__(self, unique_id, model, class_id=None):
        super().__init__(unique_id, model)
        self.status = "S"  # 状态：S, E, I, Q，R
        self.roommates = []
        self.class_id = class_id  # 班级ID
        self.incubation_time = 0  # 潜伏期计时器
        self.infection_time = 0  # 感染时间计时器
        self.prev_status = "S"  # 初始化前一个状态为易感
        self.recovery_time = 0  # 恢复延时计时器

    def step(self):
        # 更新前一个状态
        self.prev_status = self.status

        # 同班接触、宿舍接触和随机接触
        if self.status == "I":
            if self.model.time_step in [6, 7, 12, 13, 18, 19]:  # 当时间步为假期时
                self.random_contact2()  # 只进行随机接触
            else:
                self.class_contact()
                self.dorm_contact()
                self.random_contact()

        # 状态更新逻辑
        if self.status == "E":
            self.incubation_time += 1
            if self.incubation_time >= 1.5 and random.random() < 0.23:
                self.status = "I"
            elif self.incubation_time >= 3.5 and random.random() < 0.25:
                self.status = "S"
        elif self.status == "I":
            self.infection_time += 1
            if self.infection_time >= 3.5 and random.random() < 0.95:
                self.status = "R"
            elif self.infection_time >= 6 and random.random() < 0.05:
                self.status = "E"  # 回到E状态

        # 恢复状态逻辑
        elif self.status == "R":
            self.recovery_time += 1
            if self.recovery_time >= 15.625 and random.random() < 0.062:
                self.status = "E"
        return

    def class_contact(self):
        neighbors = self.model.schedule.agents  # 获取所有代理
        class_neighbors = [other for other in neighbors if other.class_id == self.class_id]  # 仅同班学生
        for _ in range(600):
            for other in class_neighbors:
                if other != self and self.status == "I" and other.status == "S":
                    if random.random() < 0.0006:  # 一个感染者与一个易感者每次接触并导致该易感者感染的概率
                        other.status = "E"

    def dorm_contact(self):
        for _ in range(300):
            for roommate in self.roommates:
                if roommate != self and self.status == "I" and roommate.status == "S":
                    if random.random() < 0.0006:
                        roommate.status = "E"

    def random_contact(self):
        contacts = random.choices(self.model.schedule.agents, k=600)
        for other in contacts:
            if other != self and self.status == "I" and other.status == "S":
                if random.random() < 0.0006:
                    other.status = "E"

    def random_contact2(self):
        contacts = random.choices(self.model.schedule.agents, k=800)
        for other in contacts:
            if other != self and self.status == "I" and other.status == "S":
                if random.random() < 0.0006:
                    other.status = "E"


class SchoolModel(Model):

    def __init__(self, N, initial_infected):
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(1065, 1065, True)
        #1700亩 = 1133333.3333平方米
        # sqrt(1133333) = 1065
        self.data_collector = DataCollector(
            agent_reporters={"Status": "status"}
        )
        self.time_step = 1  # 设置开始日期
        self.start_date = datetime(2008, 6, 1)  # 初始化开始日期
        self.dates = []
        self.new_infections = []
        self.current_infections = []
        self.assign_dorms()
        self.initial_infected = initial_infected
        self.num_classes = 290

    def run_single_model(self):
        self.schedule = RandomActivation(self)
        self.dates = []
        self.new_infections = []

        for i in range(self.num_agents):
            class_id = i % self.num_classes
            selected_indices = np.random.choice(range(self.num_agents), size=self.num_agents*0.88, replace=False)
            has_dorm = i in selected_indices

            a = StudentAgent(i, self, class_id)
            self.schedule.add(a)
            self.grid.place_agent(a, (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)))
            if i < self.initial_infected:
                a.status = "I"  # 设置初始感染者

        self.running = True

    def assign_dorms(self):
        dorms = []

        individuals_with_dorms = [agent for agent in self.schedule.agents if agent.has_dorm]

        # 分配四人间宿舍
        four_person_dorm_mates = []
        for i in range(0, len(individuals_with_dorms), 4):
            dorm_mates = individuals_with_dorms[i:i + 4]
            if len(dorm_mates) == 4:
                for mate in dorm_mates:
                    mate.roommates = dorm_mates
                four_person_dorm_mates.append(dorm_mates)

        dorms.append(four_person_dorm_mates)
        return dorms

    def step(self):
        new_infections_today = 0
        self.schedule.step()

        for agent in self.schedule.agents:
            if agent.status == "I" and agent.prev_status == "E":
                new_infections_today += 1

        self.new_infections.append(new_infections_today)

        # 更新日期
        self.time_step += 1
        self.dates.append(self.start_date + timedelta(days=len(self.dates)))
        self.data_collector.collect(self)

    def get_results(self):
        return pd.DataFrame({
            "Day": range(len(self.new_infections)),
            "New Infections": self.new_infections,
        })

    def run_single_model(self):
        # 重置模型
        self.schedule = RandomActivation(self)
        self.dates = []
        self.new_infections = []
        # 初始化学生代理
        for i in range(self.num_agents):
            class_id = i % 290
            a = StudentAgent(i, self, class_id)
            self.schedule.add(a)
            self.grid.place_agent(a, (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)))
            if i == 0:  # 只有第一个代理是感染者
                a.status = "I"

        # 模拟40天
        for _ in range(24):
            self.step()

        return self.new_infections, self.dates

    def run_model(self, **kwargs):
        num_runs = kwargs.get('num_runs', 1)  # 获取 num_runs 参数，如果不存在则默认为 1
        all_results = Parallel(n_jobs=-1)(delayed(self.run_single_model)() for _ in range(num_runs))
        all_results_list = [result[0] for result in all_results]
        dates = all_results[0][1]

        # 创建结果DataFrame
        results_df = pd.DataFrame(all_results_list).T
        results_df.columns = [f'Run {i + 1}' for i in range(num_runs)]
        results_df['Date'] = [i for i in range(1, len(dates) + 1)]
        results_df = results_df.set_index('Date')

        print(results_df)
        results_df.to_csv(r"B_0006_20_10_20.csv", index=False)


if __name__ == "__main__":
    model = SchoolModel(1000, initial_infected=10)
    # model.run_model(num_runs=20)  # 指定循环次数
    cProfile.run('model.run_model(num_runs=20)','heyue_A_hzc_joblib.prof')

# #分析prof
import pstats
p=pstats.Stats('heyue_A_hzc.prof')
p.strip_dirs()
p.sort_stats('tottime').print_stats(10)