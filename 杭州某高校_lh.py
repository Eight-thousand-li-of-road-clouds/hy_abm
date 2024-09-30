#A杭州某高校
import random
import pandas as pd
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from datetime import datetime, timedelta
import numpy as np
import cupy as cp
#
# #lh调试
import cProfile

class StudentAgent(Agent):
    """
    一个学生代理，代表学校里的每个学生
    """

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
        self.class_contact()
        self.dorm_contact()
        self.random_contact()

        # 状态更新逻辑
        if self.status == "E":
            self.incubation_time += 1
            if self.incubation_time >= 1.5 and random.random() < 0.23:
                self.status = "I"
                self.isolation_delay = 0  # 设置延迟时间，假设为2轮
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
        neighbors = self.model.schedule.agents
        class_neighbors = [other for other in neighbors if other.class_id == self.class_id]
        #修改思路：class_neighbors不能为array，无法直接搜索status，得在list中遍历后形成array
        infected = np.array([agent for agent in class_neighbors if agent.status == "I"])
        susceptible = np.array([agent for agent in class_neighbors if agent.status == "S"])

        for _ in range(900):
            if len(infected) > 0 and len(susceptible) > 0:
                contact_indices = random.choices(range(len(susceptible)), k=len(infected))
                for idx in contact_indices:
                    if random.random() < 0.00003:
                        susceptible[idx].status = "E"

    def dorm_contact(self):
        roommates_array = self.roommates
        infected = np.array([agent for agent in roommates_array if agent.status == "I"])
        susceptible = np.array([agent for agent in roommates_array if agent.status == "S"])

        for _ in range(900):
            if len(infected) > 0 and len(susceptible) > 0:
                contact_indices = random.choices(range(len(susceptible)), k=len(infected))
                for idx in contact_indices:
                    if random.random() < 0.00003:
                        susceptible[idx].status = "E"

    def random_contact(self):
        agents_array = self.model.schedule.agents
        infected = np.array([agent for agent in agents_array if agent.status == "I"])
        susceptible = np.array([agent for agent in agents_array if agent.status == "S"])

        contacts = random.choices(agents_array, k=600)
        for other in contacts:
            if other in susceptible and self.status == "I":
                if random.random() < 0.00003:
                    other.status = "E"

class SchoolModel(Model):
    """学校模型"""

    def __init__(self, N, initial_infected):
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(110, 110, True)
        self.data_collector = DataCollector(
            agent_reporters={"Status": "status"}
        )
        self.start_date = datetime.now()  # 设置开始日期
        self.dates = []  # 日期列表
        self.new_infections = []  # 每日新增感染人数
        self.current_infections = []  # 每日当前感染人数
        self.assign_dorms()

        num_classes = 290
        for i in range(self.num_agents):
            class_id = i % num_classes  # 根据代理索引分配班级

            a = StudentAgent(i, self, class_id)
            self.schedule.add(a)
            self.grid.place_agent(a, (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)))
            if i < initial_infected:
                a.status = "I"  # 设置初始感染者

        self.running = True

    def assign_dorms(self):
        # 每4个学生为一间宿舍
        for i in range(0, self.num_agents, 4):
            dorm_mates = self.schedule.agents[i:i + 4]
            for mate in dorm_mates:
                mate.roommates = dorm_mates  # 将宿舍成员分配给每个学生

    def step(self):
        new_infections_today = 0  # 新增感染人数初始化为0
        self.schedule.step()  # 更新所有代理的状态

        # 计算当前感染者人数
        current_infected = sum(1 for agent in self.schedule.agents if agent.status == "I")

        # 遍历所有代理，检查是否有新感染的
        for agent in self.schedule.agents:
            # 统计今天新增感染者（从E转为I）
            if agent.status == "I" and agent.prev_status == "E":
                new_infections_today += 1

        self.new_infections.append(new_infections_today)
        self.current_infections.append(current_infected)

        # 更新日期
        self.dates.append(self.start_date + timedelta(days=len(self.dates)))

        self.data_collector.collect(self)

    def get_results(self):
        return pd.DataFrame({
            "Date": self.dates,
            "New Infections": self.new_infections,
            "Current Infections": self.current_infections
        })

    def run_model(self, num_runs=1):
        all_results = []
        all_current_infectious=[]

        for run in range(num_runs):
            # 重置模型
            self.schedule = RandomActivation(self)
            self.dates = []
            self.new_infections = []
            self.current_infections = []

            # 初始化学生代理
            for i in range(self.num_agents):
                class_id = i % 290  # 根据代理索引分配班级
                a = StudentAgent(i, self, class_id)
                self.schedule.add(a)
                self.grid.place_agent(a, (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)))
                if i == 0:  # 只有第一个代理是感染者
                    a.status = "I"

            # 模拟24月
            for _ in range(24):
                self.step()

            all_results.append(self.new_infections)
            all_current_infectious.append(self.current_infections)

        # 创建结果DataFrame
        results_df = pd.DataFrame(all_results).T
        results_df.columns = [f'Run {i+1}' for i in range(num_runs)]
        results_df['Date'] = [date.strftime("%Y-%m-%d") for date in self.dates]
        results_df = results_df.set_index('Date')

        print(results_df)
        results_df.to_csv(r"A_5_3_5_20.csv", index=False)
            #当下感染的总人数
        all_current_infectious=pd.DataFrame(all_current_infectious).T
        all_current_infectious.columns = [f'Run {i+1}' for i in range(num_runs)]
        all_current_infectious['Date'] = [date.strftime("%Y-%m-%d") for date in self.dates]
        all_current_infectious = all_current_infectious.set_index('Date')
        print(all_current_infectious)


if __name__ == "__main__":
    model = SchoolModel(30, initial_infected=1)
    model.run_model(num_runs=5)  # 设定循环次数
    cProfile.run('model.run_model(num_runs=5)','heyue.prof')


#lh调试
import pstats
p = pstats.Stats('heyue.prof')
p.sort_stats('tottime').print_stats(10)


#————————————————————————CuPy加速————————————————————————————————
class StudentAgent(Agent):
    def __init__(self, unique_id, model, class_id=None):
        super().__init__(unique_id, model)
        self.status = "S"
        self.roommates = []
        self.class_id = class_id
        self.incubation_time = 0
        self.infection_time = 0
        self.prev_status = "S"
        self.recovery_time = 0

    def step(self):
        self.prev_status = self.status
        self.class_contact()
        self.dorm_contact()
        self.random_contact()

        if self.status == "E":
            self.incubation_time += 1
            if self.incubation_time >= 1.5 and random.random() < 0.23:
                self.status = "I"
                self.isolation_delay = 0
            elif self.incubation_time >= 3.5 and random.random() < 0.25:
                self.status = "S"
        elif self.status == "I":
            self.infection_time += 1
            if self.infection_time >= 3.5 and random.random() < 0.95:
                self.status = "R"
            elif self.infection_time >= 6 and random.random() < 0.05:
                self.status = "E"
        elif self.status == "R":
            self.recovery_time += 1
            if self.recovery_time >= 15.625 and random.random() < 0.062:
                self.status = "E"
        return

    def class_contact(self):
        neighbors = self.model.schedule.agents
        class_neighbors = [other for other in neighbors if other.class_id == self.class_id]
        infected = cp.array([agent for agent in class_neighbors if agent.status == "I"])
        susceptible = cp.array([agent for agent in class_neighbors if agent.status == "S"])

        for _ in range(900):
            if len(infected) > 0 and len(susceptible) > 0:
                contact_indices = random.choices(range(len(susceptible)), k=len(infected))
                for idx in contact_indices:
                    if random.random() < 0.00003:
                        susceptible[idx].status = "E"

    def dorm_contact(self):
        roommates_array = self.roommates
        infected = cp.array([agent for agent in roommates_array if agent.status == "I"])
        susceptible = cp.array([agent for agent in roommates_array if agent.status == "S"])

        for _ in range(900):
            if len(infected) > 0 and len(susceptible) > 0:
                contact_indices = random.choices(range(len(susceptible)), k=len(infected))
                for idx in contact_indices:
                    if random.random() < 0.00003:
                        susceptible[idx].status = "E"

    def random_contact(self):
        agents_array = self.model.schedule.agents
        infected = cp.array([agent for agent in agents_array if agent.status == "I"])
        susceptible = cp.array([agent for agent in agents_array if agent.status == "S"])

        contacts = random.choices(agents_array, k=600)
        for other in contacts:
            if other in susceptible and self.status == "I":
                if random.random() < 0.00003:
                    other.status = "E"

class SchoolModel(Model):
    """学校模型"""

    def __init__(self, N, initial_infected):
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(110, 110, True)
        self.data_collector = DataCollector(
            agent_reporters={"Status": "status"}
        )
        self.start_date = datetime.now()  # 设置开始日期
        self.dates = []  # 日期列表
        self.new_infections = []  # 每日新增感染人数
        self.current_infections = []  # 每日当前感染人数
        self.assign_dorms()

        num_classes = 290
        for i in range(self.num_agents):
            class_id = i % num_classes  # 根据代理索引分配班级

            a = StudentAgent(i, self, class_id)
            self.schedule.add(a)
            self.grid.place_agent(a, (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)))
            if i < initial_infected:
                a.status = "I"  # 设置初始感染者

        self.running = True

    def assign_dorms(self):
        # 每4个学生为一间宿舍
        for i in range(0, self.num_agents, 4):
            dorm_mates = self.schedule.agents[i:i + 4]
            for mate in dorm_mates:
                mate.roommates = dorm_mates  # 将宿舍成员分配给每个学生

    def step(self):
        new_infections_today = 0  # 新增感染人数初始化为0
        self.schedule.step()  # 更新所有代理的状态

        # 计算当前感染者人数
        current_infected = sum(1 for agent in self.schedule.agents if agent.status == "I")

        # 遍历所有代理，检查是否有新感染的
        for agent in self.schedule.agents:
            # 统计今天新增感染者（从E转为I）
            if agent.status == "I" and agent.prev_status == "E":
                new_infections_today += 1

        self.new_infections.append(new_infections_today)
        self.current_infections.append(current_infected)

        # 更新日期
        self.dates.append(self.start_date + timedelta(days=len(self.dates)))

        self.data_collector.collect(self)

    def get_results(self):
        return pd.DataFrame({
            "Date": self.dates,
            "New Infections": self.new_infections,
            "Current Infections": self.current_infections
        })

    def run_model(self, num_runs=1):
        all_results = []
        all_current_infectious=[]

        for run in range(num_runs):
            # 重置模型
            self.schedule = RandomActivation(self)
            self.dates = []
            self.new_infections = []
            self.current_infections = []

            # 初始化学生代理
            for i in range(self.num_agents):
                class_id = i % 290  # 根据代理索引分配班级
                a = StudentAgent(i, self, class_id)
                self.schedule.add(a)
                self.grid.place_agent(a, (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height)))
                if i == 0:  # 只有第一个代理是感染者
                    a.status = "I"

            # 模拟24月
            for _ in range(24):
                self.step()

            all_results.append(self.new_infections)
            all_current_infectious.append(self.current_infections)

        # 创建结果DataFrame
        results_df = pd.DataFrame(all_results).T
        results_df.columns = [f'Run {i+1}' for i in range(num_runs)]
        results_df['Date'] = [date.strftime("%Y-%m-%d") for date in self.dates]
        results_df = results_df.set_index('Date')

        print(results_df)
        results_df.to_csv(r"A_5_3_5_20.csv", index=False)
            #当下感染的总人数
        all_current_infectious=pd.DataFrame(all_current_infectious).T
        all_current_infectious.columns = [f'Run {i+1}' for i in range(num_runs)]
        all_current_infectious['Date'] = [date.strftime("%Y-%m-%d") for date in self.dates]
        all_current_infectious = all_current_infectious.set_index('Date')
        print(all_current_infectious)


if __name__ == "__main__":
    model = SchoolModel(30, initial_infected=1)
    model.run_model(num_runs=5)  # 设定循环次数
    cProfile.run('model.run_model(num_runs=5)','heyue.prof')