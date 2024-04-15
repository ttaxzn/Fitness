import random
import numpy as np
import time


activities_data = [
    {"activity_name": "SLA100A", "student_count": 50, "preferred_instructors": ["Glen", "Lock", "Banks", "Zeldin"],
     "other_instructors": ["Numen", "Richards"]},
    {"activity_name": "SLA100B", "student_count": 50, "preferred_instructors": ["Glen", "Lock", "Banks", "Zeldin"],
     "other_instructors": ["Numen", "Richards"]},
    {"activity_name": "SLA191A", "student_count": 50, "preferred_instructors": ["Glen", "Lock", "Banks", "Zeldin"],
     "other_instructors": ["Numen", "Richards"]},
    {"activity_name": "SLA191B", "student_count": 50, "preferred_instructors": ["Glen", "Lock", "Banks", "Zeldin"],
     "other_instructors": ["Numen", "Richards"]},
    {"activity_name": "SLA201", "student_count": 50, "preferred_instructors": ["Glen", "Banks", "Zeldin", "Shaw"],
     "other_instructors": ["Numen", "Richards", "Singer"]},
    {"activity_name": "SLA291", "student_count": 50, "preferred_instructors": ["Lock", "Banks", "Zeldin", "Singer"],
     "other_instructors": ["Numen", "Richards", "Shaw", "Tyler"]},
    {"activity_name": "SLA303", "student_count": 60, "preferred_instructors": ["Glen", "Zeldin", "Banks"],
     "other_instructors": ["Numen", "Singer", "Shaw"]},
    {"activity_name": "SLA304", "student_count": 25, "preferred_instructors": ["Glen", "Banks", "Tyler"],
     "other_instructors": ["Numen", "Singer", "Shaw", "Richards", "Uther", "Zeldin"]},
    {"activity_name": "SLA394", "student_count": 20, "preferred_instructors": ["Tyler", "Singer"],
     "other_instructors": ["Richards", "Zeldin"]},
    {"activity_name": "SLA449", "student_count": 60, "preferred_instructors": ["Tyler", "Singer", "Shaw"],
     "other_instructors": ["Zeldin", "Uther"]},
    {"activity_name": "SLA451", "student_count": 100, "preferred_instructors": ["Tyler", "Singer", "Shaw"],
     "other_instructors": ["Zeldin", "Uther", "Richards", "Banks"]},
]

instructors_data = ["Lock", "Glen", "Banks", "Richards", "Shaw", "Singer", "Uther", "Tyler", "Numen", "Zeldin"]

rooms_data = [
    {"room_name": "Slater 003", "capacity": 45},
    {"room_name": "Roman 216", "capacity": 30},
    {"room_name": "Loft 206", "capacity": 75},
    {"room_name": "Roman 201", "capacity": 50},
    {"room_name": "Loft 310", "capacity": 108},
    {"room_name": "Beach 201", "capacity": 60},
    {"room_name": "Beach 301", "capacity": 75},
    {"room_name": "Logos 325", "capacity": 450},
    {"room_name": "Frank 119", "capacity": 60},
]

timeslots_data = ["10 AM", "11 AM", "12 PM", "1 PM", "2 PM", "3 PM"]



class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            schedule = self.generate_random_schedule()
            population.append(schedule)
        return population

    def generate_random_schedule(self):
        schedule = []
        for activity in activities_data:
            room = random.choice(rooms_data)
            time_slot = random.choice(timeslots_data)
            preferred_instructor = random.choice(activity["preferred_instructors"])
            schedule.append({"activity_name": activity["activity_name"], "time_slot": time_slot, "room_name": room["room_name"],
                             "instructor": preferred_instructor})
        return schedule

    def evaluate_fitness(self, schedule):
        fitness = 0
        instructor_load = {instructor: 0 for instructor in instructors_data}

        for i, activity1 in enumerate(schedule):
            for j, activity2 in enumerate(schedule):
                if i != j and activity1["time_slot"] == activity2["time_slot"] and activity1["room_name"] == activity2["room_name"]:
                    fitness -= 0.5

        for activity in schedule:
            expected_students = next((a["student_count"] for a in activities_data if a["activity_name"] == activity["activity_name"]), None)
            room_capacity = next((room["capacity"] for room in rooms_data if room["room_name"] == activity["room_name"]), None)
            if room_capacity is not None:
                if room_capacity < expected_students:
                    fitness -= 0.5
                elif room_capacity > 6 * expected_students:
                    fitness -= 0.4
                elif room_capacity > 3 * expected_students:
                    fitness -= 0.2
                else:
                    fitness += 0.3

            preferred_instructors = next(
                (a["preferred_instructors"] for a in activities_data if a["activity_name"] == activity["activity_name"]), [])
            other_instructors = next(
                (a["other_instructors"] for a in activities_data if a["activity_name"] == activity["activity_name"]), [])
            if activity["instructor"] in preferred_instructors:
                fitness += 0.5
            elif activity["instructor"] in other_instructors:
                fitness += 0.2
            else:
                fitness -= 0.1

            instructor_load[activity["instructor"]] += 1

        for instructor, load in instructor_load.items():
            if instructor == "Dr. Tyler" and load < 2:
                continue
            if load > 4:
                fitness -= 0.5
            elif load > 2:
                fitness -= 0.4

        for i in range(len(schedule) - 1):
            current_time_slot = timeslots_data.index(schedule[i]["time_slot"])
            next_time_slot = timeslots_data.index(schedule[i + 1]["time_slot"])
            if next_time_slot == current_time_slot + 1:
                # Implement rules for consecutive time slots
                pass

        
        for i in range(len(schedule) - 1):
            activity1 = schedule[i]
            activity2 = schedule[i + 1]

            if (activity1["activity_name"] == "SLA101A" and activity2["activity_name"] == "SLA101B") or \
                    (activity1["activity_name"] == "SLA101B" and activity2["activity_name"] == "SLA101A"):
                if abs(timeslots_data.index(activity1["time_slot"]) - timeslots_data.index(activity2["time_slot"])) > 4:
                    fitness += 0.5
                if activity1["time_slot"] == activity2["time_slot"]:
                    fitness -= 0.5

            if (activity1["activity_name"] == "SLA191A" and activity2["activity_name"] == "SLA191B") or \
                    (activity1["activity_name"] == "SLA191B" and activity2["activity_name"] == "SLA191A"):
                if abs(timeslots_data.index(activity1["time_slot"]) - timeslots_data.index(activity2["time_slot"])) > 4:
                    fitness += 0.5
                if activity1["time_slot"] == activity2["time_slot"]:
                    fitness -= 0.5

            if (activity1["activity_name"].startswith("SLA101") and activity2["activity_name"].startswith("SLA191")) or \
                    (activity1["activity_name"].startswith("SLA191") and activity2["activity_name"].startswith("SLA101")):
                if abs(timeslots_data.index(activity1["time_slot"]) - timeslots_data.index(activity2["time_slot"])) == 1:
                    fitness += 0.25
                elif activity1["time_slot"] == activity2["time_slot"]:
                    fitness -= 0.25

            if (activity1["activity_name"].startswith("SLA101") and activity2["activity_name"].startswith("SLA191")) or \
                    (activity1["activity_name"].startswith("SLA191") and activity2["activity_name"].startswith("SLA101")):
                if abs(timeslots_data.index(activity1["time_slot"]) - timeslots_data.index(activity2["time_slot"])) == 1:
                    if activity1["room_name"].startswith("Roman") or activity1["room_name"].startswith("Beach") or \
                            activity2["room_name"].startswith("Roman") or activity2["room_name"].startswith("Beach"):
                        fitness -= 0.4

        return fitness

    def softmax(self, fitness_scores):
        exp_scores = np.exp(fitness_scores)
        probs = exp_scores / np.sum(exp_scores)
        return probs

    def select_parents(self):
        parent_indices = np.random.choice(len(self.population), size=2, replace=False)
        return self.population[parent_indices[0]], self.population[parent_indices[1]]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        return offspring1, offspring2

    def mutate(self, schedule):
        mutated_schedule = schedule[:]
        for i in range(len(mutated_schedule)):
            if random.random() < self.mutation_rate:
                mutated_schedule[i]["room_name"] = random.choice(rooms_data)["room_name"]
                mutated_schedule[i]["time_slot"] = random.choice(timeslots_data)
                mutated_schedule[i]["instructor"] = random.choice(instructors_data)
        return mutated_schedule

    def evolve(self):
        new_population = []
        fitness_scores = [self.evaluate_fitness(schedule) for schedule in self.population]
        probabilities = self.softmax(fitness_scores)

        for _ in range(self.population_size // 2):
            parent1, parent2 = self.select_parents()
            offspring1, offspring2 = self.crossover(parent1, parent2)
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)
            new_population.extend([offspring1, offspring2])

        self.population = new_population

    def optimize(self, max_generations):
        improvement_threshold = 0.01
        last_avg_fitness = float('inf')
        generations_without_improvement = 0

        for generation in range(max_generations):
            self.evolve()
            fitness_scores = [self.evaluate_fitness(schedule) for schedule in self.population]
            avg_fitness = np.mean(fitness_scores)

            if last_avg_fitness != 0 and last_avg_fitness != float('inf'):
                improvement_ratio = (last_avg_fitness - avg_fitness) / last_avg_fitness
                if improvement_ratio < improvement_threshold:
                    generations_without_improvement += 1
                else:
                    generations_without_improvement = 0
            else:
                generations_without_improvement = 0

            if generations_without_improvement >= 100:
                break

            last_avg_fitness = avg_fitness

        best_fitness_index = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_fitness_index]
        best_schedule = self.population[best_fitness_index]

        return best_fitness, best_schedule



population_size_val = 500

mutation_rate_val = 0.01

max_generations_val = 100


start_time = time.time()


ga = GeneticAlgorithm(population_size_val, mutation_rate_val)
best_fitness_val, best_schedule_val = ga.optimize(max_generations_val)


end_time = time.time()
total_time_val = end_time - start_time

print("Best Fitness:", best_fitness_val)
print("Best Schedule:", best_schedule_val)
print("Time it took to generate: ", total_time_val, " seconds")
