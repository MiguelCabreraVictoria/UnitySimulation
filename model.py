import heapq
import agentpy as ap
import numpy as np
import asyncio
import websockets
import random, json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns, IPython
from matplotlib import pyplot as plt, cm


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + 1

            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor[0]][neighbor[1]] in [-1]:
                        continue
                else:
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return False

class myAgent(ap.Agent):
    def setup(self):
        self.env = self.model.env
        self.home = self.p.home
        self.destiny = self.p.work  

    def calculate_path(self):
        grid = np.array(self.p.street)
        self.path = astar(grid, self.home, self.destiny)
        
        if self.path is False: 
            print("Error: No se encontró un camino desde 'home' hasta 'work'.")
            self.path = []
        else:
            print("Path:", len(self.path))

    def get_position(self):
        return self.env.positions[self]
    
    def move(self, new_position):
        self.env.move_to(self, new_position)
        self.record('positions', new_position)
        print('Agent moved to:', new_position)
    
    def execute(self):
      
        if self.path:
            self.move(self.path[0])
            self.path.pop(0)
        
        if self.get_position() == self.destiny:
            print('Agent arrived to work')
            self.model.end()


class myEnvironment(ap.Grid):
    def setup(self):
        self.street = self.p.street

    def modify_env(self):
        iterations = self.p.env_size[0] * self.p.env_size[1]
        
        for i in range(iterations):
            x, y = self.model.env.all[i]
            self.model.env.all[i] = self.street[x, y]

        new_env = []
        for i in range(self.p.env_size[0]):
            row = []
            for j in range(self.p.env_size[1]):
                row.append(self.model.env.all[i*self.p.env_size[1] + j])
            new_env.append(row)

        self.model.env.all = np.array(new_env)


class myModel(ap.Model):
    def setup(self):
        self.env = myEnvironment(self, self.p.env_size)
        self.agent = myAgent(self)
        self.env.add_agents([self.agent], [self.p.home])

        self.env.modify_env()

        # Calcular el camino
        self.agent.calculate_path()

    def step(self):
        self.agent.execute()


# Cargar el archivo de calles
streets = np.load('streets.npy')
streets = streets.astype(int)

# Parámetros del modelo
parameters = {
    "env_size": streets.shape,
    "street": streets,
    "home": (5, 3),
    "work": (16, 26),
    "work_parking": (16, 26)
}


def my_plot(model, ax):
    m, n = model.p.street.shape
    grid = np.zeros((m, n))
    grid[model.p.street == -1] = 3  # Celdas con valor -1
    grid[model.p.street == -10] = 8  # Celdas con valor -10
    grid[model.p.home] = 6  # Posición de inicio (home)
    grid[model.p.work] = 7  # Posición de destino (work)
    
    # Actualizar la posición del agente
    agent = list(model.env.agents)[0]
    state = model.env.positions[agent]
    grid[state] = 1  # Posición actual del agente

    # Diccionario de colores
    color_dict = {
        0: '#ffffff',  # Celdas vacías (valor 0)
        1: '#141414',  # Agente
        3: '#d10f0f',  # Celdas con -1
        6: '#bdb675',  # Home
        7: '#22d469',  # Work
        8: '#4287f5'   # Celdas con -10
    }

    # Graficar la cuadrícula
    ap.gridplot(grid, ax=ax, color_dict=color_dict, convert=True)



model = myModel(parameters)
model.run(steps=34, display=False)


async def send_coordinates(agents_positions):
    async with websockets.connect("ws://localhost:8765") as websocket:
        for idx in range(len(agents_positions)):
            x, z = agents_positions[idx]
            coordinates = {
                "x": x,
                "y": 0.9,
                "z": z
            }
            await websocket.send(json.dumps(coordinates))
            print(f"Enviando: {coordinates}")
            await asyncio.sleep(2)

if __name__ == "__main__":
    model = myModel(parameters)
    model.run(steps=34, display=False)
    agents_positions = model.agent.log['positions']
    asyncio.run(send_coordinates(agents_positions))




