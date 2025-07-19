import pygame
import sys
import time
import random
import pickle
from collections import deque
import math
import copy
import matplotlib.pyplot as plt # Import matplotlib

# --- Constants ---
BG_COLOR = (0, 0, 0)
WALL_COLOR = (0, 0, 255)
DOT_COLOR = (255, 255, 255)
POWERUP_COLOR = (0, 255, 0)
PACMAN_COLOR = (234, 179, 8)
GHOST_COLORS = [(239, 68, 68), (219, 39, 119), (14, 165, 233), (249, 115, 22)] # Blinky, Pinky, Inky, Clyde
VULNERABLE_COLOR = (100, 100, 255)  # Color for vulnerable ghosts
TEXT_COLOR = (255, 255, 255)
SCORE_TEXT_COLOR = (255, 255, 255)
FONT_NAME = 'freesansbold.ttf'
CELL_SIZE = 20
FPS = 60
BASE_AI_DELAY = 0.2  # Faster update for smoother movement
FAST_AI_DELAY = 0.04
POWERUP_DURATION = 6.0 # Power-up lasts for 6 seconds

# Speed adjustments (CRITICAL FOR AI PERCEPTION)
PACMAN_BASE_SPEED = 4.5 # Adjusted for more challenge
GHOST_BASE_SPEED = 4.0  # Adjusted for more challenge
PACMAN_DOT_EATING_SPEED_MULTIPLIER = 0.8 # Pacman slows down when eating dots
GHOST_SPEED_INCREASE_PER_1000_POINTS = 0.1 # 10% speed increase per 1000 points

# AI Tunable Parameters for Aggressiveness
AGRESSION_FACTOR = 1.5 # Multiplier for chase mode weight (higher = more aggressive chase)
FLEE_WEIGHT_REDUCTION_FACTOR = 0.3 # Probability (0.0 to 1.0) to pick a less optimal flee path when frightened.
                                  # Higher value means ghosts are less effective at fleeing.
                                  # 0.0 = always pick best flee path. 1.0 = always pick random valid path.

# Ghost AI Mode Commitment
GHOST_MODE_COMMITMENT_DURATION = 3.0 # Seconds a ghost commits to a chosen mode before re-evaluating.
                                     # This prevents rapid, indecisive mode switching.

# Dot shrink animation
DOT_SHRINK_TIME = 0.3  # seconds

BUTTON_BG_COLOR = (50, 50, 50)
BUTTON_HOVER_COLOR = (100, 100, 100)
BUTTON_TEXT_COLOR = (220, 220, 220)
BUTTON_PADDING_X = 20
BUTTON_PADDING_Y = 8
BUTTON_SPACING = 18

# UI layout constants
BOTTOM_PANEL_HEIGHT = 120 # Increased height for more space
STATUS_TEXT_HEIGHT = 30
BUTTON_AREA_HEIGHT = BOTTOM_PANEL_HEIGHT - STATUS_TEXT_HEIGHT
SCREEN_W_PADDING = 20

# --- Save / Load state ---
DUMP_FILE = 'game_stats.pkl' # Renamed for clarity, as it no longer stores Q-table

def save_dump(data):
    """Saves game statistics to a pickle file."""
    with open(DUMP_FILE, 'wb') as f:
        pickle.dump(data, f)

def load_dump():
    """Loads game statistics from a pickle file."""
    try:
        with open(DUMP_FILE, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {'best_score': 0, 'worst_score': float('inf'), 'average_score': 0, 'pacman_wins': 0, 'ghost_wins': 0}

# --- Map Definition ---
ORIGINAL_MAP = [
    list("############################"),
    list("#............##............#"),
    list("#.####.#####.##.#####.####.#"),
    list("#o####.#####.##.#####.####o#"),
    list("#.####.#####.##.#####.####.#"),
    list("#..........................#"),
    list("#.####.##.########.##.####.#"),
    list("#.####.##.########.##.####.#"),
    list("#......##....##....##......#"),
    list("######.##### ## #####.######"),
    list("     #.##### ## #####.#     "),
    list("     #.##          ##.#     "),
    list("     #.## ###--### ##.#     "),
    list("######.## #      # ##.######"),
    list("      .   #      #   .      "),
    list("######.## #      # ##.######"),
    list("     #.## ######## ##.#     "),
    list("     #.##          ##.#     "),
    list("     #.## ######## ##.#     "),
    list("######.## ######## ##.######"),
    list("#............##............#"),
    list("#.####.#####.##.#####.####.#"),
    list("#.####.#####.##.#####.####.#"),
    list("#o..##................##..o#"),
    list("###.##.##.########.##.##.###"),
    list("###.##.##.########.##.##.###"),
    list("#......##....##....##......#"),
    list("#.##########.##.##########.#"),
    list("#.##########.##.##########.#"),
    list("#..........................#"),
    list("############################"),
]

GRID_H = len(ORIGINAL_MAP)
GRID_W = len(ORIGINAL_MAP[0])
SCREEN_W = GRID_W * CELL_SIZE
SCREEN_H = GRID_H * CELL_SIZE + BOTTOM_PANEL_HEIGHT

# --- Helper Functions (Module Scope) ---
def is_collision(entity1, entity2, threshold=14):
    """
    Checks for collision between two entities based on pixel distance.
    Verifica colisão entre duas entidades com base na distância em pixels.
    """
    x1, y1 = entity1.get_pixel_pos()
    x2, y2 = entity2.get_pixel_pos()
    return math.hypot(x1 - x2, y1 - y2) < threshold

def find_spawn_for_ghosts(grid):
    """Returns fixed ghost house positions."""
    """Retorna posições fixas da casa dos fantasmas."""
    return [(13,14), (14,14), (13,15), (14,15)]

def find_pacman_spawn(grid):
    """Returns a fixed, strategic spawn point for Pacman."""
    """Retorna um ponto de spawn fixo e estratégico para o Pacman."""
    return (13, 23) # (x, y) coordinates

def bfs_find_path(grid_map, start, goal):
    """Performs Breadth-First Search to find the shortest path."""
    """Realiza uma Busca em Largura (BFS) para encontrar o caminho mais curto."""
    q = deque([(start, [start])])
    vis = set()
    
    # Ensure start and goal are valid grid positions and not walls
    # Garante que o início e o objetivo são posições de grade válidas e não paredes
    if not (0 <= start[0] < GRID_W and 0 <= start[1] < GRID_H and grid_map[start[1]][start[0]] != '#'):
        return []
    if not (0 <= goal[0] < GRID_W and 0 <= goal[1] < GRID_H and grid_map[goal[1]][goal[0]] != '#'):
        return []

    while q:
        cur, path = q.popleft()
        if cur in vis:
            continue
        vis.add(cur)
        if cur == goal:
            return path
        
        x, y = cur
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            # Check against the provided grid_map (which should be ORIGINAL_MAP for pathfinding)
            # Verifica contra o grid_map fornecido (que deve ser ORIGINAL_MAP para pathfinding)
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H and grid_map[ny][nx] != '#' and (nx, ny) not in vis:
                q.append(((nx, ny), path + [(nx, ny)]))
    return []

# --- Entity Classes ---
class Entity:
    def __init__(self, x, y, color, speed):
        self.grid_x = x
        self.grid_y = y
        self.color = color
        self.speed = speed
        self.direction = (0, 0)
        self.next_direction = (0, 0)
        self.move_progress = 0.0
        self.radius = CELL_SIZE // 2 - 2

    def can_move(self, grid_map, x, y):
        """Checks if an entity can move to a given grid position."""
        """Verifica se uma entidade pode se mover para uma dada posição na grade."""
        # Use ORIGINAL_MAP for wall checks, as 'grid' changes with dots
        # Usa ORIGINAL_MAP para verificar paredes, pois 'grid' muda com os pontos
        return 0 <= x < GRID_W and 0 <= y < GRID_H and grid_map[y][x] != '#'

    def at_center(self):
        """Returns True if entity is centered in a grid cell (move_progress close to 0)."""
        """Retorna True se a entidade está centralizada em uma célula da grade."""
        tolerance = self.speed * CELL_SIZE * 0.15
        return self.move_progress < tolerance or self.move_progress > (CELL_SIZE - tolerance)

    def update(self, dt, current_game_grid):
        """Updates entity position and handles direction changes."""
        """Atualiza a posição da entidade e lida com as mudanças de direção."""
        if self.at_center():
            # Always try to switch to next_direction if it's valid
            # Sempre tenta mudar para a próxima direção se for válida
            nx_next, ny_next = self.grid_x + self.next_direction[0], self.grid_y + self.next_direction[1]
            if self.can_move(ORIGINAL_MAP, nx_next, ny_next): # Use ORIGINAL_MAP for movement validity
                self.direction = self.next_direction
            else:
                # If next_direction is blocked, try to continue current direction
                # Se a próxima direção estiver bloqueada, tenta continuar na direção atual
                nx_curr, ny_curr = self.grid_x + self.direction[0], self.grid_y + self.direction[1]
                if not self.can_move(ORIGINAL_MAP, nx_curr, ny_curr): # Use ORIGINAL_MAP
                    # If current direction is also blocked, stop
                    # Se a direção atual também estiver bloqueada, para
                    self.direction = (0, 0)
                    self.move_progress = 0 # Reset progress if stuck
        
        # If not moving, and a next_direction is set, try to start moving
        # Se não estiver se movendo e uma próxima direção estiver definida, tenta começar a mover
        if self.direction == (0,0) and self.next_direction != (0,0):
            nx_next, ny_next = self.grid_x + self.next_direction[0], self.grid_y + self.next_direction[1]
            if self.can_move(ORIGINAL_MAP, nx_next, ny_next): # Use ORIGINAL_MAP
                self.direction = self.next_direction
                self.move_progress = 0 # Start fresh movement

        self.move_progress += dt * self.speed * CELL_SIZE

        while self.move_progress >= CELL_SIZE:
            nx = self.grid_x + self.direction[0]
            ny = self.grid_y + self.direction[1]
            if self.can_move(ORIGINAL_MAP, nx, ny): # Use ORIGINAL_MAP
                self.grid_x, self.grid_y = nx, ny
                self.move_progress -= CELL_SIZE
            else:
                self.direction = (0, 0)
                self.move_progress = 0
                break

        # Ghosts decide their next direction here, Pacman's is set by manual input
        # Fantasmas decidem sua próxima direção aqui, a do Pacman é definida por entrada manual
        if not isinstance(self, Pacman):
            # Pass the global 'pacman' and 'ghosts' list for AI decisions
            # Passa as listas globais 'pacman' e 'ghosts' para as decisões da IA
            self.decide_next_direction(current_game_grid, pacman, ghosts)

    def get_pixel_pos(self):
        """Calculates the entity's pixel position for drawing."""
        """Calcula a posição em pixels da entidade para desenho."""
        off = self.move_progress / CELL_SIZE
        px = (self.grid_x + self.direction[0] * off) * CELL_SIZE + CELL_SIZE // 2
        py = (self.grid_y + self.direction[1] * off) * CELL_SIZE + CELL_SIZE // 2
        return px, py

class Pacman(Entity):
    def __init__(self, x, y, grid_ref):
        super().__init__(x, y, PACMAN_COLOR, PACMAN_BASE_SPEED)
        self.powerup_timer = 0
        self.next_direction = (0, 0)
        self.last_valid_direction = (0, -1) # Default to facing UP when stationary
        self.grid_ref = grid_ref # Store reference to the game grid

    def set_manual_direction(self, new_direction):
        """Sets the direction based on manual keyboard input, prioritizing immediate turn."""
        """Define a direção com base na entrada manual do teclado, priorizando a virada imediata."""
        self.next_direction = new_direction
        
        # Attempt to change direction immediately if valid and not a direct reverse
        # or if currently stationary.
        # Tenta mudar de direção imediatamente se for válida e não for uma reversão direta
        # ou se estiver parado.
        if self.direction == (0,0): # If stationary, just start moving
            self.direction = new_direction
            self.move_progress = 0 # Reset progress to start new movement
        elif new_direction != (0,0) and \
             (new_direction[0] != -self.direction[0] or new_direction[1] != -self.direction[1]):
            # If not stationary and not a direct reverse, try to turn immediately
            # Se não estiver parado e não for uma reversão direta, tenta virar imediatamente
            # Check if the immediate turn is valid
            # Verifica se a virada imediata é válida
            nx, ny = self.grid_x + new_direction[0], self.grid_y + new_direction[1]
            if self.can_move(ORIGINAL_MAP, nx, ny): # Use ORIGINAL_MAP for movement validity
                self.direction = new_direction
                self.move_progress = 0 # Reset progress to start new movement
        
        if new_direction != (0,0): # Store the last non-stationary direction for drawing
            self.last_valid_direction = new_direction


class Ghost(Entity):
    def __init__(self, x, y, color, scatter_target):
        super().__init__(x, y, color, GHOST_BASE_SPEED)
        self.scatter_target = scatter_target
        self.vulnerable = False
        self.original_pos = (x, y)
        self.mode = 'scatter' # Ghosts start in scatter mode
        self.frightened_timer = 0.0 # Timer for vulnerable state

        # New: Mode Commitment Variables
        # Novo: Variáveis de Compromisso de Modo
        self.current_committed_mode = 'scatter' # The mode the ghost is currently committed to
        self.mode_start_time = time.time()      # When the current committed mode started
        self.last_evaluated_mode = 'scatter'    # The mode chosen in the last evaluation (before commitment)
        
        # New: Flag to indicate if ghost is returning to house after being eaten
        # Novo: Flag para indicar se o fantasma está retornando para a casa depois de ser comido
        self.returning_to_house = False 

    def get_blinky_target(self, pacman_inst):
        """Blinky's target: Pacman's current tile."""
        """Alvo do Blinky: a célula atual do Pacman."""
        return (pacman_inst.grid_x, pacman_inst.grid_y)

    def get_pinky_target(self, pacman_inst):
        """Pinky's target: 4 tiles ahead of Pacman's current direction."""
        """Alvo do Pinky: 4 células à frente da direção atual do Pacman."""
        target_x = pacman_inst.grid_x + pacman_inst.direction[0] * 4
        target_y = pacman_inst.grid_y + pacman_inst.direction[1] * 4
        
        # Special case for Pacman moving up (due to original game bug)
        # Caso especial para o Pacman se movendo para cima (devido a um bug do jogo original)
        if pacman_inst.direction == (0, -1):
            target_x -= 4 # Pinky targets 4 tiles up and 4 tiles left when Pacman moves up

        # Ensure target is within grid bounds and not a wall (using ORIGINAL_MAP)
        # Garante que o alvo está dentro dos limites da grade e não é uma parede (usando ORIGINAL_MAP)
        if not (0 <= target_x < GRID_W and 0 <= target_y < GRID_H and ORIGINAL_MAP[target_y][target_x] != '#'):
            return (pacman_inst.grid_x, pacman_inst.grid_y) # Fallback
        return (target_x, target_y)

    def get_inky_target(self, pacman_inst, blinky_inst):
        """Inky's target: Complex, based on Pacman's position and Blinky's position."""
        """Alvo do Inky: Complexo, baseado na posição do Pacman e do Blinky."""
        # 1. Get point 2 tiles in front of Pacman
        # 1. Obtém o ponto 2 células à frente do Pacman
        pacman_ahead_x = pacman_inst.grid_x + pacman_inst.direction[0] * 2
        pacman_ahead_y = pacman_inst.grid_y + pacman_inst.direction[1] * 2

        # Special case for Pacman moving up
        # Caso especial para o Pacman se movendo para cima
        if pacman_inst.direction == (0, -1):
            pacman_ahead_x -= 2 # Inky targets 4 tiles up and 4 tiles left when Pacman moves up

        # 2. Vector from Blinky to this point
        # 2. Vetor do Blinky para este ponto
        vector_x = pacman_ahead_x - blinky_inst.grid_x
        vector_y = pacman_ahead_y - blinky_inst.grid_y

        # 3. Double this vector from Blinky's position
        # 3. Dobra este vetor a partir da posição do Blinky
        target_x = blinky_inst.grid_x + vector_x * 2
        target_y = blinky_inst.grid_y + vector_y * 2

        # Ensure target is within grid bounds and not a wall (using ORIGINAL_MAP)
        # Garante que o alvo está dentro dos limites da grade e não é uma parede (usando ORIGINAL_MAP)
        if not (0 <= target_x < GRID_W and 0 <= target_y < GRID_H and ORIGINAL_MAP[target_y][target_x] != '#'):
            return (pacman_inst.grid_x, pacman_inst.grid_y) # Fallback
        return (target_x, target_y)

    def get_clyde_target(self, pacman_inst):
        """Clyde's target: Chase Pacman if far, scatter if close."""
        """Alvo do Clyde: Persegue o Pacman se estiver longe, dispersa se estiver perto."""
        distance_to_pacman = abs(self.grid_x - pacman_inst.grid_x) + abs(self.grid_y - pacman_inst.grid_y)
        if distance_to_pacman > 8: # If more than 8 tiles away, chase Pacman
            return (pacman_inst.grid_x, pacman_inst.grid_y)
        else: # If within 8 tiles, retreat to his scatter corner
            return self.scatter_target

    def get_powerup_target(self, current_game_grid):
        """Finds the closest power-up on the map."""
        """Encontra o power-up mais próximo no mapa."""
        powerup_locations = []
        for y, row in enumerate(current_game_grid):
            for x, val in enumerate(row):
                if val == 'o':
                    powerup_locations.append((x, y))
        
        if not powerup_locations:
            return None # No power-ups left
        
        # Find the closest power-up using Manhattan distance
        closest_powerup = None
        min_dist = float('inf')
        for px, py in powerup_locations:
            dist = abs(self.grid_x - px) + abs(self.grid_y - py)
            if dist < min_dist:
                min_dist = dist
                closest_powerup = (px, py)
        return closest_powerup

    def decide_next_direction(self, current_game_grid, pacman_inst, all_ghosts):
        """Ghost AI logic: chase Pacman, scatter, or flee when vulnerable.
           Includes distinct personalities and prevents 180-degree turns."""
        """Lógica da IA do Fantasma: persegue o Pacman, dispersa ou foge quando vulnerável.
           Inclui personalidades distintas e previne viradas de 180 graus."""
        
        # Determine vulnerability based on Pacman's powerup timer
        # Determina a vulnerabilidade com base no temporizador de power-up do Pacman
        if pacman_inst.powerup_timer > 0:
            self.mode = 'frightened'
            self.vulnerable = True
            self.frightened_timer = pacman_inst.powerup_timer # Sync ghost frightened timer with Pacman's powerup
        elif self.frightened_timer > 0: # If powerup just ended, but ghost is still vulnerable
            self.mode = 'frightened'
            self.vulnerable = True
        else:
            self.vulnerable = False # Not vulnerable if powerup is off
            # The actual mode (chase/scatter/powerup_guard) is set by the game loop's ghost_mode_manager.
            # This method will just use self.mode to determine target.

        target = None
        # --- NEW LOGIC FOR GHOST HOUSE RETURN ---
        # Lógica para retorno à casa dos fantasmas
        if self.returning_to_house:
            # Target is the original position (ghost house entrance)
            # O alvo é a posição original (entrada da casa dos fantasmas)
            target = self.original_pos
            # If ghost has reached its original position, stop returning and resume normal mode
            # Se o fantasma alcançou sua posição original, para de retornar e retoma o modo normal
            if (self.grid_x, self.grid_y) == self.original_pos:
                self.returning_to_house = False
                self.mode = 'scatter' # Or whatever default mode you want after returning
                self.current_committed_mode = 'scatter' # Reset commitment
                self.mode_start_time = time.time()
                # Immediately try to move out of the house if possible
                # Tenta imediatamente sair da casa, se possível
                # Find a path out of the ghost house (example exit point: (13, 11))
                # Encontra um caminho para fora da casa dos fantasmas (ponto de saída de exemplo: (13, 11))
                exit_path = bfs_find_path(ORIGINAL_MAP, self.original_pos, (13, 11)) 
                if len(exit_path) >= 2:
                    dx = exit_path[1][0] - self.grid_x
                    dy = exit_path[1][1] - self.grid_y
                    self.next_direction = (dx, dy)
                else:
                    # If no clear exit path, try a random valid move
                    # Se não houver um caminho de saída claro, tenta um movimento válido aleatório
                    options = [(1,0), (-1,0), (0,1), (0,-1)]
                    valid_options = [d for d in options if self.can_move(ORIGINAL_MAP, self.grid_x + d[0], self.grid_y + d[1])]
                    if valid_options:
                        self.next_direction = random.choice(valid_options)
                    else:
                        self.next_direction = (0,0) # Stuck
                return # Exit early, as we've set the next direction for returning
        # --- END NEW LOGIC FOR GHOST HOUSE RETURN ---

        # If not returning to house, determine target based on current mode
        # Se não estiver retornando para casa, determina o alvo com base no modo atual
        if self.mode == 'frightened':
            # Flee mode: move away from Pacman, or to a random safe spot
            # Modo de fuga: move-se para longe do Pacman, ou para um local seguro aleatório
            flee_options = []
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = self.grid_x + dx, self.grid_y + dy
                if self.can_move(ORIGINAL_MAP, nx, ny): # Use Entity.can_move which checks ORIGINAL_MAP
                    dist_to_pacman = abs(nx - pacman_inst.grid_x) + abs(ny - pacman_inst.grid_y)
                    flee_options.append(((nx, ny), dist_to_pacman))
            
            if flee_options:
                # Reduce flee behavior: introduce a chance to pick a less optimal flee path
                # Reduz o comportamento de fuga: introduz uma chance de escolher um caminho de fuga menos ideal
                if random.random() < FLEE_WEIGHT_REDUCTION_FACTOR: 
                    # Pick a random valid option that is not directly towards Pacman
                    # Escolhe uma opção válida aleatória que não seja diretamente em direção ao Pacman
                    valid_flee_moves = [
                        (dx, dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                        if self.can_move(ORIGINAL_MAP, self.grid_x + dx, self.grid_y + dy) and
                           (self.grid_x + dx, self.grid_y + dy) != (pacman_inst.grid_x, pacman_inst.grid_y) # Not directly into Pacman
                    ]
                    if valid_flee_moves:
                        chosen_move = random.choice(valid_flee_moves)
                        target = (self.grid_x + chosen_move[0], self.grid_y + chosen_move[1])
                    else: # Fallback if no random valid move, pick the best
                        target_pos_flee = max(flee_options, key=lambda item: item[1])[0]
                        target = target_pos_flee
                else: # Otherwise, pick the best flee path (default behavior)
                    target_pos_flee = max(flee_options, key=lambda item: item[1])[0]
                    target = target_pos_flee
            else:
                # Fallback if no valid flee options (e.g., trapped), go to scatter target
                # Fallback se não houver opções de fuga válidas (ex: preso), vai para o alvo de dispersão
                target = self.scatter_target
        elif self.mode == 'chase':
            # Determine target based on ghost personality
            # Determina o alvo com base na personalidade do fantasma
            if self.color == GHOST_COLORS[0]: # Blinky (Red)
                target = self.get_blinky_target(pacman_inst)
            elif self.color == GHOST_COLORS[1]: # Pinky (Pink)
                target = self.get_pinky_target(pacman_inst)
            elif self.color == GHOST_COLORS[2]: # Inky (Cyan)
                # Find Blinky from the list of all ghosts
                # Encontra Blinky na lista de todos os fantasmas
                blinky_ghost = next((g for g in all_ghosts if g.color == GHOST_COLORS[0]), None)
                if blinky_ghost:
                    target = self.get_inky_target(pacman_inst, blinky_ghost)
                else: # Fallback if Blinky not found (shouldn't happen in normal play)
                    target = self.get_blinky_target(pacman_inst)
            elif self.color == GHOST_COLORS[3]: # Clyde (Orange)
                target = self.get_clyde_target(pacman_inst)
            else: # Default fallback
                target = (pacman_inst.grid_x, pacman_inst.grid_y)
        elif self.mode == 'scatter':
            target = self.scatter_target
            # --- NEW LOGIC FOR SCATTER PATROLLING ---
            # Lógica para patrulhamento em modo dispersão
            # If ghost is at its scatter target, make it patrol around it
            # Se o fantasma está em seu alvo de dispersão, faz com que patrulhe ao redor dele
            if (self.grid_x, self.grid_y) == self.scatter_target:
                # Find valid moves that are not directly reversing current direction
                # Encontra movimentos válidos que não sejam diretamente o reverso da direção atual
                options = [(1,0), (-1,0), (0,1), (0,-1)]
                valid_options = [d for d in options if self.can_move(ORIGINAL_MAP, self.grid_x + d[0], self.grid_y + d[1]) and d != (-self.direction[0], -self.direction[1])]
                if valid_options:
                    # Choose a random valid direction to patrol
                    # Escolhe uma direção válida aleatória para patrulhar
                    self.next_direction = random.choice(valid_options)
                else:
                    # If stuck, allow reversing
                    # Se estiver preso, permite reverter
                    self.next_direction = (-self.direction[0], -self.direction[1])
                return # Exit early, as we've set the next direction for patrolling
            # --- END NEW LOGIC FOR SCATTER PATROLLING ---
        elif self.mode == 'powerup_guard':
            target = self.get_powerup_target(current_game_grid)
            if target is None: # If no power-ups left, revert to scatter
                target = self.scatter_target
        else: # Default to scatter if mode is undefined
            target = self.scatter_target

        # Pathfinding logic
        # Lógica de pathfinding
        # Use ORIGINAL_MAP for pathfinding to avoid issues with disappearing dots
        # Usa ORIGINAL_MAP para pathfinding para evitar problemas com pontos desaparecendo
        path = bfs_find_path(ORIGINAL_MAP, (self.grid_x, self.grid_y), target)
        if len(path) >= 2:
            dx = path[1][0] - self.grid_x
            dy = path[1][1] - self.grid_y
            
            # Prevent immediate 180-degree turns unless necessary (e.g., dead end)
            # Previne viradas de 180 graus imediatas, a menos que seja necessário (ex: beco sem saída)
            if (dx, dy) == (-self.direction[0], -self.direction[1]) and self.direction != (0,0):
                # Check if turning back is the ONLY option
                # Verifica se virar para trás é a ÚNICA opção
                can_move_forward = self.can_move(ORIGINAL_MAP, self.grid_x + self.direction[0], self.grid_y + self.direction[1])
                if can_move_forward: # If we can continue forward, don't turn back
                    # Se pudermos continuar em frente, não viramos para trás
                    # Find other valid options that are not turning back
                    # Encontra outras opções válidas que não sejam virar para trás
                    options = [(1,0), (-1,0), (0,1), (0,-1)]
                    valid_options = [d for d in options if self.can_move(ORIGINAL_MAP, self.grid_x + d[0], self.grid_y + d[1]) and d != (-self.direction[0], -self.direction[1])]
                    if valid_options:
                        # Choose the best valid option (e.g., closest to target)
                        # Escolhe a melhor opção válida (ex: mais próxima do alvo)
                        def dist_to_target(d):
                            nx, ny = self.grid_x + d[0], self.grid_y + d[1]
                            return abs(nx - target[0]) + abs(ny - target[1])
                        self.next_direction = min(valid_options, key=dist_to_target)
                    else: # If no other valid options, then turning back is allowed
                        # Se não houver outras opções válidas, então virar para trás é permitido
                        self.next_direction = (dx, dy)
                else: # If cannot move forward, turning back is the only option
                    # Se não puder mover para frente, virar para trás é a única opção
                    self.next_direction = (dx, dy)
            else:
                self.next_direction = (dx, dy)
        else:
            # If no path found or path is too short, try to move randomly but not backwards
            # Se nenhum caminho for encontrado ou o caminho for muito curto, tenta mover aleatoriamente, mas não para trás
            options = [(1,0), (-1,0), (0,1), (0,-1)]
            valid_options = [d for d in options if self.can_move(ORIGINAL_MAP, self.grid_x + d[0], self.grid_y + d[1]) and d != (-self.direction[0], -self.direction[1])]
            if valid_options:
                self.next_direction = random.choice(valid_options)
            else:
                self.next_direction = (0,0) # Stop if completely stuck

# --- Drawing functions ---
def draw_map(screen, grid):
    """Draws the game map, including walls, dots, powerups, and shrinking dots."""
    """Desenha o mapa do jogo, incluindo paredes, pontos, power-ups e pontos encolhendo."""
    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            px, py = x * CELL_SIZE, y * CELL_SIZE
            if val == '#':
                pygame.draw.rect(screen, WALL_COLOR, (px, py, CELL_SIZE, CELL_SIZE))
            elif val == '.' :
                pygame.draw.circle(screen, DOT_COLOR, (px + CELL_SIZE//2, py + CELL_SIZE//2), 4)
            elif val == 'o':
                pygame.draw.circle(screen, POWERUP_COLOR, (px + CELL_SIZE//2, py + CELL_SIZE//2), 8)
            elif isinstance(val, tuple) and val[0] == 'shrinking_dot':
                timer = val[1]
                radius = max(1, int(4 * (timer / DOT_SHRINK_TIME)))
                pygame.draw.circle(screen, DOT_COLOR, (px + CELL_SIZE//2, py + CELL_SIZE//2), radius)
            # No explicit drawing for '-' (empty space), as BG_COLOR fill handles it
            # Nenhum desenho explícito para '-' (espaço vazio), pois o preenchimento de BG_COLOR lida com isso

def draw_entity(screen, ent):
    """Draws a Pacman or Ghost entity."""
    """Desenha uma entidade Pacman ou Fantasma."""
    if not ent:
        return

    px, py = ent.get_pixel_pos()

    if isinstance(ent, Pacman):
        t = pygame.time.get_ticks() / 150
        open_amt = (math.sin(t) + 1) * 0.25 + 0.1
        
        draw_dx, draw_dy = ent.direction
        if draw_dx == 0 and draw_dy == 0:
            draw_dx, draw_dy = ent.last_valid_direction

        if draw_dx > 0: # Right
            sa, ea = open_amt * math.pi, (2 - open_amt) * math.pi
        elif draw_dx < 0: # Left
            sa, ea = math.pi + open_amt * math.pi, math.pi - open_amt * math.pi
        elif draw_dy > 0: # Down
            sa, ea = 0.5 * math.pi + open_amt * math.pi, 1.5 * math.pi - open_amt * math.pi
        else: # Up (draw_dy < 0)
            sa, ea = 1.5 * math.pi + open_amt * math.pi, 0.5 * math.pi - open_amt * math.pi
        
        pygame.draw.circle(screen, ent.color, (int(px), int(py)), ent.radius)
        mouth = [
            (int(px), int(py)),
            (int(px + ent.radius * math.cos(sa)), int(py + ent.radius * math.sin(sa))),
            (int(px + ent.radius * math.cos(ea)), int(py + ent.radius * math.sin(ea))),
        ]
        pygame.draw.polygon(screen, BG_COLOR, mouth)
    else: # Ghost
        r = ent.radius
        body = pygame.Rect(int(px - r), int(py - r), 2 * r, 2 * r)
        color = VULNERABLE_COLOR if ent.vulnerable else ent.color
        pygame.draw.rect(screen, color, body, border_radius=r // 2)
        
        # Draw eyes
        er = r//3 # Eye radius
        oy = r//3 # Eye vertical offset
        ox = r//2 # Eye horizontal offset
        for dx_eye in (-ox, ox):
            ex, ey = int(px + dx_eye), int(py - oy)
            pygame.draw.circle(screen, (255,255,255), (ex,ey), er) # White part of eye
            
            # Draw pupils based on ghost direction
            dir_x, dir_y = ent.direction
            # Normalize direction to get pupil offset
            offx = 0
            if dir_x != 0: offx = int(er//2 * (dir_x / abs(dir_x)))
            offy = 0
            if dir_y != 0: offy = int(er//2 * (dir_y / abs(dir_y)))
            
            pygame.draw.circle(screen, (0,0,0), (ex+offx, ey+offy), er//2) # Black pupil

def display_score(screen, score, font):
    """Displays the current score."""
    """Exibe a pontuação atual."""
    s = font.render(f"Score: {score}", True, SCORE_TEXT_COLOR)
    screen.blit(s, (SCREEN_W_PADDING, SCREEN_H - BOTTOM_PANEL_HEIGHT + 5))

def display_game_stats(screen, best_score, worst_score, average_score, pacman_wins, ghost_wins, font):
    """Displays best, worst, average score and win/loss counts on two separate lines."""
    """Exibe a melhor, pior, pontuação média e contagens de vitórias/derrotas em duas linhas separadas."""
    
    # Line 1: Best, Worst, Average Score
    line1_text = f"Melhor: {best_score} Pior: {worst_score if worst_score != float('inf') else 0} Média: {average_score:.0f}"
    text1_surf = font.render(line1_text, True, TEXT_COLOR)
    screen.blit(text1_surf, (SCREEN_W - text1_surf.get_width() - SCREEN_W_PADDING, SCREEN_H - BOTTOM_PANEL_HEIGHT + 5))

    # Line 2: Pacman and Ghost Wins
    line2_text = f"🟡 Pacman: {pacman_wins}    👻 Fantasmas: {ghost_wins}"
    text2_surf = font.render(line2_text, True, TEXT_COLOR)
    # Position for Line 2 (below Line 1, adjust Y coordinate)
    screen.blit(text2_surf, (SCREEN_W - text2_surf.get_width() - SCREEN_W_PADDING, SCREEN_H - BOTTOM_PANEL_HEIGHT + 35))

def display_status_message(screen, message, font):
    """Displays a status message in the center of the bottom panel."""
    """Exibe uma mensagem de status no centro do painel inferior."""
    text_surf = font.render(message, True, TEXT_COLOR)
    x = (SCREEN_W - text_surf.get_width()) // 2
    y = SCREEN_H - BOTTOM_PANEL_HEIGHT + STATUS_TEXT_HEIGHT // 2 - text_surf.get_height() // 2
    screen.blit(text_surf, (x, y))

# --- Button UI Component ---
class Button:
    def __init__(self, rect, text, font, action):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.action = action
        self.hover = False

    def draw(self, surface):
        """Draws the button on the given surface."""
        """Desenha o botão na superfície fornecida."""
        color = BUTTON_HOVER_COLOR if self.hover else BUTTON_BG_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        text_surf = self.font.render(self.text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        """Handles Pygame events for button interaction."""
        """Lida com eventos Pygame para interação do botão."""
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.action()

# Global variables for Pacman and Ghosts (needed for Entity.update and Ghost.decide_next_direction)
# Variáveis globais para Pacman e Fantasmas (necessárias para Entity.update e Ghost.decide_next_direction)
pacman = None
ghosts = []

def main():
    # Declare pacman and ghosts as global within main to ensure they are accessible
    # and modifiable by functions like reset_game_state and Entity.update
    # Declara pacman e ghosts como globais dentro de main para garantir que sejam acessíveis
    # e modificáveis por funções como reset_game_state e Entity.update
    global pacman, ghosts 

    pygame.init()

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Pacman Game")

    font = pygame.font.Font(FONT_NAME, 24)
    clock = pygame.time.Clock()

    # Game state variables (local to main, but some are modified by nested functions via nonlocal)
    # Variáveis de estado do jogo (locais a main, mas algumas são modificadas por funções aninhadas via nonlocal)
    grid = None # Will be initialized in reset_game_state
    last_ai_update_time = time.time()
    ai_delay = BASE_AI_DELAY
    paused = False
    score = 0
    game_over = False
    victory = False
    
    # Game statistics for charting (local to main, modified by nested functions via nonlocal)
    # Estatísticas do jogo para gráficos (locais a main, modificadas por funções aninhadas via nonlocal)
    current_game_time_survived = 0.0
    current_game_ghost_captures = 0
    current_game_powerup_time = 0.0
    
    # Load game progress (for best score and win/loss counts)
    # Carrega o progresso do jogo (para melhor pontuação e contagens de vitórias/derrotas)
    state_data = load_dump()
    best_score = state_data.get('best_score', 0)
    worst_score = state_data.get('worst_score', float('inf'))
    average_score = state_data.get('average_score', 0)
    pacman_wins = state_data.get('pacman_wins', 0)
    ghost_wins = state_data.get('ghost_wins', 0)
    
    # Data for charting history (local to main, modified by nested functions via nonlocal)
    # Dados para histórico de gráficos (locais a main, modificados por funções aninhadas via nonlocal)
    game_scores_history = []
    game_times_survived_history = []
    ghost_captures_history = []
    powerup_durations_history = []

    # Ghost AI mode management (local to main, modified by nested functions via nonlocal)
    # Gerenciamento de modo de IA de Fantasma (local a main, modificado por funções aninhadas via nonlocal)
    # Removed fixed sequence for dynamic mode choice
    # Removida sequência fixa para escolha de modo dinâmico
    
    def calculate_ghost_mode_weights(ghost_inst, pacman_inst, current_game_grid):
        """Calculates weights for ghost mode choices based on distances, with aggression tuning."""
        """Calcula pesos para as escolhas de modo do fantasma com base nas distâncias, com ajuste de agressividade."""
        weights = {}
        
        # Chase Pacman
        dist_to_pacman = abs(ghost_inst.grid_x - pacman_inst.grid_x) + abs(ghost_inst.grid_y - pacman_inst.grid_y)
        # Increase chase priority by multiplying its weight
        # Aumenta a prioridade de perseguição multiplicando seu peso
        weights['chase'] = (1.0 / (dist_to_pacman + 1)) * AGRESSION_FACTOR 
        
        # Scatter to corner
        dist_to_scatter = abs(ghost_inst.grid_x - ghost_inst.scatter_target[0]) + abs(ghost_inst.grid_y - ghost_inst.scatter_target[1])
        weights['scatter'] = 1.0 / (dist_to_scatter + 1)
        
        # Go for power-up (guard it)
        powerup_target = ghost_inst.get_powerup_target(current_game_grid)
        if powerup_target:
            dist_to_powerup = abs(ghost_inst.grid_x - powerup_target[0]) + abs(ghost_inst.grid_y - powerup_target[1])
            weights['powerup_guard'] = 1.0 / (dist_to_powerup + 1)
        else:
            weights['powerup_guard'] = 0.0 # No power-ups left
            
        # Normalize weights
        # Normaliza os pesos
        total_weight = sum(weights.values())
        if total_weight == 0: # Avoid division by zero if all weights are 0
            return {'chase': 1.0, 'scatter': 0.0, 'powerup_guard': 0.0} # Default to chase
        
        normalized_weights = {mode: weight / total_weight for mode, weight in weights.items()}
        return normalized_weights

    def update_ghost_modes_dynamically():
        """Dynamically updates ghost modes based on weighted choices, with commitment."""
        """Atualiza dinamicamente os modos dos fantasmas com base em escolhas ponderadas, com compromisso."""
        
        current_time = time.time() # Get current time once for efficiency

        # If Pacman is powered up, all ghosts are frightened (this overrides commitment)
        # Se o Pacman estiver com power-up, todos os fantasmas ficam assustados (isso anula o compromisso)
        if pacman.powerup_timer > 0:
            for g in ghosts:
                if g.mode != 'frightened': # Only update if not already frightened
                    g.mode = 'frightened'
                    g.current_committed_mode = 'frightened' # Commit to frightened
                    g.mode_start_time = current_time
            return

        # Otherwise, each ghost decides its mode based on commitment
        # Caso contrário, cada fantasma decide seu modo com base no compromisso
        for g in ghosts:
            # If ghost was just eaten and returning to original_pos, keep it in 'frightened' until it reaches home
            # Se o fantasma acabou de ser comido e está retornando à original_pos, mantém-no em 'frightened' até chegar em casa
            if g.vulnerable and g.frightened_timer <= 0: # Ghost was eaten and timer ran out, but still vulnerable
                g.vulnerable = False # Ensure it's no longer vulnerable
                
            if g.mode == 'frightened': # If still frightened, don't change mode yet
                continue # Frightened mode is handled above and overrides normal decision-making

            # Check if commitment duration has passed or if the ghost is currently not committed
            # Verifica se a duração do compromisso passou ou se o fantasma não está atualmente comprometido
            # Also re-evaluate if coming out of 'frightened' mode
            # Também reavalia se está saindo do modo 'frightened'
            if (current_time - g.mode_start_time >= GHOST_MODE_COMMITMENT_DURATION) or \
               (g.current_committed_mode == 'frightened' and pacman.powerup_timer <= 0): 
                
                # Re-evaluate mode weights
                # Reavalia os pesos do modo
                weights = calculate_ghost_mode_weights(g, pacman, grid)
                
                # Choose mode based on weighted random selection
                # Escolhe o modo com base na seleção aleatória ponderada
                modes = list(weights.keys())
                mode_probabilities = list(weights.values())
                
                chosen_mode = random.choices(modes, weights=mode_probabilities, k=1)[0]
                
                # Commit to the new chosen mode
                # Compromete-se com o novo modo escolhido
                g.current_committed_mode = chosen_mode
                g.mode_start_time = current_time
                g.last_evaluated_mode = chosen_mode # Store for debugging/analysis if needed
                # print(f"Ghost {g.color} re-evaluated and committed to: {chosen_mode} (weights: {weights})") # For debugging
            
            # Apply the currently committed mode
            # Aplica o modo atualmente comprometido
            g.mode = g.current_committed_mode

    def plot_game_analytics():
        """Generates and displays matplotlib plots for game analytics."""
        """Gera e exibe gráficos matplotlib para análise do jogo."""
        if not game_scores_history:
            print("No game data to plot.")
            return

        num_games = range(1, len(game_scores_history) + 1)

        plt.figure(figsize=(18, 8)) # Increased figure size for 3 subplots
        plt.suptitle('Análise de Desempenho do Jogo Pacman', fontsize=16) # Main title

        # Subplot 1: Scores (Best, Worst, Average)
        plt.subplot(1, 3, 1) # 1 row, 3 columns, first plot
        plt.plot(num_games, game_scores_history, marker='o', linestyle='-', color='blue', label='Pontuação por Jogo')
        
        # Calculate running best, worst, and average
        # Calcula o melhor, pior e média acumulados
        running_best = [max(game_scores_history[:i+1]) for i in range(len(game_scores_history))]
        running_worst = [min(game_scores_history[:i+1]) for i in range(len(game_scores_history))]
        running_average = [sum(game_scores_history[:i+1]) / (i+1) for i in range(len(game_scores_history))]

        plt.plot(num_games, running_best, marker='^', linestyle='--', color='gold', label='Melhor Pontuação (Acum.)')
        plt.plot(num_games, running_worst, marker='v', linestyle='--', color='red', label='Pior Pontuação (Acum.)')
        plt.plot(num_games, running_average, marker='s', linestyle='--', color='purple', label='Pontuação Média (Acum.)')
        
        plt.xlabel('Número do Jogo')
        plt.ylabel('Pontuação')
        plt.title('Pontuações ao Longo dos Jogos')
        plt.legend()
        plt.grid(True)

        # Subplot 2: Survival Times
        plt.subplot(1, 3, 2) # 1 row, 3 columns, second plot
        plt.plot(num_games, game_times_survived_history, marker='x', linestyle='-', color='green', label='Tempo de Sobrevivência (s)')
        plt.xlabel('Número do Jogo')
        plt.ylabel('Tempo (s)')
        plt.title('Tempos de Sobrevivência')
        plt.legend()
        plt.grid(True)

        # Subplot 3: Ghost Captures and Power-up Durations
        plt.subplot(1, 3, 3) # 1 row, 3 columns, third plot
        plt.plot(num_games, ghost_captures_history, marker='s', linestyle='-', color='cyan', label='Fantasmas Capturados')
        plt.plot(num_games, powerup_durations_history, marker='d', linestyle='-', color='orange', label='Duração Total Power-up (s)')
        plt.xlabel('Número do Jogo')
        plt.ylabel('Contagem / Tempo (s)')
        plt.title('Interações com Fantasmas e Power-ups')
        plt.legend()
        plt.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap, make space for suptitle
        plt.savefig('analytics.png') # Save the figure
        plt.show()

    def reset_game_state():
        # Declare variables from the enclosing scope (main) that this function modifies
        # Declara variáveis do escopo envolvente (main) que esta função modifica
        nonlocal grid, last_ai_update_time, score, paused, game_over, victory, \
                   best_score, worst_score, average_score, pacman_wins, ghost_wins, \
                   current_game_time_survived, current_game_ghost_captures, current_game_powerup_time, \
                   game_scores_history, game_times_survived_history, \
                   ghost_captures_history, powerup_durations_history
        
        # Declare global variables that this function modifies
        # Declara variáveis globais que esta função modifica
        global pacman, ghosts 

        # Record data for the just-finished game (if it was a full game)
        # Registra dados do jogo recém-terminado (se foi um jogo completo)
        if game_over or victory: # Only record if a game actually finished
            game_scores_history.append(score)
            game_times_survived_history.append(current_game_time_survived)
            ghost_captures_history.append(current_game_ghost_captures)
            powerup_durations_history.append(current_game_powerup_time)

            # Update best, worst, average scores
            # Atualiza as pontuações melhor, pior e média
            best_score = max(best_score, score)
            worst_score = min(worst_score, score)
            if len(game_scores_history) > 0:
                average_score = sum(game_scores_history) / len(game_scores_history)
            else:
                average_score = 0

        # Save all persistent stats
        # Salva todas as estatísticas persistentes
        save_dump({
            'best_score': best_score,
            'worst_score': worst_score,
            'average_score': average_score,
            'pacman_wins': pacman_wins,
            'ghost_wins': ghost_wins,
        })

        score = 0
        paused = False
        game_over = False
        victory = False
        last_ai_update_time = time.time()
        current_game_time_survived = 0.0
        current_game_ghost_captures = 0 # Reset per-game metrics
        current_game_powerup_time = 0.0 # Reset per-game metrics
        
        # Reset ghost mode sequence (not used with dynamic mode, but reset for consistency)
        # Reinicia a sequência de modo do fantasma (não usada com modo dinâmico, mas reiniciada para consistência)
        
        grid = copy.deepcopy(ORIGINAL_MAP) # Assign directly to the nonlocal 'grid' variable

        # Pacman spawn and ghost spawns setup
        # Configuração de spawn do Pacman e dos fantasmas
        pacman_spawn = find_pacman_spawn(grid) # Use the newly created grid
        pacman = Pacman(pacman_spawn[0], pacman_spawn[1], grid) 

        ghost_spawns = find_spawn_for_ghosts(grid)
        ghosts = [] # Clear existing ghosts
        for i, c in enumerate(GHOST_COLORS):
            spawn = ghost_spawns[i % len(ghost_spawns)]
            # Assign specific scatter targets for each ghost for distinct personalities
            # Atribui alvos de dispersão específicos para cada fantasma para personalidades distintas
            if i == 0: # Blinky (top-right)
                scatter_tgt = (GRID_W - 2, 1)
            elif i == 1: # Pinky (top-left)
                scatter_tgt = (1, 1)
            elif i == 2: # Inky (bottom-right)
                scatter_tgt = (GRID_W - 2, GRID_H - 2)
            elif i == 3: # Clyde (bottom-left)
                scatter_tgt = (1, GRID_H - 2)
            else:
                scatter_tgt = (random.choice([1, GRID_W-2]), random.choice([1, GRID_H-2]))
            ghosts.append(Ghost(spawn[0], spawn[1], c, scatter_tgt))
        
        print(f"Novo Jogo Iniciado. Vitórias Pacman: {pacman_wins}, Vitórias Fantasmas: {ghost_wins}, Melhor Pontuação: {best_score}")
        
        # No return value needed for grid, as it's modified directly via nonlocal

    # Initial game setup
    reset_game_state() # Call reset_game_state to initialize all game variables

    def toggle_pause():
        nonlocal paused
        paused = not paused

    def toggle_speed():
        nonlocal ai_delay
        ai_delay = FAST_AI_DELAY if ai_delay == BASE_AI_DELAY else BASE_AI_DELAY

    # Buttons UI layout
    # Layout da interface dos botões
    btn_width = 140
    btn_height = 38
    # Adjusted total buttons for "Stats" button
    # Botões totais ajustados para o botão "Estatísticas"
    total_buttons = 3 
    total_width = total_buttons * btn_width + (total_buttons -1) * BUTTON_SPACING
    btn_x_start = (SCREEN_W - total_width) // 2
    btn_y = SCREEN_H - BUTTON_AREA_HEIGHT + (BUTTON_AREA_HEIGHT - btn_height)//2 + STATUS_TEXT_HEIGHT

    buttons = [
        Button((btn_x_start, btn_y, btn_width, btn_height), 'Pausar', font, toggle_pause),
        Button((btn_x_start + (btn_width + BUTTON_SPACING), btn_y, btn_width, btn_height), 'Novo Jogo', font, reset_game_state),
        Button((btn_x_start + 2 * (btn_width + BUTTON_SPACING), btn_y, btn_width, btn_height), 'Estatísticas', font, plot_game_analytics)
    ]

    dot_shrink_timers = dict()

    while True:
        dt = clock.tick(FPS) / 1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Plotting the analytics before quitting
                # Plotando as análises antes de sair
                plot_game_analytics() # Call plot function on quit
                
                save_dump({
                    'best_score': best_score,
                    'worst_score': worst_score,
                    'average_score': average_score,
                    'pacman_wins': pacman_wins,
                    'ghost_wins': ghost_wins,
                })
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    toggle_pause()
                elif event.key == pygame.K_SPACE:
                    toggle_speed()
                elif event.key == pygame.K_ESCAPE:
                    # Trigger quit logic including plotting
                    # Aciona a lógica de saída, incluindo o plot
                    event = pygame.event.Event(pygame.QUIT)
                    pygame.event.post(event)
                
                # Manual Pac-Man control
                # Controle manual do Pac-Man
                if not paused and not game_over and not victory:
                    if event.key == pygame.K_w:
                        pacman.set_manual_direction((0, -1))
                    elif event.key == pygame.K_s:
                        pacman.set_manual_direction((0, 1))
                    elif event.key == pygame.K_a:
                        pacman.set_manual_direction((-1, 0))
                    elif event.key == pygame.K_d:
                        pacman.set_manual_direction((1, 0))
            elif event.type == pygame.KEYUP:
                pass # No action on KEYUP for movement
                # Nenhuma ação em KEYUP para movimento

            for btn in buttons:
                btn.handle_event(event)

        if not paused and not game_over and not victory:
            current_game_time_survived += dt
            
            # Adjust Pacman speed if eating dots
            # Ajusta a velocidade do Pacman se estiver comendo pontos
            pacman.speed = PACMAN_BASE_SPEED
            if grid[pacman.grid_y][pacman.grid_x] == '.' or grid[pacman.grid_y][pacman.grid_x] == 'o':
                pacman.speed *= PACMAN_DOT_EATING_SPEED_MULTIPLIER

            # Adaptive Ghost Speed based on score
            # Velocidade Adaptativa do Fantasma baseada na pontuação
            score_multiplier = score // 1000 # Integer division for every 1000 points
            ghost_current_base_speed = GHOST_BASE_SPEED * (1 + score_multiplier * GHOST_SPEED_INCREASE_PER_1000_POINTS)
            for g in ghosts:
                g.speed = ghost_current_base_speed

            # Pacman update
            # Atualização do Pacman
            pacman.update(dt, grid)

            # Ghost AI update (controlled by ai_delay)
            # Atualização da IA do Fantasma (controlada por ai_delay)
            now = time.time()
            if (now - last_ai_update_time) >= ai_delay:
                last_ai_update_time = now
                # Ghosts' decide_next_direction is called within their update method
                # decide_next_direction dos fantasmas é chamado dentro do método update deles
            
            # Update ghost modes dynamically
            # Atualiza os modos dos fantasmas dinamicamente
            update_ghost_modes_dynamically()

            # Update ghosts' positions
            # Atualiza as posições dos fantasmas
            for g in ghosts:
                g.update(dt, grid)

            # Dot collection + shrink animation
            # Coleta de pontos + animação de encolhimento
            cell_at_pacman_pos = grid[pacman.grid_y][pacman.grid_x]
            if cell_at_pacman_pos == '.':
                dot_shrink_timers[(pacman.grid_x, pacman.grid_y)] = DOT_SHRINK_TIME
                score += 10
                grid[pacman.grid_y][pacman.grid_x] = ('shrinking_dot', DOT_SHRINK_TIME)
            elif cell_at_pacman_pos == 'o':
                pacman.powerup_timer = POWERUP_DURATION
                dot_shrink_timers[(pacman.grid_x, pacman.grid_y)] = DOT_SHRINK_TIME
                score += 50
                grid[pacman.grid_y][pacman.grid_x] = ('shrinking_dot', DOT_SHRINK_TIME)
                print(f"Power-up ativado! Duração: {POWERUP_DURATION}s")
            
            # Decrement powerup timer and ghost frightened timer
            # Decrementa o temporizador de power-up e o temporizador de fantasma assustado
            if pacman.powerup_timer > 0:
                current_game_powerup_time += dt # Accumulate power-up time
                pacman.powerup_timer -= dt
                if pacman.powerup_timer < 0:
                    pacman.powerup_timer = 0
                    # When powerup ends, ghosts are no longer vulnerable
                    # Quando o power-up termina, os fantasmas não são mais vulneráveis
                    for g in ghosts:
                        g.vulnerable = False
                        g.frightened_timer = 0 # Reset ghost's individual frightened timer
            
            # Update individual ghost frightened timers (if they were eaten or powerup ended)
            # Atualiza os temporizadores individuais de fantasma assustado (se foram comidos ou o power-up terminou)
            for g in ghosts:
                if g.frightened_timer > 0:
                    g.frightened_timer -= dt
                    if g.frightened_timer < 0:
                        g.frightened_timer = 0
                        g.vulnerable = False # Ensure vulnerability is off

            # Update shrinking dots
            # Atualiza os pontos encolhendo
            to_remove = []
            for pos in dot_shrink_timers:
                x, y = pos
                timer = dot_shrink_timers[pos] - dt
                if timer <= 0:
                    grid[y][x] = '-' # Dot is fully gone
                    to_remove.append(pos)
                else:
                    dot_shrink_timers[pos] = timer
                    grid[y][x] = ('shrinking_dot', timer)
            for pos in to_remove:
                del dot_shrink_timers[pos]

            # Collision detection (pixel-based)
            # Detecção de colisão (baseada em pixels)
            for g in ghosts:
                if is_collision(pacman, g):
                    if pacman.powerup_timer > 0 and g.vulnerable:
                        score += 150
                        current_game_ghost_captures += 1 # Increment capture count
                        g.grid_x, g.grid_y = g.original_pos[0], g.original_pos[1]
                        g.direction = (0, 0)
                        g.next_direction = (0, 0)
                        g.vulnerable = False
                        g.frightened_timer = 0 # Reset frightened timer for eaten ghost
                        # --- NEW: Set flag for returning to house ---
                        # Novo: Define a flag para retornar à casa
                        g.returning_to_house = True 
                        # --- END NEW ---
                    else:
                        game_over = True
                        ghost_wins += 1
                        print(f"Fim de Jogo: Pacman capturado! Pontuação: {score}, Tempo de Sobrevivência: {current_game_time_survived:.2f}s")
                        break # Break from this loop as game is over

            # If game is over due to collision, reset and continue to next frame
            # Se o jogo terminou devido a colisão, reinicia e continua para o próximo frame
            if game_over:
                pygame.time.wait(2000) # Small delay to see end state
                reset_game_state() # Reset all game variables
                continue # Skip rendering for this frame and go to next loop iteration

            # Victory condition: no remaining dots or powerups or shrinking dots
            # Condição de vitória: nenhum ponto restante ou power-ups ou pontos encolhendo
            dots_left = False
            for row in grid:
                for cell in row:
                    if cell == '.' or cell == 'o' or (isinstance(cell, tuple) and cell[0] == 'shrinking_dot'):
                        dots_left = True
                        break
                if dots_left:
                    break
            
            if not dots_left:
                victory = True
                pacman_wins += 1
                print(f"Vitória! Todos os pontos coletados! Pontuação: {score}, Tempo de Sobrevivência: {current_game_time_survived:.2f}s")

            # If victory, reset for a new round
            # Se vitória, reinicia para uma nova rodada
            if victory:
                pygame.time.wait(2000) # Small delay to see end state
                reset_game_state() # Reset all game variables
                continue # Skip rendering for this frame and go to next loop iteration

        screen.fill(BG_COLOR)
        draw_map(screen, grid)
        draw_entity(screen, pacman)
        for g in ghosts:
            draw_entity(screen, g)
        display_score(screen, score, font)
        display_game_stats(screen, best_score, worst_score, average_score, pacman_wins, ghost_wins, font)

        for btn in buttons:
            btn.draw(screen)

        # Display status messages above buttons
        # Exibe mensagens de status acima dos botões
        status_msg = None
        if paused:
            status_msg = "Pausado"
        elif game_over:
            status_msg = "Fim de Jogo! Pressione Novo Jogo"
        elif victory:
            status_msg = "Vitória! Pressione Novo Jogo"

        if status_msg:
            display_status_message(screen, status_msg, font)

        pygame.display.flip()

if __name__ == "__main__":
    main()