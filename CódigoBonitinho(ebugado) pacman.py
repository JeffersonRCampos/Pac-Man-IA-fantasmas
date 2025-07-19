import pygame
import sys
import time
import random
import pickle
from collections import deque
import math
import copy
import matplotlib.pyplot as plt


BG_COLOR = (0, 0, 0)
WALL_COLOR = (0, 0, 255)
DOT_COLOR = (255, 255, 255)
POWERUP_COLOR = (0, 255, 0)
PACMAN_COLOR = (234, 179, 8)
GHOST_COLORS = [(239, 68, 68), (219, 39, 119), (14, 165, 233), (249, 115, 22)]
VULNERABLE_COLOR = (100, 100, 255) 
TEXT_COLOR = (255, 255, 255)
SCORE_TEXT_COLOR = (255, 255, 255)
FONT_NAME = 'freesansbold.ttf'
CELL_SIZE = 20
FPS = 60
BASE_AI_DELAY = 0.2  
FAST_AI_DELAY = 0.04
POWERUP_DURATION = 6.0


PACMAN_BASE_SPEED = 4.5 # velocidade jogador
GHOST_BASE_SPEED = 3.0  # Velocidade Bichin
PACMAN_DOT_EATING_SPEED_MULTIPLIER = 0.8 
GHOST_SPEED_INCREASE_PER_1000_POINTS = 0.1 


AGRESSION_FACTOR = 1.5 # Ia nível agresão
FLEE_WEIGHT_REDUCTION_FACTOR = 0.3
                                
                              


GHOST_MODE_COMMITMENT_DURATION = 3.0 
                                    


DOT_SHRINK_TIME = 0.3  

BUTTON_BG_COLOR = (50, 50, 50)
BUTTON_HOVER_COLOR = (100, 100, 100)
BUTTON_TEXT_COLOR = (220, 220, 220)
BUTTON_PADDING_X = 20
BUTTON_PADDING_Y = 8
BUTTON_SPACING = 18


BOTTOM_PANEL_HEIGHT = 130 
STATUS_TEXT_HEIGHT = 30
BUTTON_AREA_HEIGHT = BOTTOM_PANEL_HEIGHT - STATUS_TEXT_HEIGHT
SCREEN_W_PADDING = 20


DUMP_FILE = 'game_stats.pkl' 

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

#mapa
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


def is_collision(entity1, entity2, threshold=14):
    x1, y1 = entity1.get_pixel_pos()
    x2, y2 = entity2.get_pixel_pos()
    return math.hypot(x1 - x2, y1 - y2) < threshold

def find_spawn_for_ghosts(grid):

    return [(13,14), (14,14), (13,15), (14,15)]

def find_pacman_spawn(grid):
    return (13, 23) 

def bfs_find_path(grid_map, start, goal):
    q = deque([(start, [start])])
    vis = set()
    
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
            if 0 <= nx < GRID_W and 0 <= ny < GRID_H and grid_map[ny][nx] != '#' and (nx, ny) not in vis:
                q.append(((nx, ny), path + [(nx, ny)]))
    return []


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
        return 0 <= x < GRID_W and 0 <= y < GRID_H and grid_map[y][x] != '#'

    def at_center(self):
        tolerance = self.speed * CELL_SIZE * 0.15
        return self.move_progress < tolerance or self.move_progress > (CELL_SIZE - tolerance)

    def update(self, dt, current_game_grid):
        if self.at_center():
            nx_next, ny_next = self.grid_x + self.next_direction[0], self.grid_y + self.next_direction[1]
            if self.can_move(ORIGINAL_MAP, nx_next, ny_next):
                self.direction = self.next_direction
            else:
                nx_curr, ny_curr = self.grid_x + self.direction[0], self.grid_y + self.direction[1]
                if not self.can_move(ORIGINAL_MAP, nx_curr, ny_curr):
                    self.direction = (0, 0)
                    self.move_progress = 0
        
        if self.direction == (0,0) and self.next_direction != (0,0):
            nx_next, ny_next = self.grid_x + self.next_direction[0], self.grid_y + self.next_direction[1]
            if self.can_move(ORIGINAL_MAP, nx_next, ny_next):
                self.direction = self.next_direction
                self.move_progress = 0

        self.move_progress += dt * self.speed * CELL_SIZE

        while self.move_progress >= CELL_SIZE:
            nx = self.grid_x + self.direction[0]
            ny = self.grid_y + self.direction[1]
            if self.can_move(ORIGINAL_MAP, nx, ny):
                self.grid_x, self.grid_y = nx, ny
                self.move_progress -= CELL_SIZE
            else:
                self.direction = (0, 0)
                self.move_progress = 0
                break

        if not isinstance(self, Pacman):
            self.decide_next_direction(current_game_grid, pacman, ghosts)

    def get_pixel_pos(self):
        off = self.move_progress / CELL_SIZE
        px = (self.grid_x + self.direction[0] * off) * CELL_SIZE + CELL_SIZE // 2
        py = (self.grid_y + self.direction[1] * off) * CELL_SIZE + CELL_SIZE // 2
        return px, py

class Pacman(Entity):
    def __init__(self, x, y, grid_ref):
        super().__init__(x, y, PACMAN_COLOR, PACMAN_BASE_SPEED)
        self.powerup_timer = 0
        self.next_direction = (0, 0)
        self.last_valid_direction = (0, -1)
        self.grid_ref = grid_ref

    def set_manual_direction(self, new_direction):
        self.next_direction = new_direction
    
        if self.direction == (0,0):
            self.direction = new_direction
            self.move_progress = 0 
        elif new_direction != (0,0) and \
             (new_direction[0] != -self.direction[0] or new_direction[1] != -self.direction[1]):
            nx, ny = self.grid_x + new_direction[0], self.grid_y + new_direction[1]
            if self.can_move(ORIGINAL_MAP, nx, ny):
                self.direction = new_direction
                self.move_progress = 0
        
        if new_direction != (0,0):
            self.last_valid_direction = new_direction


class Ghost(Entity):
    def __init__(self, x, y, color, scatter_target):
        super().__init__(x, y, color, GHOST_BASE_SPEED)
        self.scatter_target = scatter_target
        self.vulnerable = False
        self.original_pos = (x, y)
        self.mode = 'scatter'
        self.frightened_timer = 0.0

        self.current_committed_mode = 'scatter'
        self.mode_start_time = time.time()
        self.last_evaluated_mode = 'scatter'
        self.returning_to_house = False 

    def get_blinky_target(self, pacman_inst):
        return (pacman_inst.grid_x, pacman_inst.grid_y)

    def get_pinky_target(self, pacman_inst):
        target_x = pacman_inst.grid_x + pacman_inst.direction[0] * 4
        target_y = pacman_inst.grid_y + pacman_inst.direction[1] * 4
        if pacman_inst.direction == (0, -1):
            target_x -= 4

        if not (0 <= target_x < GRID_W and 0 <= target_y < GRID_H and ORIGINAL_MAP[target_y][target_x] != '#'):
            return (pacman_inst.grid_x, pacman_inst.grid_y) 
        return (target_x, target_y)

    def get_inky_target(self, pacman_inst, blinky_inst):
        pacman_ahead_x = pacman_inst.grid_x + pacman_inst.direction[0] * 2
        pacman_ahead_y = pacman_inst.grid_y + pacman_inst.direction[1] * 2

        if pacman_inst.direction == (0, -1):
            pacman_ahead_x -= 2 

        vector_x = pacman_ahead_x - blinky_inst.grid_x
        vector_y = pacman_ahead_y - blinky_inst.grid_y

        target_x = blinky_inst.grid_x + vector_x * 2
        target_y = blinky_inst.grid_y + vector_y * 2

        if not (0 <= target_x < GRID_W and 0 <= target_y < GRID_H and ORIGINAL_MAP[target_y][target_x] != '#'):
            return (pacman_inst.grid_x, pacman_inst.grid_y) 
        return (target_x, target_y)

    def get_clyde_target(self, pacman_inst):
        distance_to_pacman = abs(self.grid_x - pacman_inst.grid_x) + abs(self.grid_y - pacman_inst.grid_y)
        if distance_to_pacman > 8:
            return (pacman_inst.grid_x, pacman_inst.grid_y)
        else: 
            return self.scatter_target

    def get_powerup_target(self, current_game_grid):
        powerup_locations = []
        for y, row in enumerate(current_game_grid):
            for x, val in enumerate(row):
                if val == 'o':
                    powerup_locations.append((x, y))
        
        if not powerup_locations:
            return None
        
        closest_powerup = None
        min_dist = float('inf')
        for px, py in powerup_locations:
            dist = abs(self.grid_x - px) + abs(self.grid_y - py)
            if dist < min_dist:
                min_dist = dist
                closest_powerup = (px, py)
        return closest_powerup

    def decide_next_direction(self, current_game_grid, pacman_inst, all_ghosts):
        if pacman_inst.powerup_timer > 0:
            self.mode = 'frightened'
            self.vulnerable = True
            self.frightened_timer = pacman_inst.powerup_timer
        elif self.frightened_timer > 0:
            self.mode = 'frightened'
            self.vulnerable = True
        else:
            self.vulnerable = False

        target = None
        if self.returning_to_house:
            target = self.original_pos
            if (self.grid_x, self.grid_y) == self.original_pos:
                self.returning_to_house = False
                self.mode = 'scatter'
                self.current_committed_mode = 'scatter'
                self.mode_start_time = time.time()
                exit_path = bfs_find_path(ORIGINAL_MAP, self.original_pos, (13, 11)) 
                if len(exit_path) >= 2:
                    dx = exit_path[1][0] - self.grid_x
                    dy = exit_path[1][1] - self.grid_y
                    self.next_direction = (dx, dy)
                else:
                    options = [(1,0), (-1,0), (0,1), (0,-1)]
                    valid_options = [d for d in options if self.can_move(ORIGINAL_MAP, self.grid_x + d[0], self.grid_y + d[1])]
                    if valid_options:
                        self.next_direction = random.choice(valid_options)
                    else:
                        self.next_direction = (0,0)
                return

        if self.mode == 'frightened':
            flee_options = []
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = self.grid_x + dx, self.grid_y + dy
                if self.can_move(ORIGINAL_MAP, nx, ny):
                    dist_to_pacman = abs(nx - pacman_inst.grid_x) + abs(ny - pacman_inst.grid_y)
                    flee_options.append(((nx, ny), dist_to_pacman))
            
            if flee_options:
                if random.random() < FLEE_WEIGHT_REDUCTION_FACTOR: 
                    valid_flee_moves = [
                        (dx, dy) for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                        if self.can_move(ORIGINAL_MAP, self.grid_x + dx, self.grid_y + dy) and
                           (self.grid_x + dx, self.grid_y + dy) != (pacman_inst.grid_x, pacman_inst.grid_y)
                    ]
                    if valid_flee_moves:
                        chosen_move = random.choice(valid_flee_moves)
                        target = (self.grid_x + chosen_move[0], self.grid_y + chosen_move[1])
                    else: 
                        target_pos_flee = max(flee_options, key=lambda item: item[1])[0]
                        target = target_pos_flee
                else:
                    target_pos_flee = max(flee_options, key=lambda item: item[1])[0]
                    target = target_pos_flee
            else:
                target = self.scatter_target
        elif self.mode == 'chase':
            if self.color == GHOST_COLORS[0]: 
                target = self.get_blinky_target(pacman_inst)
            elif self.color == GHOST_COLORS[1]:
                target = self.get_pinky_target(pacman_inst)
            elif self.color == GHOST_COLORS[2]:
                blinky_ghost = next((g for g in all_ghosts if g.color == GHOST_COLORS[0]), None)
                if blinky_ghost:
                    target = self.get_inky_target(pacman_inst, blinky_ghost)
                else:
                    target = self.get_blinky_target(pacman_inst)
            elif self.color == GHOST_COLORS[3]:
                target = self.get_clyde_target(pacman_inst)
            else:
                target = (pacman_inst.grid_x, pacman_inst.grid_y)
        elif self.mode == 'scatter':
            target = self.scatter_target
            if (self.grid_x, self.grid_y) == self.scatter_target:
                options = [(1,0), (-1,0), (0,1), (0,-1)]
                valid_options = [d for d in options if self.can_move(ORIGINAL_MAP, self.grid_x + d[0], self.grid_y + d[1]) and d != (-self.direction[0], -self.direction[1])]
                if valid_options:
                    self.next_direction = random.choice(valid_options)
                else:
                    self.next_direction = (-self.direction[0], -self.direction[1])
                return
        elif self.mode == 'powerup_guard':
            target = self.get_powerup_target(current_game_grid)
            if target is None:
                target = self.scatter_target
        else:
            target = self.scatter_target

        path = bfs_find_path(ORIGINAL_MAP, (self.grid_x, self.grid_y), target)
        if len(path) >= 2:
            dx = path[1][0] - self.grid_x
            dy = path[1][1] - self.grid_y
            
            if (dx, dy) == (-self.direction[0], -self.direction[1]) and self.direction != (0,0):
                can_move_forward = self.can_move(ORIGINAL_MAP, self.grid_x + self.direction[0], self.grid_y + self.direction[1])
                if can_move_forward:
                    options = [(1,0), (-1,0), (0,1), (0,-1)]
                    valid_options = [d for d in options if self.can_move(ORIGINAL_MAP, self.grid_x + d[0], self.grid_y + d[1]) and d != (-self.direction[0], -self.direction[1])]
                    if valid_options:
                        def dist_to_target(d):
                            nx, ny = self.grid_x + d[0], self.grid_y + d[1]
                            return abs(nx - target[0]) + abs(ny - target[1])
                        self.next_direction = min(valid_options, key=dist_to_target)
                    else:
                        self.next_direction = (dx, dy)
                else:
                    self.next_direction = (dx, dy)
            else:
                self.next_direction = (dx, dy)
        else:
            options = [(1,0), (-1,0), (0,1), (0,-1)]
            valid_options = [d for d in options if self.can_move(ORIGINAL_MAP, self.grid_x + d[0], self.grid_y + d[1]) and d != (-self.direction[0], -self.direction[1])]
            if valid_options:
                self.next_direction = random.choice(valid_options)
            else:
                self.next_direction = (0,0)

def draw_map(screen, grid):
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

def draw_entity(screen, ent):
    if not ent:
        return

    px, py = ent.get_pixel_pos()

    if isinstance(ent, Pacman):
        t = pygame.time.get_ticks() / 150
        open_amt = (math.sin(t) + 1) * 0.25 + 0.1
        
        draw_dx, draw_dy = ent.direction
        if draw_dx == 0 and draw_dy == 0:
            draw_dx, draw_dy = ent.last_valid_direction

        if draw_dx > 0:
            sa, ea = open_amt * math.pi, (2 - open_amt) * math.pi
        elif draw_dx < 0:
            sa, ea = math.pi + open_amt * math.pi, math.pi - open_amt * math.pi
        elif draw_dy > 0:
            sa, ea = 0.5 * math.pi + open_amt * math.pi, 1.5 * math.pi - open_amt * math.pi
        else:
            sa, ea = 1.5 * math.pi + open_amt * math.pi, 0.5 * math.pi - open_amt * math.pi
        
        pygame.draw.circle(screen, ent.color, (int(px), int(py)), ent.radius)
        mouth = [
            (int(px), int(py)),
            (int(px + ent.radius * math.cos(sa)), int(py + ent.radius * math.sin(sa))),
            (int(px + ent.radius * math.cos(ea)), int(py + ent.radius * math.sin(ea))),
        ]
        pygame.draw.polygon(screen, BG_COLOR, mouth)
    else:
        r = ent.radius
        body = pygame.Rect(int(px - r), int(py - r), 2 * r, 2 * r)
        color = VULNERABLE_COLOR if ent.vulnerable else ent.color
        pygame.draw.rect(screen, color, body, border_radius=r // 2)
        
        er = r//3 
        oy = r//3
        ox = r//2
        for dx_eye in (-ox, ox):
            ex, ey = int(px + dx_eye), int(py - oy)
            pygame.draw.circle(screen, (255,255,255), (ex,ey), er)
            
            dir_x, dir_y = ent.direction
            offx = 0
            if dir_x != 0: offx = int(er//2 * (dir_x / abs(dir_x)))
            offy = 0
            if dir_y != 0: offy = int(er//2 * (dir_y / abs(dir_y)))
            
            pygame.draw.circle(screen, (0,0,0), (ex+offx, ey+offy), er//2)

def display_score(screen, score, font):
    s = font.render(f"Score: {score}", True, SCORE_TEXT_COLOR)
    screen.blit(s, (SCREEN_W_PADDING, SCREEN_H - BOTTOM_PANEL_HEIGHT + 5))

def display_game_stats(screen, best_score, worst_score, average_score, pacman_wins, ghost_wins, font):
    
    line1_text = f"Melhor: {best_score} Pior: {worst_score if worst_score != float('inf') else 0} Média: {average_score:.0f}"
    text1_surf = font.render(line1_text, True, TEXT_COLOR)
    screen.blit(text1_surf, (SCREEN_W - text1_surf.get_width() - SCREEN_W_PADDING, SCREEN_H - BOTTOM_PANEL_HEIGHT + 5))

    line2_text = f" Pacman: {pacman_wins}     Fantasmas: {ghost_wins}"
    text2_surf = font.render(line2_text, True, TEXT_COLOR)
    screen.blit(text2_surf, (SCREEN_W - text2_surf.get_width() - SCREEN_W_PADDING, SCREEN_H - BOTTOM_PANEL_HEIGHT + 35))

def display_status_message(screen, message, font):
    text_surf = font.render(message, True, TEXT_COLOR)
    x = (SCREEN_W - text_surf.get_width()) // 2
    y = SCREEN_H - BOTTOM_PANEL_HEIGHT + STATUS_TEXT_HEIGHT // 2 - text_surf.get_height() // 2
    screen.blit(text_surf, (x, y))

class Button:
    def __init__(self, rect, text, font, action):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.action = action
        self.hover = False

    def draw(self, surface):
        color = BUTTON_HOVER_COLOR if self.hover else BUTTON_BG_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        text_surf = self.font.render(self.text, True, BUTTON_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                self.action()
pacman = None
ghosts = []

def main():
    global pacman, ghosts 

    pygame.init()

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Pacman Game")

    font = pygame.font.Font(FONT_NAME, 24)
    clock = pygame.time.Clock()
    grid = None
    last_ai_update_time = time.time()
    ai_delay = BASE_AI_DELAY
    paused = False
    score = 0
    game_over = False
    victory = False
    current_game_time_survived = 0.0
    current_game_ghost_captures = 0
    current_game_powerup_time = 0.0
    
    state_data = load_dump()
    best_score = state_data.get('best_score', 0)
    worst_score = state_data.get('worst_score', float('inf'))
    average_score = state_data.get('average_score', 0)
    pacman_wins = state_data.get('pacman_wins', 0)
    ghost_wins = state_data.get('ghost_wins', 0)
    
    game_scores_history = []
    game_times_survived_history = []
    ghost_captures_history = []
    powerup_durations_history = []
    
    def calculate_ghost_mode_weights(ghost_inst, pacman_inst, current_game_grid):
        weights = {}
        
        dist_to_pacman = abs(ghost_inst.grid_x - pacman_inst.grid_x) + abs(ghost_inst.grid_y - pacman_inst.grid_y)
        weights['chase'] = (1.2 / (dist_to_pacman + 1)) * AGRESSION_FACTOR 
        
        dist_to_scatter = abs(ghost_inst.grid_x - ghost_inst.scatter_target[0]) + abs(ghost_inst.grid_y - ghost_inst.scatter_target[1])
        weights['scatter'] = 1.0 / (dist_to_scatter + 1)
        
        powerup_target = ghost_inst.get_powerup_target(current_game_grid)
        if powerup_target:
            dist_to_powerup = abs(ghost_inst.grid_x - powerup_target[0]) + abs(ghost_inst.grid_y - powerup_target[1])
            weights['powerup_guard'] = 0.6 / (dist_to_powerup + 1)
        else:
            weights['powerup_guard'] = 0.0
            
        total_weight = sum(weights.values())
        if total_weight == 0:
            return {'chase': 1.0, 'scatter': 0.0, 'powerup_guard': 0.0}
        
        normalized_weights = {mode: weight / total_weight for mode, weight in weights.items()}
        return normalized_weights

    def update_ghost_modes_dynamically():
        
        current_time = time.time()
        if pacman.powerup_timer > 0:
            for g in ghosts:
                if g.mode != 'frightened':
                    g.mode = 'frightened'
                    g.current_committed_mode = 'frightened'
                    g.mode_start_time = current_time
            return
        for g in ghosts:
            if g.vulnerable and g.frightened_timer <= 0:
                g.vulnerable = False
                
            if g.mode == 'frightened':
                continue
            if (current_time - g.mode_start_time >= GHOST_MODE_COMMITMENT_DURATION) or \
               (g.current_committed_mode == 'frightened' and pacman.powerup_timer <= 0): 
                weights = calculate_ghost_mode_weights(g, pacman, grid)
                modes = list(weights.keys())
                mode_probabilities = list(weights.values())
                
                chosen_mode = random.choices(modes, weights=mode_probabilities, k=1)[0]
                g.current_committed_mode = chosen_mode
                g.mode_start_time = current_time
                g.last_evaluated_mode = chosen_mode
            g.mode = g.current_committed_mode

    def plot_game_analytics():
        if not game_scores_history:
            print("No game data to plot.")
            return

        num_games = range(1, len(game_scores_history) + 1)

        plt.figure(figsize=(18, 8))
        plt.suptitle('Análise de Desempenho do Jogo Pacman', fontsize=16)

        plt.subplot(1, 3, 1)
        plt.plot(num_games, game_scores_history, marker='o', linestyle='-', color='blue', label='Pontuação por Jogo')
        
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

        plt.subplot(1, 3, 2) 
        plt.plot(num_games, game_times_survived_history, marker='x', linestyle='-', color='green', label='Tempo de Sobrevivência (s)')
        plt.xlabel('Número do Jogo')
        plt.ylabel('Tempo (s)')
        plt.title('Tempos de Sobrevivência')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(num_games, ghost_captures_history, marker='s', linestyle='-', color='cyan', label='Fantasmas Capturados')
        plt.plot(num_games, powerup_durations_history, marker='d', linestyle='-', color='orange', label='Duração Total Power-up (s)')
        plt.xlabel('Número do Jogo')
        plt.ylabel('Contagem / Tempo (s)')
        plt.title('Interações com Fantasmas e Power-ups')
        plt.legend()
        plt.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('analytics.png')
        plt.show()

    def reset_game_state():
        nonlocal grid, last_ai_update_time, score, paused, game_over, victory, \
                   best_score, worst_score, average_score, pacman_wins, ghost_wins, \
                   current_game_time_survived, current_game_ghost_captures, current_game_powerup_time, \
                   game_scores_history, game_times_survived_history, \
                   ghost_captures_history, powerup_durations_history
        
        global pacman, ghosts 


        if game_over or victory:
            game_scores_history.append(score)
            game_times_survived_history.append(current_game_time_survived)
            ghost_captures_history.append(current_game_ghost_captures)
            powerup_durations_history.append(current_game_powerup_time)

            best_score = max(best_score, score)
            worst_score = min(worst_score, score)
            if len(game_scores_history) > 0:
                average_score = sum(game_scores_history) / len(game_scores_history)
            else:
                average_score = 0

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
        current_game_ghost_captures = 0
        current_game_powerup_time = 0.0
        
        grid = copy.deepcopy(ORIGINAL_MAP)

        pacman_spawn = find_pacman_spawn(grid)
        pacman = Pacman(pacman_spawn[0], pacman_spawn[1], grid) 

        ghost_spawns = find_spawn_for_ghosts(grid)
        ghosts = []
        for i, c in enumerate(GHOST_COLORS):
            spawn = ghost_spawns[i % len(ghost_spawns)]
            if i == 0:
                scatter_tgt = (GRID_W - 2, 1)
            elif i == 1:
                scatter_tgt = (1, 1)
            elif i == 2:
                scatter_tgt = (GRID_W - 2, GRID_H - 2)
            elif i == 3:
                scatter_tgt = (1, GRID_H - 2)
            else:
                scatter_tgt = (random.choice([1, GRID_W-2]), random.choice([1, GRID_H-2]))
            ghosts.append(Ghost(spawn[0], spawn[1], c, scatter_tgt))
        
        print(f"Novo Jogo Iniciado. Vitórias Pacman: {pacman_wins}, Vitórias Fantasmas: {ghost_wins}, Melhor Pontuação: {best_score}")

    reset_game_state()

    def toggle_pause():
        nonlocal paused
        paused = not paused

    def toggle_speed():
        nonlocal ai_delay
        ai_delay = FAST_AI_DELAY if ai_delay == BASE_AI_DELAY else BASE_AI_DELAY

    btn_width = 140
    btn_height = 38
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
                plot_game_analytics()
                
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
                    event = pygame.event.Event(pygame.QUIT)
                    pygame.event.post(event)
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
                pass

            for btn in buttons:
                btn.handle_event(event)

        if not paused and not game_over and not victory:
            current_game_time_survived += dt

            pacman.speed = PACMAN_BASE_SPEED
            if grid[pacman.grid_y][pacman.grid_x] == '.' or grid[pacman.grid_y][pacman.grid_x] == 'o':
                pacman.speed *= PACMAN_DOT_EATING_SPEED_MULTIPLIER

            score_multiplier = score // 100
            ghost_current_base_speed = GHOST_BASE_SPEED * (1 + score_multiplier * GHOST_SPEED_INCREASE_PER_1000_POINTS)
            for g in ghosts:
                g.speed = ghost_current_base_speed

            pacman.update(dt, grid)
            now = time.time()
            if (now - last_ai_update_time) >= ai_delay:
                last_ai_update_time = now
            
            update_ghost_modes_dynamically()
            for g in ghosts:
                g.update(dt, grid)

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
            
            if pacman.powerup_timer > 0:
                current_game_powerup_time += dt 
                pacman.powerup_timer -= dt
                if pacman.powerup_timer < 0:
                    pacman.powerup_timer = 0
                    for g in ghosts:
                        g.vulnerable = False
                        g.frightened_timer = 0
            for g in ghosts:
                if g.frightened_timer > 0:
                    g.frightened_timer -= dt
                    if g.frightened_timer < 0:
                        g.frightened_timer = 0
                        g.vulnerable = False

            to_remove = []
            for pos in dot_shrink_timers:
                x, y = pos
                timer = dot_shrink_timers[pos] - dt
                if timer <= 0:
                    grid[y][x] = '-'
                    to_remove.append(pos)
                else:
                    dot_shrink_timers[pos] = timer
                    grid[y][x] = ('shrinking_dot', timer)
            for pos in to_remove:
                del dot_shrink_timers[pos]

            for g in ghosts:
                if is_collision(pacman, g):
                    if pacman.powerup_timer > 0 and g.vulnerable:
                        score += 150
                        current_game_ghost_captures += 1
                        g.grid_x, g.grid_y = g.original_pos[0], g.original_pos[1]
                        g.direction = (0, 0)
                        g.next_direction = (0, 0)
                        g.vulnerable = False
                        g.frightened_timer = 0
                        g.returning_to_house = True 
                    else:
                        game_over = True
                        ghost_wins += 1
                        print(f"Fim de Jogo: Pacman capturado! Pontuação: {score}, Tempo de Sobrevivência: {current_game_time_survived:.2f}s")
                        break
            if game_over:
                pygame.time.wait(2000)
                reset_game_state()
                continue
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

            if victory:
                pygame.time.wait(2000)
                reset_game_state()
                continue

        screen.fill(BG_COLOR)
        draw_map(screen, grid)
        draw_entity(screen, pacman)
        for g in ghosts:
            draw_entity(screen, g)
        display_score(screen, score, font)
        display_game_stats(screen, best_score, worst_score, average_score, pacman_wins, ghost_wins, font)

        for btn in buttons:
            btn.draw(screen)
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