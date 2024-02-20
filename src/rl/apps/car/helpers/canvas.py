import math
import os
import random
from functools import cache
from typing import Tuple, List

import pygame
import pygame.gfxdraw
from pygame import Surface, Color

from rl.apps.car.common.constants import GREEN, DARK_GREEN, SIDE, HALF, GRAY, LIGHT_GRAY, CENTERLINE, \
    DARK_GRAY, PAD, ROAD_MAP, LIGHTEST_GRAY, MARGIN, WHITE, \
    LIGHT_BLACK, FONT_SIZE, CANVAS_AREA, CAR_LENGTH, CAR_WIDTH, COLOR_KEY, CAR_TURN_DEGREES_PER_FRAME, \
    CAR_SPEED_PIXELS_PER_FRAME, RED, TURN_SIGNAL, BLUE
from rl.apps.car.common.types import Shape, AngleDegrees, Rectangle, Vector
from rl.apps.car.environment.car import CarState, Blink
from rl.apps.car.utils.map import get_tile_position, road_next_tile, get_tile, is_right, is_left, is_up, is_down
from rl.apps.car.utils.shapes import rotate_polygon, rotate
from rl.apps.car.utils.then import then


def _copy_surface(surface: Surface) -> Surface:
    result = pygame.Surface(surface.get_size())
    result.fill(COLOR_KEY)
    result.set_colorkey(COLOR_KEY)
    result.blit(surface, (0, 0))
    return result


clone = then(_copy_surface)


def _draw_filled_pie(surface: Surface, x: int, y: int, outer_radius: int, inner_radius: int,
                     start_angle: AngleDegrees, stop_angle: AngleDegrees, color: Color):
    points = []

    start_angle_rad = math.radians(start_angle)
    stop_angle_rad = math.radians(stop_angle)

    # Outer points
    angle = start_angle_rad
    while angle <= stop_angle_rad:
        points.append((
            x + int(math.cos(angle) * outer_radius),
            y + int(math.sin(angle) * outer_radius),
        ))
        angle += math.radians(1)
    points.append((
        x + int(math.cos(stop_angle_rad) * outer_radius),
        y + int(math.sin(stop_angle_rad) * outer_radius),
    ))

    # Inner points
    angle = stop_angle_rad
    while angle >= start_angle_rad:
        points.append((
            x + int(math.cos(angle) * inner_radius),
            y + int(math.sin(angle) * inner_radius),
        ))
        angle -= math.radians(1)
    points.append((
        x + int(math.cos(start_angle_rad) * inner_radius),
        y + int(math.sin(start_angle_rad) * inner_radius),
    ))

    pygame.draw.polygon(surface, color, points)


def _draw_granules(surface: Surface, color: Color, rect: Rectangle, shape: Shape = None):
    x, y, width, height = rect

    if shape == "┌":
        center = (x + width, y + height)
    elif shape == "┐":
        center = (x, y + height)
    elif shape == "└":
        center = (x + width, y)
    elif shape == "┘":
        center = (x, y)
    else:
        center = None

    def distance(position: Vector) -> int:
        return int(math.sqrt((granule_x - position[0]) ** 2 + (granule_y - position[1]) ** 2))

    granules = int(width * height // 100)
    for _ in range(granules):

        granule_x = x + random.randint(0, width)
        granule_y = y + random.randint(0, height)

        if not center or distance(center) < width:
            pygame.draw.circle(surface, color, (granule_x, granule_y), 1)


def _draw_grass(surface: Surface):
    for x in range(surface.get_width()):
        for y in range(surface.get_height()):
            color = GREEN if y % 2 == 0 else DARK_GREEN
            grass_y = y + random.randint(-5, 5)
            pygame.draw.line(surface, color, (x, y), (x, grass_y), 1)


def _draw_asphalt(surface: Surface, tile: Vector, shape: Shape):
    x, y = get_tile_position(tile)

    if shape == "┌":
        pygame.draw.rect(surface, GRAY, (x, y + SIDE - PAD, SIDE, PAD))
        _draw_filled_pie(surface, x + SIDE - PAD, y + SIDE - PAD, SIDE - PAD, 0, 180, 270, GRAY)
        pygame.draw.rect(surface, GRAY, (x + SIDE - PAD, y, PAD, SIDE))
    elif shape == "┐":
        pygame.draw.rect(surface, GRAY, (x, y, PAD, SIDE))
        _draw_filled_pie(surface, x + PAD, y + SIDE - PAD, SIDE - PAD, 0, 270, 360, GRAY)
        pygame.draw.rect(surface, GRAY, (x, y + SIDE - PAD, SIDE, PAD))
    elif shape == "└":
        pygame.draw.rect(surface, GRAY, (x, y, SIDE, PAD))
        _draw_filled_pie(surface, x + SIDE - PAD, y + PAD, SIDE - PAD, 0, 90, 180, GRAY)
        pygame.draw.rect(surface, GRAY, (x + SIDE - PAD, y, PAD, SIDE))
    elif shape == "┘":
        pygame.draw.rect(surface, GRAY, (x, y, SIDE, PAD))
        _draw_filled_pie(surface, x + PAD, y + PAD, SIDE - PAD, 0, 0, 90, GRAY)
        pygame.draw.rect(surface, GRAY, (x, y, PAD, SIDE))
    else:
        pygame.draw.rect(surface, GRAY, (x, y, SIDE, SIDE))

    _draw_granules(surface, LIGHT_GRAY, (x, y, SIDE, SIDE), shape)


def _draw_pavement(surface: Surface, tile: Vector, shape: Shape):
    x, y = get_tile_position(tile)
    if shape == "─":
        pygame.draw.rect(surface, DARK_GRAY, (x, y, SIDE, PAD))
        pygame.draw.rect(surface, DARK_GRAY, (x, y + SIDE - PAD, SIDE, PAD))
    elif shape == "│":
        pygame.draw.rect(surface, DARK_GRAY, (x, y, PAD, SIDE))
        pygame.draw.rect(surface, DARK_GRAY, (x + SIDE - PAD, y, PAD, SIDE))
    elif shape == "┌":
        pygame.draw.rect(surface, DARK_GRAY, (x, y + SIDE - PAD, PAD, PAD))
        _draw_filled_pie(surface, x + SIDE - PAD, y + SIDE - PAD, SIDE - PAD, SIDE - PAD * 2, 180, 270, DARK_GRAY)
        _draw_filled_pie(surface, x + SIDE, y + SIDE, PAD, 0, 180, 270, DARK_GRAY)
        pygame.draw.rect(surface, DARK_GRAY, (x + SIDE - PAD, y, PAD, PAD))
    elif shape == "┐":
        pygame.draw.rect(surface, DARK_GRAY, (x, y, PAD, PAD))
        _draw_filled_pie(surface, x + PAD, y + SIDE - PAD, SIDE - PAD, SIDE - PAD * 2, 270, 360, DARK_GRAY)
        _draw_filled_pie(surface, x, y + SIDE, PAD, 0, 270, 360, DARK_GRAY)
        pygame.draw.rect(surface, DARK_GRAY, (x + SIDE - PAD, y + SIDE - PAD, PAD, PAD))
    elif shape == "└":
        pygame.draw.rect(surface, DARK_GRAY, (x, y, PAD, PAD))
        _draw_filled_pie(surface, x + SIDE - PAD, y + PAD, SIDE - PAD, SIDE - PAD * 2, 90, 180, DARK_GRAY)
        _draw_filled_pie(surface, x + SIDE, y, PAD, 0, 90, 180, DARK_GRAY)
        pygame.draw.rect(surface, DARK_GRAY, (x + SIDE - PAD, y + SIDE - PAD, PAD, PAD))
    elif shape == "┘":
        pygame.draw.rect(surface, DARK_GRAY, (x + SIDE - PAD, y, PAD, PAD))
        _draw_filled_pie(surface, x + PAD, y + PAD, SIDE - PAD, SIDE - PAD * 2, 0, 90, DARK_GRAY)
        _draw_filled_pie(surface, x, y, PAD, 0, 0, 90, DARK_GRAY)
        pygame.draw.rect(surface, DARK_GRAY, (x, y + SIDE - PAD, PAD, PAD))
    elif shape == "├":
        pygame.draw.rect(surface, DARK_GRAY, (x, y, PAD, SIDE))
        _draw_filled_pie(surface, x + SIDE, y, PAD, 0, 90, 180, DARK_GRAY)
        _draw_filled_pie(surface, x + SIDE, y + SIDE, PAD, 0, 180, 270, DARK_GRAY)
    elif shape == "┤":
        pygame.draw.rect(surface, DARK_GRAY, (x + SIDE - PAD, y, PAD, SIDE))
        _draw_filled_pie(surface, x, y + SIDE, PAD, 0, 270, 360, DARK_GRAY)
        _draw_filled_pie(surface, x, y, PAD, 0, 0, 90, DARK_GRAY)
    elif shape == "┬":
        pygame.draw.rect(surface, DARK_GRAY, (x, y, SIDE, PAD))
        _draw_filled_pie(surface, x, y + SIDE, PAD, 0, 270, 360, DARK_GRAY)
        _draw_filled_pie(surface, x + SIDE, y + SIDE, PAD, 0, 180, 270, DARK_GRAY)
    elif shape == "┴":
        pygame.draw.rect(surface, DARK_GRAY, (x, y + SIDE - PAD, SIDE, PAD))
        _draw_filled_pie(surface, x + SIDE, y, PAD, 0, 90, 180, DARK_GRAY)
        _draw_filled_pie(surface, x, y, PAD, 0, 0, 90, DARK_GRAY)
    elif shape == "┼":
        _draw_filled_pie(surface, x, y + SIDE, PAD, 0, 270, 360, DARK_GRAY)
        _draw_filled_pie(surface, x + SIDE, y + SIDE, PAD, 0, 180, 270, DARK_GRAY)
        _draw_filled_pie(surface, x + SIDE, y, PAD, 0, 90, 180, DARK_GRAY)
        _draw_filled_pie(surface, x, y, PAD, 0, 0, 90, DARK_GRAY)
    else:
        pass
    _draw_granules(surface, LIGHT_GRAY, (x, y, SIDE, SIDE), shape)


def _draw_navigation(surface: Surface, tile: Vector, direction: Vector, shape: Shape):
    x, y = get_tile_position(tile)
    if shape == "─":
        if is_left(direction):
            pygame.draw.line(surface, BLUE, (x, y + SIDE * 0.25), (x + SIDE, y + SIDE * 0.25), 1)
        elif is_right(direction):
            pygame.draw.line(surface, BLUE, (x, y + SIDE * 0.75), (x + SIDE, y + SIDE * 0.75), 1)
    elif shape == "│":
        if is_down(direction):
            pygame.draw.line(surface, BLUE, (x + SIDE * 0.25, y), (x + SIDE * 0.25, y + SIDE), 1)
        elif is_up(direction):
            pygame.draw.line(surface, BLUE, (x + SIDE * 0.75, y), (x + SIDE * 0.75, y + SIDE), 1)
    elif shape == "┌":
        if is_right(direction):
            pygame.gfxdraw.arc(surface, x + SIDE - PAD, y + SIDE - PAD, int((HALF - PAD) * 0.5), 180, 270, BLUE)
        elif is_down(direction):
            pygame.gfxdraw.arc(surface, x + SIDE - PAD, y + SIDE - PAD, int((HALF - PAD) * 1.5), 180, 270, BLUE)
    elif shape == "┐":
        if is_down(direction):
            pygame.gfxdraw.arc(surface, x + PAD, y + SIDE - PAD, int((HALF - PAD) * 0.5), 270, 360, BLUE)
        elif is_left(direction):
            pygame.gfxdraw.arc(surface, x + PAD, y + SIDE - PAD, int((HALF - PAD) * 1.5), 270, 360, CENTERLINE)
    elif shape == "└":
        if is_up(direction):
            pygame.gfxdraw.arc(surface, x + SIDE - PAD, y + PAD, int((HALF - PAD) * 0.5), 90, 180, BLUE)
        elif is_right(direction):
            pygame.gfxdraw.arc(surface, x + SIDE - PAD, y + PAD, int((HALF - PAD) * 1.5), 90, 180, BLUE)
    elif shape == "┘":
        if is_left(direction):
            pygame.gfxdraw.arc(surface, x + PAD, y + PAD, int((HALF - PAD) * 0.5), 0, 90, BLUE)
        elif is_up(direction):
            pygame.gfxdraw.arc(surface, x + PAD, y + PAD, int((HALF - PAD) * 1.5), 0, 90, BLUE)
    else:
        pass


def _draw_crosswalks(surface: Surface, tile: Vector, shape: Shape):
    def _draw_single(area: Rectangle, size: Vector, offset: Vector):
        area_x, area_y, area_width, area_height = area

        x, y = area_x, area_y
        width, height = size
        while x < (area_x + area_width) and y < (area_y + area_height):
            pygame.draw.rect(surface, LIGHTEST_GRAY, (x, y, width, height))
            _draw_granules(surface, LIGHT_GRAY, (x, y, width, height))
            x += offset[0]
            y += offset[1]

    tile_x, tile_y = get_tile_position(tile)
    if shape in "┤┬┴┼":  # Left crosswalk
        _draw_single((tile_x, tile_y + PAD * 1.25, PAD, SIDE - PAD * 2), (PAD, PAD / 2), (0, PAD))
    if shape in "├┤┴┼":  # Top crosswalk
        _draw_single((tile_x + PAD * 1.25, tile_y, SIDE - PAD * 2, PAD), (PAD / 2, PAD), (PAD, 0))
    if shape in "├┬┴┼":  # Right crosswalk
        _draw_single((tile_x + SIDE - PAD, tile_y + PAD * 1.25, PAD, SIDE - PAD * 2), (PAD, PAD / 2), (0, PAD))
    if shape in "├┤┬┼":  # Bottom crosswalk
        _draw_single((tile_x + PAD * 1.25, tile_y + SIDE - PAD, SIDE - PAD * 2, PAD), (PAD / 2, PAD), (PAD, 0))


def _draw_centerline(surface: Surface, tile: Vector, shape: Shape):
    x, y = get_tile_position(tile)
    if shape == "─":
        pygame.draw.line(surface, CENTERLINE, (x, y + HALF), (x + SIDE, y + HALF), 2)
    elif shape == "│":
        pygame.draw.line(surface, CENTERLINE, (x + HALF, y), (x + HALF, y + SIDE), 2)
    elif shape == "┌":
        pygame.draw.line(surface, CENTERLINE, (x + HALF, y + SIDE), (x + HALF, y + SIDE - PAD), 2)
        pygame.gfxdraw.arc(surface, x + SIDE - PAD, y + SIDE - PAD, HALF - PAD, 180, 270, CENTERLINE)
        pygame.gfxdraw.arc(surface, x + SIDE - PAD, y + SIDE - PAD, HALF - PAD - 1, 180, 270, CENTERLINE)
        pygame.draw.line(surface, CENTERLINE, (x + SIDE - PAD, y + HALF), (x + SIDE, y + HALF), 2)
    elif shape == "┐":
        pygame.draw.line(surface, CENTERLINE, (x, y + HALF), (x + PAD, y + HALF), 2)
        pygame.gfxdraw.arc(surface, x + PAD, y + SIDE - PAD, HALF - PAD, 270, 360, CENTERLINE)
        pygame.gfxdraw.arc(surface, x + PAD, y + SIDE - PAD, HALF - PAD - 1, 270, 360, CENTERLINE)
        pygame.draw.line(surface, CENTERLINE, (x + HALF, y + SIDE - PAD), (x + HALF, y + SIDE), 2)
    elif shape == "└":
        pygame.draw.line(surface, CENTERLINE, (x + SIDE, y + HALF), (x + SIDE - PAD, y + HALF), 2)
        pygame.gfxdraw.arc(surface, x + SIDE - PAD, y + PAD, HALF - PAD, 90, 180, CENTERLINE)
        pygame.gfxdraw.arc(surface, x + SIDE - PAD, y + PAD, HALF - PAD - 1, 90, 180, CENTERLINE)
        pygame.draw.line(surface, CENTERLINE, (x + HALF, y + PAD), (x + HALF, y), 2)
    elif shape == "┘":
        pygame.draw.line(surface, CENTERLINE, (x + HALF, y), (x + HALF, y + PAD), 2)
        pygame.gfxdraw.arc(surface, x + PAD, y + PAD, HALF - PAD, 0, 90, CENTERLINE)
        pygame.gfxdraw.arc(surface, x + PAD, y + PAD, HALF - PAD - 1, 0, 90, CENTERLINE)
        pygame.draw.line(surface, CENTERLINE, (x + PAD, y + HALF), (x, y + HALF), 2)
    else:
        pass


def _draw_background(surface: Surface):
    _draw_grass(surface)

    for tile_row, row in enumerate(ROAD_MAP):
        for tile_col, shape in enumerate(row):
            if shape == " ":
                continue

            tile = (tile_col, tile_row)
            _draw_asphalt(surface, tile, shape)
            _draw_pavement(surface, tile, shape)
            _draw_centerline(surface, tile, shape)
            _draw_crosswalks(surface, tile, shape)


@clone
@cache
def get_background() -> Surface:
    result: Surface = pygame.Surface(CANVAS_AREA)
    _draw_background(result)
    return result


def draw_car(surface: Surface, car: CarState, previous_car: CarState = None) -> Surface:
    car_x, car_y = car.position

    # Body
    body_corners = [
        (car_x - CAR_LENGTH / 2 + CAR_LENGTH / 16 * 2, car_y - CAR_WIDTH / 2),
        (car_x - CAR_LENGTH / 2 + CAR_LENGTH / 16 * 14, car_y - CAR_WIDTH / 2),
        (car_x - CAR_LENGTH / 2 + CAR_LENGTH, car_y - CAR_WIDTH / 2 + CAR_WIDTH / 8 * 3),
        (car_x - CAR_LENGTH / 2 + CAR_LENGTH, car_y - CAR_WIDTH / 2 + CAR_WIDTH / 8 * 5),
        (car_x - CAR_LENGTH / 2 + CAR_LENGTH / 16 * 14, car_y - CAR_WIDTH / 2 + CAR_WIDTH),
        (car_x - CAR_LENGTH / 2 + CAR_LENGTH / 16 * 2, car_y - CAR_WIDTH / 2 + CAR_WIDTH),
        (car_x - CAR_LENGTH / 2, car_y - CAR_WIDTH / 2 + CAR_WIDTH / 8 * 6),
        (car_x - CAR_LENGTH / 2, car_y - CAR_WIDTH / 2 + CAR_WIDTH / 8 * 2),
        (car_x - CAR_LENGTH / 2 + CAR_LENGTH / 16 * 2, car_y - CAR_WIDTH / 2),
    ]
    rotated_body_corners = rotate_polygon(body_corners, car.angle, (car_x, car_y))
    pygame.draw.polygon(surface, WHITE, rotated_body_corners)

    # Window
    blink_corners = [
        (car_x - CAR_LENGTH / 2 + CAR_LENGTH / 16 * 2, car_y - CAR_WIDTH / 2 + CAR_WIDTH / 8 * 2),
        (car_x - CAR_LENGTH / 2 + CAR_LENGTH / 16 * 11, car_y - CAR_WIDTH / 2 + CAR_WIDTH / 8 * 1),
        (car_x - CAR_LENGTH / 2 + CAR_LENGTH / 16 * 12, car_y - CAR_WIDTH / 2 + CAR_WIDTH / 8 * 3),
        (car_x - CAR_LENGTH / 2 + CAR_LENGTH / 16 * 12, car_y - CAR_WIDTH / 2 + CAR_WIDTH / 8 * 5),
        (car_x - CAR_LENGTH / 2 + CAR_LENGTH / 16 * 11, car_y - CAR_WIDTH / 2 + CAR_WIDTH / 8 * 7.0),
        (car_x - CAR_LENGTH / 2 + CAR_LENGTH / 16 * 2, car_y - CAR_WIDTH / 2 + CAR_WIDTH / 8 * 6),
    ]
    rotated_window_corners = rotate_polygon(blink_corners, car.angle, (car_x, car_y))
    pygame.draw.polygon(surface, LIGHT_BLACK, rotated_window_corners)

    # Blink
    lights: List[Tuple[int, Vector]] = []
    if car.events.crossroad:
        if car.events.crossroad.blink == Blink.LEFT:
            lights.append((TURN_SIGNAL, (car_x - CAR_LENGTH / 2 + CAR_LENGTH / 16 * 1, car_y - CAR_WIDTH / 2)))
            lights.append((TURN_SIGNAL, (car_x - CAR_LENGTH / 2 + CAR_LENGTH / 16 * 15, car_y - CAR_WIDTH / 2)))
        elif car.events.crossroad.blink == Blink.RIGHT:
            lights.append((TURN_SIGNAL, (car_x - CAR_LENGTH / 2 + CAR_LENGTH / 16 * 15, car_y + CAR_WIDTH / 2)))
            lights.append((TURN_SIGNAL, (car_x - CAR_LENGTH / 2 + CAR_LENGTH / 16 * 1, car_y + CAR_WIDTH / 2)))
    if car.decelerating:
        lights.append((RED, (car_x - CAR_LENGTH / 2 + CAR_LENGTH / 16 * 0, car_y - CAR_WIDTH / 2 + 1)))
        lights.append((RED, (car_x - CAR_LENGTH / 2 + CAR_LENGTH / 16 * 0, car_y + CAR_WIDTH / 2 - 1)))

    for color, position in lights:
        rotated = rotate(position, car.angle, (car_x, car_y))
        pygame.draw.circle(surface, color, rotated, 2)

    return surface


def draw_state(surface: Surface, car: CarState, previous_car: CarState = None) -> Surface:
    background: Surface = get_background()
    surface.blit(background, (0, 0))
    draw_car(surface, car, previous_car)
    return surface


def draw_stats(
        surface: Surface,
        car: CarState,
        fast_render: bool,
        epoch: int,
        batch: int,
        episode: int,
) -> Surface:
    rect = (MARGIN - PAD * 2, MARGIN - PAD * 2, int(SIDE * 1.5), int(SIDE * 1.5))
    x, y, width, height = rect

    # Background
    # pygame.draw.rect(surface, GRAY, rect, width)

    # Text
    lines = [
        f"Position: {car.position[0]:.0f}x{car.position[1]:.0f}",
        f"Speed: {car.speed * CAR_SPEED_PIXELS_PER_FRAME}p/f",
        f"Turn: {car.turn * CAR_TURN_DEGREES_PER_FRAME}°/f",
        f"Angle: {car.angle:.0f}°",
        "Fast render" if fast_render else "Normal render",
        "",
        f"Epoch: {epoch}",
        f"Batch: {batch}",
        f"Episode: {episode}",
        f"Shape: {car.events.crossroad.trajectory if car.events.crossroad else 'na'}",
    ]
    font = pygame.font.Font(None, FONT_SIZE)
    for index, line in enumerate(lines):
        surface.blit(font.render(line, True, CENTERLINE), (x + PAD, y + PAD + index * (FONT_SIZE + PAD // 10)))

    # When debug
    if os.environ.get("DEBUG", "false").lower() == "true":
        tile_col, tile_row = get_tile(car.position)

        # Draw next tile marker
        if tile := road_next_tile(car.position, ROAD_MAP[tile_row][tile_col]):
            (tile_col, tile_row) = tile
            x = MARGIN + (tile_col + 0.5) * SIDE
            y = MARGIN + (tile_row + 0.5) * SIDE
            pygame.draw.circle(surface, RED, (x, y), 5)

        # Draw trajectory
        if car.events.crossroad:
            _draw_navigation(
                surface,
                car.events.crossroad.tile,
                car.events.crossroad.out_direction,
                car.events.crossroad.trajectory,
            )

    return surface


def draw_observation(surface: Surface, view: Surface) -> Surface:
    x, y = (MARGIN + 8 * SIDE, MARGIN + 4.1 * SIDE)
    width, height = view.get_width(), view.get_height()

    # Blit
    surface.blit(view, (x, y))

    # Border
    pygame.draw.rect(surface, DARK_GRAY, (x, y, width, height), 4)

    return surface
