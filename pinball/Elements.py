"""
The Ball and all it's friends (the elements on the playfield) live here.

Contains test code if run as main
"""

import math
import random

import numpy as np
import cv2 as cv


def _reflect_lambda(nx, ny):
    """
    :return lambda that returns the reflection of a vector given a normal
    """
    # https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
    n = np.array([nx, ny], dtype=np.float)
    n /= np.linalg.norm(n)
    return lambda d: d - 2 * (d.dot(n)) * n + 0.5 * n


class Ball:
    """
    The ball (there can only be one)
    """
    r: int
    px: int
    py: int
    vx: float
    vy: float

    def __init__(self, x, y):
        self.r = 11  # Must be odd.
        self.px = x
        self.py = y
        self.vx = 0
        self.vy = 1
        # For collision detection, it's much better to only initialize this once.
        # The mask is 2 pixels too small intentionally, because the openCV circle function draws a circle with 4 points
        # at the cardinal coordinates, this eliminates those.
        self.mask = np.zeros((2 * self.r - 1, 2 * self.r - 1), dtype=np.uint8)
        cv.circle(self.mask, (self.r - 1, self.r - 1), self.r, 0xFF, -1)

    def draw(self, frame: np.ndarray, color=(219, 221, 197)):
        cv.circle(frame, (int(self.px), int(self.py)), self.r, color, -1)

    def is_out_of_bounds(self, width, height) -> bool:
        """
        :return: Game over
        """
        return self.px < self.r or self.py < self.r or self.px > width - self.r or self.py > height - self.r

    def move(self):
        """
        Applies gravity, velocity and drag. Also hardcaps the max speed at 8 to avoid clipping
        :return:
        """
        self.vy += 0.1

        self.px = self.px + self.vx
        self.py = self.py + self.vy

        # print('Move', self.vx, self.vy)

        v = self.vx * self.vx + self.vy * self.vy
        if v > 64:  # Hard speed limit, to avoid clipping
            v = math.sqrt(v)
            self.vx = 8 * self.vx / v
            self.vy = 8 * self.vy / v

        self.vx *= 0.999
        self.vy *= 0.999

    def __str__(self) -> str:
        return '<Ball r={} p: {},{} v: {},{}>'.format(self.r, self.px, self.py, self.vx, self.vy)


def _collide_circle(pos1: np.array, r1: int, pos2: np.array, r2: int) -> (bool, float, np.array or None):
    """
    Detect a collision between 2 circles.
    :return: hit, collision depth, collision vector
    """
    # Time for vector math :(
    # print("deflect_circle", pos2, r2)

    # delta pos
    dp = pos1 - pos2
    dp_sq = dp * dp

    # clipping_factor
    cf = (r1 * r1 + r2 * r2) - (dp_sq[0] + dp_sq[1])

    # print("Clipping Factor", cf)

    if cf < 0:
        # print("No clip")
        # No clipping
        return False, 0, None

    cf = math.sqrt(cf)

    dp_e = dp / math.sqrt(dp_sq[0] + dp_sq[1])
    # print("Hit", cf, dp_e)
    return True, cf, dp_e


class Element:
    """
    Abstract element class. Anything on the playing field must be a subclass of this.
    """
    color: (int, int, int)

    def __init__(self, color):
        self.color = color

    def can_grab(self, x, y) -> bool:
        """
        Can a mouse at x,y grab this element?
        """
        return False

    def draw(self, canvas):
        """
        Draw the element on the canvas.
        """
        return

    def draw_hitbox(self, field, color, edges=None):
        """
        Draw the hitbox on the field.
        """
        return

    def __repr__(self) -> str:
        return '<{} Color: {}>'.format(type(self).__name__, self.color)

    def mask_away(self, img):
        """
        Mask away the element on an image
        """
        return

    def reset(self):
        """
        Reset the sate of the element when the ball is lost
        """
        return

    def deflect(self, ball: Ball, overlap: np.ndarray) -> int:
        """
        Deflect the ball. The overlap is a map of which pixels overlap with the hitbox of the element.
        :return: the score/points to be added
        """
        return 0

    def step(self, is_touched: bool, lmb: bool = False, rmb: bool = False) -> bool:
        """
        Called once per step of the game
        :param is_touched: if the ball is in the hitbox
        :param lmb: Left mouse button down
        :param rmb: Right mouse button down
        :return: if the deflection logic should be called
        """
        return is_touched

    def set_pos(self, x: int, y: int, rel: bool = False):
        """
        Set position (only used during setup)
        :param x:
        :param y:
        :param rel: relative
        """
        return


class PopBumper(Element):
    """
    Pop bumper (the round things that shoot the ball away)
    """
    pos_x: int
    pos_y: int
    r: int

    def __init__(self, pos, color, radius):
        super().__init__(color)
        self.pos_x, self.pos_y = pos
        self.r = radius

    def can_grab(self, x, y) -> bool:
        x -= self.pos_x
        y -= self.pos_y
        return x * x + y * y <= self.r * self.r

    def draw(self, canvas):
        center = (self.pos_x, self.pos_y)
        cv.circle(canvas, center, self.r, (220, 200, 220), -1)
        cv.circle(canvas, center, self.r, self.color, 3)
        cv.circle(canvas, center, int(self.r / 2), self.color, 3)

    def draw_hitbox(self, field, color, edges=None):
        center = (self.pos_x, self.pos_y)
        cv.circle(field, center, int(self.r * 0.75), color, -1)

    def mask_away(self, img):
        center = (self.pos_x, self.pos_y)
        cv.circle(img, center, int(self.r), (255,), -1)
        # 8 = all 8 'touching' pixels. Default (= 4) means only orthogonal pixels.
        cv.floodFill(img, None, center, (0,), flags=8)

    def deflect(self, ball: Ball, overlap: np.ndarray):
        ball_pos = np.array([ball.px, ball.py], dtype=np.int)
        us_pos = np.array([self.pos_x, self.pos_y], dtype=np.int)
        hit, cf, v = _collide_circle(ball_pos, ball.r, us_pos, self.r)
        if hit:
            cf = max(cf, 2)
            ball.vx = v[0] * cf + random.random() * 0.01 - 0.005
            ball.vy = v[1] * cf + random.random() * 0.01 - 0.005
            return 10
        return 0

    def set_pos(self, x: int, y: int, rel: bool = False):
        if rel:
            self.pos_x += x
            self.pos_y += y
        else:
            self.pos_x = x
            self.pos_y = y

    def __str__(self) -> str:
        return '<{} at {},{} r={} Color: {}>'.format(type(self).__name__, self.pos_x, self.pos_y, self.r, self.color)


class Background(Element):
    """
    Placeholder-ish object. There can only be one.
    It should be element id 0
    """
    def __init__(self):
        super().__init__((0, 0, 0))

    def draw_hitbox(self, field, color, edges=None):
        if edges is not None:
            field |= (~edges & color)


class Wall(Element):
    """
    Placeholder-ish object. There can only be one.
    It should be element id 1

    There is a huge simplification here: Walls are assumed to be in one of the main 8 direction.
    If the ball touches a wall it's velocity is changed to a reflection of the incoming velocity.

    The main catch is that the checking is then disabled until the ball leaves the wall again.
    This is require because otherwise, if the ball is still in the wall next tick, the ball is reflected again.
    """
    def __init__(self):
        super().__init__((255, 255, 255))
        self.deflected_already = False
        self._wall_positions = [
            _reflect_lambda(1, 1),    # 0: Top Left
            _reflect_lambda(0, 1),    # 1: Top
            _reflect_lambda(-1, 1),   # 2: Top Right
            _reflect_lambda(1, 0),    # 3: Left
            None,
            _reflect_lambda(-1, 0),   # 5: Right
            _reflect_lambda(1, -1),   # 6: Bottom Left
            _reflect_lambda(0, -1),   # 7: Bottom
            _reflect_lambda(-1, -1),  # 8: Bottom Right
        ]

    def draw_hitbox(self, field, color, edges=None):
        if edges is not None:
            field |= (edges & color)

    def step(self, is_touched: bool, lmb: bool = False, rmb: bool = False) -> bool:
        if is_touched:
            ret = not self.deflected_already
            self.deflected_already = True
            return ret

        self.deflected_already = False
        return False

    def deflect(self, ball: Ball, overlap: np.ndarray) -> int:
        # Find out what approx. shape the wall is
        sides = np.ndarray((3, 3), dtype=np.uint8)
        size = overlap.shape[0]
        delta = size//3

        for side_y, y in enumerate(range(0, size, delta)):
            # print('Y:', side_y, y, y+delta)
            for side_x, x in enumerate(range(0, size, delta)):
                # print('X:', side_y, y, y + delta)
                sides[side_y, side_x] = overlap[y:y+delta, x:x+delta].sum()
        # The middle doesn't count.
        sides[1, 1] = 0

        sides = sides.reshape(9)

        old_v = np.array([ball.vx, ball.vy])
        new_v = np.zeros(2)

        # print("Old velocity:", old_v, np.linalg.norm(old_v))
        # print("Sides:", sides)
        # print("Sum sides:", sides.sum())
        # print("Nonzeros:", sides.nonzero()[0])

        # Make a weighted new speed vector
        for i in sides.nonzero()[0]:
            new_v += sides[i] * self._wall_positions[i](old_v)

        # print("Weighted new velocity:", new_v, np.linalg.norm(new_v))
        new_v /= sides.sum()
        # print((ball.vx, ball.vy), '->', new_v, np.linalg.norm(new_v))

        ball.vx = new_v[0] + random.random() * 0.01 - 0.005
        ball.vy = new_v[1] + random.random() * 0.01 - 0.005

        # cv.waitKey(0)

        return 0


class Flipper(Element):
    """
    Flipper element. There is supposed to be a left and a right flipper, but in theory there can be more then one.
    This element's hitbox is not always the same, but due to how the hit detection works (the hitbox has to be static),
    the deflection code is always called.
    """
    def __init__(self, is_left: bool, pos: (int, int)):
        super().__init__((255, 0, 0) if is_left else (0, 255, 0))
        self.is_left = is_left
        self.pos_x, self.pos_y = pos
        self.joint_radius = 20
        self.offset = 100

        self.moving_up = False
        self.min_angle = 60
        self.max_angle = 135
        self.angle_step = 5

        self.min_angle = math.radians(self.min_angle)
        self.max_angle = math.radians(self.max_angle)
        self.angle_step = math.radians(self.angle_step)

        if not self.is_left:
            self.min_angle *= -1
            self.max_angle *= -1
            self.angle_step *= -1

        self.angle = self.min_angle
        self.end_x = 0
        self.end_y = 0
        self.update_end()

    def reset(self):
        self.angle = self.min_angle
        self.update_end()

    def can_grab(self, x, y) -> bool:
        x -= self.pos_x
        y -= self.pos_y
        return x * x + y * y <= self.joint_radius * self.joint_radius

    def draw(self, canvas):
        cv.circle(canvas, (self.pos_x, self.pos_y), self.joint_radius, self.color, -1)
        cv.line(canvas, (self.pos_x, self.pos_y), (self.end_x, self.end_y), self.color, 5)
        cv.circle(canvas, (self.pos_x, self.pos_y), self.joint_radius, (0, 0, 0), 2)
        cv.circle(canvas, (self.pos_x, self.pos_y), 2, (0, 0, 0), 2)

    def draw_hitbox(self, field, color, edges=None):
        cv.circle(field, (self.pos_x, self.pos_y), self.joint_radius, color, -1)
        angle = self.min_angle
        step = math.radians(1 if self.is_left else -1)
        while angle < self.max_angle if self.is_left else angle > self.max_angle:
            end_x = self.pos_x + int(math.sin(angle) * self.offset)
            end_y = self.pos_y + int(math.cos(angle) * self.offset)
            cv.line(field, (self.pos_x, self.pos_y), (end_x, end_y), color, 5)
            angle += step

    def deflect(self, ball: Ball, overlap: np.ndarray) -> int:
        # Thanks stack overflow: https://stackoverflow.com/a/1084899/7100223
        #
        # Taking
        # E is the starting point of the ray
        # L is the end point of the ray
        # C is the center of sphere you're testing against
        # r is the radius of that sphere
        # Compute:
        # d = L - E ( Direction vector of ray, from start to end )
        # f = E - C ( Vector from center sphere to ray start )
        #
        E = np.array([self.pos_x, self.pos_y], dtype=np.float)
        C = np.array([ball.px, ball.py], dtype=np.float)

        # First, check if a collision with the joint:
        hit, cf, v = _collide_circle(C, ball.r, E, self.joint_radius)
        if hit:
            cf = max(math.sqrt(cf), 2)
            ball.vx = v[0] * cf + random.random() * 0.01 - 0.005
            ball.vy = v[1] * cf + random.random() * 0.01 - 0.005
            return 0

        L = np.array([self.end_x, self.end_y], dtype=np.float)
        r = ball.r

        d = L - E
        f = E - C

        a = d.dot(d)
        b = 2 * f.dot(d)
        c = f.dot(f) - r * r

        # t² * (d DOT d) + 2t*( f DOT d ) + ( f DOT f - r2 ) = 0

        D = b * b - 4 * a * c

        if D < 0:
            return 0

        # print("Flipper intersect")

        D = math.sqrt(D)

        #  // either solution may be on or off the ray so need to test both
        #   // t1 is always the smaller value, because BOTH discriminant and
        #   // a are nonnegative.

        t1 = (-b - D)/(2*a)
        t2 = (-b + D)/(2*a)

        if 0 <= t1 <= 1:
            # print("Intersect 1")
            t = t1
        elif 0 <= t2 <= 1:
            # print("Intersect 2")
            t = t2
        else:
            # print("Euh...")
            # cv.waitKey(0)
            return False

        p = d * t + E

        # print("Pos intersect", p)

        n = np.array([-d[1], d[0]], dtype=np.float)
        n /= np.linalg.norm(n)
        v = np.array([ball.vx, ball.vy], dtype=np.float)
        r = v - 2 * (v.dot(n)) * n + 0.5 * n

        if self.moving_up:
            r *= 2

        # print('V:', v, '->', r, n)

        ball.vx = r[0]
        ball.vy = r[1]

        # cv.waitKey(0)

        return 0

    def step(self, is_touched: bool, lmb: bool = False, rmb: bool = False) -> bool:
        self.moving_up = False
        if self.is_left:
            if lmb:
                if self.angle < self.max_angle:
                    self.moving_up = True
                    self.angle += self.angle_step
                    self.update_end()
            else:
                if self.angle > self.min_angle:
                    self.angle -= self.angle_step
                    self.update_end()
        else:
            if rmb:
                if self.angle > self.max_angle:
                    self.moving_up = True
                    self.angle += self.angle_step
                    self.update_end()
            else:
                if self.angle < self.min_angle:
                    self.angle -= self.angle_step
                    self.update_end()
        return is_touched

    def update_end(self):
        self.end_x = self.pos_x + int(math.sin(self.angle) * self.offset)
        self.end_y = self.pos_y + int(math.cos(self.angle) * self.offset)

    def set_pos(self, x: int, y: int, rel: bool = False):
        if rel:
            self.pos_x += x
            self.pos_y += y
        else:
            self.pos_x = x
            self.pos_y = y
        self.update_end()


# test filpper stuff
if __name__ == '__main__':
    _field = np.zeros((300, 300), dtype=np.uint8)

    _flipper = Flipper(True, (25, 100))
    _flipper.draw_hitbox(_field, (127,), None)

    _flipper = Flipper(False, (275, 200))
    _flipper.draw_hitbox(_field, (255,), None)

    cv.imshow("Test Flippers Hitbox", _field)
    cv.waitKey(0)
    cv.destroyAllWindows()
