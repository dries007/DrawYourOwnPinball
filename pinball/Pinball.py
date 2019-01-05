"""
The main file for image->playfield logic and the game loop.

Run via the module file.
"""

import collections
import copy
import skimage.exposure

from .Webcam import Webcam
from .Elements import *


def draw_crosshair(frame: np.ndarray, center: (int, int), color=(255, 255, 255), radius=10):
    """
    Draw a basic crosshair
    """
    cv.circle(frame, center, radius, color)
    radius = int(radius * 1.5)  # Make the lines stick out from the circle
    cv.line(frame, (center[0] - radius, center[1]), (center[0] + radius, center[1]), color)
    cv.line(frame, (center[0], center[1] - radius), (center[0], center[1] + radius), color)


def find_largest_square_contour(frame, disp=None, filter_buffer: collections.deque or None = None, min_size = None) -> None or np.ndarray:
    """
    Find the largest square contour.
    :param frame: Input (BGR)
    :param disp: If not none, draw the contour on this image (BGR).
    :param filter_buffer: If not none, used as a filter list. Only edges in all of the images in the list are used.
    :param min_size: Minimal size of the rectangle.
    """
    # Source: https://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
    # Find edges
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = skimage.exposure.rescale_intensity(gray, out_range=(0, 255))
    edged: np.ndarray = cv.Canny(cv.bilateralFilter(gray, 9, 17, 17), 30, 200)
    # cv.imshow("Raw Edges", edged)

    if filter_buffer is not None:
        # Merge close edges, helps cut out frame to frame variations
        edged = cv.dilate(edged, np.ones((5, 5), np.uint8), iterations=2)
        edged = cv.erode(edged, np.ones((5, 5), np.uint8), iterations=1)
        # cv.imshow("Edges", edged)
        # Filter out all pixels not in all of the buffered frames and the current one.
        filtered: np.ndarray = edged.copy()
        try:
            for e in filter_buffer:
                filtered &= e
        except ValueError:
            # In case the size changes (due to rotate)
            filter_buffer.clear()
        filter_buffer.append(edged)
        # cv.imshow("Buffered Edges", filtered)
        # Continue with filtered image.
        edged = filtered

    _, contours, _ = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(disp, contours, -1, (255, 0, 0), 1)
    # Find the largest contour with 4 corners ("square")
    sheet_contour = None
    contours = [(c, cv.contourArea(c)) for c in contours]
    if min_size:
        contours = filter(lambda x: x[1] >= 102400, contours)
    contours = sorted(contours, key=lambda x: x[1], reverse=True)
    for c, _ in contours:
        approx = cv.approxPolyDP(c, 0.015 * cv.arcLength(c, True), True)
        if len(approx) == 4:
            sheet_contour = approx
            break
    if sheet_contour is not None and disp is not None:
        cv.drawContours(disp, [sheet_contour], -1, (255, 0, 255), 1)
    return sheet_contour


def points2rect(points: np.ndarray, dtype="float32") -> np.ndarray:
    """
    Order 4 points into a rectangle (tl, tr, br, bl)
    """
    # Source: https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
    # the top-left point has the smallest sum whereas the bottom-right has the largest sum
    s = points.sum(axis=1)
    # compute the difference between the points -- the top-right will have the minumum difference and the bottom-left
    # will have the maximum difference
    d = np.diff(points, axis=1)
    # Return in the correct order and type
    points = [points[np.argmin(s)], points[np.argmin(d)], points[np.argmax(s)], points[np.argmax(d)]]
    return np.array(points, dtype=dtype)


def detect_circles(img, min_dist, min_radius, max_radius, th_canny_high=100, th_acc=30) -> [(float, float, float)]:
    """
    Wrapper around openCV's HoughCircles

    :param img: greyscale image
    :param min_dist: Minimal distance from one circle to another
    :param min_radius: Minimum radius
    :param max_radius: Maximum radius or negative number if you only want the centers
    :param th_canny_high: higher threshold for the canny filter (the lower one is twice smaller)
    :param th_acc: accumulator threshold for the circle centers at the detection stage
    :return: A list of tuples (actually a numpy 3 col matrix of floats OR an empty list)
    """
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, min_dist, None, th_canny_high, th_acc, min_radius, max_radius)
    if circles is None:
        return []
    return circles[0]


def mouse_empty(*_):
    """
    Since you can't "disable" a mouse callback, just set an empty one.
    """
    pass


def wait_enter(*args):
    """
    Better than wait(0) because python openCV can't do exits etc then
    :param args:
    :return:
    """
    print(*args, sep='\n')
    while True:
        key = cv.waitKey(10)
        if key == 27:  # Escape
            raise KeyboardInterrupt("Pressed ESC")
        elif key == 13:  # Enter
            break


class Pinball:
    """
    King of the Monsters

    To be used with Python's 'with' statement/block.
    You must call 'setup' and 'play' in sequence.

    The game works by using a 2d field of indexes of the elements on the playing field.
    The ball is moved over that field and any non-zero (wall=0) entry is considered a potential hit.
    """
    name: str
    _webcam: Webcam
    width: int
    height: int
    elements: [Element]
    field: np.ndarray
    disp: np.ndarray

    def __init__(self):
        self.name = type(self).__name__

    # Resource management
    def __enter__(self):
        self._webcam = Webcam(show_debug=False)
        self._webcam.set_resolution(1280, 800)
        cv.namedWindow(self.name, cv.WINDOW_AUTOSIZE | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._webcam
        cv.destroyAllWindows()

    # Misc methods
    def _set_size(self, height, width):
        """
        Arguments in order of the numpy shape output
        """
        self.width = width
        self.height = height  # print("Field Size:", self.width, self.height)

    def setup(self, load_field=None):
        """
        The setup is split into chunks to avoid it becoming even more of a monster.
        :param load_field:
        :return:
        """
        if load_field:
            frame = cv.imread(load_field)
        else:
            frame = self._setup_p1()
            cv.imwrite('field.png', frame)

        edges = self._setup_p2(frame)
        self.field, self.disp = self._setup_p3(frame, edges)

    def _setup_dragging(self, frame: np.ndarray, skip_elements=0) -> bool:
        """
        Allow dragging/deleting of elements during the setup phase.
        Intended to be used in a while loop with a backup copy of the element list handy.
        :param frame: Input frame, is not modified
        :return: keep changes? If False, the user canced the action.
        """
        print("Drag the detected elements to their proper place.")
        print("\tRight click on an element to deleted it.")
        print("\tPress R to reset the step.")
        print("\tPress Enter to confirm")

        # Serve as 'static' variables for mouse callback.
        dragging: Element = None
        px, py = None, None

        def mouse(event, x, y, *_):
            # Allow parent variables to be modified
            nonlocal dragging, px, py, self

            if event == cv.EVENT_MOUSEMOVE:
                if dragging is None:
                    return
                dragging.set_pos(x - px, y - py, True)
            elif event == cv.EVENT_LBUTTONDOWN:
                for e in self.elements[skip_elements:]:
                    if e.can_grab(x, y):
                        dragging = e
                        break
            elif event == cv.EVENT_LBUTTONUP:
                if dragging is not None:
                    dragging.set_pos(max(min(dragging.pos_x, self.width), 0), max(min(dragging.pos_y, self.height), 0))
                    dragging = None
            elif event == cv.EVENT_RBUTTONUP:
                self.elements = self.elements[:skip_elements] + \
                                [e for e in self.elements[skip_elements:] if not e.can_grab(x, y)]

            px = x
            py = y

        cv.setMouseCallback(self.name, mouse)

        while True:
            draw = frame.copy()
            for e in self.elements:
                e.draw(draw)
            cv.imshow(self.name, draw)
            key = cv.waitKey(1)
            if key != -1:
                # print("Key pressed:", key)
                if key == 27:  # Escape
                    raise KeyboardInterrupt("Pressed ESC")
                elif key == 13:  # Enter = Confirm
                    ok = True
                    break
                elif key == ord('r'):  # Reset
                    ok = False
                    break
                else:
                    print("Keypress unknown.", key, chr(key))

        # Reset mouse callback
        cv.setMouseCallback(self.name, mouse_empty)
        return ok

    def _setup_p1(self) -> np.ndarray:
        """
        Setup Part 1: Grabbing the playing field
        :return: The playing field image data
        """
        print("Setup Part 1: Make sure the entire playing field is in frame.")
        print("\tPress Enter to confirm and setup the playing field.")
        print("\tPress F to toggle between the different flipping modes. (H, V, HV)")
        print("\tPress R to toggle between the different rotation modes. (0°, 90°, 180°, 270°)")
        frame: np.ndarray
        field: np.ndarray
        edge_buffer = collections.deque(maxlen=3)

        for frame in self._webcam:
            # Handle kb input first
            key = cv.waitKey(1)
            if key != -1:
                # print("Key pressed:", key)
                if key == 27:  # Escape
                    raise KeyboardInterrupt("Pressed ESC")
                elif key == 13:  # Enter
                    if field is None:
                        print("No contour is found. Cannot continue.")
                        continue
                    break
                elif key == ord('f'):
                    print("Toggle Flip to", self._webcam.toggle_flip())
                elif key == ord('r'):
                    print("Toggle Rotate to", self._webcam.toggle_rotate())
                else:
                    print("Keypress unknown.", key, chr(key))
            # Find largest square contour
            disp = frame.copy()
            # >= 10% of the screen should be filled with the field (1280*800 = 1024000)
            field = find_largest_square_contour(frame, disp=disp, filter_buffer=edge_buffer, min_size=102400)
            cv.imshow(self.name, disp)

        # Warp the field part for the frame to a rectangle
        # Source: https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
        rect = points2rect(field.reshape(4, 2))
        tl, tr, br, bl = rect
        width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        width = max(int(width1), int(width2))
        height = max(int(height1), int(height2))
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        matrix = cv.getPerspectiveTransform(rect, dst)
        frame = cv.warpPerspective(frame, matrix, (width, height))
        # Cut off a margin so the edges of the sheet get discarded.
        margin = 10
        frame = frame[margin:-margin, margin:-margin]
        return frame

    def _setup_p2(self, frame: np.ndarray) -> np.ndarray:
        """
        Setup Part 2: Detect playing field objects
        :return the raw frame data, the other stuff is stored on the Pinball object
        """
        print("Setup Part 2: Detecting objects")
        self._set_size(*frame.shape[:2])
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = skimage.exposure.rescale_intensity(gray, out_range=(0, 255))
        gray = cv.medianBlur(gray, 5)
        edged = cv.Canny(cv.bilateralFilter(gray, 9, 17, 17), 30, 200)
        edged = cv.dilate(edged, np.ones((3, 3), np.uint8), iterations=5)
        edged = cv.erode(edged, np.ones((3, 3), np.uint8), iterations=2)

        # Holds all playfield elements, the initial ones are special.
        self.elements = [
            Background(),
            Wall(),

        ]
        skip_elements = len(self.elements)

        print("Move flippers by dragging them (click on the circle) ...")

        self.elements.append(Flipper(True, (100, self.height - 100)))
        self.elements.append(Flipper(False, (self.width - 100, self.height - 100)))

        # Backup and allow movement of new elements
        backup_elements = copy.deepcopy(self.elements)
        while not self._setup_dragging(frame, skip_elements=skip_elements):
            self.elements = copy.deepcopy(backup_elements)
        # From now, these first elements are stuck.
        skip_elements = len(self.elements)

        print("Detecting pop bumpers...")
        for cx, cy, r in detect_circles(gray, 30, 25, 50):
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            self.elements.append(PopBumper((int(cx), int(cy)), color, int(r)))

        # Backup and allow movement of new elements
        backup_elements = copy.deepcopy(self.elements)
        while not self._setup_dragging(frame, skip_elements=skip_elements):
            self.elements = copy.deepcopy(backup_elements)
        # From now, these first elements are stuck.
        skip_elements = len(self.elements)

        edged_filtered = edged.copy()
        for e in self.elements:
            e.mask_away(edged_filtered)

        # todo: Allow other things to be detected. (Not enough time)

        # # Now finding Contours
        # _, contours, _ = cv.findContours(edged_filtered, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        # contours_poly = [cv.approxPolyDP(c, 0.015 * cv.arcLength(c, True), True) for c in contours]
        #
        # cv.drawContours(frame, contours, -1, (0, 255, 255))
        # cv.drawContours(frame, contours_poly, -1, (255, 0, 255))

        #
        # # lsd: cv.LineSegmentDetector = cv.createLineSegmentDetector()
        # # lines, width, prec, nfa = lsd.detect(gray)
        # #
        # # drawn = lsd.drawSegments(frame, lines)
        #
        # rho = 1  # distance resolution in pixels of the Hough grid
        # theta = np.pi / 180  # angular resolution in radians of the Hough grid
        # threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        # min_line_length = 10  # minimum number of pixels making up a line
        # max_line_gap = 10  # maximum gap in pixels between connectable line segments
        # line_image = gray.copy()  # creating a blank to draw lines on
        #
        # # Run Hough on edge detected image
        # # Output "lines" is an array containing endpoints of detected line segments
        # lines = cv.HoughLinesP(edged_filtered, rho, theta, threshold, None, min_line_length, max_line_gap)
        #
        # for line in lines:
        #     for x1, y1, x2, y2 in line:
        #         cv.line(line_image, (x1, y1), (x2, y2), (255, ), 1)
        # #
        # lines_edges = cv.addWeighted(frame, 0.8, line_image, 1, 0)

        # cv.imshow(self.name, frame)
        # cv.imshow(self.name + " Raw Edges", edged)
        # cv.imshow(self.name + " Filtered Edges", edged_filtered)

        # cv.imshow(self.name + " Gray", gray)
        # cv.imshow(self.name + " Lines", line_image)
        # while True:
        #     key = cv.waitKey(0)
        #     if key != -1:
        #         # print("Key pressed:", key)
        #         if key == 27:  # Escape
        #             raise KeyboardInterrupt("Pressed ESC")
        #         elif key == 13:  # Enter = Confirm
        #             break
        #         else:
        #             print("Keypress unknown.", key, chr(key))

        return edged_filtered

    def _setup_p3(self, frame: np.ndarray, edges: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Setup Step 3: Draw the playing field index map
        """

        if len(self.elements) > 0xFF:
            raise RuntimeError("You cannot have more than 255 elements on the field.")

        # Lookup table (index->color) for display reasons only
        lut = np.zeros((256, 3), dtype=np.uint8)

        # Playing field = index table of elements at each position
        field = np.ndarray((self.height, self.width), dtype=np.uint8)
        field.fill(0)

        # Setup field & colors
        for i, e in enumerate(self.elements):
            e.draw_hitbox(field, i, edges)
            lut[i] = e.color

        # Turn the 1d index table into a colored image
        field3 = cv.cvtColor(field, cv.COLOR_GRAY2BGR)
        disp = cv.LUT(field3, lut.reshape((256, 1, 3)))

        # To make it easier to see what's being detected as a wall, copy those over to the display image.
        disp = cv.addWeighted(frame, 1, disp, 0.15, 0)

        # for e in self.elements:
        #     e.draw(disp)

        # cv.imshow(self.name, disp)

        # while True:
        #     key = cv.waitKey(0)
        #     if key != -1:
        #         # print("Key pressed:", key)
        #         if key == 27:  # Escape
        #             raise KeyboardInterrupt("Pressed ESC")
        #         elif key == 13:  # Enter = Confirm
        #             break
        #         else:
        #             print("Keypress unknown.", key, chr(key))

        return field, disp

    def play(self, lives=999):
        """
        The main game loop
        :param lives: How many lives you start with
        :return: The score you got
        """
        score = 0

        def mouse(event, *_):
            nonlocal lmb, rmb
            if event == cv.EVENT_LBUTTONDOWN:
                lmb = True
            elif event == cv.EVENT_LBUTTONUP:
                lmb = False
            elif event == cv.EVENT_RBUTTONDOWN:
                rmb = True
            elif event == cv.EVENT_RBUTTONUP:
                rmb = False

        while lives > 0:
            print(lives, "balls left.")
            lives -= 1
            for e in self.elements:
                e.reset()
            ball = self._spawn_ball()
            lmb = False
            rmb = False
            cv.setMouseCallback(self.name, mouse)
            while not ball.is_out_of_bounds(self.width, self.height):
                if cv.waitKey(1) == 27:  # Escape
                    raise KeyboardInterrupt("Pressed ESC")
                ball, points = self._step(ball, lmb, rmb)
                score += points
                ball.move()
            cv.setMouseCallback(self.name, mouse_empty)

            frame = self.disp.copy()
            ball.draw(frame)
            for e in self.elements:
                e.draw(frame)
            cv.imshow(self.name, frame)

            print("Ball lost! Score so far:", score)
        print("Game over!")
        return score

    def _spawn_ball(self) -> Ball:
        """
        Spawn a ball where the user clicks
        :return:
        """
        wait_enter("Press enter to start placing a ball.")
        print("Click where you want to spawn the ball...")
        done: bool = False
        ball: Ball = None

        def mouse(event, x, y, *_):
            nonlocal ball, done, self
            if x <= 0 or y <= 0 or x >= self.width or y >= self.height:
                return
            if event == cv.EVENT_MOUSEMOVE:
                if ball is not None:
                    ball.px = x
                    ball.py = y
            elif event == cv.EVENT_LBUTTONDOWN:
                ball = Ball(x, y)
            elif event == cv.EVENT_LBUTTONUP:
                if ball is None:
                    ball = Ball(x, y)
                else:
                    ball.px = x
                    ball.py = y
                done = True

        cv.setMouseCallback(self.name, mouse)

        while not done:
            frame = self.disp.copy()
            for e in self.elements:
                e.draw(frame)
            if ball is not None:
                ball.draw(frame)
            cv.imshow(self.name, frame)
            key = cv.waitKey(1)
            if key != -1:
                # print("Key pressed:", key)
                if key == 27:  # Escape
                    raise KeyboardInterrupt("Pressed ESC")
                else:
                    print("Keypress unknown.", key, chr(key))
        cv.setMouseCallback(self.name, mouse_empty)
        wait_enter("Press enter to start placing a ball.")
        return ball

    def _step(self, ball: Ball, lmb: bool, rmb: bool) -> (Ball, int):
        """
        A single game loop step
        :param ball:
        :param lmb:
        :param rmb:
        :return:
        """
        points = 0
        # Super Advanced Collision detection
        overlap = self.field[int(ball.py-ball.r):int(ball.py+ball.r-1), int(ball.px-ball.r):int(ball.px+ball.r-1)] & ball.mask
        overlap_indexes = np.transpose(overlap.nonzero())
        overlap_elements = {self.elements[overlap[y, x]] for y, x in overlap_indexes}
        frame = self.disp.copy()

        ball.draw(frame)
        for e in self.elements:
            e.draw(frame)
            if e.step(e in overlap_elements, lmb, rmb):
                points += e.deflect(ball, overlap)

        cv.imshow(self.name, frame)
        return ball, points
