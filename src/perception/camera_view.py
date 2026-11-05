import cv2
import numpy as np


### live window of what robot sees
WINDOW = "wrist camera"
PERIOD = 0.1                  
FRESH = 0.15                  
SCALE = 0.75                  
DEPTH_SHOWN = (0.15, 0.55)    
SIGMAS = 2                    

### colours (bgr)
GREEN = (80, 220, 80)
YELLOW = (0, 220, 255)
RED = (60, 60, 255)
WHITE = (255, 255, 255)
CYAN = (255, 220, 0)


### depth image 
def colour_depth(depth):
    scaled = (depth - DEPTH_SHOWN[0]) / (DEPTH_SHOWN[1] - DEPTH_SHOWN[0])
    image = cv2.applyColorMap((255 * (1 - np.clip(scaled, 0, 1))).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[depth <= 0] = 0
    return image


### text with dark outline
def text(image, words, where, colour=WHITE, size=0.6):
    cv2.putText(image, words, where, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, words, where, cv2.FONT_HERSHEY_SIMPLEX, size, colour, 1, cv2.LINE_AA)



class camera_view:
    def __init__(self, arm, camera, locator):
        self.arm = arm
        self.camera = camera
        self.locator = locator              
        self.task = None                   
        self.title = ""
        self.last = -np.inf
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    ### world points 
    def pixels(self, points, T_world_cam):
        p = (np.atleast_2d(points) - T_world_cam.t) @ T_world_cam.R  
        p = p[p[:, 2] > 0.01]
        p = p[np.all(np.abs(p[:, :2] / p[:, 2:3]) < 0.6, axis=1)]  
        if len(p) == 0:
            return np.zeros((0, 2)), p
        uv, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), self.locator.K, self.locator.dist)
        return uv.reshape(-1, 2), p

    ### called by arm at every viewpoint
    def __call__(self):
        if self.task is None or self.arm.time - self.last < PERIOD - 1e-9:
            return
        self.last = self.arm.time
        frame = self.task.frames[-1] if self.task.frames else None
        if frame is not None and self.task.seen is not None and self.arm.time - frame["t"] < FRESH:
            rgb, depth, found = self.task.seen  
        else:
            rgb, depth = self.camera.read(noise=False)  
            found = None
        self.draw(rgb, depth, found, frame)

    ### draw one frame
    def draw(self, rgb, depth, found, frame):
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        T_wc = self.task.camera_pose()
        kf = self.task.filter
        events = self.task.events
        doing = events[-1][1] if events else ""
        looking = frame is not None and self.arm.time - frame["t"] < FRESH

        ### detection of cluster outline and lid
        if found is not None:
            uv, _ = self.pixels(found["cluster"], found["camera"])
            if len(uv) > 2:
                hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
                cv2.polylines(image, [hull], True, GREEN, 2, cv2.LINE_AA)
            lid, _ = self.pixels(found["lid"], found["camera"])
            if len(lid):
                cv2.drawMarker(image, tuple(int(c) for c in lid[0]), GREEN, cv2.MARKER_TILTED_CROSS, 14, 2)

        ### filter estimate
        if kf.ready:
            centre, p = self.pixels(kf.estimate, T_wc)
            if len(centre):
                sigma = np.sqrt(np.diag(kf.covariance()))
                radius = SIGMAS * sigma[:2].max() * self.locator.K[0, 0] / p[0, 2]
                c = tuple(int(v) for v in centre[0])
                cv2.circle(image, c, max(3, int(radius)), YELLOW, 2, cv2.LINE_AA)

                ### pbvs
                if doing == "pbvs approach":
                    goal = tuple(int(v) for v in self.locator.K[:2, 2])
                    cv2.drawMarker(image, goal, CYAN, cv2.MARKER_CROSS, 30, 2)
                    cv2.arrowedLine(image, c, goal, CYAN, 1, cv2.LINE_AA, tipLength=0.05)

        ### labels/text
        text(image, self.title, (10, 25))
        text(image, doing, (10, 50), RED if "lost" in doing or "again" in doing or "missed" in doing else YELLOW)
        if looking:
            status = frame["status"]
            text(image, f"detection: {status}", (10, 75), RED if status in ("rejected", "none") else GREEN)
        error = next((f for f in reversed(self.task.frames) if f["error"] is not None), None)
        if doing == "pbvs approach" and error is not None:
            text(image, f"e_t {error['error'] * 1000:6.1f} mm   e_theta {np.degrees(error['angle']):5.1f} deg", (10, 455), CYAN)
        if kf.ready:
            sigma = np.sqrt(np.diag(kf.covariance())) * 1000
            text(image, f"sigma {sigma[0]:.1f} {sigma[1]:.1f} {sigma[2]:.1f} mm", (400, 25), YELLOW)

        ### depth beside it
        coloured = colour_depth(depth)
        text(coloured, "depth (black = no return)", (10, 25))
        if np.mean(depth > 0) < 0.05:
            text(coloured, "too close for the d435", (10, 50), RED)
        both = np.hstack([image, coloured])
        both = cv2.resize(both, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        cv2.imshow(WINDOW, both)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow(WINDOW)