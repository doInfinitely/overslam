"""Frontier-based explorer for Mei Cartographer.

Algorithm:
  1. Build a 2D nav grid from the voxel cloud: project to XZ, keep cells
     whose vertical voxel span is < WALL_HEIGHT (floor) and discard tall
     columns (walls).  Cache the grid and rebuild every NAV_REBUILD ticks.
  2. Score frontier cells (coverage boundary) by number of unseen 8-neighbors.
  3. A* from current position to the best reachable frontier cell.  If the
     best cell is unreachable, walk to the closest reachable cell to it.
  4. Follow the path: steer yaw toward next waypoint + gentle look-around
     oscillation so we accumulate geometry off the direct heading.
  5. Reassess when: path exhausted, target reached, stuck for STALE_LIMIT
     ticks, or NAV_REASSESS ticks elapsed.
"""
from __future__ import annotations

import heapq
import math

import numpy as np


# --------------------------------------------------------------------------
# Tuning knobs
# --------------------------------------------------------------------------

OBS_MIN       = 0.15  # m  – ignore voxels this close to column floor (noise)
OBS_MAX       = 1.90  # m  – Mei body height: voxels above floor+OBS_MIN and
                      #       below floor+OBS_MAX mean the cell is impassable
CLEARANCE     = 0.4   # m  – inflate blocked cells outward by this much
NAV_REBUILD   = 15    # ticks between nav-grid rebuilds
NAV_REASSESS  = 12    # ticks between path reassessments
STALE_LIMIT   = 2     # stale ticks before triggering escape (was 4)
REACH_DIST    = 0.6   # m  – distance to consider a waypoint "reached"
MAX_TURN_DEG  = 40    # °  – max yaw correction per tick
LOOK_AMP_DEG  = 12    # °  – look-around oscillation amplitude (while walking only)
LOOK_FREQ     = 0.28  # rad/tick – look-around frequency (slower = wider sweeps)
FRONTIER_TOP  = 32    # check only the top-N frontier cells for A*
MOTION_STUCK  = 4.0   # mean pixel diff below this = bot didn't move during burst
LOOKAHEAD     = 1     # cells ahead to check before committing to walk (1 = 0.5m)


class FrontierExplorer:
    def __init__(self, rad_per_px: float, cov_cell: float = 0.5):
        self.rad_per_px = rad_per_px
        self.cov_cell   = cov_cell

        self._nav: set[tuple[int,int]]     = set()
        self._blocked: set[tuple[int,int]] = set()   # cells with obstacle voxels
        self._nav_age   = NAV_REBUILD    # force rebuild on first tick
        self.path: list[tuple[int,int]] = []
        self.target: tuple[int,int] | None = None
        self.reassess_in = 0
        self.stale   = 0
        self.t       = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _px(self, rad: float) -> int:
        return int(rad / self.rad_per_px)

    def _xz_to_ij(self, x: float, z: float) -> tuple[int,int]:
        cs = self.cov_cell
        return (int(math.floor(x / cs)), int(math.floor(z / cs)))

    def _ij_to_xz(self, i: int, j: int) -> tuple[float,float]:
        cs = self.cov_cell
        return ((i + 0.5) * cs, (j + 0.5) * cs)

    def _build_nav(self, cloud) -> set[tuple[int,int]]:
        """Project voxel cloud to 2D; keep cells with floor but no body-height
        obstacle (OBS_MIN..OBS_MAX above column floor).

        A 1m box has voxels at 0-1m above floor → blocked (0.15-1m lands in
        the obstacle band).  An open floor has voxels only at floor_y →
        navigable.  A tall wall also has voxels in the band → blocked.
        """
        pts = cloud.points()   # Mx3 float32
        if len(pts) == 0:
            return set(), set()

        cs = self.cov_cell
        ci = np.floor(pts[:, 0] / cs).astype(np.int32)
        cj = np.floor(pts[:, 2] / cs).astype(np.int32)
        y  = pts[:, 1]

        # Encode (ci, cj) as a single int64 for fast groupby-sort
        SHIFT = 20000
        encoded = (ci.astype(np.int64) + SHIFT) * 40000 + (cj.astype(np.int64) + SHIFT)
        order   = np.argsort(encoded)
        enc_s   = encoded[order]
        y_s     = y[order]

        bounds  = np.where(np.diff(enc_s) != 0)[0] + 1
        segs    = np.split(y_s, bounds)
        keys    = enc_s[np.concatenate([[0], bounds])]

        nav: set[tuple[int,int]] = set()
        blocked: set[tuple[int,int]] = set()

        for k, seg in zip(keys, segs):
            floor_y = float(seg.min())
            # Any voxel in body-clearance band → impassable
            in_band = seg[(seg > floor_y + OBS_MIN) & (seg < floor_y + OBS_MAX)]
            i = int(k // 40000) - SHIFT
            j = int(k  % 40000) - SHIFT
            if len(in_band) > 0:
                blocked.add((i, j))
            else:
                nav.add((i, j))

        # Thin wall buffer — 1 cell only, avoid destroying narrow corridors
        padded: set[tuple[int,int]] = set()
        for bi, bj in blocked:
            for di, dj in ((-1,0),(1,0),(0,-1),(0,1)):   # 4-connected only
                padded.add((bi + di, bj + dj))
        nav -= padded
        return nav, blocked

    def _score_frontiers(self, frontier_cells, seen: set) -> list[tuple]:
        """Score frontier cells for exploration priority.

        Three signals combined:
          unseen_density  — count of unseen 8-neighbors (existing signal).
          corner_penalty  — number of blocked cardinal sides; corners/dead-ends
                            have 2-4 blocked sides and are expensive to escape.
          corridor_bonus  — length of the longest unobstructed run in any
                            cardinal direction; hallway entrances score highest.
        """
        nav     = self._nav
        blocked = self._blocked
        scored  = []
        for (i, j) in frontier_cells:
            unseen = sum(
                1 for di in (-1, 0, 1) for dj in (-1, 0, 1)
                if (di or dj) and (i + di, j + dj) not in seen
            )

            blocked_cardinal = sum(
                1 for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if (i + di, j + dj) in blocked
            )

            # Longest free run in any cardinal direction (up to 8 cells = 4m).
            # A hallway entrance has a long run; a dead end has run ≤ 1.
            max_run = 0
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                run = 0
                for k in range(1, 9):
                    if (i + di * k, j + dj * k) in nav:
                        run += 1
                    else:
                        break
                max_run = max(max_run, run)

            score = (unseen
                     - 2 * max(0, blocked_cardinal - 1)   # 0 pen for ≤1 wall
                     + 2 * max_run)                        # corridor reward
            scored.append(((i, j), score))

        scored.sort(key=lambda x: -x[1])
        return scored

    def _astar(self, start: tuple[int,int],
                goal: tuple[int,int]) -> list[tuple[int,int]]:
        """8-connected A* on self._nav. Returns waypoint list (excl. start)."""
        nav = self._nav
        if not nav:
            return []

        # If goal unreachable, find closest nav cell to it
        if goal not in nav:
            goal = min(nav,
                       key=lambda c: (c[0] - goal[0])**2 + (c[1] - goal[1])**2)

        if start == goal:
            return []
        if start not in nav:
            # Snap start to nearest nav cell
            start = min(nav,
                        key=lambda c: (c[0] - start[0])**2 + (c[1] - start[1])**2)

        def h(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # heap: (f, g, node, path)
        heap = [(h(start, goal), 0.0, start, [])]
        visited: set[tuple[int,int]] = set()
        best_reached = (start, [])

        while heap:
            f, g, cur, path = heapq.heappop(heap)
            if cur in visited:
                continue
            visited.add(cur)
            full = path + [cur]

            # Track closest-to-goal reached node for fallback
            if h(cur, goal) < h(best_reached[0], goal):
                best_reached = (cur, full)

            if cur == goal:
                return full[1:]   # skip start

            for di, dj in ((-1,0),(1,0),(0,-1),(0,1),
                           (-1,-1),(-1,1),(1,-1),(1,1)):
                nb = (cur[0] + di, cur[1] + dj)
                if nb not in nav or nb in visited:
                    continue
                cost = 1.0 if (di == 0 or dj == 0) else 1.414
                ng = g + cost
                heapq.heappush(heap, (ng + h(nb, goal), ng, nb, full))

        # Goal unreachable — return path to closest reached cell
        return best_reached[1][1:]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _wall_ahead(self, cam_x: float, cam_z: float, yaw: float) -> bool:
        """Return True if any of the LOOKAHEAD cells directly ahead are blocked."""
        fwd_x = math.sin(yaw)
        fwd_z = math.cos(yaw)
        cs = self.cov_cell
        for step in range(1, LOOKAHEAD + 1):
            cx = cam_x + fwd_x * cs * step
            cz = cam_z + fwd_z * cs * step
            ij = self._xz_to_ij(cx, cz)
            if ij in self._blocked:
                return True
        return False

    def best_open_yaw(self, cam_x: float, cam_z: float,
                      n_dirs: int = 16, lookahead: int = 12
                      ) -> "tuple[float, int]":
        """Best heading (yaw) based on nav-grid free-run length.

        Samples n_dirs evenly around 360°.  For each direction counts
        consecutive cells that are in nav and NOT in blocked (so freshly
        mark_obstacle_ahead() cells are already excluded).

        Returns (best_yaw_radians, best_run_length).
        Caller can check run_length == 0 to detect a fully surrounded bot."""
        cs      = self.cov_cell
        nav     = self._nav
        blocked = self._blocked
        best_yaw = 0.0
        best_run = -1
        for k in range(n_dirs):
            angle = 2.0 * math.pi * k / n_dirs
            dy    = math.sin(angle)
            dz    = math.cos(angle)
            run   = 0
            for step in range(1, lookahead + 1):
                ij = self._xz_to_ij(cam_x + dy * cs * step,
                                    cam_z + dz * cs * step)
                if ij in nav and ij not in blocked:
                    run += 1
                else:
                    break
            if run > best_run:
                best_run = run
                best_yaw = angle
        return best_yaw, best_run

    def mark_obstacle_ahead(self, cam_x: float, cam_z: float, yaw: float,
                            steps: int = 2):
        """Ground-truth wall contact: mark the cell(s) directly ahead as
        blocked so A* routes around them on the very next reassess."""
        fwd_x = math.sin(yaw)
        fwd_z = math.cos(yaw)
        cs = self.cov_cell
        for step in range(1, steps + 1):
            ij = self._xz_to_ij(cam_x + fwd_x * cs * step,
                                 cam_z + fwd_z * cs * step)
            self._blocked.add(ij)
            self._nav.discard(ij)

    def clear_obstacles_near(self, cam_x: float, cam_z: float, radius: float = 2.0):
        """Remove manually-marked blocked cells within radius metres of cam.
        Called after a backup burst so A* can route through the cleared area."""
        cs  = self.cov_cell
        r_c = int(math.ceil(radius / cs))
        ci  = int(math.floor(cam_x / cs))
        cj  = int(math.floor(cam_z / cs))
        for di in range(-r_c, r_c + 1):
            for dj in range(-r_c, r_c + 1):
                self._blocked.discard((ci + di, cj + dj))
        self.path   = []
        self.target = None
        self.reassess_in = 0

    def report_motion(self, score: float):
        """Legacy frame-diff stuck signal (mean abs pixel delta on thumbnail)."""
        if score < MOTION_STUCK:
            self.stale += 1

    def report_wall_contact(self):
        """Multi-signal wall detection: radial flow + pixel diff + center depth
        all agree the camera didn't translate.  Force immediate reassess."""
        self.stale  = max(self.stale + 2, STALE_LIMIT)
        self.path   = []    # drop current path — it drove us into a wall
        self.target = None
        self.reassess_in = 0
        self.t      = 0     # reset oscillation phase so first post-reorient step has no twitch

    def apply_radar_obstacles(self, flow_radar: "np.ndarray",
                              scan_base_yaw: float,
                              cam_x: float, cam_z: float,
                              block_steps: int = 2) -> None:
        """Mark nav-grid cells as blocked based on radar wall detections.

        The top-quartile bins (highest flow = most texture = nearest walls)
        are treated as confirmed obstacles and their forward cells blocked.
        This gives A* full 360° wall awareness after every scan, not just
        the single forward direction from mark_obstacle_ahead().
        """
        import math, numpy as np
        n = len(flow_radar)
        valid = flow_radar[~np.isnan(flow_radar)]
        if not len(valid):
            return
        wall_thresh = float(np.percentile(valid, 75))
        cs = self.cov_cell
        for b in range(n):
            if np.isnan(flow_radar[b]) or flow_radar[b] < wall_thresh:
                continue
            world_yaw = scan_base_yaw + b * 2 * math.pi / n
            fwd_x = math.sin(world_yaw)
            fwd_z = math.cos(world_yaw)
            for step in range(1, block_steps + 1):
                ij = self._xz_to_ij(cam_x + fwd_x * cs * step,
                                    cam_z + fwd_z * cs * step)
                self._blocked.add(ij)
                self._nav.discard(ij)

    def radar_best_yaw(self, flow_radar: "np.ndarray") -> "float | None":
        """Find the centre of the longest clear arc in the radar.

        Clear bins = bottom quartile of flow magnitude (low texture = open).
        Returns the bot-relative yaw in radians toward the arc centre,
        or None if the radar has no usable data.
        Prefer a wide arc over a narrow one: a 30° opening is safer to
        drive into than a 10° slot even if the slot has lower flow.
        """
        import math, numpy as np
        n = len(flow_radar)
        valid = flow_radar[~np.isnan(flow_radar)]
        if not len(valid):
            return None
        mag_lo = float(np.percentile(valid, 25))
        is_open = np.array(
            [(not np.isnan(flow_radar[b])) and (flow_radar[b] <= mag_lo)
             for b in range(n)], dtype=bool)

        # Find longest consecutive run of open bins (circular wrap).
        best_len, best_start = 0, 0
        cur_len, cur_start = 0, 0
        for i in range(2 * n):
            if is_open[i % n]:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
            else:
                cur_len = 0

        if best_len == 0:
            # No clearly open bin — fall back to global minimum.
            best_b = int(np.nanargmin(flow_radar))
        else:
            best_b = (best_start + best_len // 2) % n

        rel_rad = best_b * 2 * math.pi / n
        if rel_rad > math.pi:
            rel_rad -= 2 * math.pi
        return rel_rad

    def propose(self, cloud, coverage, pose, added_voxels: int) -> dict:
        """Return an action plan dict: {hold, look, tap}.

        Parameters
        ----------
        cloud       : G.VoxelCloud
        coverage    : G.CoverageGrid
        pose        : (yaw, pitch, t_vec) current camera pose
        added_voxels: voxels added this tick (geometry novelty signal)
        """
        self.t += 1
        yaw, pitch, t_vec = pose
        cam_x, cam_z = float(t_vec[0]), float(t_vec[2])
        cur_ij = self._xz_to_ij(cam_x, cam_z)

        # Stale: count when following a path OR during the no-path forward walk.
        # Low voxels = already-mapped geometry = probably pressed against a wall.
        # Motion-based stale increments come from report_motion() (called after burst).
        if added_voxels < 30:
            self.stale += 1
        elif added_voxels >= 30:
            self.stale = 0

        # Rebuild nav grid periodically; always fold in visited cells (ground
        # truth — if the bot stood there it's walkable regardless of geometry).
        self._nav_age += 1
        if self._nav_age >= NAV_REBUILD:
            self._nav, self._blocked = self._build_nav(cloud)
            self._nav_age = 0
        self._nav |= coverage.visited   # visited overrides any block classification

        # ---- Reassess path? -----------------------------------------------
        need_reassess = (
            self.reassess_in <= 0
            or self.target is None
            or not self.path
            or self.stale >= STALE_LIMIT
        )

        if need_reassess:
            if self.stale >= STALE_LIMIT:
                # Stuck: drop plan and reassess immediately.
                # Path-following look command handles the turn toward the new
                # frontier — no separate spin phase needed.
                self.stale  = 0
                self.path   = []
                self.target = None

            frontier = coverage.frontier_cells()
            scored   = self._score_frontiers(frontier, coverage.seen)

            new_path: list[tuple[int,int]] = []
            new_target = None
            for (fi, fj), _score in scored[:FRONTIER_TOP]:
                candidate = self._astar(cur_ij, (fi, fj))
                if candidate:
                    new_path   = candidate
                    new_target = (fi, fj)
                    break

            if new_path:
                self.path        = new_path
                self.target      = new_target
                self.reassess_in = NAV_REASSESS
                self.stale       = 0
            else:
                # No frontier reachable — spin to scan
                self.path        = []
                self.target      = None
                self.reassess_in = 8

        self.reassess_in -= 1

        # ---- Pop reached waypoints ----------------------------------------
        while self.path:
            wp_x, wp_z = self._ij_to_xz(*self.path[0])
            dist = math.hypot(wp_x - cam_x, wp_z - cam_z)
            if dist < REACH_DIST:
                self.path.pop(0)
            else:
                break

        # ---- Build movement plan ------------------------------------------
        if self.path:
            wp_x, wp_z = self._ij_to_xz(*self.path[0])
            dx, dz = wp_x - cam_x, wp_z - cam_z
            target_yaw = math.atan2(dx, dz)   # forward=(sin(y),cos(y)) in XZ: R@[0,0,1]=col2

            dyaw = target_yaw - yaw
            dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi   # to [-π,π]
            dyaw = max(-math.radians(MAX_TURN_DEG),
                       min(math.radians(MAX_TURN_DEG), dyaw))

            look_osc = math.radians(LOOK_AMP_DEG) * math.sin(self.t * LOOK_FREQ)
            turn_px = self._px(dyaw + look_osc)
            return {"hold": ["w"], "look": (turn_px, 0), "tap": []}
        else:
            # No A* path — steer toward clearest nav-grid heading and walk.
            # No look oscillation here: we want a clean turn that the SLAM can
            # track accurately, and the burst's wall-contact detection will
            # trigger a scan if we walk into something.
            open_yaw, _run = self.best_open_yaw(cam_x, cam_z)
            dyaw    = open_yaw - yaw
            dyaw    = (dyaw + math.pi) % (2 * math.pi) - math.pi
            dyaw    = max(-math.radians(MAX_TURN_DEG),
                          min(math.radians(MAX_TURN_DEG), dyaw))
            turn_px = self._px(dyaw)
            return {"hold": ["w"], "look": (turn_px, 0), "tap": []}
