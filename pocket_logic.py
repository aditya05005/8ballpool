import numpy as np

class PocketDetector:
    def __init__(self, pocket_coords, pocket_radius=30, cooldown_frames=20):
        """
        pocket_coords: List of (x, y) tuples.
        pocket_radius: The strict zone (if a ball is SEEN here, it counts).
        cooldown_frames: Prevent double counting.
        """
        self.pockets = np.array(pocket_coords)
        self.pocket_radius = pocket_radius
        self.cooldown_frames = cooldown_frames
        
        # Timers to prevent spamming a pocket
        self.pocket_timers = np.zeros(len(pocket_coords), dtype=int)
        self.total_pocketed_count = 0
        
        # MEMORY SYSTEM: Stores data from the previous frame
        # Format: list of dicts {'pos': (x,y), 'cls': class_id}
        self.prev_balls = []

    def update(self, curr_detections, curr_classes):
        """
        Check for 2 things:
        1. Is a ball CURRENTLY in the pocket?
        2. Did a ball VANISH near a pocket?
        """
        events = []
        
        # Decrement cooldowns
        self.pocket_timers = np.maximum(0, self.pocket_timers - 1)
        
        # --- PREPARE DATA ---
        # Combine detections and classes into a structured list
        current_balls = []
        if len(curr_detections) > 0:
            for i, box in enumerate(curr_detections):
                current_balls.append({
                    'pos': np.array(box[:2]), 
                    'cls': curr_classes[i]
                })

        # --- LOGIC 1: DIRECT HITS (The ball is currently visible in the pocket) ---
        used_pockets = set() # Track pockets hit in this frame to avoid double trigger
        
        for ball in current_balls:
            # Dist to all pockets
            dists = np.linalg.norm(self.pockets - ball['pos'], axis=1)
            
            # Find pockets within RADIUS
            hit_indices = np.where(dists < self.pocket_radius)[0]
            
            for p_idx in hit_indices:
                if self.pocket_timers[p_idx] == 0 and p_idx not in used_pockets:
                    self._trigger_event(p_idx, ball['cls'], events)
                    used_pockets.add(p_idx)

        # --- LOGIC 2: THE "VANISHED BALL" CHECK ---
        # If a ball existed last frame, but is NOT close to any ball this frame,
        # it has "vanished". If it vanished NEAR a pocket, it counts.
        
        # Tolerance: If a ball moves > 20px, we assume it's a different ball or jump.
        # Vanishing Radius: We allow a slightly larger radius for vanishing balls 
        # (e.g. ball disappears 5px outside the official ring).
        MOVEMENT_THRESHOLD = 50 
        VANISHING_RADIUS = self.pocket_radius + 10 # slightly larger leniency
        
        for old_ball in self.prev_balls:
            # Check if this old_ball still exists in the current frame
            # We look for any current ball within MOVEMENT_THRESHOLD pixels
            is_still_here = False
            
            if len(current_balls) > 0:
                # Calculate distances from this old ball to ALL current balls
                curr_positions = np.array([b['pos'] for b in current_balls])
                d_to_current = np.linalg.norm(curr_positions - old_ball['pos'], axis=1)
                
                if np.min(d_to_current) < MOVEMENT_THRESHOLD:
                    is_still_here = True
            
            # If the ball is GONE, check where it went
            if not is_still_here:
                # Calculate distance to pockets
                d_to_pockets = np.linalg.norm(self.pockets - old_ball['pos'], axis=1)
                close_pockets = np.where(d_to_pockets < VANISHING_RADIUS)[0]
                
                for p_idx in close_pockets:
                    # Only trigger if pocket is cool AND we haven't triggered it in Logic 1
                    if self.pocket_timers[p_idx] == 0 and p_idx not in used_pockets:
                        print(f"Logic: Ball vanished {int(d_to_pockets[p_idx])}px from pocket. Counting it.")
                        self._trigger_event(p_idx, old_ball['cls'], events)
                        used_pockets.add(p_idx)

        # --- UPDATE MEMORY ---
        self.prev_balls = current_balls
        return self.total_pocketed_count, events

    def _trigger_event(self, p_idx, ball_cls, events_list):
        self.total_pocketed_count += 1
        events_list.append((p_idx, ball_cls))
        self.pocket_timers[p_idx] = self.cooldown_frames
