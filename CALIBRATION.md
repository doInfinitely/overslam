# overslam calibration protocol (Overwatch Workshop)

The depth pipeline needs five constants. None are reliably published; all
have to be measured in a controlled Workshop scene. Once filled in,
`calibration.json` is read by the depth/SLAM scripts.

  1. Mei-wall footprint in OW-units (defines 1 Mei / 1 Ling / 1 Zhou)
  2. Focal length in px at the capture resolution (camera intrinsic)
  3. Mouse sensitivity → camera-rotation radians per raw count
  4. Hero base move speed in OW-m/s (for translating WASD hold time)
  5. OW-m ↔ real-m factor (only if you ever want metric output)

Run all measurements in **Workshop > Practice Range** with one player, no
bots, no movement abilities other than what you're measuring. Pin a
Workshop script that prints positions to the HUD; the core expressions
you need are:

```
Position Of(Event Player)              ; X, Y, Z in OW-m
Facing Direction Of(Event Player)      ; unit vector
Horizontal Facing Angle Of(Event Player); degrees
Vertical Facing Angle Of(Event Player) ; degrees
```

The `Custom String` action ("X: {0}, Y: {1}, Z: {2}", X Component Of(...))
plus `Hud Text` gives you a live readout. Use the third-person camera
mode (Workshop `Start Camera`) only for visual checks; all numbers
should come from first-person.


## 1. Mei wall footprint — 1 Mei / 1 Ling / 1 Zhou

Goal: determine the OW-unit length, depth, and height of Mei's Ice Wall
in its default orientation.

  1. Find a long flat strip on Practice Range with no geometry close by.
  2. Stand at point A facing along the strip. Note Position A.
  3. Place an ice wall straight ahead at max range (20 m). The wall
     orients perpendicular to your facing.
  4. **Length (1 Mei):** strafe sideways until your camera is exactly at
     one end pillar's outer face (line up against the pillar so it
     touches the screen edge). Note Position B. Walk to the other end
     and align with the far pillar's outer face. Note Position C.
     `|C - B|` in the strafe axis = wall length in OW-m. Repeat 3×; the
     numbers should agree to <0.1 OW-m.
  5. **Depth (1 Ling):** stand against the wall's front face, note
     Position D. Walk through to the back face (wait for the wall to
     expire, then re-place a fresh wall in the same spot using a
     ground marker; or skip Mei and use the Workshop `Create Effect` of
     a known cube at the same coords as reference). `|E - D|` = depth.
  6. **Height (1 Zhou):** stand at the base, look straight up at the top
     edge of the pillar directly in front of you. Vertical facing angle
     θ + horizontal distance d (also Workshop-measurable: the wall
     hitbox depth from step 5 / 2) → height = `d × tan(θ)` + camera
     eye-height (Mei's eye height ≈ 1.5 OW-m, verify with the Z value
     of `Position Of(Event Player)` while standing on a flat floor at
     Z=0).
  7. Cross-check height by jumping next to it (jump apex Z relative to
     floor Z, compared against the pillar top).

Record the three numbers in `calibration.json`.


## 2. Focal length / FOV at capture resolution

Goal: f_px such that an object of OW-unit width W at distance D appears
W·f_px/D pixels wide in the captured frame.

  1. Set OW2 graphics: target resolution, FOV (default 103° in OW2),
     and disable any FOV-altering settings. Note the FOV in
     `calibration.json` → `game_settings.fov_deg`.
  2. Run `./record-windows.sh -n` and stand at a known OW-unit
     distance D from a freshly placed Mei wall (perpendicular to view).
     D ≈ 10 OW-m works well — close enough for measurable pixel width,
     far enough that perspective foreshortening on the wall ends is
     small.
  3. Stop recording. Open one frame, measure the wall's pixel width P
     (use the central pillar to avoid foreshortening; multiply by the
     pillar count if measuring just one).
  4. `f_px = P × D / W_OW`. Record this and the resolution it was
     measured at — f_px scales linearly with capture height in pixels,
     so if you change `--downscale` you can rescale.
  5. Sanity: derived horizontal FOV = `2 × atan((image_width/2) / f_px)`
     should equal the game setting to within ~1°. If not, your D or
     wall measurement is off.


## 3. Mouse sensitivity → radians per count

Goal: `yaw_rad_per_count` and `pitch_rad_per_count` so a logged
`mouse_move(dx,dy)` event converts to (Δyaw, Δpitch) in radians.

The standard OW formula is `yaw_deg_per_count = sens × m_yaw × dpi_factor`
but the constants are model-specific. Empirical is more reliable:

  1. Set in-game sensitivity to a known value and write it into
     `calibration.json` → `game_settings.mouse_sensitivity`. Disable
     mouse acceleration in Windows.
  2. Start `./log-input.sh -o calib_mouse.jsonl`.
  3. Face a fixed reference (paint a wall, or stand pointing at a
     workshop-marker placed at a known position). Record start yaw
     from Workshop HUD.
  4. Do a slow, deliberate 360° turn (or N×360° for higher precision —
     5 turns averages out noise). End facing the same reference.
  5. Stop the logger. Sum `dx` across all `mouse_move` events between
     start/end: `total_dx`. Then
     `yaw_rad_per_count = 2π × N / total_dx`.
  6. Repeat for pitch (look up to ceiling and back to known reference)
     for `pitch_rad_per_count`. OW typically uses the same factor for
     both axes unless "scale aim with FOV" is on, but verify.


## 4. WASD → translational velocity

Goal: per-hero `move_speed_ow_m_per_s` and modifiers for sprint /
crouch / strafe-penalty.

  1. Pin a script that prints `Position Of(Event Player)` every tick.
  2. Stand still, note start position P0 and start wall-clock from
     `t_wall_ns` in the input log.
  3. Hold W for exactly 2 s (use the input log to find the W-down and
     W-up timestamps — should be accurate to sub-ms).
  4. Read end position P1. Speed = `|P1 - P0| / Δt_seconds`.
  5. Repeat for A, S, D individually (some heroes strafe slower than
     they run forward; check). Record one row per hero you plan to
     play. Default Mei base speed is 5.5 OW-m/s but verify in your
     actual build.


## 5. OW-m ↔ real-m (optional, only if you need metric output)

Two hypotheses:

  - H1: 1 OW-m = 1 real m (units are real meters as labeled)
  - H2: 1 OW-m = 0.3048 real m (units are feet relabeled "m")

Test against two independent real-world references:

  - **Reinhardt's stated height:** 7'4" ≈ 2.24 real m. Workshop-measure
    Rein's model height (have him stand still; eye position Z minus
    floor Z, then add the chin-to-crown offset, ~0.18 OW-m).
    - H1 prediction: Rein model ≈ 2.24 OW-m
    - H2 prediction: Rein model ≈ 7.35 OW-m
  - **Door frames on standard maps** (e.g., King's Row spawn doors).
    Real-world doors are 2.0–2.1 m. Measure in-game door height via
    Workshop. If they come out ~2 OW-m → H1; if ~6.5 OW-m → H2.

Two consistent measurements resolve which hypothesis is correct. Don't
trust just one — model proportions in stylised games are not
necessarily lore-accurate.


## After calibration

Fill in `calibration.json`. The depth and SLAM scripts in this repo
read it on startup. Re-measure (1) and (2) any time you change capture
resolution, FOV, or game patch.
