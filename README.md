# 3D Hand Gesture Experience | Angkor Wat ➔ 3D Numbers 1–5

An interactive real-time 3D experience where gestures **1, 2, 3, 4, and 5** smoothly transform the initial **Angkor Wat (ប្រាសាទអង្គរវត្ត)** model into detailed volumetric **3D Numbers 1 to 5**, each activating a unique color theme with cinematic particle animations and dynamic lighting.

---

## 🌟 Key Features

### 1. Initial State: Angkor Wat (ប្រាសាទអង្គរវត្ត)
- The application **always starts with Angkor Wat** as the initial 3D object.
- Features the iconic 5 lotus-bud *Prasat* towers, concentric stepped galleries, cruciform causeways, and 50,000 metallic particles in glowing Gold.

### 2. 3D Object Transformations (Gestures 1 to 5)
When you show hand gestures 1 through 5, Angkor Wat dissolves into thousands of golden particles, spirals through 3D space, and gathers to construct a recognizable, volumetric 3D number:
- **Gesture 1** $\to$ **3D Number 1** (Volumetric Roman numeral with vertical spine, top flag, and base).
- **Gesture 2** $\to$ **3D Number 2** (Volumetric curved upper arch, diagonal sweep, and plinth).
- **Gesture 3** $\to$ **3D Number 3** (Volumetric double-lobed loops and central junction cusp).
- **Gesture 4** $\to$ **3D Number 4** (Volumetric vertical stem, diagonal arm, and crossbar).
- **Gesture 5** $\to$ **3D Number 5** (Volumetric top bar, upper stem, and rounded lower bowl).
- **Reset (`R` / `0` / `Tab`)** $\to$ Smoothly morphs back into **Angkor Wat**.

### 3. The 5 Color Themes

| Number | Color Theme | Visual Environment |
|:---:|:---|:---|
| **1** | **Electric Blue** | Deep vivid blue particles, cyan specular highlights, electric blue lighting |
| **2** | **Emerald Green** | Lush vibrant emerald particles, jade lighting, green ground reflections |
| **3** | **Royal Purple** | Rich royal purple particles, violet highlights, lavender ambient fill |
| **4** | **Crimson Red** | Fiery crimson red particles, ruby specular gleam, scarlet lighting |
| **5** | **Gold** | Radiant metallic gold particles, warm amber gleam, golden lighting |

- Colors transition smoothly across the entire 3D environment (particles, lighting, ground reflections, and HUD).
- **One-Shot Animation**: Animations trigger on gesture change and do not repeatedly restart while the same number remains detected.

### 4. Significantly Bigger Webcam Display
- Enlarged to **360 x 240** pixels (~3× the original area).
- Preserves the clean, unadorned, high-performance layout and style.
- Hand tracking landmarks remain visible with clean green joint markers `(80, 230, 120)` and bone connections.
- Floating glowing number badge attached smoothly near the hand with $EMA$ jitter suppression.

### 5. Hand Gesture Recognition (Numbers 1 to 5 Only)
- Landmark-based finger analysis using MediaPipe:
  - **1**: Index finger raised (Middle, Ring, Pinky folded).
  - **2**: Index + Middle raised (V / Peace sign).
  - **3**: Index + Middle + Ring raised (or Thumb + Index + Middle).
  - **4**: Four fingers raised, Thumb tucked.
  - **5**: Open palm (all 5 fingers spread).
- Gestures outside 1–5 are ignored to prevent accidental triggering.
- Temporal smoothing eliminates flickering.

---

## 🎮 Controls

### Hand Gestures
- Show **1, 2, 3, 4, or 5** fingers to transform the 3D model into that 3D number and switch to its color theme.
- Move hand in front of camera to rotate the 3D model with smooth rotational inertia.

### Keyboard Shortcuts
- `1` – `5`: Directly transform to 3D Numbers 1 through 5.
- `0` / `Tab` / `R`: Reset transformation back to Angkor Wat.
- `M`: Toggle virtual hand demo (cycles 1–5).
- `O`: Toggle structural architectural wireframe overlay.
- `H`: Toggle HUD help overlay.
- `Esc`: Quit application.

---

## 🚀 Running the Project

```bash
cd /home/los/Downloads/ComputerVS
.venv/bin/python main.py
```
