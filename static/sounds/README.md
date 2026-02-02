# Emergency Sounds

## Required Files

### emergency_siren.mp3
An emergency siren sound that plays when hazards are detected.

**To add the siren sound:**
1. Download a royalty-free siren sound (e.g., from freesound.org or similar)
2. Convert to MP3 format
3. Save as `emergency_siren.mp3` in this directory

**Recommended properties:**
- Format: MP3
- Duration: 3-5 seconds (loopable)
- Volume: Moderate (will be controlled by browser)

**Example sources:**
- https://freesound.org/search/?q=siren
- https://pixabay.com/sound-effects/search/emergency/
- https://www.zapsplat.com/

### Fallback
If no sound file is provided, the frontend will use the browser's built-in audio alert API or display a visual-only alert.
