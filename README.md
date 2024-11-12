# Spatial Informatics Project

## Project Title
Optimal Trajectory Planning for a Mars Rover Using Spatial Data collected by Mars Orbiters.

## Programming Language
Python ('cause QGIS has a python API)

## Goal
Make it as easy as possible but as convincing and possible.

## Things to complete before mid-evaluation (4th November, 2024)
- [ ] Write Down assumptions / criteria
- [ ] Data – Source, Characteristics (in as much detail as possible)
  - [ ] List all the data
  - [ ] Use them to at least generate a few derivates.
    - [ ] Make DEM for the elevation data
    - [ ] Make a proximity fading map for places with higher (or closer to surface) ice. (idk what is that actually called)
  - [ ] In doing the above, list down all the spatial methods used.
  
## Things to complete until end-evaluation
- [ ] More data post-processing.
    - [ ] Do something with the Mars Odyssey THEMIS: (For temperature and solar exposure data)
    - [ ] Identify danger zones - (deep craters, big rocks)
    - [ ] Make local slope data analysis also (limit the rover to maximum slope of 30 degrees and max rock encounter to be 2 * diameter of the wheels)
    - [ ] Also do something like: Raster Reclassification – Simplify terrain data for easier rover navigation.
- [ ] Design an algorithm that plans a trajectory for the mars rover.
- [ ] Implement that algorithm with all the processed data and QGIS python API.
- [ ] Make a robust visualisation enabling the user to input the start and the end-points of the trajectory and generate a trajectory.

## Extras (to get brownie points)
- [ ] Make a 3D surface visualiser to make the robot traverse it (like a game). A gaebo simulation like thingy.

## Some important points to note:
1. I would rather get more marks if I talk about keywords and terminologies than if I do great work. That is how the course spatial informatics is.
2. Focus on terms and keywords and references. Especially to the slides.

## Things completed till mid-evaluation
1. Explored Mars CRS
2. Went with the MOLA dataset first but then the problems (show calculations)
3. Then I went with specific HiRISE dataset that seemed to work after many problems.
   - Convting the .img file to a .tiff file
   - Then georeferencing it (how did I do so?)
   - Then performing r.walk.rast to get the cummulative cost and the movement directions from any point