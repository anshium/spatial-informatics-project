# Spatial Informatics Project

## Project Title
Optimal Trajectory Planning for a Mars Rover Using Spatial Data collected by Mars Orbiters.

## Programming Language
Python ('cause QGIS has a python API)

Ended up not using QGIS for most tasks, rather used custom scripts.

## Things to complete before mid-evaluation (4th November, 2024)
- [X] Write Down assumptions / criteria
- [X] Data – Source, Characteristics (in as much detail as possible)
  - [X] List all the data
  - [X] Use them to at least generate a few derivates.
    - [X] Make DEM for the elevation data
  - [X] In doing the above, list down all the spatial methods used.
  
## Things to complete until end-evaluation
- [X] More data post-processing.
    - [X] Identify danger zones - (deep craters, big rocks)
    - [X] Make local slope data analysis also (limit the rover to maximum slope of 30 degrees and max rock encounter to be 2 * diameter of the wheels)
    - [X] Also do something like: Raster Reclassification – Simplify terrain data for easier rover navigation.
    - [X] Use CRISM data to enhance trajectory.
- [X] Design an algorithm that plans a trajectory for the mars rover.
- [X] Make a robust visualisation enabling the user to input the start and the end-points of the trajectory and generate a trajectory.

`(Note: more exhaustive list is in the final report)`

## Extras
- [X] Make a 3D surface visualiser to make the robot traverse it (like a game). A gaebo simulation like thingy.

## Things completed till mid-evaluation
1. Explored Mars CRS
2. Went with the MOLA dataset first but then the problems (show calculations)
3. Then I went with specific HiRISE dataset that seemed to work after many problems.
   - Convting the .img file to a .tiff file
   - Then georeferencing it (how did I do so?)
   - Then performing r.walk.rast to get the cummulative cost and the movement directions from any point

`(more things in the final report - find in reports)`