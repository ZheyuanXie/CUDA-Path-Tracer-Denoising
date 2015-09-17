CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

Instructions (delete me)
========================

This is due Thursday, September 24 evening at midnight.

**Summary:**
In this project, you'll implement a CUDA-based path tracer capable of rendering
globally-illuminated images very quickly.
Since in this class we are concerned with working in GPU programming,
performance, and the generation of actual beautiful images (and not with
mundane programming tasks like I/O), this project includes base code for
loading a scene description file, described below, and various other things
that generally make up a framework for previewing and saving images.

The core renderer is left for you to implement. Finally, note that, while this
base code is meant to serve as a strong starting point for a CUDA path tracer,
you are not required to use it if you don't want to. You may also change any
part of the base code as you please. **This is YOUR project.**

**Recommendation:** Every image you save should automatically get a different
filename. Don't delete all of them! For the benefit of your README, keep a
bunch of them around so you can pick a few to document your progress at the
end.

### Contents

* `src/` C++/CUDA source files.
* `scenes/` Example scene description files.
* `img/` Renders of example scene description files.
  (These probably won't match precisely with yours.)
* `external/` Includes and static libraries for 3rd party libraries.


### Running the code

The main function requires a scene description file. Call the program with
one as an argument: `cis565_path_tracer scene/sphere.txt`.
(In Visual Studio, `../scene/sphere.txt`.)

If you are using Visual Studio, you can set this in the Debugging > Command
Arguments section in the Project properties. Make sure you get the path right -
read the console for errors.

#### Controls

* Esc to save an image and exit.
* Space to save an image. Watch the console for the output filename.
* W/A/S/D and R/F move the camera. Arrow keys rotate.

## Requirements

**Ask on the mailing list for clarifications.**

In this project, you are given code for:

* Loading and reading the scene description format
* Sphere and box intersection functions
* Support for saving images
* Working CUDA-GL interop for previewing your render while it's running
* A function which generates random screen noise (instead of an actual render).

You will need to implement the following features:

* Raycasting from the camera into the scene through an imaginary grid of pixels
  (the screen)
  * Implement antialiasing (by jittering rays within each pixel)
* Diffuse surfaces
* Perfectly specular-reflective (mirrored) surfaces
* Stream compaction optimization. You may use any of:
  * Your global-memory work-efficient stream compaction implementation.
  * A shared-memory work-efficient stream compaction (see below).
  * `thrust::remove_if` or any of the other Thrust stream compaction functions.

You are also required to implement at least 2 of the following features.
Please ask if you need good references (they will be added to this README
later on). If you find good references, share them! **Extra credit**: implement
more features on top of the 2 required ones, with point value up to +20/100 at
the grader's discretion (based on difficulty and coolness).

* Work-efficient stream compaction using shared memory across multiple blocks
  (See *GPU Gems 3* Chapter 39).
* These 2 smaller features:
  * Refraction (e.g. glass/water) with Frensel effects using Schlick's
    approximation or more accurate methods
  * Physically-based depth-of-field (by jittering rays within an aperture)
* Texture mapping
* Bump mapping
* Direct lighting (by taking a final ray directly to a random point on an
  emissive object acting as a light source)
* Some method of defining object motion, and motion blur
* Subsurface scattering
* Arbitrary mesh loading and rendering (e.g. `obj` files). You can find these
  online or export them from your favorite 3D modeling application.
  With approval, you may use a third-party OBJ loading code to bring the data
  into C++.
  * You can use the triangle intersection function `glm::intersectRayTriangle`.

This 'extra features' list is not comprehensive. If you have a particular idea
you would like to implement (e.g. acceleration structures, etc.), please
contact us first.

For each extra feature, you must provide the following analysis:

* Overview write-up of the feature
* Performance impact of the feature
* If you did something to accelerate the feature, what did you do and why?
* Compare your GPU version of the feature to a HYPOTHETICAL CPU version
  (you don't have to implement it!) Does it benefit or suffer from being
  implemented on the GPU?
* How might this feature be optimized beyond your current implementation?

## Base Code Tour

You'll be working in the following files. Look for important parts of the code:
search for `CHECKITOUT`. You'll have to implement parts labeled with `TODO`.
(But don't let these constrain you - you have free rein!)

* `src/pathtrace.cu`: path tracing kernels, device functions, and calling code
  * `pathtraceInit` initializes the path tracer state - it should copy
    scene data (e.g. geometry, materials) from `Scene`.
  * `pathtraceFree` frees memory allocated by `pathtraceInit`
  * `pathtrace` performs one iteration of the rendering - it handles kernel
    launches, memory copies, transferring some data, etc.
    * See comments for a low-level path tracing recap.
* `src/intersections.h`: ray intersection functions
  * `boxIntersectionTest` and `sphereIntersectionTest`, which take in a ray and
    a geometry object and return various properties of the intersection.
* `src/interactions.h`: ray scattering functions
  * `calculateRandomDirectionInHemisphere`: a cosine-weighted random direction
    in a hemisphere. Needed for implementing diffuse surfaces.
  * `scatterRay`: this function should perform all ray scattering, and will
    call `calculateRandomDirectionInHemisphere`. See comments for details.
* `src/main.cpp`: you don't need to do anything here, but you can change the
  program to save `.hdr` image files, if you want (for postprocessing).

### Generating random numbers

```
thrust::default_random_engine rng(hash(index));
thrust::uniform_real_distribution<float> u01(0, 1);
float result = u01(rng);
```

There is a convenience function for generating a random engine using a
combination of index, iteration, and depth as the seed:

```
thrust::default_random_engine rng = random_engine(iter, index, depth);
```

### Notes on GLM

This project uses GLM for linear algebra.

On NVIDIA cards pre-Fermi (pre-DX12), you may have issues with mat4-vec4
multiplication. If you have one of these cards, be careful! If you have issues,
you might need to grab `cudamat4` and `multiplyMV` from the
[Fall 2014 project](https://github.com/CIS565-Fall-2014/Project3-Pathtracer).
Let us know if you need to do this.

### Scene File Format

This project uses a custom scene description format. Scene files are flat text
files that describe all geometry, materials, lights, cameras, and render
settings inside of the scene. Items in the format are delimited by new lines,
and comments can be added using C-style `// comments`.

Materials are defined in the following fashion:

* MATERIAL (material ID) //material header
* RGB (float r) (float g) (float b) //diffuse color
* SPECX (float specx) //specular exponent
* SPECRGB (float r) (float g) (float b) //specular color
* REFL (bool refl) //reflectivity flag, 0 for no, 1 for yes
* REFR (bool refr) //refractivity flag, 0 for no, 1 for yes
* REFRIOR (float ior) //index of refraction for Fresnel effects
* SCATTER (float scatter) //scatter flag, 0 for no, 1 for yes
* ABSCOEFF (float r) (float b) (float g) //absorption coefficient for scattering
* RSCTCOEFF (float rsctcoeff) //reduced scattering coefficient
* EMITTANCE (float emittance) //the emittance of the material. Anything >0
  makes the material a light source.

Cameras are defined in the following fashion:

* CAMERA //camera header
* RES (float x) (float y) //resolution
* FOVY (float fovy) //vertical field of view half-angle. the horizonal angle is calculated from this and the reslution
* ITERATIONS (float interations) //how many iterations to refine the image,
  only relevant for supersampled antialiasing, depth of field, area lights, and
  other distributed raytracing applications
* DEPTH (int depth) //maximum depth (number of times the path will bounce)
* FILE (string filename) //file to output render to upon completion
* EYE (float x) (float y) (float z) //camera's position in worldspace
* VIEW (float x) (float y) (float z) //camera's view direction
* UP (float x) (float y) (float z) //camera's up vector

Objects are defined in the following fashion:

* OBJECT (object ID) //object header
* (cube OR sphere OR mesh) //type of object, can be either "cube", "sphere", or
  "mesh". Note that cubes and spheres are unit sized and centered at the
  origin.
* material (material ID) //material to assign this object
* TRANS (float transx) (float transy) (float transz) //translation
* ROTAT (float rotationx) (float rotationy) (float rotationz) //rotation
* SCALE (float scalex) (float scaley) (float scalez) //scale

Two examples are provided in the `scenes/` directory: a single emissive sphere,
and a simple cornell box made using cubes for walls and lights and a sphere in
the middle.

## Third-Party Code Policy

* Use of any third-party code must be approved by asking on our Google Group.
* If it is approved, all students are welcome to use it. Generally, we approve
  use of third-party code that is not a core part of the project. For example,
  for the path tracer, we would approve using a third-party library for loading
  models, but would not approve copying and pasting a CUDA function for doing
  refraction.
* Third-party code **MUST** be credited in README.md.
* Using third-party code without its approval, including using another
  student's code, is an academic integrity violation, and will, at minimum,
  result in you receiving an F for the semester.

## README

Please see: [**TIPS FOR WRITING AN AWESOME README**](https://github.com/pjcozzi/Articles/blob/master/CIS565/GitHubRepo/README.md)

* Sell your project.
* Assume the reader has a little knowledge of path tracing - don't go into
  detail explaining what it is. Focus on your project.
* Don't talk about it like it's an assignment - don't say what is and isn't
  "extra" or "extra credit." Talk about what you accomplished.
* Use this to document what you've done.
* *DO NOT* leave the README to the last minute! It is a crucial part of the
  project, and we will not be able to grade you without a good README.

In addition:

* This is a renderer, so include images that you've made!
* Be sure to back your claims for optimization with numbers and comparisons.
* If you reference any other material, please provide a link to it.
* You wil not be graded on how fast your path tracer runs, but getting close to
  real-time is always nice!
* If you have a fast GPU renderer, it is very good to show case this with a
  video to show interactivity. If you do so, please include a link!

### Analysis

* Stream compaction helps most after a few bounces. Print and plot the
  effects of stream compaction within a single iteration (i.e. the number of
  unterminated rays after each bounce) and evaluate the benefits you get from
  stream compaction.
* Compare scenes which are open (like the given cornell box) and closed
  (i.e. no light can escape the scene). Again, compare the performance effects
  of stream compaction! Remember, stream compaction only affects rays which
  terminate, so what might you expect?


## Submit

If you have modified any of the `CMakeLists.txt` files at all (aside from the
list of `SOURCE_FILES`), you must test that your project can build in Moore
100B/C. Beware of any build issues discussed on the Google Group.

1. Open a GitHub pull request so that we can see that you have finished.
   The title should be "Submission: YOUR NAME".
2. Send an email to the TA (gmail: kainino1+cis565@) with:
   * **Subject**: in the form of `[CIS565] Project N: PENNKEY`.
   * Direct link to your pull request on GitHub.
   * Estimate the amount of time you spent on the project.
   * If there were any outstanding problems, or if you did any extra
     work, *briefly* explain.
   * Feedback on the project itself, if any.
