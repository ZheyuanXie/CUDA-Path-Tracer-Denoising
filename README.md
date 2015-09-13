CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

Include analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

Instructions (delete me)
========================

This is due **INSTRUCTOR TODO** evening at midnight.

**Summary:**


### Controls

* W/A/S/D and R/F fly. Arrow keys rotate.

### INSTRUCTOR TODO

* Look for important parts of the code in the following files. You can search
  for `CHECKITOUT` in the code.
  * `src/interactions.h`: ray scattering functions
  * `src/intersections.h`: ray intersection functions
  * `src/pathtrace.cu`: path tracing kernels, device functions, and calling code
  * `src/main.cpp`: optionally, allows you to save HDR image files

```
    thrust::default_random_engine rng(hash(index));
    thrust::uniform_real_distribution<float> u01(0, 1);
    float result = u01(rng);
```

There is a convenience function for generating a random engine using a
combination of index, iteration, and depth as the seed:

```
  thrust::default_random_engine rng = random_engine(time, iter, depth);
```
   

## Submit

If you have modified any of the `CMakeLists.txt` files at all (aside from the
list of `SOURCE_FILES`), you must test that your project can build in Moore
100B/C. Beware of any build issues discussed on the Google Group.

1. Open a GitHub pull request so that we can see that you have finished.
   The title should be "Submission: YOUR NAME".
2. Send an email to the TA (gmail: kainino1+cis565@) with:
   * **Subject**: in the form of `[CIS565] Project 2: PENNKEY`
   * Direct link to your pull request on GitHub
   * Estimate the amount of time you spent on the project.
   * If there were any outstanding problems, or if you did any extra work,
     briefly explain for grading purposes.
   * Feedback on the project itself, if any.
