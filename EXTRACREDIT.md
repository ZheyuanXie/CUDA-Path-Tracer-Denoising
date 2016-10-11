 Path Tracer Extra Credit: Optimizations
========================

This week (ending 10/16/2016) you will have the opportunity to add some additional
optimizations to your pathtracer. This is 100% optional!
Optimizations will mostly be based on [concepts discussed in lecture](https://github.com/CIS565-Fall-2016/cis565-fall-2016.github.io/blob/master/lectures/5-CUDA-Performance.pdf):
- Warp Partitioning
- Memory Coalescing
- Overcoming Bank Conflicts in Shared Memory
- Dynamic Resource Partitioning (more fine grained kernel launch tuning)
- Data Prefetching
- Instruction Mixing (to hide latency for sharing Special Function Units)
- Loop Unrolling
- Modified Thread Granularity

Spend some time thinking about ways these tools may fit in your implementation.
To give you more material to work with, you may work on adapting your own
stream compaction code from Project 2 into the Pathtracer so you can also
optimize/analyze that.

If you haven't done so already, we also encourage you to implement shared-memory
stream compaction for the sake of investigating bank conflicts.

This is also your time to impress anyone who might look at your project, so
feel free to tackle algorithmic optimizations as well, including acceleration
features you saw in the standard assignment but didn't want to tackle due to
time constraints. If in doubt as to whether or not a feature counts as an
"optimization," please ask in the [Google Group](https://groups.google.com/forum/#!forum/cis-565-fall-2016).

A thorough exploration of any single optimization area will be worth 5 points.
Larger algorithmic changes with analysis will generally be worth 10 points.

## `git`ing Started
In your project directory, open a new branch off the branch you used for your
submission pull request:

`git checkout -b YourBranchNameHere`

The `-b` instructs git to open a new branch named "YourBranchNameHere" and
switch your project directory over to this branch. Commit your work as usual,
and when you're ready push it to Github with:

`git push origin YourBranchNameHere`

To switch back to your submission branch (for most of you, the `master` branch):

`git checkout master`

When switching branches you may need to `commit` your current changes, or discard
them with `git checkout .` if you don't want them.
You can also `git stash` if you just want to temporarily hop to another branch,
and `git stash apply` when you want your changes back.

You can also `git diff` between branches to see how your code has changed.

Commit working code early and often. You never know when you will have to revert
your entire project directory to an earlier state.

## Presenting Your Findings
Please describe what you did in a separate OPTMIZATION.md.
For credit, every optimization should come with:
- a brief explanation of what you did and where to look in your code
- before-and-after performance comparisons
- explanation of any observed changes
- additional Nsight data where appropriate (register counts, occupancy, etc.)

Your optimizations do NOT have to be successful. However, please prepare a
detailed analysis of what you were trying to do and what you think went wrong
for credit.

## Submission
This is due midnight, Sunday 10/16/2016.

In your own repo, open a new pull request from your optimization branch into
the branch you submitted (typically master).
Use the usual submission template in the comment section:
* [Repo Link](https://link-to-your-repo)
* `Your PENNKEY`
* (Briefly) mention features/optimizations that you've completed:
    * Optimization 0
    * Optimization 1
    * ...

Also EMAIL this header to the TAs with a link to the pull request.
Opening a pull request into your own submission branch will help us see what
you have changed.
