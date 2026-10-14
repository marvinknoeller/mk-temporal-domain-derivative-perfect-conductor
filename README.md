## License

Copyright (c) 2026, Marvin Knöller

This software is released under GNU General Public License, Version 3.
The full license text is provided in the file LICENSE included in this repository 
or can be obtained from http://www.gnu.org/licenses/

## Content of this project

This is a guide to generate the figures that have been used in the work

**„A Domain Derivative for Electromagnetic Scattering by Perfect Conductors in the Time Domain“**

_Marvin Knöller_

You find all the needed Python and Matlab files in the provided project.

## An overview
In Section 5.2 the decay of the difference quotient, i.e., the validation of the temporal domain derivative is tested.
To do that run
Taylor-test.sh
and to plot it later,
plot_taylor_test.m

In Section 5.2 the order of the temporal domain derivative's time and space discretization based on a Radau IIA Runge--Kutta convolution quadrature method in time and a Galerkin method in space is validated numerically.
To do that run first
Convergence-reference.sh
to obtain a reference solution.
Then, run
Convergence-time.sh ... for the numerical validation of the Runge--Kutta Convolution quadrature
and 
Convergence-space.sh ... for the numerical validation of the Galerkin discretization with (rotated) Raviart--Thomas functions.
Then, use
plot_convergence_test.m
to plot the results.

This work features 3 numerical examples for the shape reconstruction, each with noiseless data, 15% relative noise and 30% relative noise. Run it with
Ex1-1.sh, Ex1-2.sh, Ex1-3.sh,
Ex2-1.sh, Ex2-2.sh, Ex2-3.sh,
Ex3-1.sh, Ex3-2.sh, Ex3-3.sh.

Then, in order to plot the pictures use the Matlab files
plot_reconstruction1.m,
plot_reconstruction1.m,
plot_reconstruction1.m.


# A short explanation of the scripts

All **.py** files hold descriptions. In particular, all Python defs are described in detail.

The folder Precomputed results includes all the precomputed results that were computed in order to generate the numerical examples.

## Requirements and additional information
The numerical examples were run on the Roihu supercomputer provided by CSC-Center of Science, Finland.
In their current release the examples cannot be run locally. Neither the local workspace nor the number of workers would suffice.
