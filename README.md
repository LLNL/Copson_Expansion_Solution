# Copson_Expansion_Solution

This package evaluates Copson's solution for the free expansion of an ideal gas (&gamma; = 5/3, P = k &#961;<sup>5/3</sup>) into vacuum. 
For x < 0 the initial conditions are zero velocity and constant sound speed. 
There is a surface layer for the range 0 < x < h. 
In the surface layer the sound speed goes smoothly to zero as described by a cubic function for x(3c/2). 
The slope at x=0, called alpha, and at x=h, called beta, fully specify the cubic. 
Alpha and beta must be in the triangle with corners (&alpha;, &beta;) = (-1,-1), (0, -5/3), (0, -3) for the solution to have no shocks and be valid. 
To initialize the surface layer you need to solve a cubic equation for the sound speed given the value of x. 
**A routine is provided to do this**.

The solution for the characteristics (r and s) are given along with a routine to evaluate the which r and s characteristics go through any (x,t) location. 
From the values of r and s any physical quantity can be calculated. 
For example, velocity is v = r - s, soundc speed c = (r + s)/3, density is &#961; = [3 &frasl; (5k)]<sup>3/2</sup>c<sup>3</sup> = (15k)<sup>-3/2</sup>(r+s)<sup>3</sup>, internal energy e = (9 &frasl;10) c<sup>2</sup> = 0.1 (r+s)<sup>2</sup>, and pressure is P = k<sup>-3&frasl;2</sup>(3 &frasl; 5)<sup>5&frasl;2</sup> c<sup>5</sup> = (15)<sup>-5&frasl;2</sup> k<sup>-3&frasl;2</sup> (r+s)<sup>5</sup>.


## Getting Started

Clone the git repository to a convenient location. You can copy the CopsonFuncs.py module to your Python site-packages directory if you want it globally accessible.

A draft copy of the journal article describing the solution, [Modeling the Free Expansion of an Ideal Gas without Shocks](MultiMat2019_Managan.pdf), is included.



### Prerequisites

The Python scripts load [NumPy](https://numpy.org) and [SciPy](https://www.scipy.org). The plotting script, Copson_plots.py, also uses the [Matplotlib](https://matplotlib.org) module to create the plots.

## Running the example

In the directory containing the three python files you can generate plots of the characteristics for the cases 
(&alpha;=-1/3, &beta;=-17/9), (&alpha;=0, &beta;=-5/3), and (&alpha;=0, &beta;=-3) by this command
```
python Copson_plots.py
```

### Test script

The script TestCopson.py tests the case of (&alpha;=-1/3, &beta;=-17/9). 
The first set of tests looks at the region near the free surface where s < 0. 
The values of x and t are solved for using the general expressions of r and s and then using the factored form that avoids dividing by small numbers. 
The factored form is only valid when s < 0. 
The factored form is used when appropriate in the general solutions for x and t.

The second set of tests check that the routines can calculate r and s given a point x and t. 
For t = 0.5 the test checks for points near the free surface where numerical problems can occur with the unfactored equations.

```
python TestCopson.py
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Robert Managan** - *Initial work* - [Managan](https://people.llnl.gov/managan1)

See also the list of [contributors](CONTRIBUTING.md) who participated in this project.

## License

This project is licensed under the BSD-3 License - see the [LICENSE.md](LICENSE.md) file for details

Unlimited Open Source - BSD 3-clause Distribution LLNL-CODE-802401

## SPDX usage

Individual files contain SPDX tags instead of the full license text.
This enables machine processing of license information based on the SPDX
License Identifiers that are available here: https://spdx.org/licenses/

Files that are licensed as BSD 3-Clause contain the following
text in the license header:

    SPDX-License-Identifier: (BSD-3-Clause)



