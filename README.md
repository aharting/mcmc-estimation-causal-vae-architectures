# MCMC estimation of causal VAE architectures

Algorithms and simulated experiments

Author: Alice Harting, supervised by Liam Solus

## About this repo
This repo contains the algorithms and simulated experiments in the paper *MCMC estimation of causal VAE architectures*. We reproduce the results in Chapter 3 *MCMC estimation of UEC* and Section 4.4.1 *Simulated randomized control trial*. To protect business information, no Spotify data or code is included. The code herein has been developed independently of proprietary code and methods of Spotify.

## Organisation
There are three main folders:
- `scripts`
- `experiments`
- `storage`

`scripts` contains classes and functions. Its subfolder `thesis` contains the algorithms developed in our paper. Subfolders `medil` and `grues` contain code from public repositories associated with academic papers.

`experiments` contains scripts triggering experiments with simulated data. It imports algorithms from `scripts`. 

`storage` contains a sample of randomly generated matrices (UDGs). 

## Reproduction of results
To reproduce, run `reproduce_results.sh`.

