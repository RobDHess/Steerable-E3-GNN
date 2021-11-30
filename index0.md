---
layout: default
title:  "Steerable Equivariant Message Passing on Molecular Graphs"
description: Blog post
date:   2021-06-08 17:03:39 +0200
usemathjax: true
---

<link rel="stylesheet" href="assets/css/accordion.css">

Blog post to the paper [Steerable Equivariant Message Passing on Molecular Graphs](https://arxiv.org/abs/2009.14794).

The implementation of Steerable E(3) Equivariant Graph Neural Networks (SEGNNs) is built on the [e3nn](https://docs.e3nn.org/en/stable/) software library of Mario Geiger et al.

## Key insights of the paper

The paper has three main pillars:

* Generalisation of equivariant graph neural networks
* Framing of various (equivariant) message passing algorithms in a unified convolutional form
* Generalisation of message passing allows SEGNNs to operate on graphs with small cutoff radii

This blog post tries to focus on the motivation and the intuition behind SEGNNs.

## Motivation

Our objective is to build graph neural networks that are robust to rotations, reflections, translations and permutations.
This is a desirable property since many tasks, such as molecular energy prediction, require such invariances.
For example, the XXXX molecule in the figure below should have the same ground state energy, regardless of its orientation.
Different atoms of the XXXX molecule are plotted with different colours.

{:refdef: style="text-align: center;"}
![not found](/assets/molecular_graph.png)
{: refdef}

Other tasks, like force prediction require equivariance.
As the molecule rotates, so do the atoms and therefore the force vectors depend on the rotated position.
First rotating the molecule and then putting it through a network should give the same results as doing the rotation afterwards.


In this work, we go one step further and introduce Steerable E(3) Equivariant Graph Neural Networks (SEGNNs).
SEGNNs generalise equivariant graph neural networks, such that (input) information at nodes and edges is not restricted to be invariant (scalar), but can also be covariant (vector, tensor).
Even if information at nodes is vector-valued (e.g. position, velocity, force) equivariance should be preserved.
Furthermore, we can also integrate relative orientation (pose) information in the message passing updates, which we found is particularly effective.
The figure below shows the commutative diagram sketched with vector-valued information at the nodes.

{:refdef: style="text-align: center;"}
![not found](/assets/Equivariance.png)
{: refdef}

This idea is actually coming from Multiagent Reinforcement Learning.

TODO: include sketch and description from Elise

The steps towards building SEGNNs are the following:

* Use of **steerable vectors** to be equivariant with respect to the transformation group of rotations and reflections
* Use of **Clebsch-Gordan tensor product** to map between **steerable vector spaces**
* Putting **relative orientation** into the Clebsch-Gordan tensor product to build a more powerful message passing scheme


## SEGNN builing blocks

### Steerable vectors, steerable MLPs
The essence of our approach is to build O(3) equivariant graph neural networks
where O(3) is the group of all rotations and reflections.
We do so by using the concept of steerable vectors.
Steerability of a vector $$\tilde{\mathbf{h}}$$ means that for a certain transformation group with transformation parameters $$g$$, the vector transforms via matrix-vector multiplication $$\mathbf{D}(g)$$. For example, a Euclidean vector in $$\mathbb{R}^3$$ is steerable for rotations and reflections $$g \in \text{O}(3)$$ by multiplying the vector with a rotation matrix, thus $$\mathbf{D}(g) = \mathbf{R}$$.

Like regular MLPs, steerable MLPs are constructed by interleaving linear mappings (matrix-vector multiplications)
with non-linearities. Now however, the linear maps transform between steerable vector spaces, for which we define how they transform under the action of a group.
This then sets an equivariance constraint on the operator that maps between these spaces. By only working with such equivariant operators we can guarantee that the entire learning framework is equivariant.

The figure below shows a vector $$\mathbf{x}$$ embedded into the space which is spanned by the spherical harmonics $$Y^{(l)}_m$$.
Each subspace of the sphercial harmonics transforms via the so-called Wigner D-matrices $$\mathbf{D}^l$$ acting on it separately.
The Wigner D-matrices are the representations of the orthogonal group O(3), the group of rotations and reflections.
In the appendix of our paper, we show that the mapping from vectors into spherical harmonics coefficients is O(3) equivariant, and further that vectors
embedded into the basis spanned by spherical harmonics are steerable by the Wigner D-matrices.


{:refdef: style="text-align: center;"}
![not found](/assets/steerable_1.png){:width="410px"}
![not found](/assets/steerable_2.png){:width="410px"}
{: refdef}

Just like the 1D Fourier basis forms a complete orthonormal basis for 1D functions, the spherical harmonics $$Y^{(l)}_m$$ form an orthonormal basis for $$\mathbb{L}_2(S^2)$$, the space of square integrable functions on the sphere $$S^2$$. Any function on the sphere $$f(\mathbf{n})$$ can thus be represented by a steerable vector when it is expressed in a spherical harmonic basis via:

$$
f(\mathbf{n}) = \sum_{l\geq 0} \sum_{m=-l}^l h_m^{(l)} Y_m^{(l)}(\mathbf{n}) \ .
\label{eq1}\tag{1}
$$

We visualize such functions on $$S^2$$ via glyph-visualizations which are obtained as surface plots and each point on this surface is color-coded with the function value $$f(\mathbf{n})$$. The visualisations are thus color-coded spheres that are stretched in each direction $$\mathbf{n}$$ via $$\lVert f(\mathbf{n})\rVert$$.

![not found](/assets/single_harmonic.png){:width="410px" style="float: right"}

$$
~ \\
~ \\
~ \\
~ \\
\left\{ \; \mathbf{n} \, |f(\mathbf{n})| \;\;\; \left| \;\;\; \mathbf{n} \in S^2 \right. \; \right\} \;\;\; \Longleftrightarrow \;\;\;
~ \\
~ \\
~ \\
~ \\
$$

### Clebsch-Gordan tensor product

In a regular MLP one maps between input and output vector spaces linearly via matrix-vector multiplication and applies non-linearities afterwards.
In steerable MLPs, one maps between steerable input and output vector spaces via the Clebsch-Gordan tensor product and applies non-linearities afterwards.
Akin to the learnable weight matrix in regular MLPs, the learnable Glebsch-Gordan tensor product is the main workhorse of our steerable MLPs.

Let $$\tilde{\mathbf{h}}^{(l)} \in V_l = \mathbb{R}^{2l+1}$$ denote a steerable vector of type $$l$$ and $$h^{(l)}_m$$ its components with $$m=-l,-l+1,\dots,l$$.
Then the $$m$$-th component of the type $$l$$ sub-vector of the output of the tensor product between two steerable vectors of type $$l_1$$ and $$l_2$$ is given by

$$
\begin{align}
(\tilde{\mathbf{h}}^{(l_1)} \otimes_{cg} \tilde{\mathbf{h}}^{(l_2)})^{(l)}_{m} = \sum_{m_1=-l_1}^{l_1} \sum_{m_2=-l_2}^{l_2} C^{(l,m)}_{(l_1,m_1)(l_2,m_2)} h^{(l_1)}_{m_1} h^{(l_2)}_{m_2} \ ,
\label{eq2}\tag{2}
\end{align}
$$

in which $$C^{(l,m)}_{(l_1,m_1)(l_2,m_2)}$$ are the Clebsch-Gordan coefficients.
The Clebsch-Gordan tensor product is a sparse tensor product, as generally many of the $$C^{(l,m)}_{(l_1,m_1)(l_2,m_2)}$$ components are zero.
The Clebsch-Gordan coefficients carry out a change of basis such that the resulting vector is again steerable relative to our standard basis of Wigner-D matrices.
Well known examples of the Clebsch-Gordan product are e.g.
* the scalar product ($$l_1=0, l_2=1, l=1$$), which takes as input a scalar and a type-1 vector to generate a type-1 vector,
* the dot product ($$l_1=1, l_2=1, l=0$$) and cross product ($$l_1=1, l_2=1, l=1$$).


### Steerable E(3) Equivariant Graph Neural Networks

Consider a graph $$\mathcal{G} =(\mathcal{V},\mathcal{E})$$, with nodes $$v_i \in \mathcal{V}$$ and edges $$e_{ij} \in \mathcal{E}$$.
An SEGNN message passing step looks as follows:

$$
\begin{align}
\text{compute message $\tilde{\textbf{m}}_{ij}$ from node $v_j$ to $v_i$:}\hspace{10mm} & \tilde{\mathbf{m}}_{ij} = \phi_m\left(\tilde{\mathbf{f}}_i, \tilde{\mathbf{f}}_j, \lVert \mathbf{x}_j - \mathbf{x}_i \rVert^2, \tilde{\mathbf{a}}_{ij}\right) \label{eq:segnn1} \ , \\
\text{aggregate messages:}\hspace{10mm} & \tilde{\mathbf{m}}_i = \sum_{j \in \mathcal{N}(i)} \tilde{\mathbf{m}}_{ij} \label{eq:segnn2} \ , \\
\text{update node features at node $v_i$:}\hspace{10mm} & \tilde{\mathbf{f}}'_i = \phi_f\left(\tilde{\mathbf{f}}_i, \tilde{\mathbf{m}}_i, \tilde{\mathbf{a}}_i \right) \ , \label{eq:segnn3}
\end{align}
$$

where $$\mathcal{N}(i)$$ represents the set of neighbours of node $$v_i$$, and $$\phi_m$$ and $$\phi_f$$ are steerable MLPs.
We now have a closer look at the message update network $$\phi_m$$:

* Every layer of $$\phi_m$$ is a Clebsch-Gordan tensor product as in equation (\ref{eq2}).
* The first input $$\tilde{\mathbf{h}}^{(l_1)}$$ of equation (\ref{eq2}) is given by the concatenation of $$\tilde{\mathbf{f}}_i, \tilde{\mathbf{f}}_j$$, and $$\lVert \mathbf{x}_j - \mathbf{x}_i \rVert^2$$.
* The second input $$\tilde{\mathbf{h}}^{(l_2)}$$ of equation (\ref{eq2}) is given by embedding the relative position $$\mathbf{x}_j - \mathbf{x}_i$$ vector via spherical harmonics as done in equation (\ref{eq1}).
* In this way we not only are able to encode vector-valued node features, but also send relative orientation information as messages.

A similar consideration applies to the node update network $$\phi_f$$ where $$\tilde{\mathbf{a}}_i$$ are the summed spherical harmonics embeddings of all relative position vectors.


Which order $$l$$ of spherical harmonics are used in each layer is a hyperparameter of the model.
In following sketches an SEGNN forward pass where TODO: what orders?

{:refdef: style="text-align: center;"}
![not found](/assets/SEGNN_forward.gif)
{: refdef}


## QM9 Experiments

One of our key insights while looking at the QM9 dataset was that using SEGNNs allows us to operate on graphs with small cutoff radii ("local" graphs).
This is mostly due to relative orientations of the different atoms which are sent as messages.
Operating on local graphs results in a sharp reduction of the number of messages per layer, as shown in the figure below.
While previous methods use relatively large cutoff radii of 4.5-11 Å, we use a cutoff radius of 2 Å.

{:refdef: style="text-align: center;"}
![not found](/assets/radius.png){:width="510px"}
{: refdef}

## OC20 challenge

The [Open Catalyst Project](https://opencatalystproject.org/index.html) of Facebook AI Research (FAIR) and Carnegie Mellon University’s (CMU) Department of Chemical Engineering
consists of molecular adsorptions onto catalyst surfaces.
The adsorbats are made up out of light atoms where 82 different adsorbats are considered.
As seen on the animation below (taken from the [Open Catalyst Project webside](https://opencatalystproject.org/index.html)) the adsorbats have rotation, translation and permutation symmetries.
SEGNNs are ideally suited for solving such tasks and perform very well. Have a look at the [Open Catalyst Project Leaderboard](https://opencatalystproject.org/leaderboard_is2re.html).

{:refdef: style="text-align: center;"}
![not found](/assets/OC20.gif)
{: refdef}


## Correspondence

This blog post was written by
