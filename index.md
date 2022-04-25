---
layout: default
title:  "Steerable Equivariant Message Passing on Molecular Graphs"
description: Blog post
date:   2021-06-08 17:03:39 +0200
usemathjax: true
---

<link rel="stylesheet" href="assets/css/accordion.css">

This is a blog post accompanying the paper [Steerable Equivariant Message Passing on Molecular Graphs](https://arxiv.org/abs/2009.14794). The goal is to give a quick overview of our method, its motivations and its consequences, maybe in a way that is more accesible than in the paper itself. This blog provides us with the space to include some extra figures, which might clarify our approach more thoroughly.


# Key insights of the paper

The paper has three main pillars:

1. The introduction of Steerable Equivariant Graph Neural Networks (**SEGNNs**) as **generalisation** of equivariant graph neural networks
2. Showing that SEGNNs are effective on **local graphs** with small cutoff radii
3. Framing of various (equivariant) message passing algorithms in a **unified convolutional form**

In this blog post we focus on point 1 and point 2. Point 3 is discussed at length in the paper.


# Credit

Before we continue, we must give credit where it's due: much of the mathematics we present here has been worked out previously. Our implementation of Steerable E(3) Equivariant Graph Neural Networks (SEGNNs) is built on the excellent [e3nn](https://docs.e3nn.org/en/stable/) software library of Mario Geiger et al., which implements all the building that blocks we will discuss shortly.


# Steerable equivariant message passing

### Motivation: deep learning on molecules

**NOTE**: the next two subsections serve as an introduction to deep learning for molecules and the concept of equivariance. Readers already familiar with these might wish to skip this section.

Deep learning methods have found their place in more and more different types of data. Graphs, point clouds, manifolds and groups all have specialised deep learning tools—with quite some succes! For this work, we wish to predict molecular properties by treating them as graphs. In order to do so, we choose treat the atoms as nodes and use a cutoff radius to define edges between them. The larger the cutoff radius, the more connections there will be.

One such molecule is shown in the figure below. The different atom sizes and colours correspond to different atoms and the edges were determined using a cutoff radius of 2Å.

{:refdef: style="text-align: center;"}
![not found](/assets/molecular_graph.png)
{: refdef}

One approach for learning on graphs is called [Message Passing](https://arxiv.org/abs/1704.01212v1). It uses a multilayer perceptron (MLP) to compute messages from neighbours, aggregates them and applies yet another MLP to update the central node. This means that the new node feature vector is a highly non-linear function of its neighbours—this is what makes this approach so powerful. Message passing networks have already been shown to match the accuracy of principled density functional theory (DFT) calculations, with far shorter inference times.

### Equivariance

As any chemist may tell you, the three-dimensional structure of a molecule is of great importance for predicting its properties. However, we should not overfit on the way the molecule is presented to us. For example, the predicting the energy should be invariant to rotations, reflections and translations. For other tasks, such as predicting the force on each atoms, the predictions should rotate as the molecule is rotated. This is the key to equivariance: as the input is transformed, the output transforms _predictably_. Formally, we can say that for any element $$g$$ in some group $$G$$, our function $$\phi$$ satisfies

$$ T'_g[\phi(x)] = \phi(T_g[x])  $$

where $$T$$ and $$T'$$ are the group action in the input space and the output space respectively. This property allows a model to generalise to all possible rotations, while only ever seeing a molecule in a single pose. As one can imagine, this greatly increases data efficiency as it removes the need for showing the molecule under every angle using data augmentation.

Approaching this idea with another perspective, we can say that an invariant function _removes_ information—the orientation can no longer be reconstructed from the output. Similarly, an equivariant function _preserves_ information, since all geometric information is preserved throughout the network. This implies that even though the output is invariant, it is beneficial to discard geometric information only at the last possible moment and have all preceding functions be equivariant.

{:refdef: style="text-align: center;"}
![not found](/assets/Equivariance.png)
{: refdef}

<!-- ## Steerable E(3) GNN
This blog is about the steerable E(3) graph neural network (SEGNN). This model generalises existing models, such as the [tensor field fetwork](https://arxiv.org/abs/1802.08219), the [SE(3) transformer](https://arxiv.org/abs/2006.10503) and especially the [E(N)GNN](https://arxiv.org/abs/2102.09844). Instead of using standard Euclidean feature vectors, the SEGNN produces features that are coefficients of steerable functions: _spherical harmonics_. Spherical harmonics are functions that live on the sphere $$S^2$$. -->

### Equivariance for vector (or tensor) valued features


In this work, we want to go one step further. We want to sustain equivariance if information at nodes or edges is vector or tensor valued. This means that as a molecule rotates, so do the vectors at each node. A good example is force prediction, where we want the three-dimensional output vector to behave equivariantly. SEGNNs achieve this by representing nodes and messages as steerable vectors as explained in the next section.




# SEGNN builing blocks
Let's take a look at how to build a Steerable E(3) GNN. For a short overview, we need:

* **Steerable feature vectors**, which are equivariant with respect to the transformation group of rotations and reflections.
* The **Clebsch-Gordan tensor product**, the workhorse of our method, analogous to the linear transform in standard MLPs.
* Including **relative positions** into the Clebsch-Gordan tensor product to build a more powerful message passing scheme.

We'll go over these one by one and try to give an intuitive explanation.


### Steerable vectors
The essence of our approach is to build graph neural networks equivariant to $$O(3)$$, the group of rotations and reflections. Equivariance to this group is easily extended to $$E(3)$$ by working only with relative distances. We are used to applying elements of $$O(3)$$ to three-dimensional Euclidean vectors, like so: $$\mathbf{x} \rightarrow \mathbf{R}\mathbf{x}$$. However, by using representations of $$O(3)$$ called [Wigher-D matrices](https://docs.e3nn.org/en/stable/api/o3/o3_irreps.html), the group can act on any $$2l+1 $$ dimensional vector space $$V_l$$, as long as this vector space consists of coefficients in a spherical harmonic basis. Any such vector will be called steerable and denoted with a tilde. For example, $$\tilde{\mathbf{h}}^{(l)}$$ is a steerable vector of order/type $$l$$.

So what exactly are spherical harmonics? Just like the 1D Fourier basis forms a complete orthonormal basis for 1D functions, the spherical harmonics $$Y^{(l)}_m$$ form an orthonormal basis for $$\mathbb{L}_2(S^2)$$, the space of square integrable functions on the sphere $$S^2$$. Any function on the sphere $$f(\mathbf{n})$$ can thus be represented by a steerable vector when it is expressed in a spherical harmonic basis via:

$$
f(\mathbf{n}) = \sum_{l\geq 0} \sum_{m=-l}^l h_m^{(l)} Y_m^{(l)}(\mathbf{n}) \ .
\label{eq1}\tag{1}
$$

We visualize such functions on $$S^2$$ via glyph-visualizations by stretching and colour-coding a sphere based on the function value $$f(\mathbf{n})$$. The figure below shows such a glyph plot.

![not found](/assets/single_harmonic.png){:width="410px" style="float: right"}

$$
~ \\
~ \\
~ \\
~ \\
\left\{ \; \mathbf{n} \, ||f(\mathbf{n})|| \;\;\; \left| \;\;\; \mathbf{n} \in S^2 \right. \; \right\} \;\;\; \Longleftrightarrow \;\;\;
~ \\
~ \\
~ \\
~ \\
$$

The next figure visualises what this looks like for our model specifically. We can embed any vector $$\mathbf{x}$$ in a spherical harmonic basis, here shown up to second order. Whenever the original vector is rotated, the coefficients transform predictably under the Wigner-D matrices.

{:refdef: style="text-align: center;"}
![not found](/assets/steerable_1.png){:width="410px"}
![not found](/assets/steerable_2.png){:width="410px"}
{: refdef}

### Clebsch-Gordan tensor product

As long as our features consist of spherical harmonics, equivariance is guaranteed. But we need to be careful to preserve this property throughout the network, when applying linear transformations and non-linear activation functions. Generally, the feature vectors consist of multiple instances of spherical harmonics of different orders. We could apply linear transformations to each type separately, but ideally the information contained in different types interacts. Fortunately, there exists a linear transform that allows us to do exactly this: the Clebsch-Gordan tensor product.

Let $$\tilde{\mathbf{h}}^{(l)} \in V_l = \mathbb{R}^{2l+1}$$ denote a steerable vector of type $$l$$ and $$h^{(l)}_m$$ its components with $$m=-l,-l+1,\dots,l$$.
Then the $$m$$-th component of the type $$l$$ sub-vector of the output of the tensor product between two steerable vectors of type $$l_1$$ and $$l_2$$ is given by

$$
\begin{align}
(\tilde{\mathbf{h}}^{(l_1)} \otimes_{cg} \tilde{\mathbf{h}}^{(l_2)})^{(l)}_{m} = \sum_{m_1=-l_1}^{l_1} \sum_{m_2=-l_2}^{l_2}  C^{(l,m)}_{(l_1,m_1)(l_2,m_2)} h^{(l_1)}_{m_1} h^{(l_2)}_{m_2} \ ,
\label{eq2}\tag{2}
\end{align}
$$

in which $$C^{(l,m)}_{(l_1,m_1)(l_2,m_2)}$$ are the [Clebsch-Gordan coefficients](https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients).
The Clebsch-Gordan (CG) tensor product  is a sparse tensor product, as generally many of the $$C^{(l,m)}_{(l_1,m_1)(l_2,m_2)}$$ components are zero. Most notably, $$C^{(l,m)}_{(l_1,m_1)(l_2,m_2)} = 0 $$ whenever $$l < |l_1 − l_2|$$ or $$l > l_1 + l_2$$. The Clebsch-Gordan coefficients carry out a change of basis such that the resulting vector is again steerable.

In fact, most of us are already familiar with some versions of the Clebsch-Gordan tensor product. For example, combining to type-1 features to form a type-0 feature is simply the dot-product. Combining two type-1 features to create another type-1 feature is the cross product.

By interleaving Clebsch-Gordan products and non-linearities, we can create MLPs for steerable feature vectors. However, the use of non-linearities is somewhat restricted. For type-0 features, one can use any pointwise non-linearity. For higher types, however, the pointwise non-linearities do not commute with the wigner-D matrices, hereby breaking equivariance. We therefore use specially designed [gated non-linearities](https://arxiv.org/pdf/1807.02547.pdf), which map features to scalar values and apply sigmoid gates, somewhat reminiscent of the Swish activation.

<!-- The essence of our approach is to build O(3) equivariant graph neural networks
where O(3) is the group of all rotations and reflections.
We do so by using the concept of steerable vectors.
Steerability of a vector $$\tilde{\mathbf{h}}$$ means that for a certain transformation group with transformation parameters $$g$$, the vector transforms via matrix-vector multiplication $$\mathbf{D}(g)$$. For example, a Euclidean vector in $$\mathbb{R}^3$$ is steerable for rotations and reflections $$g \in \text{O}(3)$$ by multiplying the vector with a rotation matrix, thus $$\mathbf{D}(g) = \mathbf{R}$$. -->

<!-- Like regular MLPs, steerable MLPs are constructed by interleaving linear mappings (matrix-vector multiplications)
with non-linearities. Now however, the linear maps transform between steerable vector spaces, for which we define how they transform under the action of a group.
This then sets an equivariance constraint on the operator that maps between these spaces. By only working with such equivariant operators we can guarantee that the entire learning framework is equivariant. -->

<!-- The figure below shows a vector $$\mathbf{x}$$ embedded into the space which is spanned by the spherical harmonics $$Y^{(l)}_m$$.
Each subspace of the sphercial harmonics transforms via the so-called Wigner D-matrices $$\mathbf{D}^l$$ acting on it separately.
The Wigner D-matrices are the representations of the orthogonal group O(3), the group of rotations and reflections.
In the appendix of our paper, we show that the mapping from vectors into spherical harmonics coefficients is O(3) equivariant, and further that vectors
embedded into the basis spanned by spherical harmonics are steerable by the Wigner D-matrices. -->


### Including relative positions

The Clebsch-Gordan product allows for the inclusion of directional information in messages and updates. Reminding ourselves that it takes two steerable vectors as input, we can concatenate the features and distance between sender and receiver to form one input $$ \tilde{\mathbf{f}}_i \oplus \tilde{\mathbf{f}}_j \oplus \lVert \mathbf{x}_j - \mathbf{x}_i \rVert^2 $$ and use a spherical harmonic embedding of the relative position $$\tilde{\mathbf{a}}_{ij}$$ as the other input. Similarly, when can update node features using the average relative position to neighbours denoted by $$\tilde{\mathbf{a}}_i$$. This endows messages with a sense of direction and endows nodes with a sense for the local geometry.

The order at which we embed the directional information can be thought of as a spatial resolution. The following animation shows the effect of adding higher order spherical harmonics to the embedding of local geometry, starting from a maximum order of zero up until third order. We consider the maximum order an important hyperparameter.

While there exist methods that can include some directional information into messages, they generally do so by computing angles between neighbours of neighbours, as is done in [DimeNet](https://arxiv.org/abs/2003.03123) and [SphereNet](https://arxiv.org/abs/2102.05013). Many other methods, such as [E(N)GNN](https://arxiv.org/abs/2102.09844), send messages that contain no directional information at all. Being able to directly incorporate this information is a contributor to SEGNNs success.

### Steerable E(3) Equivariant Graph Neural Networks

Let's see what the SEGNN actually looks like in practice. Consider a graph $$\mathcal{G} =(\mathcal{V},\mathcal{E})$$, with nodes $$v_i \in \mathcal{V}$$ and edges $$e_{ij} \in \mathcal{E}$$.
An SEGNN message passing step looks as follows:

$$
\begin{align}
\text{compute message $\tilde{\textbf{m}}_{ij}$ from node $v_j$ to $v_i$:}\hspace{10mm} & \tilde{\mathbf{m}}_{ij} = \phi_m\left(\tilde{\mathbf{f}}_i, \tilde{\mathbf{f}}_j, \lVert \mathbf{x}_j - \mathbf{x}_i \rVert^2; \tilde{\mathbf{a}}_{ij}\right) \label{eq:segnn1} \ , \\
\text{aggregate messages:}\hspace{10mm} & \tilde{\mathbf{m}}_i = \sum_{j \in \mathcal{N}(i)} \tilde{\mathbf{m}}_{ij} \label{eq:segnn2} \ , \\
\text{update node features at node $v_i$:}\hspace{10mm} & \tilde{\mathbf{f}}'_i = \phi_f\left(\tilde{\mathbf{f}}_i, \tilde{\mathbf{m}}_i; \tilde{\mathbf{a}}_i \right) \ , \label{eq:segnn3}
\end{align}
$$

where $$\mathcal{N}(i)$$ represents the set of neighbours of node $$v_i$$, and $$\phi_m$$ and $$\phi_f$$ are steerable MLPs, created by interleaving Clebsch-Gordan tensor products and non-linearities. So what does this all look like? The following animation shows the forward passes for an actual model trained to predict heat capacity. This model contains multiple spherical harmonics up until second order. We randomly choose one of each order, so we can visualise them. Just know that there's more going on than is shown in this animation!

{:refdef: style="text-align: center;"}
![not found](/assets/forward_pass_faster_larger.gif)
{: refdef}


# Experiments

### QM9 experiment

One of our key insights while looking at the QM9 dataset was that using SEGNNs allows us to operate on graphs with small cutoff radius, (local graphs). By sending directional messages, the model is able to leverage molecular structure, which many other models cannot do. Operating on local graphs results in a sharp reduction of the number of messages per layer, as shown in the figure below.
While previous methods use relatively large cutoff radii of 4.5-11 Å, we use a cutoff radius of 2 Å. The following figures show the effects of this, showing the growing number of connections on the left and on the right the mean number of messages per cutoff radius, including standard deviation over the QM9 train partition. Note that there exists a minimum cutoff radius, at the transition from disconnected to disconnected graphs.

{:refdef: style="text-align: center;"}
![not found](/assets/edges.gif){:width="380px"}
![not found](/assets/radius.png){:width="440px"}
{: refdef}


### OC20 challenge

The [Open Catalyst Project](https://opencatalystproject.org/index.html) of Facebook AI Research (FAIR) and Carnegie Mellon University’s (CMU) Department of Chemical Engineering consists of molecular adsorptions onto catalyst surfaces. We participate in the initial structure to relaxed energy (IS2RE). The catalyst consists of three layers of atoms with periodic boundaries and a small adsorbate molecule that relaxes onto the adsorbate. This is a much more challenging task than energy prediction on QM9, because additionally the equilibrium state needs to be taken into account—although we predict energy directly from the initial states, without updating positions.   

The adsorbates are made up out of light atoms where 82 different adsorbats are considered. As seen on the animation below, taken from the [Open Catalyst Project website](https://opencatalystproject.org/index.html), the adsorbates have rotation, translation and permutation symmetries. SEGNNs are ideally suited for solving such tasks and perform very well. Have a look at the [Open Catalyst Project Leaderboard](https://opencatalystproject.org/leaderboard_is2re.html). It contains both in-distribution and out-of-distribution categories, which are far more challenging.

We want to note that the SEGNN was trained on the "small" dataset, consisting of 460.000 samples. Other methods in the leaderboard use the large dataset, which has 138.000.000 samples. As far as we are able to tell, we have the leading entry for methods on this smaller dataset. Unfortunately, we do not possess the computational resources to train a model on the larger dataset. If you happen to have them, feel free to give it a try!

{:refdef: style="text-align: center;"}
![not found](/assets/OC20.gif){:width="400px"}
{: refdef}

# Conclusion
We hope that SEGNNs grow to be a viable model on many tasks, not merely on molecular data, but all forms of point clouds. We are currently working hard to publish the code so that people can easily apply SEGNNs to their problems. We believe that the effectiveness on local graphs is especially promising for larger structures, such as proteins, since models that operate on fully connected graphs might become too computationally expensive.


# Material
If after you've read this, you find yourself hungry for more, please check out our github repository where you can find the SEGNN implementation and apply it to your data.

[Github](https://github.com/RobDHess/Steerable-E3-GNN)
[Paper](https://arxiv.org/abs/2110.02905)


#### Correspondence
This blog post was written by Rob Hesselink and Johannes Brandstetter. If you have questions about our paper or the blog, please contact us at r.d.hesselink[at]uva.nl or brandstetter[at]ml.jku.at.
