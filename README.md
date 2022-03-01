# Active Feature Acquisition using Classifier Gradients

<p float="center">
  <img src="https://i.imgur.com/6HamZ2O.gif" height="336" /> 
  <img src="https://i.imgur.com/USWpdDw.gif" height="336" />
</p>
Left: unacquired features shown in blue. Acquired features shown in grayscale (darkest corresponds to pixel color black).
Right: Classifier prediction of digit [0-9].

## Overview
It is not always feasible to acquire all features immediately. The goal of this project is to explore an *active* feature selection strategy for classification, in which classification can be performed at each stage of feature acquisition. Unlike static feature selection, active feature selection works on a per-instance basis in which the next feature to acquire is conditioned on the values of already observed features. This in principle provides more flexibility than choosing the same feature set for each instance. In this project, features are selected sequentially based on *how much the <img src="https://render.githubusercontent.com/render/math?math=\ell_2">-norm of the gradient of a trained classifier is affected*. The main code is found in `afa_jax.ipynb` and consists of two parts - first, training a classifier, and then using it for feature selection. With the exception of the dataloader everything is written in JAX.

## Details

### Classifier Training

This is done using flax + optax following a very standard procedure. The model is a CNN with 2 convolutional layers and 2 fc layers, along with a dropout layer<sup>1</sup>. Cross-entropy loss is used as the objective. It is worth noting that the classifier is trained on noisy images which have certain pixels set to zero independently at random with probability p between 0.4 and 0.8. This is to accomodate classification with missing features. Classifier training is completely decoupled from feature acquisition; once the classifier is trained, its weights are then frozen.

### Feature Acquisition

For a given sample, feature acquisition is performed with regards to how much each unacquired feature would affect the <img src="https://render.githubusercontent.com/render/math?math=\ell_2">-norm of the gradient of the trained classifier, and we will refer to this as a *feature importance measure*. Although alternatives are possible, using just vanilla gradients as a proxy for feature importance has the advantage of being very efficient. This technique relies on having an initial value for each missing feature to feed the classifier. For this project, missing features are zero-imputed, but again alternatives are possible. Importantly, no ground truth labels are required as we are operating only on the classifier output. `jax.vmap` is used to speed up gradient acquisition over batches.

## Results
In the below image, we have a digit, a gradient heatmap and the classifier predictions. Initially, with few features acquired the gradients will be most sensitive around the middle of the image, which is to be expected.
<p float="center">
  <img src="https://i.imgur.com/7jCuwQo.png" height="224" /> 
</p>
After some more features are acquired, the gradients will concentrate on areas which are most likely to change the prediction. For example, after 50 features below the pixels near the top could potentially change the prediction of "6" to "8" or "2". However, the heatmaps are not always so easily interpretable. (Note that the presence of the checkerboard artifacts is due to convolutional overlap.)
<p float="center">
  <img src="https://i.imgur.com/II4RKxp.png" height="224" /> 
</p>

Although this gradient sensitivity method operates on a per-sample basis, we can also compare it to a static feature acquisition method as a baseline. The most natural one is to average gradient sensitivities of all digits across a batch and choose the most sensitive pixel overall. This turns out to actually outperform per-sample feature selection.

<p float="center">
  <img src="https://i.imgur.com/GPaSxoM.png"/> 
</p>

Nonetheless, we see that both active selection and the baseline need less than 50 features to achieve roughly 80% classification accuracy!

## Improvements

Overall, the fact that static acquisition currently works better hints that there is significant room improve this technique. The acquisition process consists of two steps - (1) imputation and (2) feature selection with a sensitivity metric. Currently zero imputation is used. To improve upon this, we may try to sample from the space of all possible digits conditioned on the revealed pixel values, using some sort of generative model like a VAE or a GAN. Feature selection can also be improved by using alternative selection strategies, of which there are many. For example, we may instead choose the feature which changes most the <img src="https://render.githubusercontent.com/render/math?math=\ell_2">-norm of the difference between the current classifier output and the new classifier output conditioned on the feature (pixel) being white. It is also possible to use techniques to generate feature relevance maps from classifier explainability literature. 
