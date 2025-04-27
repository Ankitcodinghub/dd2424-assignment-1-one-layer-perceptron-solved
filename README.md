# dd2424-assignment-1-one-layer-perceptron-solved
**TO GET THIS SOLUTION VISIT:** [DD2424 Assignment 1-One-Layer-Perceptron Solved](https://www.ankitcodinghub.com/product/dd2424-assignment-1-one-layer-perceptron-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;51827&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;DD2424  Assignment 1-One-Layer-Perceptron Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
In this assignment you will train and test a one layer network with multiple outputs to classify images from the CIFAR-10 dataset. You will train the network using mini-batch gradient descent applied to a cost function that computes the cross-entropy loss of the classifier applied to the labelled training data and an <em>L</em><sub>2 </sub>regularization term on the weight matrix.

<strong>Background 1</strong>: Mathematical background

The mathematical details of the network are as follows. Given an input vector, <strong>x</strong>, of size <em>d </em>× 1 our classifier outputs a vector of probabilities, <strong>p </strong>(<em>K </em>× 1), for each possible output label:

<strong>s </strong>= <em>W</em><strong>x </strong>+ <strong>b&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </strong>(1) <strong>p </strong>= SOFTMAX(<strong>s</strong>)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (2)

where the matrix <em>W </em>has size <em>K </em>× <em>d</em>, the vector <strong>b </strong>is <em>K </em>× 1 and SOFTMAX is

defined as

<h2>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; SOFTMAX&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (3)</h2>
The predicted class corresponds to the label with the highest probability:

<em>k</em>∗ = arg max {<em>p</em><sub>1</sub><em>,…,p<sub>K</sub></em>}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (4)

1≤<em>k</em>≤<em>K</em>

<table width="0">
<tbody>
<tr>
<td width="198"></td>
<td width="14"></td>
<td width="250"></td>
</tr>
</tbody>
</table>
<h3>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; a) Classification function&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; b) Loss function</h3>
Figure 1: Computational graph of the classification and loss function that is applied to each input <strong>x </strong>in this assignment.

The parameters <em>W </em>and <strong>b </strong>of our classifier are what we have to learn by exploiting labelled training data. Let , with each <em>y<sub>i </sub></em>∈

{1<em>,…,K</em>} and <strong>x</strong><em><sub>i </sub></em>∈ R<em><sup>d</sup></em>, represent our labelled training data. In the lectures we have described how to set the parameters by minimizing the cross-entropy loss plus a regularization term on <em>W</em>. Mathematically this cost function is

cross(<strong>x</strong><em>,y,W,</em><strong>b</strong>) + <em>λ</em><sup>X</sup><em>W<sub>ij</sub></em><sup>2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </sup>(5)

<em>i,j</em>

where

<em>l</em><sub>cross</sub>(<strong>x</strong><em>,y,W,</em><strong>b</strong>) = −log(<em>p<sub>y</sub></em>)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (6)

and <strong>p </strong>has been calculated using equations (1, 2). (Note if the label is encoded by a one-hot representation then the cross-entropy loss is defined as −log(<strong>y</strong><em><sup>T </sup></em><strong>p</strong>).) The optimization problem we have to solve is

<em>W</em>∗<em>,</em><strong>b</strong>∗ = argmin <em>J</em>(D<em>,λ,W,b</em>)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (7)

<em>W,</em><strong>b</strong>

In this assignment (as described in the lectures) we will solve this optimization problem via mini-batch gradient descent.

For mini-batch gradient descent we begin with a sensible random initialization of the parameters <em>W,</em><strong>b </strong>and we then update our estimate for the parameters with

(8)

<strong>b</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (9)

where <em>η </em>is the learning rate and B<sup>(<em>t</em>+1) </sup>is called a mini-batch and is a random subset of the training data D and

<em>∂J</em>(B<sup>(<em>t</em>+1)</sup><em>,λ,W,</em><strong>b</strong>)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <sub>X&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </sub><em>∂l</em><sub>cross</sub>(<strong>x</strong><em>,y,W,</em><strong>b</strong>)

=&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (11)

<em>∂</em><strong>b&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </strong>|B(<em>t</em>+1)|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (<em>t</em>+1)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <em>∂</em><strong>b</strong>

(<strong>x</strong><em>,y</em>)∈B

To compute the relevant gradients for the mini-batch, we then have to compute the gradient of the loss w.r.t. each training example in the mini-batch. You should refer to the lecture notes for the explicit description of how to compute these gradients.

<h3>Before Starting</h3>
I assume that you will complete the assignment in <em>Matlab</em>. You can complete the assignment in another programming language. If you do though I will not answer programming specific questions and you will also probably have to find a way to display, plot and graph your results.

Besides invoking <em>Matlab </em>commands, you will be required to run a few operating system commands. For these commands I will assume your computer’s

Figure 2: Computational graph of the cost function applied to a mini-batch containing one training example <strong>x</strong>. If you have a mini-batch of size greater than one, then the loss computations are repeated for each entry in the mini-batch (as in the above graph), but the regularization term is only computed once.

operating system is either linux or unix. If otherwise, you’ll have to fend for yourself. But all the non-<em>Matlab </em>commands needed are more-or-less trivial.

The notes for this assignment, and those to follow, will give you pointers about which <em>Matlab </em>commands to use. However, I will not give detailed explanations about their usage. I assume you have some previous experience with <em>Matlab</em>, are aware of many of the in-built functions, how to manipulate vectors and matrices and how to write your own functions etc. Keep in mind the function help can be called to obtain information about particular functions. So for example

&gt;&gt; help plot will display information about the <em>Matlab </em>command plot.

<h1><strong>Background 2</strong>: Getting Started</h1>
<em>Set up your environment</em>

Create a new directory to hold all the <em>Matlab </em>files you will write for this course:

$ mkdir DirName

$ cd DirName

$ mkdir Datasets

$ mkdir Result Pics

Download the CIFAR-10 dataset stored in its <em>Matlab </em>format from <a href="https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz">this link</a><a href="https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz">.</a>

Move the cifar-10-matlab.tar.gz file to the Datasets directory you have just created, untar the file and then move up to the parent directory. Also download the file montage.m from the Canvas webpage for Assignment 1 and move it to DirName.

$ mv montage.m DirName/

$ mv cifar-10-matlab.tar.gz DirName/Datasets

$ cd DirName/Datasets $ tar xvfz cifar-10-matlab.tar.gz $ cd ..

<h1><strong>Background 3</strong>: Useful Display Function</h1>
You have copied a function called montage.m, which is a slightly modified version of the function available at: <a href="http://www.mathworks.com/matlabcentral/fileexchange/22387">http://www.mathworks.com/matlabcentral/fileexchange/22387</a>

This is a useful function as it allows you to efficiently view the images in a directory or in a <em>Matlab </em>array or a cell array. To look at some of the images from the CIFAR-10 dataset you can use the following set of commands:

&gt;&gt; addpath DirName/Datasets/cifar-10-batches-mat/;

&gt;&gt; A = load(’data batch 1.mat’);

&gt;&gt; I = reshape(A.data’, 32, 32, 3, 10000);

&gt;&gt; I = permute(I, [2, 1, 3, 4]);

&gt;&gt; montage(I(:, :, :, 1:500), ’Size’, [5,5]);

This sequence of commands tells <em>Matlab </em>to add the directory of the CIFAR-10 dataset to its path. Then it loads one of the .mat files containing image and label data. You access the image data with the command A.data. It has size 10000 × 3072. Each row of A.data corresponds to an image of size 32 × 32 × 3 that has been flattened into a row vector. We can re-arrange A.data into an array format expected by montage (32×32×3×10000) using the command reshape. After reshaping the array, the rows and columns of each image still need to be permuted and this is achieved with the permute command. Now you have an array format montage expects. In the above code we just view the first 500 images. Use help to find out the different ways montage can be called.

You have looked at some of the CIFAR-10 images. Now it is time to start writing some code.

<h1><strong>Exercise 1</strong>: Training a multi-linear classifier</h1>
For this assignment you will just use data in the file data batch 1.mat for training, the file data batch 2.mat for validation and the file test batch.mat for testing. Create a file Assignment1.m. In this file you will write the code for this assignment and the necessary (sub-)functions. Here are my recommendations for which functions to write and the order in which to write them:

<ol>
<li>Write a function that reads in the data from a CIFAR-10 batch file and returns the image and label data in separate files. Make sure to convert your image data to single or double I would suggest the function has the following input and outputs function [X, Y, y] = LoadBatch(filename)</li>
</ol>
where

<ul>
<li>X contains the image pixel data, has size d×n, is of type double or single and has entries between 0 and 1. n is the number of images (10000) and d the dimensionality of each image (3072=32×32×3).</li>
<li>Y is K×n (K= # of labels = 10) and contains the one-hot representation of the label for each image.</li>
<li>y is a vector of length <em>n </em>containing the label for each image. A note of caution. CIFAR-10 encodes the labels as integers between 0-9 but <em>Matlab </em>indexes matrices and vectors starting at 1. Therefore it may be easier to encode the labels between 1-10.</li>
</ul>
This file will not be long. You just need to call A = load(fname); and then rearrange and store the values in A.data and A.labels.

<strong>Top-level</strong>: Read in and store the training, validation and test data.

<ol start="2">
<li>Next we should pre-process the raw input data as it helps training. You should transform training data to have zero mean. If trainX is the d×n image data matrix (each column corresponds to an image) for the training data then</li>
</ol>
mean X = mean(trainX, 2); std X = std(trainX, 0, 2);

Both mean X and std X have size d×1.

You should normalize the training, validation and test data with respect to the mean and standard deviation values computed from the training data as follows. If X is an d×n image data matrix then you can normalize X as

X = X – repmat(meanX, [1, size(X, 2)]);

X = X ./ repmat(stdX, [1, size(X, 2)]);

<strong>Top-level</strong>: Compute the mean and standard deviation vector for the training data and then normalize the training, validation and test data w.r.t. these mean and standard deviation vectors.

<ol start="3">
<li><strong>Top-Level</strong>: After reading in and pre-processing the data, you can initialize the parameters of the model W and b as you now know what size they should be. W has size K×d and b is K×1. Initialize each entry to have Gaussian random values with zero mean and standard deviation .01. You should use the Matlab function randn to create this data.</li>
<li>Write a function that evaluates the network function, i.e. equations (1, 2), on multiple images and returns the results. I would suggest the function has the following form function P = EvaluateClassifier(X, W, b)</li>
</ol>
where

<ul>
<li>each column of X corresponds to an image and it has size d×n.</li>
<li>W and b are the parameters of the network.</li>
<li>each column of P contains the probability for each label for the image in the corresponding column of X. P has size K×n.</li>
</ul>
<strong>Top-level</strong>: Check the function runs on a subset of the training data given a random initialization of the network’s parameters:

P = EvaluateClassifier(trainX(:, 1:100), W, b).

<ol start="5">
<li>Write the function that computes the cost function given by equation (5) for a set of images. I suggest the function has the following inputs and outputs function J = ComputeCost(X, Y, W, b, lambda)</li>
</ol>
where

<ul>
<li>each column of X corresponds to an image and X has size d×n.</li>
<li>each column of Y (K×n) is the one-hot ground truth label for the corresponding column of X or Y is the (1×n) vector of ground truth labels.</li>
<li>J is a scalar corresponding to the sum of the loss of the network’s predictions for the images in X relative to the ground truth labels and the regularization term on W.</li>
</ul>
<ol start="6">
<li>Write a function that computes the accuracy of the network’s predictions given by equation (4) on a set of data. Remember the accuracy of a classifier for a given set of examples is the percentage of examples for which it gets the correct answer. I suggest the function has the following inputs and outputs function acc = ComputeAccuracy(X, y, W, b) where
<ul>
<li>each column of X corresponds to an image and X has size d×n.</li>
<li>y is the vector of ground truth labels of length n.</li>
<li>acc is a scalar value containing the accuracy.</li>
</ul>
</li>
<li>Write the function that evaluates, for a mini-batch, the gradients of the cost function w.r.t. W and b, that is equations (10, 11). I suggest the function has the form</li>
</ol>
function [grad W, grad b] = ComputeGradients(X, Y, P, W, lambda)

where

<ul>
<li>each column of X corresponds to an image and it has size d×n.</li>
<li>each column of Y (K×n) is the one-hot ground truth label for the corresponding column of X.</li>
<li>each column of P contains the probability for each label for the image in the corresponding column of X. P has size K×n.</li>
<li>grad W is the gradient matrix of the cost <em>J </em>relative to W and has size K×d.</li>
<li>grad b is the gradient vector of the cost <em>J </em>relative to b and has size K×1.</li>
</ul>
Be sure to check out how you can efficiently compute the gradient for a batch from the last slide of Lecture 3. This can lead to a much faster implementation (<em>&gt; </em>3 times faster) than looping through each training example in the batch.

Everyone makes mistakes when computing gradients. Therefore you must always check your analytic gradient computations against numerical estimations of the gradients! Download code from the Canvas webpage that computes the gradient vectors numerically. Note there are two versions <strong>1</strong>) a slower but more accurate version based on the <em>centered difference </em>formula and <strong>2</strong>) a faster but less accurate based on the <em>finite difference method</em>. You will probably have to make small changes to the functions to make them compatible with your code. It will take some time to run the numerical gradient calculations as your network has 32×32×3×10 different parameters in <em>W</em>. Initially, you should just perform your checks on mini-batches of size 1 and with no regularization (lambda=0). Afterwards you can increase the size of the mini-batch and include regularization into the computations. Another trick is that you should send in a reduced dimensionality version of trainX and W, so instead of

[ngrad b, ngrad W] = ComputeGradsNumSlow(trainX(:, 1), trainY(:, 1),

W, b, lambda, 1e-6); you can compute the gradients numerically for smaller inputs (the first 20 dimensions of the first training example)

[ngrad b, ngrad W] = ComputeGradsNumSlow(trainX(1:20, 1), trainY(:, 1),

W(:, 1:20), b, lambda, 1e-6);

You should then also compute your analytical gradients on this reduced version of the input data with reduced dimensionality. This will speed up computations and also reduce the risk of numerical precision issues (very possible when the full W is initialized with very small numbers and trainX also contains small numbers).

You could compare the numerically and analytically computed gradient vectors (matrices) by examining their absolute differences and declaring, if all these absolutes difference are small (<em>&lt;</em>1e-6), then they have produced the same result. However, when the gradient vectors have small values this approach may fail. A more reliable method is to compute the relative error between a numerically computed gradient value <em>g<sub>n </sub></em>and an analytically computed gradient value <em>g<sub>a</sub></em>

where eps a very small positive number max(eps

and check this is small. There are potentially more issues that can plague numerical gradient checking (especially when you start to train deep rectifier networks), so I suggest you read the relevant section of the <a href="http://cs231n.github.io/neural-networks-3/">Additional material for lecture</a> <a href="http://cs231n.github.io/neural-networks-3/">3</a> from Standford’s course <strong>Convolutional Neural Networks for Visual Recognition </strong>for a more thorough exposition especially for the subsequent assignments.

Do not continue with the rest of this assignment until you are sure your analytic gradient code is correct. If you are having problems, set the seed of the random number generator with the command rng to ensure at each test W and b have the same values and double/triple check that you have a correct implementation of the gradient equations from the lecture notes.

<ol start="8">
<li>Once you have the gradient computations debugged you are now ready to write the code to perform the mini-batch gradient descent algorithm to learn the network’s parameters where the updates are defined in equations (8, 9). You have a couple of parameters controlling the learning algorithm (for this assignment you will just implement the most vanilla version of the mini-batch gradient descent algorithm, with no adaptive tuning of the learning rate or momentum terms):</li>
</ol>
<ul>
<li>n batch the size of the mini-batches</li>
<li>eta the learning rate</li>
<li>n epochs the number of runs through the whole training set.</li>
</ul>
As the images in the CIFAR-10 dataset are in random order, the easiest to generate each mini-batch is to just run through the images sequentially. Let n batch be the number of images in a mini-batch. Then for one epoch (a complete run through all the training images), you can generate the set of mini-batches with this snippet of code: for j=1:n/n batch j start = (j-1)*n batch + 1; j end = j*n batch; inds = j start:j end; Xbatch = Xtrain(:, j start:j end);

Ybatch = Ytrain(:, j start:j end); end

(A slight upgrade of this default implementation is to randomly shuffle your training examples before each epoch. One efficient way to do this is via the command randperm which when given the input n returns a vector containing a random permutation of the integers 1:n.) I suggest the mini-batch learning function has these inputs and outputs

function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)

where X contains all the training images, Y the labels for the training images, W, b are the initial values for the network’s parameters, lambda is the regularization factor in the cost function and

<ul>
<li>GDparams is an object containing the parameter values n batch, eta and n epochs</li>
</ul>
For my initial experiments I set n batch=100, eta=.001, n epochs=20 and lambda=0. To help you debug I suggest that after each epoch you compute the cost function and print it out (and save it) on all the training data. For these parameter settings you should see that the training cost decreases for each epoch. After the first epoch my cost score on all the training data was 1.981428 where I had set the random number seed generator to rng(400) and I had initialized the weight matrix before the bias vector. In figure 8 you can see the training cost score when I run these parameter settings for 40 epochs. The cost score on the validation set is plotted in red in the same figure.

(Note: in Tensorflow and other software packages they count in the number of update steps as opposed to the number of training epochs. Given n training images and mini-batches of size n batch then one epoch will result in n/n batch update steps of the parameter values. Obviously, the performance of the resulting network depends on the number of update steps and how much training data is used. Therefore to make fair comparisons one should make sure the number of update steps is consistent across runs – for instance if you change the size of the mini-batch, number of training images, etc.. you may need to run more or fewer epochs of training.)

When you have finished training you can compute the accuracy of your learnt classifier on the test data. My network achieves (after 40

10&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 20&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 30&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 40

Figure 3: The graph of the training and validation loss computed after every epoch. The network was trained with the following parameter settings: n batch=100, eta=.001, n epochs=40 and lambda=0.

epochs) an accuracy of 38.83% (with a random shuffling of the training example at the beginning of each epoch). Much better than random but not great. Hopefully, we will achieve improved accuracies in the assignments to come when we build and train more complex networks.

After training you can also visualization the weight matrix W as an image and see what <em>class templates </em>your network has learnt. Figure 4 shows the templates my network learnt. Here is a code snippet to re-arrange each row of W (assuming W is a 10×d matrix) into a set of images that can be displayed by <em>Matlab</em>.

for i=1:10

im = reshape(W(i, :), 32, 32, 3); s im{i} = (im – min(im(:))) / (max(im(:)) – min(im(:))); s im{i} = permute(s im{i}, [2, 1, 3]);

end

Then you can use either imshow, imagesc or montage to display the images in the cell array s im.

Figure 4: The learnt W matrix visualized as class template images. The network was trained with the following parameter settings: n batch=100, eta=.001, n epochs=40 and lambda=0.
