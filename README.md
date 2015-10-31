# t-SNE.js

t-distributed stochastic neighbor embedding (t-SNE) algorithm implemented in JavaScript

### Background

t-SNE is a powerful manifold technique for embedding data into low-dimensional space (typically 2-d or 3-d for visualization purposes) while preserving small pairwise distances or local data structures in the original high-dimensional space. In practice, this results in a much more intuitive layout within the low-dimensional space as compared to other techniques. The low-dimensional embedding is learned by minimizing the Kullback-Leibler divergence between the pairwise-similarity probability distributions over the original data space and distribuions over the embedding space.

An important note is that the objective function is non-convex, and thus results are non-deterministic.

Currently implemented is the exact fomulation, which has computational complexity O(_N^2_). Implementation of the O(_N*logN_) Barnes-Hut approximation variant is planned.

<p align="center">
  <img src="http://lvdmaaten.github.io/tsne/examples/caltech101_tsne.jpg" width="400" />
</p>
[source](http://lvdmaaten.github.io/tsne/)

### Usage

Can be run in node.js or the browser. In the browser, should ideally be run in a web worker.

```
$ npm install tsne-js --save
```

```js
import TSNE from 'tsne-js';

let model = new TSNE({
  dim: 2,
  perplexity: 30.0,
  earlyExaggeration: 4.0,
  learningRate: 100.0,
  nIter: 1000,
  tolerance: 1e-5,
  metric: 'euclidean'
});

// inputData is a nested array which can be converted into an ndarray
model.init(inputData);
// use `let [error, iter] = model.run()` for final error and iteration number
model.run();
// output is unpacked ndarray (regular nested javascript array)
let output = model.getOutput();
// outputScaled is output scaled to [-1, 1]
let outputScaled = model.getOutputScaled();
```

###### Model Parameters

+ `dim`: number of embedding dimensions, typically 2 or 3

+ `perplexity`: approximately related to number of nearest neighbors used during learning, typically between 5 and 50

+ `earlyExaggeration`: parameter which influences spacing between clusters, must be at least 1.0

+ `learningRate`: learning rate for gradient descent, should be between 100 and 1000

+ `nIter`: maximum number of iterations, should be at least 200

+ `tolerance`: perplexity tolerance, default is `1e-5`

+ `metric`: distance measure to use for input data, currently implemented measures include
  + `euclidean`
  + `jaccard`

### References

The original paper on t-SNE:

```
L.J.P. van der Maaten and G.E. Hinton.
Visualizing High-Dimensional Data Using t-SNE.
Journal of Machine Learning Research 9(Nov):2579-2605, 2008.
```

Paper on Barnes-Hut variant t-SNE:

```
L.J.P. van der Maaten.
Accelerating t-SNE using Tree-Based Algorithms.
Journal of Machine Learning Research 15(Oct):3221-3245, 2014.
```

### Tests

```sh
$ npm test
```

### License

[Apache 2.0](https://github.com/scienceai/bh-tsne/blob/master/LICENSE)
