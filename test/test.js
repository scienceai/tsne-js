import assert from 'assert';
import TSNE from '../src/index';
import ndarray from 'ndarray';
import unpack from 'ndarray-unpack';

describe('TSNE', function() {
  this.timeout(10*1000);

  let model, inputData;

  before(done => {

    let sample_coo = require('./fixtures/sample_coo.json');
    const samples = 68;
    const features = 585;
    inputData = ndarray(new Float64Array(samples * features), [samples, features]);
    for (let coord of sample_coo) {
      inputData.set(coord[0], coord[1], 1);
    }
    inputData = unpack(inputData);

    model = new TSNE({
      dim: 2,
      perplexity: 30.0,
      earlyExaggeration: 2.0,
      learningRate: 120.0,
      nIter: 1000,
      tolerance: 1e-5,
      metric: 'jaccard'
    });

    done();
  });

  it('should reach reasonable local minima', done => {

    const nTrials = 10;

    let trialErrors = [];

    for (let trial = 0; trial < nTrials; trial++) {
      model.init(inputData);
      let [error, iter] = model.run();
      trialErrors.push(error);
    }

    let meanError = trialErrors.reduce((a, b) => a + b) / trialErrors.length;
    console.log(`Errors over ${nTrials} trials: ${trialErrors.map(e => e.toPrecision(3)).join(', ')}`);
    console.log(`Mean error: ${meanError.toPrecision(3)}`);

    assert(meanError < 0.8);

    done();
  });

});
